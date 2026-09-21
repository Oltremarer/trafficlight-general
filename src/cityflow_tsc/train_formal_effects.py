"""Approved v3 staged training: six A jobs, three B1 jobs and three separate R heads."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_model import (Ranker, create_model, group_mean, multiscale_loss,
                                        ranking_features, root_metadata, tensor_scales, validation)
from .train_effect_world_model import learning_rate, save_checkpoint

_RUNTIME_INITIALIZED = False
SEEDS = (42, 43, 44)
VARIANTS = ("A_ref", "A_MS")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def initialize(seed):
    global _RUNTIME_INITIALIZED
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    if not _RUNTIME_INITIALIZED:
        torch.set_num_interop_threads(1)
        _RUNTIME_INITIALIZED = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _static(run):
    # Loading metadata-only static never opens held-out targets.
    with np.load(Path(run) / "derived/static.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_a(run, seed, device, variant=None):
    run = Path(run)
    selection = json.loads((run / "selected_A.json").read_text())
    variant = selection["variant"] if variant is None else variant
    if variant not in VARIANTS:
        raise ValueError("Unknown A variant")
    path = run / variant / f"seed_{seed}" / "best.pt"
    saved = torch.load(path, map_location=device, weights_only=False)
    model = create_model(_static(run)).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, {**{k: v for k, v in saved.items() if k != "state_dict"},
                   "checkpoint": str(path), "sha256": sha256(path)}


def load_b(run, seed, device):
    path = Path(run) / "B1" / f"seed_{seed}" / "best.pt"
    saved = torch.load(path, map_location=device, weights_only=False)
    model = create_model(_static(run)).to(device)
    model.start_pair_stage()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, {**{k: v for k, v in saved.items() if k != "state_dict"},
                   "checkpoint": str(path), "sha256": sha256(path)}


def load_ranker(run, seed, device):
    path = Path(run) / "ranker" / f"seed_{seed}" / "best.pt"
    saved = torch.load(path, map_location=device, weights_only=False)
    model = Ranker().to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, {**{k: v for k, v in saved.items() if k != "state_dict"},
                   "checkpoint": str(path), "sha256": sha256(path)}


def optimizer_for(parameters):
    return torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4,
                            betas=(.9, .999), eps=1e-8)


def write_epoch(directory, record):
    with (directory / "epochs.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    atomic_json(directory / "status.json", {"stage": "training", "pid": os.getpid(), **record})


def check_split(roots):
    counts = {s: sum(r["split"] == s for r in roots) for s in ("train", "validation")}
    if counts != {"train": 36, "validation": 12} or any(r["split"] == "test" for r in roots):
        raise ValueError("Training requires exactly 36 training and 12 validation roots; no test labels")
    for split in counts:
        subset = [r for r in roots if r["split"] == split]
        if len({r["cohort_id"] for r in subset}) != 3:
            raise ValueError("Expected three base cohorts in " + split)
    for root in roots:
        if root["single"].shape != (64, 272, 48) or root["joint"].shape != (65, 272, 48):
            raise ValueError("Formal factor and reference-first candidate shape mismatch")
        if np.any(root["s_incidence"][0]) or np.any(root["joint"][0]):
            raise ValueError("Candidate zero must be exact reference")


def cached_encodings(model, roots, device):
    result = []
    model.eval()
    with torch.no_grad():
        for root in roots:
            result.append(model.encode(torch.tensor(root["normalized_history"][None], device=device),
                                       torch.tensor(root["action_bank"][None], device=device)))
    return {key: torch.cat([row[key] for row in result]).detach() for key in result[0]}


def select_checkpoint(directory, model, metrics, epoch, variant, seed, best, mae_best):
    regret, mae = metrics["regret"], metrics["joint_mae"]
    if not np.isfinite([regret, mae]).all():
        raise FloatingPointError("Nonfinite validation metric")
    key = (float(regret), float(mae), int(epoch))
    metadata = {"variant": variant, "seed": seed, "epoch": epoch,
                "validation_regret": regret, "validation_joint_mae": mae,
                "selection": "equal cohort regret, equal cohort joint MAE, earlier epoch",
                "horizon_s": 240, "test_used": False}
    if key < best:
        best = key
        save_checkpoint(directory / "best.pt", model, metadata)
        atomic_json(directory / "best_validation.json", {**metadata, "metrics": metrics})
    if (mae, epoch) < mae_best:
        mae_best = (mae, epoch)
        save_checkpoint(directory / "mae_best_diagnostic.pt", model, metadata)
    return best, mae_best


def train_factor(run, directory, variant, seed):
    from .effect_model.formal_data import load_data
    initialize(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Approved formal GPU training requires CUDA; no silent CPU fallback")
    device = torch.device("cuda:0")
    pair_stage = variant == "B1"
    started = time.perf_counter()
    roots, static, stats = load_data(run, include_pairs=pair_stage, splits=("train", "validation"))
    check_split(roots)
    training = [r for r in roots if r["split"] == "train"]
    validation_roots = [r for r in roots if r["split"] == "validation"]
    if pair_stage:
        model, source_metadata = load_a(run, seed, device)
        model.start_pair_stage()
    else:
        model, source_metadata = create_model(static).to(device), None
        for parameter in model.pair_head.parameters():
            parameter.requires_grad_(False)
    kind = "pair" if pair_stage else "single"
    per_root, batch, epochs = (1920, 64, 100) if pair_stage else (64, 32, 300)
    count = len(training) * per_root
    if count != (69120 if pair_stage else 2304):
        raise ValueError("Full approved factor query count changed")
    scales = tensor_scales(stats, kind, device)
    nodes = torch.tensor(np.concatenate([r[kind + "_nodes"] for r in training]), dtype=torch.long, device=device)
    targets = torch.tensor(np.concatenate([r[kind + "_actions"] for r in training]), dtype=torch.long, device=device)
    base = torch.tensor(np.stack([r["base_phase"] for r in training]), dtype=torch.long, device=device)
    histories = torch.tensor(np.stack([r["normalized_history"] for r in training]), device=device)
    actions = torch.tensor(np.stack([r["action_bank"] for r in training]), device=device)
    cached = cached_encodings(model, training, device) if pair_stage else None
    # Singles fit in GPU memory; pair labels stay file-backed and only batches move.
    single_labels = None if pair_stage else torch.tensor(np.concatenate([r["single"] for r in training]), device=device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_for(parameters)
    generator = np.random.default_rng(seed)
    best, mae_best = (float("inf"),) * 3, (float("inf"), float("inf"))
    if pair_stage:
        metrics = validation(model, validation_roots, stats, device, include_pairs=True)
        best, mae_best = select_checkpoint(directory, model, metrics, 0, variant, seed, best, mae_best)
        write_epoch(directory, {"epoch": 0, "updates": 0, "validation_regret": metrics["regret"],
                                "validation_joint_mae": metrics["joint_mae"], "seed": seed, "variant": variant})
    trained_at = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        rate = learning_rate(epoch, epochs)
        for group in optimizer.param_groups:
            group["lr"] = rate
        order, losses = generator.permutation(count), []
        for offset in range(0, count, batch):
            ix_np = order[offset:offset + batch]
            ix = torch.tensor(ix_np, device=device)
            root_ids = ix // per_root
            optimizer.zero_grad(set_to_none=True)
            if pair_stage:
                labels = torch.tensor(np.stack([training[int(q // per_root)]["pair"][int(q % per_root)]
                                                for q in ix_np]), device=device)
                prediction = model.pair(cached, root_ids, nodes[ix], targets[ix], base)
                logits = None
            else:
                unique, inverse = torch.unique(root_ids, return_inverse=True)
                encoded = model.encode(histories[unique], actions[unique])
                prediction, logits = model.single(encoded, inverse, nodes[ix, 0], targets[ix, 0],
                                                  base[unique], return_gate=True)
                labels = single_labels[ix]
            loss = multiscale_loss(prediction, labels, scales, variant, logits)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {variant}/{seed}/{epoch} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "updates": epoch * ((count + batch - 1) // batch),
                  "loss": float(np.mean(losses)), "lr": rate, "variant": variant, "seed": seed,
                  "elapsed_s": time.perf_counter() - trained_at}
        if epoch % 5 == 0:
            metrics = validation(model, validation_roots, stats, device, include_pairs=pair_stage)
            best, mae_best = select_checkpoint(directory, model, metrics, epoch, variant, seed, best, mae_best)
            record.update(validation_regret=metrics["regret"], validation_joint_mae=metrics["joint_mae"])
        record["selected_epoch"] = int(best[2]) if np.isfinite(best[2]) else None
        write_epoch(directory, record)
    result = {"variant": variant, "seed": seed, "stage": "complete", "epochs_run": epochs,
              "updates": epochs * ((count + batch - 1) // batch), "queries_per_epoch": count,
              "training_wall_s": time.perf_counter() - trained_at, "total_wall_s": time.perf_counter() - started,
              "selected_epoch": int(best[2]), "validation_regret": best[0], "validation_joint_mae": best[1],
              "mae_best_diagnostic_epoch": int(mae_best[1]), "source_A": source_metadata,
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in parameters),
              "checkpoint_sha256": sha256(directory / "best.pt"), "finished_at": now(),
              "test_used": False, "timing": "Concurrent training, not isolated inference"}
    atomic_json(directory / "status.json", result)
    return result


def train_ranker(run, directory, seed):
    from .effect_model.formal_data import load_data
    initialize(seed)
    device = torch.device("cuda:0")
    start = time.perf_counter()
    roots, _, stats = load_data(run, include_pairs=True, splits=("train", "validation"))
    check_split(roots)
    a, source_metadata = load_a(run, seed, device)
    for p in a.parameters():
        p.requires_grad_(False)
    encoded = cached_encodings(a, roots, device)
    features = ranking_features(a, encoded).detach()
    labels = torch.tensor(np.log1p(np.stack([r["rank_targets"] for r in roots]) / float(stats["rank_scale"])),
                          dtype=torch.float32, device=device).reshape(len(roots), 120)
    train_ix = [i for i, r in enumerate(roots) if r["split"] == "train"]
    val_ix = [i for i, r in enumerate(roots) if r["split"] == "validation"]
    x, y = features[train_ix].reshape(-1, 197), labels[train_ix].reshape(-1)
    vx, vy = features[val_ix], labels[val_ix]
    if len(x) != 4320:
        raise ValueError("Expected 4320 rank training pairs")
    ranker = Ranker().to(device)
    parameters = list(ranker.parameters())
    optimizer = optimizer_for(parameters)
    generator = np.random.default_rng(seed)
    best = (float("inf"), float("inf"))
    trained_at = time.perf_counter()
    for epoch in range(1, 101):
        ranker.train()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(epoch, 100)
        losses = []
        order = generator.permutation(len(x))
        for offset in range(0, len(x), 64):
            ix = torch.tensor(order[offset:offset + 64], device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = ranker(x[ix]).reshape(-1)
            loss = torch.nn.functional.smooth_l1_loss(prediction, y[ix], beta=1.)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite ranker loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "updates": epoch * 68, "loss": float(np.mean(losses)),
                  "seed": seed, "elapsed_s": time.perf_counter() - trained_at}
        if epoch % 5 == 0:
            ranker.eval()
            with torch.no_grad():
                error = torch.nn.functional.smooth_l1_loss(ranker(vx).squeeze(-1), vy,
                                                           beta=1., reduction="none").mean(1).cpu().numpy()
            rows = [{**root_metadata(roots[i]), "loss": float(value)} for i, value in zip(val_ix, error)]
            score = group_mean(rows, "loss")
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite rank validation loss")
            record["validation_log_strength_smooth_l1"] = score
            if (score, epoch) < best:
                best = (score, epoch)
                save_checkpoint(directory / "best.pt", ranker,
                                {"seed": seed, "epoch": epoch, "validation_loss": score,
                                 "rank_scale": float(stats["rank_scale"]), "source_A": source_metadata,
                                 "selection": "validation log-strength SmoothL1, earlier epoch", "test_used": False})
        record["selected_epoch"] = int(best[1]) if np.isfinite(best[1]) else None
        write_epoch(directory, record)
    result = {"seed": seed, "stage": "complete", "epochs_run": 100, "updates": 6800,
              "parameters": sum(p.numel() for p in parameters), "selected_epoch": int(best[1]),
              "validation_loss": best[0], "training_wall_s": time.perf_counter() - trained_at,
              "total_wall_s": time.perf_counter() - start, "checkpoint_sha256": sha256(directory / "best.pt"),
              "independent_optimizer": True, "test_used": False, "finished_at": now()}
    atomic_json(directory / "status.json", result)
    return result


def committed_job(run, variant, seed):
    import fcntl
    run = Path(run)
    directory = run / variant / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "job.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "complete.json").exists():
            result = json.loads((directory / "complete.json").read_text())
            if result["stage"] != "complete" or result["checkpoint_sha256"] != sha256(directory / "best.pt"):
                raise RuntimeError("Committed training checkpoint missing or altered")
            return result
        if (directory / "started.json").exists():
            raise RuntimeError("Incomplete attempt retained; no blind restart: " + str(directory))
        atomic_json(directory / "started.json", {"variant": variant, "seed": seed, "pid": os.getpid(), "started_at": now()})
        try:
            result = train_ranker(run, directory, seed) if variant == "ranker" else train_factor(run, directory, variant, seed)
            atomic_json(directory / "complete.json", result)
            return result
        except Exception as exc:
            atomic_json(directory / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            raise


def run_b_job(run, seed):
    b = committed_job(run, "B1", seed)
    ranker = committed_job(run, "ranker", seed)
    return {**b, "ranker": ranker}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("A", "B"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not Path("/mnt/pan").is_mount() or not run.is_relative_to(Path("/mnt/pan")):
        raise ValueError("Mounted /mnt/pan run is required")
    if not 1 <= args.workers <= 3:
        raise ValueError("At most three concurrent GPU training jobs")
    from .effect_model.formal_data import prepare_dataset
    directory = run / ("training_" + args.stage)
    directory.mkdir(exist_ok=True)
    import fcntl
    with (directory / "pool.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "summary.json").exists():
            prior = json.loads((directory / "summary.json").read_text())
            if prior.get("stage") == "complete":
                return
            raise RuntimeError("Existing failed stage retained, requires explicit recovery")
        if args.stage == "B" and not (run / "selected_A.json").exists():
            raise RuntimeError("Unified A must be locked before B")
        prepare_dataset(run, include_pairs=args.stage == "B", splits=("train", "validation"))
        jobs = [(v, s) for s in SEEDS for v in VARIANTS] if args.stage == "A" else [("B1", s) for s in SEEDS]
        atomic_json(directory / "config.json", {"stage": args.stage, "jobs": jobs, "workers_max": args.workers,
                    "cpu_threads": 2, "interop_threads": 1, "FP32": True, "AMP": False, "TF32": False,
                    "test_used": False, "started_at": now(), "source_sha256": sha256(Path(__file__))})
        results, errors = [], []
        atomic_json(directory / "queue.json", {"completed": 0, "total": len(jobs), "errors": [], "pid": os.getpid()})
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(committed_job, str(run), variant, seed) if args.stage == "A" else
                       pool.submit(run_b_job, str(run), seed): (variant, seed) for variant, seed in jobs}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({"job": futures[future], "error": repr(exc)})
                atomic_json(directory / "queue.json", {"completed": len(results), "total": len(jobs), "errors": errors})
        if not errors and len(results) == len(jobs):
            if args.stage == "A":
                scores = {variant: [float(np.mean([r[key] for r in results if r["variant"] == variant]))
                                    for key in ("validation_regret", "validation_joint_mae")] for variant in VARIANTS}
                selected = min(VARIANTS, key=lambda v: (*scores[v], VARIANTS.index(v)))
                atomic_json(run / "selected_A.json", {"variant": selected, "validation_scores": scores,
                            "test_used": False, "locked_at": now(), "uniform_for_all_seeds": True})
            else:
                artifacts = [run / family / f"seed_{seed}" / "best.pt" for seed in SEEDS
                             for family in (*VARIANTS, "B1", "ranker")]
                atomic_json(run / "checkpoints_locked.json", {"locked_at": now(), "test_used": False,
                            "checkpoints": {str(p.relative_to(run)): sha256(p) for p in artifacts},
                            "selected_A_sha256": sha256(run / "selected_A.json"),
                            "normalization_sha256": sha256(run / "derived/normalization.npz"),
                            "pair_normalization_sha256": sha256(run / "derived/pair_normalization.npz")})
        atomic_json(directory / "summary.json", {"stage": "failed" if errors else "complete", "results": results,
                    "errors": errors, "finished_at": now(), "test_used": False})
        if errors:
            raise RuntimeError("Formal training stage had failed jobs; see summary.json")


if __name__ == "__main__":
    main()
