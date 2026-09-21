"""Nine approved old-data loss experiments. No recollection or implicit tuning."""
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
from .effect_model.data import normalize_history
from .effect_model.evaluation import error_stats, joint_metrics
from .effect_model.revision import RevisedEffectModel, checkpoint_key, revision_loss
from .train_effect_world_model import learning_rate, save_checkpoint

_RUNTIME_INITIALIZED = False


def load_old(source):
    manifest = json.loads((source / "split_and_data.json").read_text())
    with np.load(source / "data/static.npz") as z:
        static = {k: z[k] for k in z.files}
    with np.load(source / "data/normalization.npz") as z:
        stats = {"mean": z["mean"], "std": z["std"]}
        scale, total = z["output_scales"][0], z["total_scales"][0]
    roots = []
    for info in manifest["roots"]:
        # Read only single/joint training tensors, not thousands of raw physical rollouts or pair labels.
        with np.load(source / "data" / (info["root_id"] + ".npz")) as z:
            row = {k: z[k] for k in ("history", "action_bank", "base_phase", "single", "joint",
                                     "single_nodes", "single_actions", "s_incidence", "joint_sizes", "joint_ids")}
        row["normalized_history"] = normalize_history(row["history"], stats)
        roots.append({**info, **row})
    if {s: sum(r["split"] == s for r in roots) for s in ("train", "validation", "development_holdout")} != {
            "train": 8, "validation": 4, "development_holdout": 6}:
        raise ValueError("Old development split must remain 8/4/6")
    return roots, static, scale, total


@torch.no_grad()
def predict(model, root, scale, device):
    model.eval()
    h = torch.as_tensor(root["normalized_history"][None], device=device)
    a = torch.as_tensor(root["action_bank"][None], device=device)
    base = torch.as_tensor(root["base_phase"][None], device=device)
    encoded = model.encode(h, a)
    nodes = torch.as_tensor(root["single_nodes"][:, 0], device=device)
    targets = torch.as_tensor(root["single_actions"][:, 0], device=device)
    factors = model.single(encoded, torch.zeros_like(nodes), nodes, targets, base) * scale
    joint = (torch.as_tensor(root["s_incidence"], device=device) @ factors.flatten(1)).reshape(
        -1, 272, model.windows)
    return factors, joint


@torch.no_grad()
def validation(model, roots, scale, device):
    errors, regrets = [], []
    for root in roots:
        _, prediction = predict(model, root, scale, device)
        truth = torch.as_tensor(root["joint"], device=device)
        choice = int(prediction.sum((1, 2), dtype=torch.float64).argmin())
        scores = truth.sum((1, 2), dtype=torch.float64)
        regrets.append(float(scores[choice] - scores.min()))
        errors.append(float((prediction - truth).abs().mean(dtype=torch.float64)))
    return float(np.mean(regrets)), float(np.mean(errors))


def detailed_error(prediction, truth):
    stats = error_stats(prediction, truth)
    mask = truth != 0
    stats["nonzero_sign_accuracy"] = float((np.sign(prediction[mask]) == np.sign(truth[mask])).mean()) if mask.any() else None
    delta = prediction.sum((1, 2), dtype=np.float64) - truth.sum((1, 2), dtype=np.float64)
    stats.update(total_wait_change_mae=float(np.abs(delta).mean()), total_wait_change_bias=float(delta.mean()),
                 prediction_exact_zero_fraction=float((prediction == 0).mean()))
    return stats


def evaluate(model, roots, scale, device):
    records = []
    for root in roots:
        single, joint = predict(model, root, scale, device)
        single, joint = single.cpu().numpy(), joint.cpu().numpy()
        record = {k: root[k] for k in ("root_id", "flow_id", "policy", "time_s", "split")}
        record.update(single=detailed_error(single, root["single"]), joint=joint_metrics(joint, root),
                      joint_detailed=detailed_error(joint, root["joint"]))
        # Additional cumulative horizons preserve 180-second old-data semantics.
        for horizon in (90,):
            record["joint"]["horizons"][str(horizon)] = error_stats(
                joint[:, :, :horizon // 5].sum(2, dtype=np.float64),
                root["joint"][:, :, :horizon // 5].sum(2, dtype=np.float64))
        records.append(record)
    return records


def run_job(run_dir, source_dir, variant, seed):
    run, source = Path(run_dir), Path(source_dir)
    directory = run / "old_training" / f"{variant}_seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (directory / "job.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "complete.json").exists():
            return json.loads((directory / "complete.json").read_text())
        if (directory / "started.json").exists():
            raise RuntimeError("Incomplete prior attempt retained; no blind automatic restart")
        atomic_json(directory / "started.json", {"pid": os.getpid(), "variant": variant, "seed": seed,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            result = train(directory, source, variant, seed)
            atomic_json(directory / "complete.json", result)
            return result
        except Exception as exc:
            atomic_json(directory / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            raise


def train(directory, source, variant, seed):
    global _RUNTIME_INITIALIZED
    start = time.perf_counter()
    torch.set_num_threads(2)
    if not _RUNTIME_INITIALIZED:
        torch.set_num_interop_threads(1)
        _RUNTIME_INITIALIZED = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    roots, static, raw_scale, raw_total = load_old(source)
    train_roots = [r for r in roots if r["split"] == "train"]
    validation_roots = [r for r in roots if r["split"] == "validation"]
    model = RevisedEffectModel(static, variant).to(device)
    for p in model.pair_head.parameters():
        p.requires_grad_(False)
    scale, total = torch.tensor(raw_scale, device=device), torch.tensor(raw_total, device=device)
    histories = torch.as_tensor(np.stack([r["normalized_history"] for r in train_roots]), device=device)
    actions = torch.as_tensor(np.stack([r["action_bank"] for r in train_roots]), device=device)
    base = torch.as_tensor(np.stack([r["base_phase"] for r in train_roots]), device=device)
    nodes = torch.as_tensor(np.concatenate([r["single_nodes"][:, 0] for r in train_roots]), device=device)
    targets = torch.as_tensor(np.concatenate([r["single_actions"][:, 0] for r in train_roots]), device=device)
    labels = torch.as_tensor(np.concatenate([r["single"] for r in train_roots]), device=device)
    if len(labels) != 384:
        raise ValueError("Expected exactly 384 old single queries")
    root_index = torch.arange(8, device=device).repeat_interleave(48)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
    generator = torch.Generator(device=device).manual_seed(seed)
    best, mae_best = (float("inf"),) * 3, (float("inf"), float("inf"))
    trained_at = time.perf_counter()
    for epoch in range(1, 301):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(epoch, 300)
        order, losses = torch.randperm(384, device=device, generator=generator), []
        for offset in range(0, 384, 32):
            ix = order[offset:offset + 32]
            unique, inverse = torch.unique(root_index[ix], return_inverse=True)
            optimizer.zero_grad(set_to_none=True)
            encoded = model.encode(histories[unique], actions[unique])
            prediction, logits = model.single(encoded, inverse, nodes[ix], targets[ix], base[unique], return_gate=True)
            loss = revision_loss(prediction, labels[ix], scale, total, variant, logits)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss, {variant}/{seed}/{epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": learning_rate(epoch, 300),
                  "elapsed_s": time.perf_counter() - trained_at, "updates": epoch * 12}
        if epoch % 5 == 0:
            regret, mae = validation(model, validation_roots, scale, device)
            if not np.isfinite([regret, mae]).all():
                raise FloatingPointError("Nonfinite validation metric")
            record.update(validation_regret=regret, validation_joint_mae=mae)
            key = checkpoint_key(regret, mae, epoch)
            metadata = {"variant": variant, "seed": seed, "epoch": epoch,
                        "validation_regret": regret, "validation_joint_mae": mae}
            if key < best:
                best = key; save_checkpoint(directory / "best.pt", model, metadata)
            if (mae, epoch) < mae_best:
                mae_best = mae, epoch; save_checkpoint(directory / "mae_best_diagnostic.pt", model, metadata)
        with (directory / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        atomic_json(directory / "status.json", {"stage": "training", "variant": variant, "seed": seed,
                    "pid": os.getpid(), **record, "selected_epoch": None if not np.isfinite(best[2]) else best[2]})
    train_seconds = time.perf_counter() - trained_at
    selected = torch.load(directory / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["state_dict"])
    model.eval()
    atomic_json(directory / "status.json", {"stage": "evaluating", "variant": variant, "seed": seed, "pid": os.getpid()})
    records = evaluate(model, roots, scale, device)
    summaries = {}
    for split in ("train", "validation", "development_holdout"):
        rows = [r for r in records if r["split"] == split]
        summaries[split] = {"root_count": len(rows), "regret": float(np.mean([r["joint"]["decision"]["regret_vehicle_s"] for r in rows])),
                            "joint_mae": float(np.mean([r["joint"]["all"]["mae_vehicle_s"] for r in rows])),
                            "optimal_choice_rate": float(np.mean([r["joint"]["decision"]["optimal_choice"] for r in rows]))}
    result = {"variant": variant, "seed": seed, "stage": "complete", "epochs_run": 300,
              "selected_epoch": selected["epoch"], "mae_best_diagnostic_epoch": mae_best[1],
              "training_wall_s": train_seconds, "total_wall_s": time.perf_counter() - start,
              "summary": summaries, "parameters": sum(p.numel() for p in model.parameters()),
              "timing": "Concurrent training/evaluation; not an isolated inference benchmark",
              "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(directory / "evaluation.json", {**result, "records": records})
    atomic_json(directory / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    if not Path("/mnt/pan").is_mount() or not args.run_dir.resolve().is_relative_to(Path("/mnt/pan")):
        raise ValueError("Mounted /mnt/pan is required")
    run = args.run_dir
    directory = run / "old_training"; directory.mkdir(exist_ok=True)
    config = {"source": str(args.source), "source_manifest_sha256": sha256(args.source / "split_and_data.json"),
              "variants": ["A0", "A1", "A2"], "seeds": [42, 43, 44], "workers_max": 3,
              "cpu_threads_per_worker": 2, "epochs": 300, "batch": 32, "validation_every": 5,
              "checkpoint": "min(validation mean-root regret, mean-root joint MAE, epoch)",
              "version_selection": "min(3-seed mean validation regret, mean validation joint MAE, variant order)",
              "loss_A0": "original equal-group/nonzero SmoothL1 + 0.1 total SmoothL1",
              "loss_A1": "natural all-element normalized L1 + 0.1 total SmoothL1",
              "loss_A2": "A1 on scale*p*v + 0.01 unweighted natural BCE(p, true_nonzero)",
              "scale": "reuse frozen old train-only P95 single scale and total scale",
              "horizon_s": 180, "no_early_stop": True, "no_joint_training_labels": True,
              "optimizer": {"name": "AdamW", "lr": .0003, "min_lr": .00003, "warmup_epochs": 5,
                            "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8, "clip": 1.},
              "precision": "FP32; AMP and TF32 off", "old_data_only_development": True,
              "source_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).parent / "effect_model/revision.py")}}
    atomic_json(directory / "config.json", config)
    jobs = [(v, s) for s in (42, 43, 44) for v in ("A0", "A1", "A2")]
    results, errors = [], []
    import fcntl
    with (directory / "pool.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "summary.json").exists():
            return
        with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
            future_jobs = {pool.submit(run_job, str(run), str(args.source), v, s): (v, s) for v, s in jobs}
            for future in concurrent.futures.as_completed(future_jobs):
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({"job": future_jobs[future], "error": repr(exc)})
                atomic_json(directory / "queue.json", {"completed": len(results), "total": 9, "errors": errors})
        if not errors and len(results) == 9:
            scores = {v: [float(np.mean([r["summary"]["validation"][k] for r in results if r["variant"] == v]))
                          for k in ("regret", "joint_mae")] for v in ("A0", "A1", "A2")}
            chosen = min(scores, key=lambda v: (*scores[v], v))
            atomic_json(run / "selected_A.json", {"variant": chosen, "validation_scores": scores,
                        "selection_uses_development_holdout": False, "fixed_for_future_new_training": True})
        atomic_json(directory / "summary.json", {"stage": "complete" if not errors else "failed", "results": results,
                                                  "errors": errors, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()
