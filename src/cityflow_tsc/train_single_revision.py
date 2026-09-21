"""Expanded-root single-effect training; no pair or joint targets enter the loss."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_model import (create_model, decision_metrics, group_mean,
    multiscale_loss, root_metadata, tensor_scales, validation)
from .train_effect_world_model import learning_rate
from .train_formal_effects import initialize, now, optimizer_for, select_checkpoint, write_epoch

VARIANTS = ("A_ref", "A_contrast")
SEEDS = (42, 43, 44)
CONTRAST_WEIGHT = .1


def action_groups(roots):
    """Indices in flattened singles, ordered root/node then plan IDs 1..4."""
    groups = []
    for ri, root in enumerate(roots):
        nodes = np.asarray(root["single_nodes"]).reshape(-1)
        actions = np.asarray(root["single_actions"]).reshape(-1)
        lookup = {(int(n), int(a)): i for i, (n, a) in enumerate(zip(nodes, actions))}
        if len(lookup) != 64 or set(lookup) != {(n, a) for n in range(16) for a in range(1, 5)}:
            raise ValueError("Every root needs 16 nodes and four unique actions each")
        groups.extend([[64 * ri + lookup[n, a] for a in range(1, 5)] for n in range(16)])
    return np.asarray(groups, dtype=np.int64)


def contrast_scale(training):
    """Train-only P95 nonzero absolute differences of signed action totals."""
    if not training or any(r["split"] != "train" for r in training):
        raise ValueError("Contrast normalization accepts training roots only")
    groups = action_groups(training)
    totals = np.concatenate([np.asarray(r["single"]).sum((1, 2), dtype=np.float64) for r in training])
    grouped = totals[groups]
    i, j = np.triu_indices(4, 1)
    values = np.abs(grouped[:, i] - grouped[:, j]).ravel()
    values = values[values != 0]
    return max(1., float(np.percentile(values, 95))) if len(values) else 1.


def action_contrast_loss(prediction, truth, s5, scale):
    """Six same-root/node contrasts per group; no cross-root comparisons."""
    if len(prediction) % 4 or prediction.shape != truth.shape:
        raise ValueError("Complete four-action groups are required")
    p = (prediction * s5).sum((1, 2)).reshape(-1, 4)
    y = truth.sum((1, 2)).reshape(-1, 4)
    i, j = torch.triu_indices(4, 4, 1, device=p.device)
    return torch.nn.functional.smooth_l1_loss((p[:, i] - p[:, j]) / scale,
                                             (y[:, i] - y[:, j]) / scale, beta=1.)


def check_roots(roots):
    if not roots or set(r["split"] for r in roots) != {"train", "validation"}:
        raise ValueError("Training requires nonempty train/validation only")
    for root in roots:
        if root["single"].shape != (64, 272, 48) or root["joint"].shape != (65, 272, 48):
            raise ValueError("Single and reference-first joint dimensions changed")
        if np.any(root["s_incidence"][0]) or np.any(root["joint"][0]):
            raise ValueError("Reference candidate must be zero")


def train_job(run, variant, seed):
    import fcntl
    from .effect_model.formal_data import load_data
    run = Path(run)
    directory = run / variant / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "job.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "complete.json").exists():
            result = json.loads((directory / "complete.json").read_text())
            if result["checkpoint_sha256"] != sha256(directory / "best.pt"):
                raise RuntimeError("Completed checkpoint changed")
            return result
        if (directory / "started.json").exists():
            raise RuntimeError("Incomplete training retained; explicit recovery required")
        atomic_json(directory / "started.json", {"variant": variant, "seed": seed, "pid": os.getpid(), "started_at": now()})
        try:
            initialize(seed)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required; no CPU training fallback")
            device = torch.device("cuda:0")
            roots, static, stats = load_data(run, include_pairs=False, splits=("train", "validation"))
            check_roots(roots)
            training = [r for r in roots if r["split"] == "train"]
            heldout = [r for r in roots if r["split"] == "validation"]
            groups = action_groups(training)
            scale = contrast_scale(training)
            atomic_json(directory / "contrast_normalization.json", {"scale": scale, "fit_split": "train", "train_roots": len(training), "weight": CONTRAST_WEIGHT if variant == "A_contrast" else 0.})
            model = create_model(static).to(device)
            for p in model.pair_head.parameters():
                p.requires_grad_(False)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = optimizer_for(parameters)
            scales = tensor_scales(stats, "single", device)
            nodes = torch.as_tensor(np.concatenate([r["single_nodes"] for r in training]), device=device)
            targets = torch.as_tensor(np.concatenate([r["single_actions"] for r in training]), device=device)
            base = torch.as_tensor(np.stack([r["base_phase"] for r in training]), device=device)
            histories = torch.as_tensor(np.stack([r["normalized_history"] for r in training]), device=device)
            actions = torch.as_tensor(np.stack([r["action_bank"] for r in training]), device=device)
            generator = np.random.default_rng(seed)
            best, mae_best = (float("inf"),) * 3, (float("inf"),) * 2
            started = time.perf_counter()
            batches = (len(groups) + 7) // 8
            for epoch in range(1, 301):
                model.train()
                rate = learning_rate(epoch, 300)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                order = generator.permutation(len(groups))
                losses, contrasts = [], []
                for offset in range(0, len(groups), 8):
                    ix_np = groups[order[offset:offset + 8]].ravel()
                    ix = torch.as_tensor(ix_np, device=device)
                    unique, inverse = torch.unique(ix // 64, return_inverse=True)
                    optimizer.zero_grad(set_to_none=True)
                    encoded = model.encode(histories[unique], actions[unique])
                    prediction, logits = model.single(encoded, inverse, nodes[ix, 0], targets[ix, 0], base[unique], return_gate=True)
                    labels = torch.as_tensor(np.stack([training[int(q // 64)]["single"][int(q % 64)] for q in ix_np]), device=device)
                    loss = multiscale_loss(prediction, labels, scales, "A_ref", logits)
                    extra = action_contrast_loss(prediction, labels, scales["s5"], scale) if variant == "A_contrast" else loss.new_zeros(())
                    loss = loss + CONTRAST_WEIGHT * extra
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite single-effect training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    contrasts.append(float(extra.detach()))
                record = {"epoch": epoch, "updates": epoch * batches, "loss": float(np.mean(losses)), "contrast_loss": float(np.mean(contrasts)), "lr": rate, "variant": variant, "seed": seed, "elapsed_s": time.perf_counter() - started}
                if epoch % 5 == 0:
                    metrics = validation(model, heldout, stats, device, include_pairs=False)
                    best, mae_best = select_checkpoint(directory, model, metrics, epoch, variant, seed, best, mae_best)
                    record.update(validation_regret=metrics["regret"], validation_joint_mae=metrics["joint_mae"])
                record["selected_epoch"] = int(best[2]) if np.isfinite(best[2]) else None
                write_epoch(directory, record)
            result = {"stage": "complete", "variant": variant, "seed": seed, "epochs_run": 300, "updates": 300 * batches, "queries_per_epoch": len(groups) * 4, "selected_epoch": int(best[2]), "validation_regret": best[0], "validation_joint_mae": best[1], "checkpoint_sha256": sha256(directory / "best.pt"), "training_wall_s": time.perf_counter() - started, "test_used": False, "finished_at": now()}
            atomic_json(directory / "status.json", result)
            atomic_json(directory / "complete.json", result)
            return result
        except Exception as exc:
            atomic_json(directory / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            raise


def train(run, workers=3):
    from .effect_model.formal_data import prepare_dataset
    if not 1 <= workers <= 6:
        raise ValueError("Expected one to six workers")
    prepare_dataset(run, include_pairs=False, splits=("train", "validation"))
    directory = run / "training_single"
    directory.mkdir(exist_ok=True)
    results, errors = [], []
    jobs = [(v, s) for s in SEEDS for v in VARIANTS]
    atomic_json(directory / "config.json", {"jobs": jobs, "workers": workers, "batch_queries": 32, "grouped_sampler": "eight root-node groups, four actions each", "epochs": 300, "contrast_weight": CONTRAST_WEIGHT, "test_used": False})
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(train_job, str(run), v, s): (v, s) for v, s in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"job": futures[future], "error": repr(exc)})
            atomic_json(directory / "queue.json", {"completed": len(results), "total": 6, "errors": errors})
    atomic_json(directory / "summary.json", {"stage": "failed" if errors else "complete", "results": results, "errors": errors})
    if errors:
        raise RuntimeError("Training failures retained; see training_single/summary.json")
    scores = {v: [float(np.mean([r[k] for r in results if r["variant"] == v])) for k in ("validation_regret", "validation_joint_mae")] for v in VARIANTS}
    selected = min(VARIANTS, key=lambda v: (*scores[v], VARIANTS.index(v)))
    atomic_json(run / "selected_A.json", {"variant": selected, "validation_scores": scores, "uniform_for_all_seeds": True, "test_used": False, "locked_at": now()})
    checkpoints = {str(Path(v) / f"seed_{s}" / "best.pt"): sha256(run / v / f"seed_{s}" / "best.pt") for v, s in jobs}
    atomic_json(run / "checkpoints_locked.json", {"checkpoints": checkpoints, "selected_A_sha256": sha256(run / "selected_A.json"), "normalization_sha256": sha256(run / "derived/normalization.npz"), "test_used": False, "locked_at": now()})


def evaluate(run):
    from .effect_model.formal_data import load_data, prepare_dataset
    lock = json.loads((run / "checkpoints_locked.json").read_text())
    for relative, expected in lock["checkpoints"].items():
        if sha256(run / relative) != expected:
            raise RuntimeError("Frozen checkpoint changed")
    if sha256(run / "derived/normalization.npz") != lock["normalization_sha256"] or sha256(run / "selected_A.json") != lock["selected_A_sha256"]:
        raise RuntimeError("Frozen normalization or selection changed")
    prepare_dataset(run, include_pairs=False, splits=("test",))
    roots, static, stats = load_data(run, include_pairs=False, splits=("test",))
    initialize(42)
    device = torch.device("cuda:0")
    results = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            saved = torch.load(run / variant / f"seed_{seed}" / "best.pt", map_location=device, weights_only=False)
            model = create_model(static).to(device)
            model.load_state_dict(saved["state_dict"])
            model.eval()
            results[f"{variant}/seed_{seed}"] = validation(model, roots, stats, device, include_pairs=False)
            del model
    for mode in ("Zero", "True-S"):
        rows = []
        for root in roots:
            truth = np.asarray(root["joint"], dtype=np.float64)
            prediction = np.zeros_like(truth) if mode == "Zero" else (root["s_incidence"] @ np.asarray(root["single"], dtype=np.float64).reshape(64, -1)).reshape(truth.shape)
            rows.append({**root_metadata(root), **decision_metrics(prediction.sum((1, 2)), truth.sum((1, 2))), "joint_mae": float(np.abs(prediction - truth).mean())})
        results[mode] = {"regret": group_mean(rows, "regret"), "joint_mae": group_mean(rows, "joint_mae"), "records": rows}
    atomic_json(run / "evaluation/summary.json", {"stage": "complete", "results": results, "finished_at": now(), "selection_used_test": False,
                "test_scope": "diagnostic on previously inspected demand, not pristine unseen-demand holdout"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not Path("/mnt/pan").is_mount() or not run.is_relative_to(Path("/mnt/pan")):
        raise ValueError("Mounted /mnt/pan run required")
    train(run, args.workers) if args.stage == "train" else evaluate(run)


if __name__ == "__main__":
    main()
