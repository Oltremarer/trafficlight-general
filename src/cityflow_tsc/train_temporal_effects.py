"""Fixed-update A_ref comparison: early versus multi-stage traffic coverage."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_model import (create_model, decision_metrics, group_mean,
    multiscale_loss, root_metadata, tensor_scales, validation)
from .train_effect_world_model import save_checkpoint
from .train_formal_effects import initialize, now, optimizer_for, write_epoch
from .train_single_revision import action_groups, check_roots

SEEDS = (42, 43, 44)
MAX_UPDATES = 132_000
WARMUP_UPDATES = 2_200
VALIDATE_EVERY = 2_200
GROUPS_PER_BATCH = 8
TRAIN_ROOTS = {"early": 220, "multi": 660}
TIME_POINTS = (600, 1800, 2700)


def ensure_file_limit(root_count):
    """Reserve descriptors for two persistent memmaps per root and runtime IO."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = 2 * root_count + 256
    if soft == resource.RLIM_INFINITY or soft >= required:
        return
    if hard != resource.RLIM_INFINITY and hard < required:
        raise RuntimeError(f'Need {required} file descriptors; hard limit is {hard}')
    resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))


def update_learning_rate(update, maximum=MAX_UPDATES, warmup=WARMUP_UPDATES):
    if not 1 <= update <= maximum or not 0 < warmup < maximum:
        raise ValueError("Invalid update schedule")
    if update <= warmup:
        return 3e-4 * update / warmup
    fraction = (update - warmup) / (maximum - warmup)
    return 3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * fraction))


def grouped_batches(groups, seed, maximum=MAX_UPDATES):
    """Permute complete root-node groups each cycle; stop at an exact update."""
    if groups.ndim != 2 or groups.shape[1] != 4 or len(groups) % GROUPS_PER_BATCH:
        raise ValueError("Expected complete four-action groups divisible by eight")
    if not len(groups) or maximum < 1:
        raise ValueError("Nonempty groups and positive update budget required")
    rng = np.random.default_rng(seed)
    update, cycle = 0, 0
    while update < maximum:
        cycle += 1
        order = rng.permutation(len(groups))
        for offset in range(0, len(groups), GROUPS_PER_BATCH):
            update += 1
            yield update, cycle, groups[order[offset:offset + GROUPS_PER_BATCH]].ravel()
            if update == maximum:
                return


def resolve_arm(run, arm=None):
    selection = json.loads((Path(run) / "selection.json").read_text())
    stored = selection.get("temporal_arm", selection.get("arm"))
    if arm is not None and stored is not None and arm != stored:
        raise ValueError("CLI arm disagrees with frozen selection")
    arm = stored if arm is None else arm
    if arm not in TRAIN_ROOTS:
        raise ValueError("Expected early or multi arm")
    expected = {"train": TRAIN_ROOTS[arm], "validation": 78, "test": 12}
    if selection.get("split_roots") != expected:
        raise ValueError("Frozen arm root counts disagree with approved design")
    return arm


def check_temporal_roots(roots, arm):
    check_roots(roots)
    for split, count, times in (("train", TRAIN_ROOTS[arm], (600,) if arm == "early" else TIME_POINTS),
                                ("validation", 78, TIME_POINTS)):
        selected = [r for r in roots if r["split"] == split]
        if len(selected) != count:
            raise ValueError(f"Unexpected {split} root count")
        counts = {t: sum(float(r["time_s"]) == t for r in selected) for t in times}
        if any(n != count // len(times) for n in counts.values()):
            raise ValueError(f"Unexpected {split} temporal coverage")


def summarize_records(rows):
    keys = ("regret", "joint_mae", "optimal_choice", "worse_than_reference",
            "excess_wait_vs_reference", "benefit_vs_reference")
    return {key: group_mean(rows, key) for key in keys}


def temporal_metrics(metrics):
    rows = metrics["records"]
    times = sorted(set(float(r["time_s"]) for r in rows))
    return {**metrics, **summarize_records(rows), "by_time": {
        str(int(t)): {"root_count": sum(float(r["time_s"]) == t for r in rows),
                     **summarize_records([r for r in rows if float(r["time_s"]) == t])}
        for t in times}}


def select_update_checkpoint(directory, model, metrics, update, cycle, arm, seed, best):
    regret, mae = metrics["regret"], metrics["joint_mae"]
    if not np.isfinite([regret, mae]).all():
        raise FloatingPointError("Nonfinite validation metric")
    key = (float(regret), float(mae), int(update))
    if key < best:
        metadata = {"variant": "A_ref", "arm": arm, "seed": seed, "updates": update,
                    "sampling_cycle": cycle, "validation_regret": regret,
                    "validation_joint_mae": mae, "selection": "equal cohort regret, equal cohort joint MAE, earlier update",
                    "horizon_s": 240, "test_used": False}
        save_checkpoint(directory / "best.pt", model, metadata)
        atomic_json(directory / "best_validation.json", {**metadata, "metrics": metrics})
        return key
    return best


def train_job(run, arm, seed):
    import fcntl
    from .effect_model.formal_data import load_data
    run = Path(run)
    directory = run / "A_ref" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "job.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "complete.json").exists():
            result = json.loads((directory / "complete.json").read_text())
            if result["checkpoint_sha256"] != sha256(directory / "best.pt") or result["updates"] != MAX_UPDATES or result["arm"] != arm:
                raise RuntimeError("Completed training artifact changed")
            return result
        if (directory / "started.json").exists():
            raise RuntimeError("Incomplete training retained; explicit recovery required")
        atomic_json(directory / "started.json", {"arm": arm, "variant": "A_ref", "seed": seed,
                                                "pid": os.getpid(), "started_at": now()})
        try:
            initialize(seed)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required; no CPU fallback")
            device = torch.device("cuda:0")
            ensure_file_limit(TRAIN_ROOTS[arm] + 78)
            roots, static, stats = load_data(run, include_pairs=False, splits=("train", "validation"))
            check_temporal_roots(roots, arm)
            training = [r for r in roots if r["split"] == "train"]
            heldout = [r for r in roots if r["split"] == "validation"]
            groups = action_groups(training)
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
            best = (float("inf"),) * 3
            started = time.perf_counter()
            losses = []
            for update, cycle, ix_np in grouped_batches(groups, seed):
                model.train()
                rate = update_learning_rate(update)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                ix = torch.as_tensor(ix_np, device=device)
                unique, inverse = torch.unique(ix // 64, return_inverse=True)
                optimizer.zero_grad(set_to_none=True)
                encoded = model.encode(histories[unique], actions[unique])
                prediction, logits = model.single(encoded, inverse, nodes[ix, 0], targets[ix, 0], base[unique], return_gate=True)
                # Only this batch's targets move from CPU-backed arrays to GPU.
                labels = torch.as_tensor(np.stack([training[int(q // 64)]["single"][int(q % 64)] for q in ix_np]), device=device)
                loss = multiscale_loss(prediction, labels, scales, "A_ref", logits)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite single-effect loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
                if update % 440 == 0 or update % VALIDATE_EVERY == 0 or update == MAX_UPDATES:
                    record = {"updates": update, "sampling_cycle": cycle, "loss": float(np.mean(losses)),
                              "lr": rate, "arm": arm, "variant": "A_ref", "seed": seed,
                              "elapsed_s": time.perf_counter() - started}
                    losses.clear()
                    if update % VALIDATE_EVERY == 0 or update == MAX_UPDATES:
                        metrics = temporal_metrics(validation(model, heldout, stats, device, include_pairs=False))
                        best = select_update_checkpoint(directory, model, metrics, update, cycle, arm, seed, best)
                        record.update(validation_regret=metrics["regret"], validation_joint_mae=metrics["joint_mae"], validation_by_time=metrics["by_time"])
                    record["selected_update"] = int(best[2]) if np.isfinite(best[2]) else None
                    write_epoch(directory, record)
            result = {"stage": "complete", "arm": arm, "variant": "A_ref", "seed": seed,
                      "updates": update, "sampling_cycles": cycle, "selected_update": int(best[2]),
                      "validation_regret": best[0], "validation_joint_mae": best[1],
                      "checkpoint_sha256": sha256(directory / "best.pt"),
                      "training_wall_s": time.perf_counter() - started, "test_used": False, "finished_at": now()}
            atomic_json(directory / "status.json", result)
            atomic_json(directory / "complete.json", result)
            return result
        except Exception as exc:
            atomic_json(directory / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            raise


def train(run, workers=3, arm=None):
    from .effect_model.formal_data import prepare_dataset
    run = Path(run)
    arm = resolve_arm(run, arm)
    if not 1 <= workers <= 3:
        raise ValueError("One to three workers per arm required")
    prepare_dataset(run, include_pairs=False, splits=("train", "validation"))
    directory = run / "training_temporal"
    directory.mkdir(exist_ok=True)
    atomic_json(directory / "config.json", {"arm": arm, "variant": "A_ref", "seeds": SEEDS,
        "workers": workers, "batch_queries": 32, "max_updates": MAX_UPDATES,
        "warmup_updates": WARMUP_UPDATES, "validate_every_updates": VALIDATE_EVERY,
        "lr_peak": 3e-4, "lr_final": 3e-5, "normalization": "this arm training roots only",
        "grouped_sampler": "eight root-node groups, four actions each; permuted complete cycles",
        "contrast_loss": False, "test_used": False})
    results, errors = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(train_job, str(run), arm, s): s for s in SEEDS}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"seed": futures[future], "error": repr(exc)})
            atomic_json(directory / "queue.json", {"completed": len(results), "total": 3, "errors": errors})
    atomic_json(directory / "summary.json", {"stage": "failed" if errors else "complete", "arm": arm,
                                            "results": results, "errors": errors})
    if errors:
        raise RuntimeError("Training failures retained; see training_temporal/summary.json")
    atomic_json(run / "selected_A.json", {"variant": "A_ref", "arm": arm, "predeclared": True,
                "uniform_for_all_seeds": True, "test_used": False, "locked_at": now()})
    checkpoints = {str(Path("A_ref") / f"seed_{s}" / "best.pt"): sha256(run / "A_ref" / f"seed_{s}" / "best.pt") for s in SEEDS}
    atomic_json(run / "checkpoints_locked.json", {"checkpoints": checkpoints, "arm": arm,
        "selected_A_sha256": sha256(run / "selected_A.json"),
        "normalization_sha256": sha256(run / "derived/normalization.npz"), "test_used": False, "locked_at": now()})


def evaluate(run, arm=None):
    from .effect_model.formal_data import load_data, prepare_dataset
    run = Path(run)
    arm = resolve_arm(run, arm)
    lock = json.loads((run / "checkpoints_locked.json").read_text())
    expected = {str(Path("A_ref") / f"seed_{s}" / "best.pt") for s in SEEDS}
    if set(lock["checkpoints"]) != expected or lock["arm"] != arm:
        raise RuntimeError("Expected three locked A_ref checkpoints for this arm")
    for relative, digest in lock["checkpoints"].items():
        if sha256(run / relative) != digest:
            raise RuntimeError("Frozen checkpoint changed")
    if sha256(run / "derived/normalization.npz") != lock["normalization_sha256"] or sha256(run / "selected_A.json") != lock["selected_A_sha256"]:
        raise RuntimeError("Frozen normalization or selection changed")
    prepare_dataset(run, include_pairs=False, splits=("test",))
    roots, static, stats = load_data(run, include_pairs=False, splits=("test",))
    initialize(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    device = torch.device("cuda:0")
    results, validation_results = {}, {}
    for seed in SEEDS:
        saved = torch.load(run / "A_ref" / f"seed_{seed}" / "best.pt", map_location=device, weights_only=False)
        model = create_model(static).to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        results[f"A_ref/seed_{seed}"] = temporal_metrics(validation(model, roots, stats, device, include_pairs=False))
        selected_validation = json.loads((run / "A_ref" / f"seed_{seed}" / "best_validation.json").read_text())
        validation_results[f"A_ref/seed_{seed}"] = {"selected_update": selected_validation["updates"],
                                                  **selected_validation["metrics"]}
        del model, saved
    for mode in ("Zero", "True-S"):
        rows = []
        for root in roots:
            truth = np.asarray(root["joint"], dtype=np.float64)
            prediction = np.zeros_like(truth) if mode == "Zero" else (root["s_incidence"] @ np.asarray(root["single"], dtype=np.float64).reshape(64, -1)).reshape(truth.shape)
            rows.append({**root_metadata(root), **decision_metrics(prediction.sum((1, 2)), truth.sum((1, 2))),
                         "joint_mae": float(np.abs(prediction - truth).mean())})
        results[mode] = temporal_metrics({"records": rows})
    atomic_json(run / "evaluation/summary.json", {"stage": "complete", "arm": arm, "results": results,
        "validation_results": validation_results,
        "finished_at": now(), "selection_used_test": False,
        "test_scope": "diagnostic on previously inspected demand, not pristine unseen-demand holdout"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--arm", choices=tuple(TRAIN_ROOTS))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not Path("/mnt/pan").is_mount() or not run.is_relative_to(Path("/mnt/pan")):
        raise ValueError("Mounted /mnt/pan run required")
    train(run, args.workers, args.arm) if args.stage == "train" else evaluate(run, args.arm)


if __name__ == "__main__":
    main()
