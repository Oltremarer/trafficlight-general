"""Isolated v3 test-time correction. Default: prepare only, never train.

Reuse locked models/statistics, recollect twelve test roots, then evaluate and
time the same methods. The original run is read-only. No outcome-based root
selection, consistency rollout, extra pilot, or automatic timing retry.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np

from .collect_counterfactual import root_identity, stamp
from .collect_formal_effects import BRANCHES_PER_ROOT, _check_task_sources, collect_root
from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_data import TEST_REPAIR_SCHEMA, _selected_tasks
from .evaluate_formal_effects import verify_checkpoint_lock


# Fixed before corrected counterfactual labels. Not searched against model scores.
TEST_ROOT_TIMES = {"stress_peak": 1800, "stress_late": 2700}
COLLECTION_BRANCHES = 12 * BRANCHES_PER_ROOT
TIMING_BRANCHES = 12 * 65 * 3
MAX_WORKERS = 12


def read(path):
    return json.loads(Path(path).read_text())


def corrected_selection(source, parent):
    if source.get("schema") == TEST_REPAIR_SCHEMA:
        raise ValueError("Repair must reference the original sixty-root run, not another repair")
    original = _selected_tasks(source, ("test",))
    tasks = []
    for old in original:
        task = copy.deepcopy(old)
        t = TEST_ROOT_TIMES[task["profile"]]
        if t % 30 or not 150 <= t <= 3600 - 240:
            raise ValueError("Root needs 150 s history, 30 s alignment and a full 240 s future")
        task.update(time_s=t, previous_root_id=old["root_id"], previous_time_s=old["time_s"])
        task["collection"]["root_times"] = [t]
        task["root_id"] = root_identity(task, t)
        if task["root_id"] == old["root_id"]:
            raise ValueError("Corrected roots must not reuse an original root ID")
        tasks.append(task)
    profiles = {name: sum(t["profile"] == name for t in tasks) for name in TEST_ROOT_TIMES}
    if profiles != {"stress_peak": 6, "stress_late": 6}:
        raise ValueError("Expected six test roots per frozen stress profile")
    for flow in {t["flow_id"] for t in tasks}:
        pair = [t for t in tasks if t["flow_id"] == flow]
        if len(pair) != 2 or {t["policy"] for t in pair} != {"fixed_time", "max_pressure"}:
            raise ValueError("Each test demand requires both original source policies")
    selection = {"schema": TEST_REPAIR_SCHEMA, "dataset": source["dataset"],
                 "source_run": str(parent), "tasks": tasks, "root_count": 12,
                 "split_roots": {"test": 12}, "test_root_times_s": dict(TEST_ROOT_TIMES),
                 "candidate_policy": "unchanged formal_t1_joint_v2, keyed by corrected root ID; shared across models"}
    _selected_tasks(selection, ("test",))
    return selection


def temporal_coverage(parent, source, selection):
    """Read existing roots/source observations only, never future branch labels.

    Movement sums can double-count vehicles across movements: they are coverage
    descriptors, not a unique vehicle count or the formal waiting objective.
    No traffic threshold filters roots or changes the predefined root times.
    """
    original_rows, corrected_rows = [], []
    for task in source["tasks"]:
        root = read(parent / "roots" / task["root_id"] / "root.json")
        if root["root_id"] != task["root_id"] or root["time_s"] != task["time_s"]:
            raise ValueError("Original root metadata does not match frozen selection")
        flows = read(task["manifest"]["scenario"]["flow_path"])
        original_rows.append({**{k: task[k] for k in ("root_id", "split", "profile", "time_s")},
                              "active_vehicles": root["active_vehicles"], "backlog": root["backlog"],
                              "first_departure_s": min(f["startTime"] for f in flows)})
    for task in selection["tasks"]:
        _check_task_sources(task)
        manifest = read(task["manifest_path"])
        names = manifest["feature_names"]
        with np.load(Path(task["manifest_path"]).parent / "trajectory.npz", allow_pickle=False) as z:
            index = np.flatnonzero(z["observation_time_s"] == task["time_s"])
            if len(index) != 1:
                raise ValueError("Corrected time is missing or duplicated in original source trajectory")
            features = z["observation_features"][index[0]]
            valid = z["observation_valid_mask"][index[0]].astype(bool)
            if valid.shape != features.shape:
                raise ValueError("Source observations require a per-feature validity mask")
            ni, qi = names.index("incoming_vehicle_count"), names.index("incoming_queue_count")
            vehicles = np.where(valid[..., ni], features[..., ni], 0)
            queues = np.where(valid[..., qi], features[..., qi], 0)
        corrected_rows.append({**{k: task[k] for k in ("root_id", "flow_id", "policy", "profile", "time_s")},
                               "incoming_vehicle_movement_sum": float(vehicles.sum()),
                               "incoming_queue_movement_sum": float(queues.sum()),
                               "intersections_with_incoming_vehicles": int(np.any(vehicles > 0, axis=1).sum()),
                               "intersections_with_incoming_queue": int(np.any(queues > 0, axis=1).sum())})
    return {"original_roots": original_rows, "corrected_source_observations": corrected_rows,
            "scope": "existing observations at/before intervention; no future counterfactual outcomes",
            "caution": "movement sums are not unique vehicle counts; no outcome-based selection or zero-root deletion"}


def require_destination(run, parent):
    storage = Path("/mnt/pan")
    if not storage.is_mount() or not run.is_relative_to(storage):
        raise ValueError("Repair outputs require mounted /mnt/pan; no system-disk fallback")
    if run.is_relative_to(parent) or parent.is_relative_to(run):
        raise ValueError("Repair directory must be separate from the original run")


def prepare(run, parent):
    run, parent = Path(run).resolve(), Path(parent).resolve()
    require_destination(run, parent)
    if read(parent / "complete.json").get("stage") != "complete":
        raise ValueError("Original run must be complete before an isolated test repair")
    verify_checkpoint_lock(parent)
    source = read(parent / "selection.json")
    selection = corrected_selection(source, parent)
    if (run / "test_repair.json").exists():
        repair = read(run / "test_repair.json")
        if (repair["source_run"] != str(parent)
                or repair["source_selection_sha256"] != sha256(parent / "selection.json")
                or repair["source_checkpoint_lock_sha256"] != sha256(parent / "checkpoints_locked.json")
                or read(run / "selection.json") != selection):
            raise ValueError("Refusing to change a frozen test repair")
        verify_checkpoint_lock(run)
        return selection
    if run.exists() and any(p.name != "repair.lock" for p in run.iterdir()):
        raise ValueError("Nonempty output without repair provenance; choose a fresh directory")
    protocol = read(parent / "protocol.json")
    if sha256(protocol["engine_binary"]) != protocol["engine_binary_sha256"]:
        raise IOError("Original CityFlow binary changed")
    coverage = temporal_coverage(parent, source, selection)
    lock = read(parent / "checkpoints_locked.json")
    expected = {f"{model}/seed_{seed}/best.pt" for model in ("A_ref", "A_MS", "B1", "ranker")
                for seed in (42, 43, 44)}
    if set(lock["checkpoints"]) != expected:
        raise ValueError("Expected exactly the twelve original locked checkpoints")
    files = sorted(expected | {"selected_A.json", "checkpoints_locked.json", "derived/static.npz",
                              "derived/normalization.npz", "derived/pair_normalization.npz", "static/geometry.json"})
    reused = {name: sha256(parent / name) for name in files}
    for name in ("logs", "tmp", "cache", "protocol"):
        (run / name).mkdir(parents=True, exist_ok=True)
    # Copy individual frozen inputs, never symlink mutable directories into parent.
    # In particular do not inherit complete/summary/status/derived root caches.
    for name in files:
        target = run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, target)
        if sha256(target) != reused[name]:
            raise IOError("Frozen input changed during copy: " + name)
    protocol.update(schema=TEST_REPAIR_SCHEMA, source_run=str(parent), root_count=12,
                    branch_count=COLLECTION_BRANCHES, branch_upper_bound=COLLECTION_BRANCHES,
                    initial_non_test_roots=0, initial_non_test_branches=0,
                    training_jobs=0, test_root_times_s=dict(TEST_ROOT_TIMES),
                    timing_branch_count=TIMING_BRANCHES,
                    test_use="corrected after original test inspection; all original models remain locked")
    atomic_json(run / "selection.json", selection)
    atomic_json(run / "protocol.json", protocol)
    atomic_json(run / "protocol/temporal_coverage.json", coverage)
    atomic_json(run / "test_repair.json", {
        "schema": TEST_REPAIR_SCHEMA, "created_at": stamp(), "source_run": str(parent),
        "source_selection_sha256": sha256(parent / "selection.json"),
        "source_checkpoint_lock_sha256": sha256(parent / "checkpoints_locked.json"),
        "selection_sha256": sha256(run / "selection.json"), "protocol_sha256": sha256(run / "protocol.json"),
        "reused_files": reused, "test_root_times_s": dict(TEST_ROOT_TIMES),
        "reason": "Original stress interventions preceded first departures; excluded from main traffic efficacy claims",
        "original_run_policy": "preserved read-only, no deletion, no old test labels or completion caches reused",
        "test_status": "corrected protocol, not an untouched original blind test",
        "training_jobs": 0, "collection_branches": COLLECTION_BRANCHES, "timing_branches": TIMING_BRANCHES,
        "code_sha256": sha256(__file__)})
    verify_checkpoint_lock(run)
    return selection


def configure_workers(run, workers):
    """Amend only pre-launch resource capacity; scientific inputs stay frozen."""
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError("Use 1..12 one-thread CityFlow workers")
    protocol = read(run / "protocol.json")
    previous = protocol.get("workers_max", 8)
    if workers <= previous:
        return
    if any((run / name).exists() for name in ("status.json", "summary.json", "roots", "shards")):
        raise ValueError("Cannot amend worker capacity after test collection starts")
    verify_checkpoint_lock(run)
    manifest = read(run / "test_repair.json")
    amendment = {"changed_at": stamp(), "reason": "user requested maximum parallel launch",
                 "previous_workers_max": previous, "workers_max": workers, "engine_threads": 1,
                 "previous_protocol_sha256": manifest["protocol_sha256"],
                 "previous_code_sha256": manifest["code_sha256"],
                 "scientific_selection_and_models_unchanged": True}
    protocol["workers_max"] = workers
    atomic_json(run / "protocol.json", protocol)
    manifest.update(protocol_sha256=sha256(run / "protocol.json"), code_sha256=sha256(__file__),
                    resource_amendment=amendment)
    atomic_json(run / "test_repair.json", manifest)
    verify_checkpoint_lock(run)


def collect_tests(run, selection, workers):
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError("Use 1..12 one-thread CityFlow workers")
    tasks = _selected_tasks(selection, ("test",))
    if (run / "summary.json").exists() and read(run / "summary.json").get("stage") == "complete":
        return read(run / "summary.json")
    results, errors = [], []
    def progress(stage):
        row = {"stage": stage, "scope": "test_only_time_repair", "updated_at": stamp(),
               "root_count": len(results), "expected_root_count": 12,
               "branch_count": sum(r["branch_count"] for r in results),
               "expected_branch_count": COLLECTION_BRANCHES, "workers": workers,
               "errors": errors, "results": results, "training_jobs": 0,
               "consistency_checks": False, "unresolved_events": sum(r["unresolved_events"] for r in results)}
        atomic_json(run / "status.json", row)
        return row
    progress("running")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(collect_root, str(run), task, "remaining"): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if (result["root_id"] != task["root_id"] or result["split"] != "test"
                        or result["branch_count"] != BRANCHES_PER_ROOT or result["stage"] != "complete"):
                    raise ValueError("Test collection returned mismatched completion metadata")
                results.append(result)
            except Exception as exc:
                errors.append({"root_id": task["root_id"], "error": repr(exc)})
            progress("running")
    summary = progress("failed" if errors else "complete")
    atomic_json(run / "summary.json", summary)
    return summary


def execute(run, parent, stage, workers=8):
    run, parent = Path(run).resolve(), Path(parent).resolve()
    require_destination(run, parent)
    run.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (run / "repair.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = prepare(run, parent)
        if stage == "prepare":
            return 0
        from .run_formal_effects import command_environment
        os.environ.update(command_environment(run, 1))
        if stage in ("collect", "all"):
            configure_workers(run, workers)
            if collect_tests(run, selection, workers)["stage"] != "complete":
                return 1
        if stage in ("evaluate", "timing", "all"):
            if read(run / "summary.json").get("stage") != "complete":
                raise ValueError("Corrected test collection is not complete")
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if stage in ("evaluate", "all"):
            from .effect_model.formal_data import prepare_dataset
            from .evaluate_formal_effects import evaluate
            verify_checkpoint_lock(run)
            prepare_dataset(run, include_pairs=True, splits=("test",))
            evaluate(run)
        if stage in ("timing", "all"):
            if read(run / "evaluation/summary.json").get("stage") != "complete":
                raise ValueError("Corrected evaluation is not complete")
            from .benchmark_formal_effects import benchmark
            result = benchmark(run)
            if result:
                return result  # Isolation unavailable: retain state, never stop others or auto-retry.
            atomic_json(run / "complete.json", {"stage": "complete", "finished_at": stamp(),
                "scope": "test_only_time_repair", "source_run": str(parent), "training_jobs": 0,
                "collection_branches": COLLECTION_BRANCHES, "timing_branches": TIMING_BRANCHES,
                "total_budget_branches": COLLECTION_BRANCHES + TIMING_BRANCHES})
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("prepare", "collect", "evaluate", "timing", "all"), default="prepare")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error("--workers must be in 1..12")
    return execute(args.run_dir, args.source_run, args.stage, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
