"""Collect only the additional fixed-time roots for the controlled temporal study."""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import copy
import importlib
import json
import multiprocessing
import os
from pathlib import Path

from . import collect_formal_effects as formal
from .collect_counterfactual import root_identity, stamp
from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_data import SINGLE_REVISION_SCHEMA

SCHEMA = SINGLE_REVISION_SCHEMA
EXPECTED_SPLITS = {"train": 660, "validation": 78, "test": 12}
MAX_WORKERS = 24


def read(path):
    return json.loads(Path(path).read_text())


def expand_tasks(parent):
    tasks = copy.deepcopy(parent["tasks"])
    if dict(Counter(t["split"] for t in tasks)) != {"train": 220, "validation": 26, "test": 12}:
        raise ValueError("Source must have 220/26/12 roots")
    if any(t["time_s"] != 600 for t in tasks if t["split"] != "test"):
        raise ValueError("Source training/validation roots must all be at 600 seconds")
    for task in tasks:
        task["temporal_reused"] = True
    added = []
    for original in tasks:
        if original["split"] == "test":
            continue
        for time_s in (1800, 2700):
            task = copy.deepcopy(original)
            task.update(time_s=time_s, task_id=f"{original['task_id']}_t{time_s}",
                        root_time_rule="fixed_600_1800_2700_no_outcome_filtering", temporal_reused=False)
            task["collection"] = {"root_times": [time_s], "root_order": None, "full_all_roots": True}
            task["root_id"] = root_identity(task, time_s)
            added.append(task)
    # Submit all 52 new validation roots first, allowing the early arm to overlap collection.
    added.sort(key=lambda t: (t["split"] != "validation", t["time_s"], t["task_id"]))
    combined = tasks + added
    for ordinal, task in enumerate(combined):
        task["ordinal"] = ordinal
    if len({t["root_id"] for t in combined}) != 750 or len({t["task_id"] for t in combined}) != 750:
        raise ValueError("Task/root identities collide")
    return combined


def link_directory(source, target):
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source:
            raise ValueError(f"Raw link target changed: {target}")
    elif target.exists():
        raise ValueError(f"Refusing to adopt a mutable raw directory: {target}")
    else:
        target.symlink_to(source, target_is_directory=True)


def verify_reused(source, task):
    """Verify existing commit files only; never replay or modify source data."""
    rid = task["root_id"]
    marker = source / "indexes" / f"{rid}.initial.complete.json"
    result = read(marker)
    root_dir = source / "roots" / rid
    root = read(root_dir / "root.json")
    if (result["root_id"] != rid or result["branch_count"] != 129
            or result["stage"] != "initial_complete" or result["split"] != task["split"]
            or root["root_id"] != rid or root["time_s"] != task["time_s"]
            or root["flow_sha256"] != task["manifest"]["flow_sha256"]
            or root["roadnet_sha256"] != task["manifest"]["roadnet_sha256"]
            or root["simulator_seed"] != task["manifest"]["scenario"]["seed"]
            or root["policy"] != task["policy"]):
        raise ValueError(f"Reused root identity/count mismatch: {rid}")
    if (sha256(root_dir / "history.npz") != root["history_sha256"]
            or sha256(root_dir / "branches.json") != result["branch_plan_sha256"]):
        raise ValueError(f"Reused history/plan changed: {rid}")
    files = {str(p.relative_to(source)): sha256(p) for p in root_dir.iterdir() if p.is_file()}
    count = 0
    for meta_path in sorted((source / "shards" / rid).glob("*.npz.json")):
        meta = read(meta_path)
        shard = Path(str(meta_path)[:-5])
        if meta["stage"] != "initial" or sha256(shard) != meta["sha256"]:
            raise ValueError(f"Reused shard changed: {shard}")
        count += meta["branch_count"]
        files[str(shard.relative_to(source))] = meta["sha256"]
        files[str(meta_path.relative_to(source))] = sha256(meta_path)
    if count != 129:
        raise ValueError(f"Reused shard count differs: {rid}")
    files[str(marker.relative_to(source))] = sha256(marker)
    return files


def prepare_arm(shared, arm):
    if arm not in ("early", "multi"):
        raise ValueError("arm must be early or multi")
    shared = Path(shared).resolve()
    selection = read(shared / "selection.json")
    tasks = [t for t in selection["tasks"] if arm == "multi" or t["split"] != "train" or t["time_s"] == 600]
    run = shared / "arms" / arm
    run.mkdir(parents=True, exist_ok=True)
    arm_selection = {**selection, "tasks": tasks, "temporal_arm": arm,
                     "shared_run": str(shared), "root_count": len(tasks),
                     "split_roots": dict(Counter(t["split"] for t in tasks))}
    formal._freeze_json(run / "selection.json", arm_selection)
    formal._freeze_json(run / "protocol.json", {**read(shared / "protocol.json"), "temporal_arm": arm,
                        "root_count": len(tasks), "branch_count": len(tasks) * 129,
                        "normalization_scope": "this arm's training roots only"})
    # Raw links are only read by derivation. Derived files and statistics are arm-local.
    for task in tasks:
        for name in ("roots", "shards"):
            link_directory(shared / name / task["root_id"], run / name / task["root_id"])
    (run / "static").mkdir(exist_ok=True)
    geometry = shared / "static/geometry.json"
    formal._freeze_json(run / "static/geometry.json", read(geometry))
    return run


def prepare(run, source_run):
    run, source = Path(run).resolve(), Path(source_run).resolve()
    if not Path("/mnt/pan").is_mount() or not run.is_relative_to(Path("/mnt/pan")):
        raise ValueError("Outputs require mounted /mnt/pan")
    if run == source or source.is_relative_to(run) or run.is_relative_to(source):
        raise ValueError("Use a separate run; preserve the source")
    parent = read(source / "selection.json")
    if parent.get("schema") != SCHEMA or read(source / "summary.json").get("stage") != "complete":
        raise ValueError("Completed single-revision source required")
    parent_protocol = read(source / "protocol.json")
    engine = importlib.import_module("cityflow")
    if sha256(engine.__file__) != parent_protocol["engine_binary_sha256"]:
        raise ValueError("CityFlow binary differs from source")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "reuse_manifest.json").exists():
        manifest = read(run / "reuse_manifest.json")
        if (manifest["source_run"] != str(source)
                or manifest["source_selection_sha256"] != sha256(source / "selection.json")
                or manifest["source_protocol_sha256"] != sha256(source / "protocol.json")):
            raise ValueError("Frozen parent metadata changed")
        selection = read(run / "selection.json")
        for arm in ("early", "multi"):
            prepare_arm(run, arm)
        return selection
    tasks = expand_tasks(parent)
    selection = {**parent, "source_run": str(source), "tasks": tasks, "root_count": 750,
                 "split_roots": EXPECTED_SPLITS, "temporal_revision": "v5",
                 "trainval_root_time_rule": "fixed 600, 1800, 2700 seconds; no outcome filtering"}
    formal._freeze_json(run / "selection.json", selection)
    formal._freeze_json(run / "protocol.json", {**parent_protocol, "source_run": str(source),
        "collector_sha256": sha256(__file__), "root_count": 750, "branch_count": 96750,
        "branch_upper_bound": 96750, "reused_roots": 258, "reused_branches": 33282,
        "new_roots": 492, "new_branches": 63468, "total_budget_branches": 63468,
        "initial_non_test_roots": 738, "initial_non_test_branches": 95202,
        "workers_max": MAX_WORKERS, "training_jobs": 6, "temporal_revision": "v5",
        "training_max_updates": 132000, "training_seeds": [42, 43, 44],
        "training_loss": "A_ref", "validation_roots_shared": 78,
        "normalization_scope": "arm-local training roots only", "test_use": "previously_inspected_diagnostic_not_fresh_holdout"})
    formal._freeze_json(run / "static/geometry.json", read(source / "static/geometry.json"))
    hashes = {}
    for task in tasks:
        if not task["temporal_reused"]:
            continue
        hashes.update(verify_reused(source, task))
        for name in ("roots", "shards"):
            link_directory(source / name / task["root_id"], run / name / task["root_id"])
        (run / "indexes").mkdir(exist_ok=True)
        for name in (f"{task['root_id']}.initial.complete.json", f"{task['task_id']}.roots.json"):
            path = source / "indexes" / name
            if path.exists():
                formal._freeze_json(run / "indexes" / name, read(path))
    formal._freeze_json(run / "reuse_manifest.json", {"source_run": str(source), "files": hashes,
        "source_selection_sha256": sha256(source / "selection.json"),
        "source_protocol_sha256": sha256(source / "protocol.json"), "reused_roots": 258})
    for arm in ("early", "multi"):
        prepare_arm(run, arm)
    return selection


def emit_readiness(run, tasks, results):
    ids = set(results)
    for arm in ("early", "multi"):
        required = [t for t in tasks if t["split"] != "test" and
                    (arm == "multi" or t["split"] == "validation" or t["time_s"] == 600)]
        if all(t["root_id"] in ids for t in required):
            marker = run / f"{arm}_ready.json"
            if not marker.exists():
                atomic_json(marker, {"stage": f"{arm}_ready", "root_count": len(required),
                            "branch_count": len(required) * 129, "finished_at": stamp()})


def collect(run, source_run, workers=20):
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be 1..{MAX_WORKERS}")
    selection = prepare(run, source_run)
    run = Path(run).resolve()
    import fcntl
    with (run / "collector.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        results, errors = {}, []
        pending = []
        for task in selection["tasks"]:
            marker = run / "indexes" / f"{task['root_id']}.initial.complete.json"
            if marker.exists():
                results[task["root_id"]] = read(marker)
            elif task["temporal_reused"]:
                raise ValueError("Missing reused commit")
            else:
                pending.append(task)
        emit_readiness(run, selection["tasks"], results)
        def status(stage):
            value = {"stage": stage, "pid": os.getpid(), "workers": workers,
                     "completed_roots": len(results), "root_count": len(results), "total_roots": 750,
                     "branch_count": sum(r["branch_count"] for r in results.values()),
                     "branch_upper_bound": 96750, "reused_roots": 258, "new_roots": len(results) - 258,
                     "errors": errors, "updated_at": stamp()}
            atomic_json(run / "status.json", value)
            return value
        status("running")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(formal.collect_root, str(run), t, "initial"): t for t in pending}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    if (result["root_id"] != task["root_id"] or result["branch_count"] != 129
                            or result["stage"] != "initial_complete"):
                        raise ValueError("Unexpected collection commit")
                    results[task["root_id"]] = result
                except Exception as exc:
                    errors.append({"task_id": task["task_id"], "error": repr(exc)})
                emit_readiness(run, selection["tasks"], results)
                status("running")
        summary = status("failed" if errors or len(results) != 750 else "complete")
        summary["results"] = list(results.values())
        summary["unresolved_events"] = sum(r["unresolved_events"] for r in results.values())
        atomic_json(run / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "collect"), default="prepare")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args(argv)
    result = prepare(args.run_dir, args.source_run) if args.stage == "prepare" else collect(
        args.run_dir, args.source_run, args.workers)
    print(json.dumps({k: v for k, v in result.items() if k not in ("tasks", "results")}), flush=True)
    return int(result.get("stage") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
