"""Expanded single-effect collection, without pair rollouts or new demand.

All existing training/validation flows use one metadata-timed root per policy.
The previously inspected repaired test is retained as diagnostic, not fresh test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import copy
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

from . import collect_formal_effects as formal
from .collect_counterfactual import root_identity, stamp
from .counterfactual.geometry import Geometry
from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_data import SINGLE_REVISION_SCHEMA

SCHEMA = SINGLE_REVISION_SCHEMA
MAX_WORKERS = 24
EXPECTED_SPLITS = {"train": 220, "validation": 26, "test": 12}


def read(path):
    return json.loads(Path(path).read_text())


def demand_root_time(flow_rows):
    """Use demand timing only; never inspect traffic outcomes or model scores."""
    if not flow_rows:
        raise ValueError("Empty flow definition cannot determine a demand-relative root")
    first = min(float(row["startTime"]) for row in flow_rows)
    root = int(math.ceil((first + 600) / 30) * 30)
    if root > 3360:
        raise ValueError("Demand-relative root leaves insufficient 240-second source horizon")
    return root, first


def select_tasks(plan, index, dataset, parent_selection):
    trajectories = defaultdict(list)
    for row in index["trajectories"]:
        if int(row.get("episode", 0)) == 0:
            trajectories[row["flow_id"], row["policy"]].append(row)
    tasks = []
    for flow in sorted(plan["conditions"], key=lambda x: x["flow_id"]):
        if flow["split"] not in ("train", "validation"):
            continue
        config = flow["generator_config"]
        flow_path = Path(flow["path"])
        if not flow_path.is_absolute():
            flow_path = Path(dataset) / flow_path
        root_time, first = demand_root_time(read(flow_path))
        for policy in formal.POLICIES:
            rows = trajectories[flow["flow_id"], policy]
            if len(rows) != 1:
                raise ValueError(f"Unique episode 0 required: {flow['flow_id']} {policy}")
            row = rows[0]
            manifest_path = Path(row["manifest_path"])
            if not manifest_path.is_absolute():
                manifest_path = Path(dataset) / manifest_path
            manifest = read(manifest_path)
            control = manifest["control"]
            if (control["decision_interval_s"], control["simulator_step_s"], control["yellow_time_s"],
                    control["green_phase_ids"], control["all_red_time_s"], control.get("yellow_phase_id", 0)) != (
                    30, 1.0, 5, [1, 2, 3, 4], 0, 0):
                raise ValueError("Source signal protocol differs")
            if manifest["roadnet_sha256"] != formal.ROADNET_SHA256 or manifest["flow_sha256"] != flow["flow_sha256"]:
                raise ValueError("Source roadnet or flow differs")
            task = {"flow_id": flow["flow_id"], "split": flow["split"], "profile": config["profile"],
                    "demand_scale": config["demand_scale"], "source_id": config["source_id"],
                    "source_sha256": flow["source_sha256"], "cohort_sha256": flow["cohort_sha256"],
                    "cohort_id": flow["cohort_sha256"], "group_id": flow["source_sha256"] + ":" + flow["cohort_sha256"],
                    "policy": policy, "flow": copy.deepcopy(flow), "time_s": root_time,
                    "first_departure_s": first, "root_time_rule": "ceil30(first_departure+600)",
                    "task_id": flow["flow_id"] + "_" + policy, "ordinal": len(tasks),
                    "manifest_path": str(manifest_path), "manifest_file_sha256": row["manifest_sha256"],
                    "manifest": manifest, "collection": {"root_times": [root_time], "root_order": None,
                                                          "full_all_roots": True}}
            formal._check_task_sources(task)
            task["root_id"] = root_identity(task, root_time)
            tasks.append(task)
    tests = [copy.deepcopy(t) for t in parent_selection["tasks"] if t["split"] == "test"]
    if len(tests) != 12 or any(t["time_s"] != {"stress_peak": 1800, "stress_late": 2700}.get(t["profile"])
                               for t in tests):
        raise ValueError("Parent must contain the twelve repaired diagnostic roots")
    for task in tests:
        formal._check_task_sources(task)
        task["ordinal"] = len(tasks)
        task["test_role"] = "previously_inspected_diagnostic_not_fresh_holdout"
        tasks.append(task)
    counts = dict(Counter(t["split"] for t in tasks))
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"Expected all 110/13 flows plus 12 repaired roots, got {counts}")
    groups = defaultdict(set)
    for task in tasks:
        task["collected_branch_count"] = 129
        groups[task["source_sha256"], task["cohort_id"]].add(task["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Demand cohort crosses split boundary")
    return formal.balanced_order([t for t in tasks if t["split"] != "test"]) + tests


def prepare(run, dataset, source_run=None):
    run, dataset = Path(run).resolve(), Path(dataset).resolve()
    if not Path("/mnt/pan").is_mount() or not run.is_relative_to(Path("/mnt/pan")):
        raise ValueError("All outputs must be under mounted /mnt/pan")
    if source_run is None:
        raise ValueError("Explicit repaired source run required")
    source_run = Path(source_run).resolve()
    if run == source_run:
        raise ValueError("Use a new run; preserve the repaired parent")
    run.mkdir(parents=True, exist_ok=True)
    selection_path = run / "selection.json"
    if selection_path.exists():
        selection = read(selection_path)
        if selection.get("schema") != SCHEMA or selection["dataset"] != str(dataset) or selection["source_run"] != str(source_run):
            raise ValueError("Existing selection belongs to another run")
        return selection
    if any((run / "shards").glob("*/*.npz")):
        raise ValueError("Existing shards without selection cannot be adopted")
    plan_path, index_path = dataset / "dataset/dataset_plan.json", dataset / "hangzhou_dataset.index.json"
    tasks = select_tasks(read(plan_path), read(index_path), dataset, read(source_run / "selection.json"))
    selection = {"schema": SCHEMA, "dataset": str(dataset), "source_run": str(source_run),
                 "tasks": tasks, "split_roots": EXPECTED_SPLITS, "root_count": len(tasks),
                 "independent_source_cohort_groups": len({t["group_id"] for t in tasks}),
                 "trainval_root_time_rule": "ceil30(first_departure+600), required <=3360; no outcome filtering",
                 "test_role": "previously_inspected_diagnostic_not_fresh_holdout",
                 "unused_test_flow_count": 7}
    engine = importlib.import_module("cityflow")
    parent_protocol = read(source_run / "protocol.json")
    if sha256(engine.__file__) != parent_protocol["engine_binary_sha256"]:
        raise ValueError("CityFlow binary differs from the fixed repaired parent")
    retained = {key: parent_protocol[key] for key in (
        "roadnet_sha256", "master_seed", "plan_generator", "formal_setting", "branch_seconds",
        "history_seconds", "window_seconds", "interval_s", "decision_interval_s", "transition_phase",
        "transition_s", "engine_green_phases", "lane_change", "waiting_speed_lt_mps", "event_semantics",
        "root_restore", "archive_file_disabled_reason", "time_definition", "shard_size", "engine_threads")
        if key in parent_protocol}
    protocol = {**retained, "schema": SCHEMA, "source_run": str(source_run),
                "dataset": str(dataset), "dataset_plan_sha256": sha256(plan_path),
                "dataset_index_sha256": sha256(index_path), "collector_sha256": sha256(__file__),
                "engine_binary": str(engine.__file__), "engine_binary_sha256": sha256(engine.__file__),
                "root_count": len(tasks), "branch_count": len(tasks) * 129,
                "branch_upper_bound": len(tasks) * 129, "branches_per_root": 129,
                "initial_non_test_roots": 246, "initial_non_test_branches": 246 * 129,
                "pair_branches_collected": 0, "workers_max": MAX_WORKERS,
                "training_jobs": 6, "timing_branch_count": 0, "total_budget_branches": len(tasks) * 129,
                "collection_stages": ["initial"], "consistency_checks": False,
                "repeat_rollout_check": False, "conservation_gate": False,
                "shard_layout": "129 initial branches only; full2049 branch plan retained for shared query encoding",
                "test_use": selection["test_role"]}
    formal._freeze_json(selection_path, selection)
    formal._freeze_json(run / "protocol.json", protocol)
    geometry = Geometry(Path(tasks[0]["manifest"]["scenario"]["roadnet_path"]))
    formal._freeze_json(run / "static/geometry.json", geometry.to_dict())
    return selection


def collect(run, dataset, workers=20, source_run=None):
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be 1..{MAX_WORKERS}")
    selection = prepare(run, dataset, source_run)
    run = Path(run).resolve()
    import fcntl
    with (run / "collector.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        results, errors, started = [], [], time.perf_counter()
        atomic_json(run / "status.json", {"stage": "running", "pid": os.getpid(), "workers": workers,
                    "completed_roots": 0, "total_roots": len(selection["tasks"]),
                    "completed_root_branches": 0, "branch_upper_bound": len(selection["tasks"]) * 129,
                    "errors": [], "updated_at": stamp()})
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(formal.collect_root, str(run), task, "initial"): task
                       for task in selection["tasks"]}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    if (result.get("root_id") != task["root_id"] or result.get("stage") != "initial_complete"
                            or result.get("branch_count") != 129 or result.get("split") != task["split"]):
                        raise ValueError("Completed initial collection differs from declared root/count")
                    results.append(result)
                except Exception as exc:
                    errors.append({"task_id": task["task_id"], "error": repr(exc)})
                non_test = [r for r in results if r["split"] != "test"]
                if len(non_test) == 246 and not (run / "trainval_ready.json").exists():
                    atomic_json(run / "trainval_ready.json", {"stage": "trainval_ready", "root_count": 246,
                                "branch_count": 246 * 129, "finished_at": stamp()})
                atomic_json(run / "status.json", {"stage": "running", "pid": os.getpid(), "workers": workers,
                            "completed_roots": len(results), "total_roots": len(selection["tasks"]),
                            "completed_root_branches": sum(r["branch_count"] for r in results),
                            "branch_upper_bound": len(selection["tasks"]) * 129,
                            "errors": errors, "updated_at": stamp()})
        summary = {"stage": "failed" if errors else "complete", "root_count": len(results),
                   "branch_count": sum(r["branch_count"] for r in results), "errors": errors,
                   "expected_root_count": len(selection["tasks"]), "expected_branch_count": len(selection["tasks"]) * 129,
                   "unresolved_events": sum(r["unresolved_events"] for r in results),
                   "results": results, "wall_s": time.perf_counter() - started, "finished_at": stamp()}
        atomic_json(run / "summary.json", summary)
        atomic_json(run / "status.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "collect"), default="prepare")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        selected = prepare(args.run_dir, args.dataset, args.source_run)
        result = {"stage": "prepared", "root_count": selected["root_count"], "split_roots": selected["split_roots"]}
    else:
        result = collect(args.run_dir, args.dataset, args.workers, args.source_run)
    print(json.dumps(result), flush=True)
    return int(result.get("stage") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
