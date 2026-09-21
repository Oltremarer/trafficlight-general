"""Six pre-state-selected roots, sharing one <=8-process CityFlow pool."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from .collect_counterfactual import create_roots, stamp, task_progress
from .config import ControlConfig, ScenarioConfig
from .simulator import CityFlowBackend
from .counterfactual.geometry import Geometry
from .counterfactual.recorder import SignalRunner, stack_rows
from .counterfactual.route_aware_recorder import RouteAwareRecorder
from .counterfactual.timed_plan import timing_plan
from .counterfactual.writer import atomic_json, atomic_npz, sha256, shard_committed


def select_roots(roots):
    """Receiving proxy first, then lower/upper backlog tertiles; no future filtering."""
    ordered = sorted(roots, key=lambda r: (r["backlog"], r["root_id"]))
    third = max(1, len(ordered) // 3)
    groups = {
        "receiving_unavailable": sorted([r for r in roots if r["unavailable_lanes"] > 0],
                                         key=lambda r: (-r["unavailable_lanes"], r["root_id"])),
        "lower_queue": ordered[:third],
        "higher_queue": list(reversed(ordered[-third:])),
    }
    selected, used, coverage = [], set(), {}
    for name in ("receiving_unavailable", "lower_queue", "higher_queue"):
        choices = [r for r in groups[name] if r["root_id"] not in used][:2]
        coverage[name] = len(choices)
        for root in choices:
            used.add(root["root_id"])
            selected.append({"selection_stratum": name, "root": root})
    return selected, coverage


def inventory_demands(old_dataset, output):
    old = json.loads((old_dataset / "protocol.json").read_text())
    source = Path(old["dataset"])
    plan_path, index_path = source / "dataset/dataset_plan.json", source / "hangzhou_dataset.index.json"
    plan, index = json.loads(plan_path.read_text()), json.loads(index_path.read_text())
    used_ids = {json.loads(p.read_text())["flow_id"] for p in (old_dataset / "roots").glob("*/root.json")}
    groups = defaultdict(list)
    for row in plan["conditions"]:
        groups[(row["source_sha256"], row["cohort_sha256"])].append(row)
    records = []
    for (source_sha, cohort), rows in sorted(groups.items()):
        records.append({"source_sha256": source_sha, "cohort_sha256": cohort,
                        "flow_ids": [r["flow_id"] for r in rows], "source_splits": sorted({r["split"] for r in rows}),
                        "used_in_retained_old_roots": bool(used_ids.intersection(r["flow_id"] for r in rows)),
                        "conditions": [{"flow_id": r["flow_id"], "path": r["path"],
                                        "flow_sha256": r["flow_sha256"], "generator_config": r["generator_config"]} for r in rows]})
    result = {"dataset_plan": str(plan_path), "dataset_plan_sha256": sha256(plan_path),
              "dataset_index": str(index_path), "index_sha256": sha256(index_path),
              "flow_files": len(plan["conditions"]), "trajectory_records": len(index["trajectories"]),
              "group_definition": "same source_sha256 and cohort_sha256; scaling/profile/jitter variants stay grouped",
              "total_groups": len(records), "unused_by_retained_old_roots_upper_bound": sum(not r["used_in_retained_old_roots"] for r in records),
              "required_new_groups": 30, "groups": records,
              "formal_collection_state": "needs_user_direction" if len(records) < 30 else "requires_group_selection",
              "limitation": "Current indexed dataset only; not a claim about every dataset elsewhere or historical deleted experiments"}
    atomic_json(output / "demand_inventory.json", result)
    return result


def prepare(run, old_dataset):
    timing = run / "timing"; timing.mkdir(exist_ok=True)
    selected, coverage = select_roots([json.loads(p.read_text()) for p in sorted((old_dataset / "roots").glob("*/root.json"))])
    prior = json.loads((old_dataset / "selection.json").read_text())
    lookup = {(t["flow"]["flow_id"], t["policy"]): t for t in prior["tasks"]}
    tasks = []
    for selected_root in selected:
        root = selected_root["root"]
        task = copy.deepcopy(lookup[root["flow_id"], root["policy"]])
        task["task_id"] = root["root_id"] + "_timing"
        task["source_root_id"] = root["root_id"]
        task["selection_stratum"] = selected_root["selection_stratum"]
        task["manifest_file_sha256"] = sha256(Path(task["manifest_path"]))
        task["collection"] = {"root_times": [root["time_s"]], "root_order": None, "full_all_roots": True}
        tasks.append(task)
    geometry = Geometry(Path(tasks[0]["manifest"]["scenario"]["roadnet_path"]))
    atomic_json(timing / "static/geometry.json", geometry.to_dict())
    config = {"old_dataset": str(old_dataset), "formal_setting": "T1", "branch_seconds": 240,
              "history_seconds": 150, "window_seconds": 5, "coverage": coverage,
              "root_selection": "pre-root proxy receiving>0 descending first; then lowest/highest backlog tertiles, unique roots",
              "pair_selection": "8 adjacent +4 two-hop +4 farther; root-keyed fixed seed; lower-index A to higher-index B",
              "branch_upper_bound": len(tasks) * 16 * 57, "actual_roots": len(tasks), "workers_max": 8,
              "engine_threads": 1, "event_semantics": "route-completion-v2", "shard_size": 4,
              "consistency_checks": False, "conservation_gate": False,
              "limitations": ["T1 changes duration too; not an isolated propagation-delay experiment",
                              "T2 60-second offset is not a physical travel-time cutoff",
                              "receiving_unavailable and speed_loss remain proxies, not full spillback or trip delay"],
              "tasks": tasks}
    path = timing / "selection.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Frozen timing selection changed")
    atomic_json(path, config)
    return tasks, config


def collect_root(run_path, task):
    run, tid = Path(run_path), task["task_id"]
    done = run / "indexes" / (tid + ".complete.json")
    if done.exists():
        return json.loads(done.read_text())
    import fcntl
    (run / "locks").mkdir(exist_ok=True)
    with (run / "locks" / (tid + ".lock")).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _collect_root(run, task, done)


def _collect_root(run, task, done):
    tid, backend, started = task["task_id"], None, time.perf_counter()
    try:
        m = task["manifest"]
        for source, digest in ((Path(task["manifest_path"]), task["manifest_file_sha256"]),
                               (Path(m["scenario"]["roadnet_path"]), m["roadnet_sha256"]),
                               (Path(m["scenario"]["flow_path"]), m["flow_sha256"]),
                               (Path(task["manifest_path"]).parent / "trajectory.npz", m["trajectory_sha256"])):
            if sha256(source) != digest:
                raise IOError(f"Input source has changed: {source}")
        g = Geometry(Path(m["scenario"]["roadnet_path"]))
        flows = json.loads(Path(m["scenario"]["flow_path"]).read_text())
        with np.load(Path(task["manifest_path"]).parent / "trajectory.npz") as z:
            actions = z["actions"]
        backend = CityFlowBackend(ControlConfig())
        backend.reset(ScenarioConfig(roadnet_path=Path(m["scenario"]["roadnet_path"]),
                      flow_path=Path(m["scenario"]["flow_path"]), output_dir=run / "runtime" / tid,
                      duration_s=3600, seed=int(m["scenario"]["seed"]), thread_num=1, save_replay=False))
        recorder = RouteAwareRecorder(g, flows)
        runner = SignalRunner(backend, g, recorder); runner.initialize()
        roots, archives = create_roots(run, task, backend, runner, recorder, actions)
        root = roots[0]; rid = root["root_id"]
        plan = timing_plan(rid, root["signal_context"]["current_phase"], g.adjacency)
        directory = run / "roots" / rid
        plan_path = directory / "branches.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing branch requests differ from frozen full-sequence plan")
        atomic_json(plan_path, plan)
        context = json.loads((directory / "recorder_context.json").read_text())
        total_unknown, total_bytes = 0, 0
        for offset in range(0, plan["branch_count"], 4):
            chunk = plan["branches"][offset:offset + 4]; ids = [b["branch_id"] for b in chunk]
            shard = run / "shards" / rid / f"part_{offset // 4:04d}.npz"
            if not shard_committed(shard, ids):
                payloads, timings = [], []
                for branch in chunk:
                    begin = time.perf_counter(); backend.restore_archive(archives[rid])
                    runner.restore_context(root["signal_context"]); recorder.restore_context(context)
                    restored = time.perf_counter(); rows = []
                    for request in branch["requests"]:
                        # No phase/elapsed reset when returning to common absolute-time requests.
                        rows.extend(runner.step(request))
                    payloads.append(stack_rows(rows)); timings.append([restored - begin, time.perf_counter() - restored])
                arrays = {k: np.stack([p[k] for p in payloads]) for k in payloads[0]}
                arrays.update(branch_ids=np.asarray(ids, dtype="U24"), requests=np.asarray([b["requests"] for b in chunk], dtype=np.uint8),
                              timing_restore_rollout_s=np.asarray(timings),
                              compositional_holdout=np.asarray([b["compositional_holdout"] for b in chunk]))
                checksum = atomic_npz(shard, arrays)
                atomic_json(Path(str(shard) + ".json"), {"branch_ids": ids, "branch_count": len(chunk),
                            "root_id": rid, "sha256": checksum, "bytes": shard.stat().st_size,
                            "unresolved_events": int(arrays["unresolved_events"].sum(dtype=np.uint64)), "finished_at": stamp()})
            meta = json.loads(Path(str(shard) + ".json").read_text())
            total_unknown += meta["unresolved_events"]; total_bytes += meta["bytes"]
            task_progress(run, tid, stage="collecting", root_id=rid, root_branches_committed=offset + len(chunk),
                          root_branch_count=plan["branch_count"], unresolved_events=total_unknown,
                          elapsed_s=time.perf_counter() - started)
        result = {"task_id": tid, "root_id": rid, "branch_count": plan["branch_count"],
                  "unresolved_events": total_unknown, "bytes": total_bytes, "wall_s": time.perf_counter() - started,
                  "stage": "complete", "finished_at": stamp(), "selection_stratum": task["selection_stratum"]}
        atomic_json(done, result); task_progress(run, **result)
        return result
    except Exception as exc:
        error = {"task_id": tid, "stage": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
        atomic_json(run / "logs" / (tid + ".failure.json"), error)
        raise
    finally:
        if backend is not None:
            backend.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--old-dataset", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/mnt/pan").is_mount() or not args.run_dir.resolve().is_relative_to(Path("/mnt/pan")):
        raise ValueError("Mounted /mnt/pan required")
    run = args.run_dir
    inventory_demands(args.old_dataset, run)
    tasks, config = prepare(run, args.old_dataset)
    timing = run / "timing"
    import fcntl
    with (timing / "pool.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (timing / "summary.json").exists():
            return
        results, errors = [], []
        analysis_errors = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(8, len(tasks)), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(collect_root, str(timing), task): task["task_id"] for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    # Run once per completed root; remaining CityFlow workers continue independently.
                    from .counterfactual.timing_analysis import analyze_root
                    try:
                        analyze_root(timing, result["root_id"])
                    except Exception as exc:
                        analysis_errors.append({"root_id": result["root_id"], "error": repr(exc), "traceback": traceback.format_exc()})
                except Exception as exc:
                    errors.append({"task_id": futures[future], "error": repr(exc)})
                atomic_json(timing / "queue.json", {"completed_roots": len(results), "total_roots": len(tasks), "errors": errors,
                                                    "analysis_errors": analysis_errors})
        atomic_json(timing / "summary.json", {"stage": "failed" if errors else "analysis_failed" if analysis_errors else "complete", "results": results,
                    "errors": errors, "analysis_errors": analysis_errors, "coverage": config["coverage"], "finished_at": stamp()})


if __name__ == "__main__":
    main()
