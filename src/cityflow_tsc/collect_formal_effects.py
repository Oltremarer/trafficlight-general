"""Frozen v3 collection: 48 initial roots first, then all remaining factors.

The source binary has a known truncated archive-dump limitation. True Engine
snapshots therefore stay in their owning process; the second phase rebuilds the
original source prefix once per root, as in the established collector. This is
normal root construction, not an extra consistency replay or an admission test.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import importlib
import json
import multiprocessing
import os
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .collect_counterfactual import create_roots, root_identity, stamp, task_progress
from .config import ControlConfig, ScenarioConfig
from .counterfactual.geometry import Geometry
from .counterfactual.plan import MASTER_SEED
from .counterfactual.recorder import SignalRunner, stack_rows
from .counterfactual.route_aware_recorder import RouteAwareRecorder
from .counterfactual.timed_plan import formal_t1_plan
from .counterfactual.writer import atomic_json, atomic_npz, sha256, shard_committed
from .simulator import CityFlowBackend


SCHEMA = "formal-effect-v3-20260911"
ROADNET_SHA256 = "11e2fe89f632e43e81f56ea87a308d544b66f9c5af8a410e16149668d6d376c1"
SHARD_SIZE = 4
BRANCHES_PER_ROOT = 2049
INITIAL_PER_ROOT = 129
PAIR_PER_ROOT = 1920
POLICIES = ("fixed_time", "max_pressure")
SPLITS = ("train", "validation", "test")
# Each row is profile, per-source demand scales, root time, and base_00/01/02 IDs.
CONDITION_ROWS = (
    ("train", "flat", (.65, .65, .65), 600, ("train_000", "train_040", "train_020")),
    ("train", "flat", (1., 1., 1.), 1800, ("train_072", "train_052", "train_032")),
    ("train", "flat", (1.4, 1.4, 1.4), 600, ("train_024", "train_004", "train_044")),
    ("train", "early", (.8, .8, .8), 600, ("train_021", "train_001", "train_041")),
    ("train", "late", (1., 1., 1.), 1800, ("train_042", "train_022", "train_002")),
    ("train", "bimodal", (1.2, 1.2, 1.2), 1800, ("train_003", "train_043", "train_023")),
    ("validation", "sharp_early", (1.15, 1.35, .75), 600,
     ("validation_004", "validation_002", "validation_000")),
    ("validation", "sharp_late", (1.15, 1.35, .75), 1800,
     ("validation_001", "validation_005", "validation_003")),
    ("test", "stress_peak", (.9, 1.3, 1.5), 600, ("test_000", "test_004", "test_002")),
    ("test", "stress_late", (.9, 1.3, 1.5), 1800, ("test_003", "test_001", "test_005")),
)


def fixed_conditions():
    return [{"split": split, "profile": profile, "demand_scale": scales[i],
             "time_s": t, "flow_id": flow_id, "source_id": f"base_{i:02d}"}
            for split, profile, scales, t, ids in CONDITION_ROWS
            for i, flow_id in enumerate(ids)]


def balanced_order(tasks):
    """Round-robin source, policy, root time and split without outcome sorting."""
    buckets = defaultdict(deque)
    for task in tasks:
        key = (task["source_id"], task["policy"], task["time_s"], task["split"])
        buckets[key].append(task)
    keys = sorted(buckets, key=lambda k: (k[0], k[2], k[1], SPLITS.index(k[3])))
    ordered = []
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].popleft())
    return ordered


def select_tasks(plan, index, dataset):
    """Resolve exact declared conditions; inspect only their source metadata."""
    conditions = {r["flow_id"]: r for r in plan["conditions"]}
    trajectories = defaultdict(list)
    for row in index["trajectories"]:
        if int(row.get("episode", 0)) == 0:
            trajectories[row["flow_id"], row["policy"]].append(row)
    digests = {}

    def checked(path, expected):
        path = Path(path)
        if path not in digests:
            digests[path] = sha256(path)
        if digests[path] != expected:
            raise IOError(f"Frozen source checksum differs: {path}")

    tasks, groups, flow_hashes = [], defaultdict(set), set()
    for ordinal, expected in enumerate(fixed_conditions()):
        flow = copy.deepcopy(conditions[expected["flow_id"]])
        config = flow["generator_config"]
        if (flow["split"] != expected["split"] or config["source_id"] != expected["source_id"]
                or config["profile"] != expected["profile"]
                or not np.isclose(config["demand_scale"], expected["demand_scale"], rtol=0, atol=1e-12)):
            raise ValueError(f"Source condition differs from approved v3 row: {expected['flow_id']}")
        group_id = flow["source_sha256"] + ":" + flow["cohort_sha256"]
        groups[group_id].add(expected["split"])
        flow_hashes.add(flow["flow_sha256"])
        for policy in POLICIES:
            choices = trajectories[flow["flow_id"], policy]
            if len(choices) != 1:
                raise ValueError(f"Need unique episode 0: {flow['flow_id']} {policy}")
            row = choices[0]
            manifest_path = Path(row["manifest_path"])
            if not manifest_path.is_absolute():
                manifest_path = Path(dataset) / manifest_path
            checked(manifest_path, row["manifest_sha256"])
            manifest = json.loads(manifest_path.read_text())
            control = manifest["control"]
            if (control["decision_interval_s"], control["simulator_step_s"], control["yellow_time_s"],
                control["green_phase_ids"], control["all_red_time_s"], control.get("yellow_phase_id", 0)) != (
                    30, 1.0, 5, [1, 2, 3, 4], 0, 0):
                raise ValueError("Source trajectory has a different signal protocol")
            if manifest["roadnet_sha256"] != ROADNET_SHA256 or manifest["flow_sha256"] != flow["flow_sha256"]:
                raise ValueError("Source manifest does not match frozen roadnet/condition")
            checked(Path(manifest["scenario"]["roadnet_path"]), manifest["roadnet_sha256"])
            checked(Path(manifest["scenario"]["flow_path"]), manifest["flow_sha256"])
            checked(manifest_path.parent / "trajectory.npz", manifest["trajectory_sha256"])
            task = {**expected, "ordinal": ordinal, "flow": flow, "policy": policy,
                    "task_id": flow["flow_id"] + "_" + policy, "group_id": group_id,
                    "source_sha256": flow["source_sha256"], "cohort_sha256": flow["cohort_sha256"],
                    "cohort_id": flow["cohort_sha256"], "manifest_path": str(manifest_path),
                    "manifest_file_sha256": row["manifest_sha256"], "manifest": manifest,
                    "collection": {"root_times": [expected["time_s"]], "root_order": None, "full_all_roots": True}}
            task["root_id"] = root_identity(task, expected["time_s"])
            tasks.append(task)
    if len(groups) != 9 or any(len(splits) != 1 for splits in groups.values()) or len(flow_hashes) != 30:
        raise ValueError("Expected 9 split-disjoint source/cohort groups and 30 distinct flow files")
    return balanced_order(tasks)


def _freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to change frozen formal run metadata: {path}")
    else:
        atomic_json(path, value)


def prepare(run, dataset):
    """Freeze 60 exact source tasks at run/selection.json and return that object."""
    run, dataset = Path(run).resolve(), Path(dataset).resolve()
    run.mkdir(parents=True, exist_ok=True)
    plan_path, index_path = dataset / "dataset/dataset_plan.json", dataset / "hangzhou_dataset.index.json"
    plan, index = json.loads(plan_path.read_text()), json.loads(index_path.read_text())
    tasks = select_tasks(plan, index, dataset)
    selection = {"schema": SCHEMA, "dataset": str(dataset), "tasks": tasks,
                 "selected_flows": [t["flow"] for t in tasks if t["policy"] == POLICIES[0]],
                 "conditions": fixed_conditions(), "root_count": 60, "demand_conditions": 30,
                 "independent_source_cohort_groups": 9, "split_roots": {"train": 36, "validation": 12, "test": 12}}
    geometry = Geometry(Path(tasks[0]["manifest"]["scenario"]["roadnet_path"]))
    if (len(geometry.intersections), len(geometry.lanes), len(geometry.boundaries)) != (16, 240, 16):
        raise ValueError("Formal model requires the declared 16/240/16 position layout")
    engine = importlib.import_module("cityflow")
    protocol = {"schema": SCHEMA, "dataset": str(dataset), "dataset_plan_sha256": sha256(plan_path),
                "dataset_index_sha256": sha256(index_path), "roadnet_sha256": ROADNET_SHA256,
                "engine_binary": str(engine.__file__), "engine_binary_sha256": sha256(engine.__file__),
                "engine_version": getattr(engine, "__version__", "not_exposed"),
                "collector_sha256": sha256(__file__), "master_seed": MASTER_SEED,
                "plan_generator": "formal_t1_joint_v2", "formal_setting": "T1", "branch_seconds": 240,
                "history_seconds": 150, "window_seconds": 5, "interval_s": 1, "decision_interval_s": 30,
                "transition_phase": 0, "transition_s": 5, "engine_green_phases": [1, 2, 3, 4],
                "lane_change": False, "waiting_speed_lt_mps": .1, "event_semantics": "route-completion-v2",
                "root_count": 60, "branch_count": 122940, "branch_upper_bound": 122940,
                "initial_non_test_roots": 48, "initial_non_test_branches": 6192,
                "shard_size": SHARD_SIZE, "workers_max": 8, "engine_threads": 1,
                "consistency_checks": False, "repeat_rollout_check": False, "conservation_gate": False,
                "root_restore": "true_engine_memory_archive; original source prefix rebuilt once per collection phase",
                "archive_file_disabled_reason": "pinned binary previously emitted truncated archive JSON",
                "shard_layout": "129 initial branches then 1920 pairs; branch_indices refer to unchanged branches.json",
                "time_definition": "right endpoint states t+1..t+240; signal applies on [t,t+1)",
                "test_use": "collection progress and errors only until all model/checkpoint selection is locked"}
    # A new run must not silently adopt already-existing data without its own contract.
    if not (run / "selection.json").exists() and any((run / "shards").glob("*/*.npz")):
        raise ValueError("Existing shards without a frozen formal selection; use a new run directory")
    _freeze_json(run / "selection.json", selection)
    _freeze_json(run / "protocol.json", protocol)
    _freeze_json(run / "static/geometry.json", geometry.to_dict())
    return selection


def stage_chunks(plan, stage):
    """Stable shard slots across stages, while leaving factor plan order intact."""
    branches = plan["branches"]
    initial = [i for i, b in enumerate(branches) if len(b["changed_intersections"]) != 2]
    pairs = [i for i, b in enumerate(branches) if len(b["changed_intersections"]) == 2]
    if (len(branches), len(initial), len(pairs)) != (BRANCHES_PER_ROOT, INITIAL_PER_ROOT, PAIR_PER_ROOT):
        raise ValueError("Unexpected formal factor count")
    if stage not in ("initial", "pairs"):
        raise ValueError("stage must be initial or pairs")
    chosen = initial if stage == "initial" else pairs
    first = 0 if stage == "initial" else (len(initial) + SHARD_SIZE - 1) // SHARD_SIZE
    return [(first + j // SHARD_SIZE, chosen[j:j + SHARD_SIZE])
            for j in range(0, len(chosen), SHARD_SIZE)]


def _root_groups(task):
    return {k: task[k] for k in ("source_id", "cohort_id", "source_sha256", "cohort_sha256", "group_id",
                                 "profile", "demand_scale", "split")}


def _check_task_sources(task):
    m = task["manifest"]
    for path, expected in ((Path(task["manifest_path"]), task["manifest_file_sha256"]),
                           (Path(m["scenario"]["roadnet_path"]), m["roadnet_sha256"]),
                           (Path(m["scenario"]["flow_path"]), m["flow_sha256"]),
                           (Path(task["manifest_path"]).parent / "trajectory.npz", m["trajectory_sha256"])):
        if sha256(path) != expected:
            raise IOError(f"Frozen source changed before collection: {path}")


def _build_root(run, task, backend, runner, recorder, actions):
    """On resume rebuild only the Engine prefix, never rewrite frozen history."""
    directory = run / "roots" / task["root_id"]
    saved = directory / "root.json"
    if not saved.exists():
        roots, archives = create_roots(run, task, backend, runner, recorder, actions)
        root = roots[0]
        root.update(_root_groups(task))
        root["archive_storage"] = "true_engine_memory_archive; source actions rebuild on second phase/restart"
        atomic_json(saved, root)
        atomic_json(run / "indexes" / (task["task_id"] + ".roots.json"), {"roots": [root]})
        return root, archives[root["root_id"]]
    root = json.loads(saved.read_text())
    if (root["root_id"] != task["root_id"] or root["time_s"] != task["time_s"]
            or root["flow_sha256"] != task["manifest"]["flow_sha256"]
            or root["policy"] != task["policy"] or root["roadnet_sha256"] != ROADNET_SHA256
            or root["simulator_seed"] != task["manifest"]["scenario"]["seed"]):
        raise ValueError("Saved root belongs to another immutable source")
    if sha256(directory / "history.npz") != root["history_sha256"]:
        raise IOError("Saved root history checksum differs")
    if any(root.get(k) != value for k, value in _root_groups(task).items()):
        # A process may stop after create_roots committed base metadata, before
        # this module added the frozen grouping fields. Fill only absent fields.
        if any(k in root and root[k] != value for k, value in _root_groups(task).items()):
            raise ValueError("Saved root grouping differs from frozen selection")
        root.update(_root_groups(task))
        atomic_json(saved, root)
    recorder.observe(backend.engine)
    for step in range(task["time_s"] // 30):
        runner.step(actions[step])
    return root, backend.capture_archive()


def _collect_stage(run, task, root, archive, backend, runner, recorder, plan, context, stage, started):
    rid, tid = root["root_id"], task["task_id"]
    marker = run / "indexes" / f"{rid}.{stage}.complete.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["branch_plan_sha256"] != sha256(run / "roots" / rid / "branches.json"):
            raise ValueError("Committed stage belongs to another branch plan")
        return result
    branch_count, unknown, total_bytes = 0, 0, 0
    for slot, indices in stage_chunks(plan, stage):
        chunk = [plan["branches"][i] for i in indices]
        ids = [b["branch_id"] for b in chunk]
        shard = run / "shards" / rid / f"part_{slot:04d}.npz"
        if not shard_committed(shard, ids):
            payloads, timings = [], []
            for branch in chunk:
                begin = time.perf_counter()
                backend.restore_archive(archive)
                runner.restore_context(root["signal_context"])
                recorder.restore_context(context)
                restored = time.perf_counter()
                rows = []
                for request in branch["requests"]:
                    rows.extend(runner.step(request))
                payloads.append(stack_rows(rows))
                timings.append([restored - begin, time.perf_counter() - restored])
            arrays = {k: np.stack([p[k] for p in payloads]) for k in payloads[0]}
            arrays.update(branch_ids=np.asarray(ids, dtype="U24"), branch_indices=np.asarray(indices, dtype=np.int32),
                          requests=np.asarray([b["requests"] for b in chunk], dtype=np.uint8),
                          timing_restore_rollout_s=np.asarray(timings, dtype=np.float64),
                          compositional_holdout=np.asarray([b["compositional_holdout"] for b in chunk]))
            checksum = atomic_npz(shard, arrays)
            atomic_json(Path(str(shard) + ".json"), {
                "root_id": rid, "stage": stage, "branch_indices": indices, "branch_ids": ids,
                "branch_count": len(ids), "sha256": checksum, "bytes": shard.stat().st_size,
                "raw_bytes": sum(a.nbytes for a in arrays.values()), "event_semantics": "route-completion-v2",
                "unresolved_events": int(arrays["unresolved_events"].sum(dtype=np.uint64)), "finished_at": stamp()})
        meta = json.loads(Path(str(shard) + ".json").read_text())
        if meta.get("branch_indices") != indices or meta.get("stage") != stage:
            raise ValueError(f"Committed shard stage/indices differ: {shard}")
        branch_count += meta["branch_count"]
        unknown += meta["unresolved_events"]
        total_bytes += meta["bytes"]
        task_progress(run, tid, stage="collecting_" + stage, root_id=rid, split=task["split"],
                      stage_branches_committed=branch_count,
                      root_branches_committed=branch_count + (INITIAL_PER_ROOT if stage == "pairs" else 0),
                      root_branch_count=task.get("collected_branch_count", BRANCHES_PER_ROOT), stage_unresolved_events=unknown,
                      elapsed_s=time.perf_counter() - started)
    result = {"task_id": tid, "root_id": rid, "split": task["split"], "stage": stage + "_complete",
              "branch_count": branch_count, "unresolved_events": unknown, "bytes": total_bytes,
              "branch_plan_sha256": sha256(run / "roots" / rid / "branches.json"), "finished_at": stamp()}
    atomic_json(marker, result)
    return result


def collect_root(run_path, task, phase):
    """One process owns each Engine; initial/pair writes never share a root lock."""
    run, tid, rid = Path(run_path), task["task_id"], task["root_id"]
    suffix = "initial.complete.json" if phase == "initial" else "complete.json"
    done = run / "indexes" / f"{rid}.{suffix}"
    if done.exists():
        return json.loads(done.read_text())
    if phase not in ("initial", "remaining"):
        raise ValueError("phase must be initial or remaining")
    import fcntl
    (run / "locks").mkdir(exist_ok=True)
    backend, started = None, time.perf_counter()
    with (run / "locks" / (rid + ".lock")).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            _check_task_sources(task)
            m = task["manifest"]
            g = Geometry(Path(m["scenario"]["roadnet_path"]))
            flows = json.loads(Path(m["scenario"]["flow_path"]).read_text())
            for flow in flows:
                if float(flow["vehicle"]["length"]) != 5. or float(flow["vehicle"]["minGap"]) != 2.5:
                    raise ValueError("Source differs from the recorder's 5 m / 2.5 m receiving proxy")
            with np.load(Path(task["manifest_path"]).parent / "trajectory.npz", allow_pickle=False) as z:
                actions = z["actions"].copy()
            if actions.shape != (120, 16):
                raise ValueError(f"Expected 120 source decisions for 16 intersections, got {actions.shape}")
            backend = CityFlowBackend(ControlConfig())
            backend.reset(ScenarioConfig(roadnet_path=Path(m["scenario"]["roadnet_path"]),
                          flow_path=Path(m["scenario"]["flow_path"]), output_dir=run / "runtime" / tid,
                          duration_s=3600, seed=int(m["scenario"]["seed"]), thread_num=1, save_replay=False))
            recorder = RouteAwareRecorder(g, flows)
            runner = SignalRunner(backend, g, recorder)
            runner.initialize()
            root, archive = _build_root(run, task, backend, runner, recorder, actions)
            if root["root_id"] != rid:
                raise ValueError("Constructed root differs from frozen identity")
            directory = run / "roots" / rid
            plan = formal_t1_plan(rid, root["signal_context"]["current_phase"], g.adjacency)
            _freeze_json(directory / "branches.json", plan)
            context = json.loads((directory / "recorder_context.json").read_text())
            initial = _collect_stage(run, task, root, archive, backend, runner, recorder,
                                     plan, context, "initial", started)
            if phase == "initial":
                task_progress(run, tid, **{k: v for k, v in initial.items() if k != "task_id"})
                return initial
            pairs = _collect_stage(run, task, root, archive, backend, runner, recorder,
                                   plan, context, "pairs", started)
            result = {"task_id": tid, "root_id": rid, "split": task["split"], "stage": "complete",
                      "branch_count": initial["branch_count"] + pairs["branch_count"],
                      "unresolved_events": initial["unresolved_events"] + pairs["unresolved_events"],
                      "bytes": initial["bytes"] + pairs["bytes"], "wall_s": time.perf_counter() - started,
                      "branch_plan_sha256": initial["branch_plan_sha256"], "finished_at": stamp()}
            atomic_json(done, result)
            task_progress(run, tid, **{k: v for k, v in result.items() if k != "task_id"})
            return result
        except Exception as exc:
            error = {"task_id": tid, "root_id": rid, "phase": phase, "stage": "failed",
                     "error": repr(exc), "traceback": traceback.format_exc(), "failed_at": stamp()}
            atomic_json(run / "logs" / f"{tid}.{phase}.failure.json", error)
            task_progress(run, tid, **{k: v for k, v in error.items() if k != "task_id"})
            raise
        finally:
            if backend is not None:
                backend.close()


def collect(run, dataset, workers=8):
    """Blocking two-phase queue. Readiness is a dependency, never a metric gate."""
    run, dataset = Path(run).resolve(), Path(dataset).resolve()
    if not Path("/mnt/pan").is_mount() or not run.is_relative_to(Path("/mnt/pan")):
        raise ValueError("All formal outputs must be under mounted /mnt/pan")
    if not 1 <= workers <= 8:
        raise ValueError("Use 1..8 CityFlow workers; each Engine has one thread")
    run.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (run / "collector.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = prepare(run, dataset)
        previous = run / "summary.json"
        if previous.exists() and json.loads(previous.read_text()).get("stage") == "complete":
            return json.loads(previous.read_text())
        tasks = selection["tasks"]
        initial_tasks = [t for t in tasks if t["split"] != "test"]
        initial_results, results, errors = [], [], []
        started = time.perf_counter()

        def progress(phase):
            atomic_json(run / "status.json", {"stage": "running", "phase": phase, "pid": os.getpid(),
                        "workers": workers, "engine_threads": 1, "updated_at": stamp(),
                        "initial_non_test_roots_completed": len(initial_results), "initial_non_test_roots": 48,
                        "completed_roots": len(results), "total_roots": 60,
                        "completed_root_branches": sum(r["branch_count"] for r in results),
                        "branch_upper_bound": 122940, "errors": errors, "consistency_checks": False})

        progress("initial_non_test")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(collect_root, str(run), task, "initial"): task for task in initial_tasks}
            failed_initial = set()
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    initial_results.append(future.result())
                except Exception as exc:
                    failed_initial.add(task["root_id"])
                    errors.append({"task_id": task["task_id"], "root_id": task["root_id"],
                                   "phase": "initial", "error": repr(exc)})
                progress("initial_non_test")
            if len(initial_results) == 48:
                atomic_json(run / "initial_ready.json", {"stage": "initial_ready", "root_count": 48,
                            "branch_count": 6192, "root_ids": sorted(r["root_id"] for r in initial_results),
                            "splits": ["train", "validation"], "finished_at": stamp()})
            # Complete train/validation factors first so B can overlap final test collection.
            # Initial failures do not suppress independent roots or trigger extra automatic trials.
            remaining = [t for t in initial_tasks if t["root_id"] not in failed_initial]
            remaining += [t for t in tasks if t["split"] == "test"]
            futures = {pool.submit(collect_root, str(run), task, "remaining"): task for task in remaining}
            progress("remaining_factors")
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({"task_id": task["task_id"], "root_id": task["root_id"],
                                   "phase": "remaining", "error": repr(exc)})
                non_test = [r for r in results if r["split"] != "test"]
                if len(non_test) == 48 and not (run / "trainval_pairs_ready.json").exists():
                    atomic_json(run / "trainval_pairs_ready.json", {
                        "stage": "trainval_pairs_ready", "root_count": 48,
                        "branch_count": 98352, "pair_count": 92160,
                        "root_ids": sorted(r["root_id"] for r in non_test), "finished_at": stamp()})
                progress("remaining_factors")
        summary = {"stage": "failed" if errors else "complete", "root_count": len(results),
                   "branch_count": sum(r["branch_count"] for r in results),
                   "unresolved_events": sum(r["unresolved_events"] for r in results),
                   "bytes": sum(r["bytes"] for r in results), "expected_root_count": 60,
                   "expected_branch_count": 122940, "initial_results": initial_results, "results": results,
                   "errors": errors, "wall_s": time.perf_counter() - started, "finished_at": stamp(),
                   "consistency_checks": False, "scope": "collection_only; no model or test-result evaluation"}
        atomic_json(run / "summary.json", summary)
        atomic_json(run / "status.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    result = collect(args.run_dir, args.dataset, args.workers)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["stage"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
