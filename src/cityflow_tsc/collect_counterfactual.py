from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

from .config import ControlConfig, ScenarioConfig
from .simulator import CityFlowBackend
from .counterfactual import SCHEMA
from .counterfactual.geometry import Geometry
from .counterfactual.plan import ROOT_TIMES, MASTER_SEED, digest, select_flows, make_branches, full_root_index
from .counterfactual.recorder import PhysicalRecorder, SignalRunner, stack_rows
from .counterfactual.route_aware_recorder import RouteAwareRecorder
from .counterfactual.writer import atomic_json, atomic_npz, sha256, shard_committed


RECORDER_CLASSES = {"legacy": PhysicalRecorder, "route-completion-v2": RouteAwareRecorder}


class CollectionBudgetExpired(Exception):
    pass


def check_budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise CollectionBudgetExpired()


def committed_task_stats(run, task_id):
    roots = list((run / "roots").glob(f"{task_id}_t*/root.json"))
    records = [json.loads(p.read_text()) for p in (run / "shards").glob(f"{task_id}_t*/*.npz.json")]
    return {"root_count": len(roots),
            **{key: sum(r.get(key, 0) for r in records)
               for key in ("branch_count", "unresolved_events", "internal_trip_completion_count",
                           "trip_completion_count", "boundary_exit_count", "bytes")}}


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def task_progress(run, task_id, **values):
    atomic_json(run / "logs" / f"{task_id}.progress.json", {"updated_at": stamp(), **values})


def collection_config(root_times=None, root_order=None, full_all_roots=False):
    times = list(ROOT_TIMES if root_times is None else root_times)
    if (not times or len(times) != len(set(times))
            or any(not isinstance(t, int) or t < 150 or t > 3420 or t % 30 for t in times)):
        raise ValueError("Root times must be unique 30-second boundaries in 150..3420")
    times.sort()
    if not full_all_roots and times != list(ROOT_TIMES):
        raise ValueError("Custom root times require full-all-selected-roots")
    order = None if root_order is None else list(root_order)
    if order is not None and (len(order) != len(times) or set(order) != set(times)):
        raise ValueError("Root order must contain every selected root time exactly once")
    return {"root_times": times, "root_order": order, "full_all_roots": bool(full_all_roots)}


def ordered_tasks(tasks, requested=None):
    if requested is None:
        return tasks
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate task IDs are not allowed")
    lookup = {task["task_id"]: task for task in tasks}
    unknown = set(requested) - lookup.keys()
    if unknown:
        raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
    # Preserve source ordinals: they determine the four-intersection panel.
    return [lookup[tid] for tid in requested]


def collection_counts(tasks, settings):
    roots = len(tasks) * len(settings["root_times"])
    full = roots if settings["full_all_roots"] else len(tasks)
    return {"root_count": roots, "full_root_count": full,
            "branch_upper_bound": full * 1382 + (roots - full) * 337}


def order_roots(roots, settings):
    if settings["root_order"] is not None:
        rank = {t: i for i, t in enumerate(settings["root_order"])}
        return sorted(roots, key=lambda r: rank[r["time_s"]])
    return sorted(roots, key=lambda r: (not r["full_pair_root"], r["time_s"]))


def prepare(run, dataset, event_semantics="legacy", task_ids=None, collection=None):
    settings = collection_config(**(collection or {}))
    protocol_path = run / "protocol.json"
    selection_path = run / "selection.json"
    existing = None
    if protocol_path.exists() != selection_path.exists():
        raise ValueError("Incomplete protocol/selection pair; use a new run directory")
    if protocol_path.exists() and selection_path.exists():
        existing = json.loads(protocol_path.read_text())
        if existing.get("schema") != SCHEMA or existing.get("consistency_checks") is not False:
            raise ValueError("Existing run uses a different collection protocol")
        if existing.get("event_semantics", "legacy") != event_semantics:
            raise ValueError("Refusing to mix recorder event semantics in an existing run")
        if existing.get("integrity_contract") != 1:
            raise ValueError("Existing run lacks the immutable integrity contract; use a new run directory")
        if existing.get("dataset") != str(dataset):
            raise ValueError("Refusing to resume with a different dataset")
    plan_path = dataset / "dataset" / "dataset_plan.json"
    index_path = dataset / "hangzhou_dataset.index.json"
    plan = json.loads(plan_path.read_text())
    index = json.loads(index_path.read_text())
    selected = select_flows(plan["conditions"])
    tasks = []
    for ordinal, flow in enumerate(selected):
        for policy in ("fixed_time", "max_pressure"):
            tid = flow["flow_id"] + "_" + policy
            if task_ids is not None and tid not in task_ids:
                continue
            choices = [r for r in index["trajectories"] if r["flow_id"] == flow["flow_id"]
                       and r["policy"] == policy and int(r.get("episode", 0)) == 0]
            if len(choices) != 1:
                raise ValueError(f"Cannot resolve unique episode for {flow['flow_id']} {policy}")
            row = choices[0]
            manifest_path = Path(row["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            if sha256(manifest_path) != row["manifest_sha256"]:
                raise IOError(f"Manifest file has changed: {manifest_path}")
            control = manifest["control"]
            if (control["decision_interval_s"], control["simulator_step_s"], control["yellow_time_s"],
                control["green_phase_ids"], control["all_red_time_s"]) != (30, 1.0, 5, [1, 2, 3, 4], 0):
                raise ValueError("Source trajectory has a different action protocol")
            for path, expected in ((Path(manifest["scenario"]["flow_path"]), manifest["flow_sha256"]),
                                   (Path(manifest["scenario"]["roadnet_path"]), manifest["roadnet_sha256"]),
                                   (manifest_path.parent / "trajectory.npz", manifest["trajectory_sha256"])):
                if sha256(path) != expected:
                    raise IOError(f"Input file checksum differs from manifest: {path}")
            tasks.append({"ordinal": ordinal, "flow": flow, "policy": policy,
                          "task_id": tid, "collection": settings,
                          "manifest_path": str(manifest_path), "manifest": manifest})
    tasks = ordered_tasks(tasks, task_ids)
    selected_ids = {task["flow"]["flow_id"] for task in tasks}
    selected = [flow for flow in selected if flow["flow_id"] in selected_ids]
    g = Geometry(Path(plan["roadnet_path"]))
    import cityflow
    source_dir = Path(__file__).parent
    source_hashes = {str(p.relative_to(source_dir)): sha256(p)
                     for p in [Path(__file__), source_dir / "simulator.py", source_dir / "config.py",
                               *sorted((source_dir / "counterfactual").glob("*.py"))]}
    protocol = {
        "schema": SCHEMA, "integrity_contract": 1,
        "event_semantics": event_semantics, "created_at": stamp(), "dataset": str(dataset),
        "dataset_plan_sha256": sha256(plan_path), "dataset_index_sha256": sha256(index_path),
        "roadnet_sha256": sha256(Path(plan["roadnet_path"])),
        "engine_binary": cityflow.__file__, "engine_binary_sha256": sha256(cityflow.__file__),
        "source_hashes": source_hashes, "master_seed": MASTER_SEED,
        "consistency_checks": False, "repeat_rollout_check": False, "conservation_gate": False,
        "interval_s": 1, "decision_interval_s": 30, "transition_phase": 0, "transition_s": 5,
        "engine_green_phases": [1, 2, 3, 4], "lane_change": False,
        "branch_seconds": 180, "window_seconds": 5, "history_seconds": 150,
        "root_times": settings["root_times"], "collection": settings,
        "spatial_end_m": 100, "waiting_speed_lt_mps": 0.1,
        **collection_counts(tasks, settings),
        "collection_scope": "physical_counterfactual_data_only; no model training",
        "root_restore": "engine_in_memory_archive; rebuild roots by original action replay on restart",
        "time_definition": "phase_used on [t,t+1); physical state after next_step; right-endpoint costs",
        "lane_events_columns": ["enter", "leave"],
        "movement_events_columns": ["release_to_lanelink", "receive_from_lanelink"],
        "intersection_nq_columns": ["vehicle_count", "waiting_count"],
        "lane_tail_columns": ["distance_from_lane_entrance_m", "speed_mps"],
        "tail_validity": "valid iff sum(lane_n)>0; zeros in empty lane are masked placeholders",
        "loss_definition": "sum max(0,1-v/min(vehicle_max_speed,lane_speed_limit)); lanelink uses both connected lanes",
        "receiving_definition": "unavailable iff nonempty and tail_distance<=tail_length+5 and tail_speed<2",
        "limitations": ["No independent same-root replay consistency check, per user instruction",
                        "No per-tick conservation gate; unresolved events recorded explicitly",
                        "Generated/deleted within a single tick may be absent from public vehicle-ID readouts",
                        "receiving_unavailable is not proof that an upstream vehicle was blocked",
                        "Future flow/route metadata is recorder-only, not model input",
                        "Engine archive dump/load disabled after truncated JSON files occurred in actual collection"],
    }
    if event_semantics == "route-completion-v2":
        protocol["completion_definitions"] = {
            "exit_count": "geographical boundary exits; unchanged legacy meaning",
            "internal_trip_completion_count": "vanished on declared terminal internal road within one-tick speed-bound reach of lane end",
            "internal_trip_completion_by_lane": "internal trip completion count indexed by physical lane",
            "trip_completion_count": "exit_count + internal_trip_completion_count",
            "unresolved_events": "remaining unexplained transitions after route-aware completion classification"}
    selection = {"selected_flows": selected, "tasks": tasks}
    if existing is not None:
        verify_resume_contract(existing, protocol, json.loads(selection_path.read_text()), selection)
        if json.loads((run / "static" / "geometry.json").read_text()) != g.to_dict():
            raise ValueError("Frozen geometry differs from current input")
        return existing, tasks
    atomic_json(run / "static" / "geometry.json", g.to_dict())
    atomic_json(protocol_path, protocol)
    atomic_json(selection_path, selection)
    return protocol, tasks


def verify_resume_contract(existing, current, saved_selection, current_selection):
    """Reject mixed versions before rewriting any roots, history, or protocol."""
    ignored = {"created_at", "execution"}
    old = {k: v for k, v in existing.items() if k not in ignored}
    new = {k: v for k, v in current.items() if k not in ignored}
    if old != new:
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        raise ValueError(f"Immutable collection contract changed: {changed}")
    if saved_selection != current_selection:
        raise ValueError("Frozen task selection or source manifests changed")


def freeze_execution(run, protocol, tasks, shard_size):
    settings = protocol.get("collection", collection_config())
    execution = {"task_ids": [task["task_id"] for task in tasks],
                 "root_upper_bound": len(tasks) * len(settings["root_times"]), "shard_size": shard_size,
                 "collection": settings,
                 "stop_boundary": "complete shard or root-building decision step"}
    if "execution" in protocol:
        if protocol["execution"] != execution:
            raise ValueError("Frozen task selection or shard-size changed; use a new run directory")
    else:
        protocol["execution"] = execution
        atomic_json(run / "protocol.json", protocol)


def verify_completed_task(run, task, shard_size, result):
    """A completion marker is not evidence without its planned, intact payloads."""
    roots = json.loads((run / "indexes" / f"{task['task_id']}.roots.json").read_text())["roots"]
    settings = task.get("collection", collection_config())
    expected_roots = {root_identity(task, t) for t in settings["root_times"]}
    if len(roots) != len(expected_roots) or {r["root_id"] for r in roots} != expected_roots:
        raise ValueError("Completed task root IDs do not match the frozen plan")
    branch_total = 0
    for root in roots:
        rid = root["root_id"]
        directory = run / "roots" / rid
        saved_root = json.loads((directory / "root.json").read_text())
        if saved_root != root or sha256(directory / "history.npz") != root["history_sha256"]:
            raise IOError(f"Completed root metadata/history mismatch: {rid}")
        plan = json.loads((directory / "branches.json").read_text())
        branches = plan["branches"]
        ids = [b["branch_id"] for b in branches]
        marker = json.loads((run / "indexes" / f"{rid}.complete.json").read_text())
        if marker.get("branch_plan_sha256") != sha256(directory / "branches.json"):
            raise IOError(f"Completed root branch plan checksum mismatch: {rid}")
        if (plan["root_id"] != rid or plan["branch_count"] != len(ids)
                or len(ids) != len(set(ids)) or marker["branch_count"] != len(ids)
                or marker["root_id"] != rid):
            raise ValueError(f"Completed root branch plan/count mismatch: {rid}")
        expected_shards = set()
        for offset in range(0, len(ids), shard_size):
            shard = run / "shards" / rid / f"part_{offset // shard_size:04d}.npz"
            expected_shards.add(shard.name)
            if not shard_committed(shard, ids[offset:offset + shard_size]):
                raise IOError(f"Completed task has a missing/uncommitted shard: {shard}")
        actual = {p.name for p in (run / "shards" / rid).glob("*.npz")}
        metas = {p.name[:-5] for p in (run / "shards" / rid).glob("*.npz.json")}
        if actual != expected_shards or metas != expected_shards:
            raise ValueError(f"Completed root shard inventory mismatch: {rid}")
        branch_total += len(ids)
    if (result["task_id"] != task["task_id"] or result["root_count"] != len(roots)
            or result["branch_count"] != branch_total):
        raise ValueError("Completed task totals differ from committed branch plans")


def root_identity(task, t):
    m = task["manifest"]
    suffix = digest([SCHEMA, m["flow_sha256"], m["roadnet_sha256"],
                     task["policy"], m["scenario"]["seed"], t])[:10]
    return f"{task['task_id']}_t{t:04d}_{suffix}"


def create_roots(run, task, backend, runner, recorder, actions, deadline=None):
    settings = task.get("collection", collection_config())
    root_times = settings["root_times"]
    ready = run / "indexes" / f"{task['task_id']}.roots.json"
    roots = []
    archives = {}
    history = deque(maxlen=150)
    recorder.observe(backend.engine)
    for step in range(root_times[-1] // 30):
        check_budget(deadline)
        history.extend(runner.step(actions[step]))
        t = (step + 1) * 30
        if t not in root_times:
            continue
        rid = root_identity(task, t)
        directory = run / "roots" / rid
        directory.mkdir(parents=True, exist_ok=True)
        # Keep the actual Engine snapshot in its owning process. The pinned
        # Engine can emit truncated archive JSON, so restart rebuilds this
        # prefix from the immutable source actions instead of loading that file.
        archives[rid] = backend.capture_archive()
        history_hash = atomic_npz(directory / "history.npz", stack_rows(list(history)))
        last = history[-1]
        root = {
            "root_id": rid, "time_s": t, "flow_id": task["flow"]["flow_id"],
            "split": task["flow"]["split"], "policy": task["policy"],
            "manifest_path": task["manifest_path"],
            "flow_sha256": task["manifest"]["flow_sha256"],
            "roadnet_sha256": task["manifest"]["roadnet_sha256"],
            "simulator_seed": task["manifest"]["scenario"]["seed"],
            "signal_context": runner.context(),
            "backlog": int(last["lane_q"].sum(dtype=np.int64)
                           + last["intersection_nq"][:, 1].sum(dtype=np.int64)
                           + last["boundary_pending"].sum(dtype=np.int64)),
            "active_vehicles": int(last["active_vehicles"]),
            "unavailable_lanes": int(last["receiving_unavailable"].sum()),
            "history_sha256": history_hash,
            "archive_storage": "engine_memory; rebuilt_from_original_actions_on_restart",
        }
        atomic_json(directory / "recorder_context.json", recorder.context())
        atomic_json(directory / "root.json", root)
        roots.append(root)
        task_progress(run, task["task_id"], stage="building_roots", roots_saved=len(roots), time_s=t)
    complete_index = None if settings["full_all_roots"] else full_root_index(roots, task["ordinal"], task["policy"])
    for i, root in enumerate(roots):
        root["full_pair_root"] = settings["full_all_roots"] or i == complete_index
        atomic_json(run / "roots" / root["root_id"] / "root.json", root)
    atomic_json(ready, {"roots": roots})
    return roots, archives


def collect_task(run_path, task, shard_size, event_semantics="legacy", deadline=None):
    run = Path(run_path)
    tid = task["task_id"]
    done_file = run / "indexes" / f"{tid}.complete.json"
    start = time.perf_counter()
    backend = None
    try:
        if done_file.exists():
            result = json.loads(done_file.read_text())
            verify_completed_task(run, task, shard_size, result)
            task_progress(run, stage="complete", **result)
            return result
        check_budget(deadline)
        m = task["manifest"]
        g = Geometry(Path(m["scenario"]["roadnet_path"]))
        flows = json.loads(Path(m["scenario"]["flow_path"]).read_text())
        for flow in flows:
            v = flow["vehicle"]
            if float(v["length"]) != 5.0 or float(v["minGap"]) != 2.5:
                raise ValueError("Selected flow differs from fixed 5 m / 2.5 m receiving semantics")
        with np.load(Path(task["manifest_path"]).parent / "trajectory.npz", allow_pickle=False) as data:
            actions = data["actions"].copy()
        backend = CityFlowBackend(ControlConfig())
        backend.reset(ScenarioConfig(
            roadnet_path=Path(m["scenario"]["roadnet_path"]), flow_path=Path(m["scenario"]["flow_path"]),
            output_dir=run / "runtime" / tid, duration_s=3600, seed=int(m["scenario"]["seed"]),
            thread_num=1, save_replay=False))
        recorder = RECORDER_CLASSES[event_semantics](g, flows)
        runner = SignalRunner(backend, g, recorder)
        runner.initialize()
        roots, archives = create_roots(run, task, backend, runner, recorder, actions, deadline)
        panel_ids = (["intersection_1_1", "intersection_1_2", "intersection_2_1", "intersection_2_2"]
                     if task["ordinal"] % 2 == 0 else
                     ["intersection_2_2", "intersection_2_3", "intersection_3_2", "intersection_3_3"])
        panel = [g.node_index[i] for i in panel_ids]
        total = 0
        unresolved_total = 0
        roots = order_roots(roots, task.get("collection", collection_config()))
        prepared_roots = []
        # Freeze all selected-root action lists before collecting the first root.
        for root in roots:
            rid = root["root_id"]
            directory = run / "roots" / rid
            a0 = root["signal_context"]["current_phase"]
            branches, pairs = make_branches(rid, a0, root["full_pair_root"], g.adjacency, panel)
            plan_path = directory / "branches.json"
            branch_plan = {"root_id": rid, "branch_count": len(branches), "selected_pairs": pairs,
                           "panel_intersection_ids": panel_ids if root["full_pair_root"] else [],
                           "branches": branches}
            if plan_path.exists() and digest(json.loads(plan_path.read_text())) != digest(branch_plan):
                raise ValueError(f"Refusing to mix branch plans for {rid}")
            atomic_json(plan_path, branch_plan)
            prepared_roots.append((root, branches))
        for root_no, (root, branches) in enumerate(prepared_roots):
            rid = root["root_id"]
            directory = run / "roots" / rid
            plan_path = directory / "branches.json"
            archive = archives[rid]
            context = json.loads((directory / "recorder_context.json").read_text())
            root_unresolved = 0
            for offset in range(0, len(branches), shard_size):
                check_budget(deadline)
                chunk = branches[offset:offset + shard_size]
                ids = [r["branch_id"] for r in chunk]
                shard = run / "shards" / rid / f"part_{offset // shard_size:04d}.npz"
                if shard_committed(shard, ids):
                    info = json.loads(Path(str(shard) + ".json").read_text())
                    root_unresolved += info.get("unresolved_events", 0)
                    continue
                payloads = []
                timings = []
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
                arrays["branch_ids"] = np.asarray(ids, dtype="U24")
                arrays["requests"] = np.asarray([r["requests"] for r in chunk], dtype=np.uint8)
                arrays["timing_restore_rollout_s"] = np.asarray(timings, dtype=np.float64)
                arrays["compositional_holdout"] = np.asarray([r["compositional_holdout"] for r in chunk])
                write_start = time.perf_counter()
                checksum = atomic_npz(shard, arrays)
                unresolved = int(arrays["unresolved_events"].sum(dtype=np.uint64))
                root_unresolved += unresolved
                info = {"root_id": rid, "branch_ids": ids, "sha256": checksum,
                        "branch_count": len(chunk), "unresolved_events": unresolved,
                        "event_semantics": event_semantics,
                        "boundary_exit_count": int(arrays["exit_count"].sum(dtype=np.uint64)),
                        "internal_trip_completion_count": int(arrays.get("internal_trip_completion_count", np.asarray(0)).sum(dtype=np.uint64)),
                        "trip_completion_count": int(arrays.get("trip_completion_count", arrays["exit_count"]).sum(dtype=np.uint64)),
                        "raw_bytes": sum(a.nbytes for a in arrays.values()), "bytes": shard.stat().st_size,
                        "write_s": time.perf_counter() - write_start, "finished_at": stamp()}
                atomic_json(Path(str(shard) + ".json"), info)
                task_progress(run, tid, stage="collecting", root_id=rid,
                              roots_completed=root_no, root_branch_count=len(branches),
                              root_branches_committed=offset + len(chunk),
                              branches_committed=total + offset + len(chunk),
                              mean_branch_restore_rollout_s=float(np.mean(np.sum(timings, axis=1))),
                              unresolved_events=unresolved_total + root_unresolved,
                              last_shard_bytes=info["bytes"])
            total += len(branches)
            unresolved_total += root_unresolved
            atomic_json(run / "indexes" / f"{rid}.complete.json",
                        {"root_id": rid, "branch_count": len(branches), "unresolved_events": root_unresolved,
                         "branch_plan_sha256": sha256(plan_path),
                         "finished_at": stamp()})
        result = {"task_id": tid, "root_count": len(roots), "branch_count": total,
                  "unresolved_events": unresolved_total, "wall_time_s": time.perf_counter() - start,
                  "finished_at": stamp(), **committed_task_stats(run, tid)}
        atomic_json(done_file, result)
        task_progress(run, stage="complete", **result)
        return result
    except CollectionBudgetExpired:
        result = {"task_id": tid, "stage": "budget_exhausted", "event_semantics": event_semantics,
                  "wall_time_s": time.perf_counter() - start, "finished_at": stamp(),
                  **committed_task_stats(run, tid)}
        task_progress(run, **result)
        return result
    except Exception as exc:
        error = {"task_id": tid, "error": repr(exc), "traceback": traceback.format_exc(), "failed_at": stamp()}
        atomic_json(run / "logs" / f"{tid}.error.json", error)
        task_progress(run, tid, stage="failed", error=repr(exc))
        raise
    finally:
        if backend is not None:
            backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect the agreed physical counterfactual dataset; no consistency pilot")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--event-semantics", choices=RECORDER_CLASSES, default="legacy")
    parser.add_argument("--max-wall-seconds", type=float, help="Stop between root steps or complete shards after this collection budget")
    parser.add_argument("--task-id", action="append", help="Collect only these existing flow-policy tasks; repeatable")
    parser.add_argument("--root-times", type=int, nargs="+", help="Selected root times in seconds")
    parser.add_argument("--root-order", type=int, nargs="+", help="Collection order, a permutation of selected root times")
    parser.add_argument("--full-all-selected-roots", action="store_true", help="Collect all pairs and joint/panel actions at every selected root")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    settings = collection_config(args.root_times, args.root_order, args.full_all_selected_roots)
    run = args.run_dir.resolve()
    if not run.is_relative_to(Path("/mnt/pan")) or not Path("/mnt/pan").is_mount():
        raise ValueError("All collection outputs must use the mounted /mnt/pan disk")
    if args.workers < 1 or args.shard_size < 1:
        raise ValueError("workers and shard-size must be positive")
    if args.max_wall_seconds is not None and (not np.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0):
        raise ValueError("max-wall-seconds must be finite and positive")
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    if args.detach:
        argv = [v for v in (sys.argv[1:] if argv is None else argv) if v != "--detach"]
        environment = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        with (run / "logs" / "launcher.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "cityflow_tsc.collect_counterfactual", *argv],
                                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True, env=environment)
        atomic_json(run / "launcher.json", {"pid": process.pid, "run_dir": str(run), "launched_at": stamp()})
        print(json.dumps({"pid": process.pid, "run_dir": str(run), "status": "launched"}))
        return 0
    # A live-process lock prevents accidental duplicate collection on resume.
    import fcntl
    with (run / "collector.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, tasks = prepare(run, args.dataset.resolve(), args.event_semantics, args.task_id, settings)
        freeze_execution(run, protocol, tasks, args.shard_size)
        collection_start = time.monotonic()
        deadline = None if args.max_wall_seconds is None else collection_start + args.max_wall_seconds
        atomic_json(run / "status.json", {"stage": "running", "started_at": stamp(), "pid": os.getpid(),
                                          "workers": args.workers, "tasks": len(tasks), "consistency_checks": False,
                                          "event_semantics": args.event_semantics, "max_wall_seconds": args.max_wall_seconds})
        print(f"Starting main collection: {len(tasks)} flow-policy tasks, {args.workers} workers", flush=True)
        results, errors = [], []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                                                   mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = {pool.submit(collect_task, str(run), task, args.shard_size, args.event_semantics, deadline): task["task_id"] for task in tasks}
            for future in concurrent.futures.as_completed(jobs):
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({"task_id": jobs[future], "error": repr(exc)})
                atomic_json(run / "status.json", {"stage": "running", "updated_at": stamp(), "pid": os.getpid(),
                                                  "tasks": len(tasks), "completed_tasks": sum(r.get("stage") != "budget_exhausted" for r in results),
                                                  "stopped_tasks": sum(r.get("stage") == "budget_exhausted" for r in results), "errors": errors,
                                                  "completed_task_branches": sum(x["branch_count"] for x in results),
                                                  "consistency_checks": False})
        stopped = any(r.get("stage") == "budget_exhausted" for r in results)
        summary = {"stage": "failed" if errors else "budget_exhausted" if stopped else "complete", "finished_at": stamp(),
                   "collection_wall_time_s": time.monotonic() - collection_start,
                   "event_semantics": args.event_semantics, "max_wall_seconds": args.max_wall_seconds,
                   "root_count": sum(x["root_count"] for x in results),
                   "branch_count": sum(x["branch_count"] for x in results),
                   "unresolved_events": sum(x["unresolved_events"] for x in results),
                   **{key: sum(x.get(key, 0) for x in results)
                      for key in ("internal_trip_completion_count", "trip_completion_count", "boundary_exit_count", "bytes")},
                   "results": results, "errors": errors, "consistency_checks": False,
                   "scope": "data_collection_only_not_interaction_or_controller_evidence"}
        atomic_json(run / "summary.json", summary)
        atomic_json(run / "status.json", summary)
        return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
