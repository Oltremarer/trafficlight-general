"""Bounded, isolated replay diagnostic for the completed counterfactual dataset."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .config import ControlConfig, ScenarioConfig
from .simulator import CityFlowBackend
from .counterfactual.event_audit import EventAuditRecorder
from .counterfactual.geometry import Geometry
from .counterfactual.recorder import SignalRunner, stack_rows
from .counterfactual.route_aware_recorder import RouteAwareRecorder
from .counterfactual.writer import atomic_json, atomic_npz, sha256

TASKS = ("train_004_fixed_time", "train_004_max_pressure", "train_012_fixed_time", "train_089_max_pressure")


def build_root(source, out, tid):
    selection = json.loads((source / "selection.json").read_text())
    task = next(x for x in selection["tasks"] if x["task_id"] == tid)
    roots = json.loads((source / "indexes" / f"{tid}.roots.json").read_text())["roots"]
    root = next(x for x in roots if x["full_pair_root"])
    m = task["manifest"]
    geometry = Geometry(Path(m["scenario"]["roadnet_path"]))
    flows = json.loads(Path(m["scenario"]["flow_path"]).read_text())
    backend = CityFlowBackend(ControlConfig())
    backend.reset(ScenarioConfig(roadnet_path=Path(m["scenario"]["roadnet_path"]),
                                 flow_path=Path(m["scenario"]["flow_path"]),
                                 output_dir=out / "runtime" / tid, seed=int(m["scenario"]["seed"])))
    recorder = EventAuditRecorder(geometry, flows)
    runner = SignalRunner(backend, geometry, recorder)
    runner.initialize()
    recorder.observe(backend.engine)
    with np.load(Path(task["manifest_path"]).parent / "trajectory.npz", allow_pickle=False) as z:
        actions = z["actions"]
        for request in actions[:root["time_s"] // 30]:
            runner.step(request)
    plan = json.loads((source / "roots" / root["root_id"] / "branches.json").read_text())
    return task, root, plan, backend, runner, recorder


def trace_one(source, out):
    tid = TASKS[0]
    task, root, plan, backend, runner, recorder = build_root(source, out, tid)
    baseline = next(x for x in plan["branches"] if "baseline" in x["roles"])
    events = []
    original_observe = recorder.observe

    def observed(engine):
        row = original_observe(engine)
        events.extend(recorder.last_events)
        return row

    recorder.observe = observed
    rows = []
    for request in baseline["requests"]:
        rows.extend(runner.step(request))
    arrays = stack_rows(rows)
    atomic_npz(out / "trace_baseline.npz", arrays)
    summary = {"root_id": root["root_id"], "branch_id": baseline["branch_id"],
               "unresolved": int(arrays["unresolved_events"].sum()),
               "reasons": dict(Counter(e["reason"] for e in events)), "events": events}
    atomic_json(out / "trace_baseline.json", summary)
    backend.close()
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def five_second_targets(arrays, corrected):
    def windows(key):
        a = arrays[key].astype(np.float64)
        return a.reshape(36, 5, *a.shape[1:]).sum(axis=1)

    return {"lane_wait": windows("lane_q").sum(axis=-1),
            "queue": arrays["lane_q"][4::5].astype(np.float64),
            "movement_events": windows("movement_events"),
            "lane_events": windows("lane_events"),
            "boundary_exit": windows("exit_count"),
            # This comparison deliberately tests the invalid interpretation of
            # legacy boundary exits as total completed trips. It does NOT change
            # the meaning of the preserved boundary_exit target above.
            "total_completion_vs_legacy_boundary_proxy": windows("trip_completion_count" if corrected else "exit_count")}


def compare_effects(old, new):
    outputs = {}
    summary = {}
    for metric in old[0]:
        summary[metric] = {}
        for mask in (1, 2, 4, 3, 5, 6, 7):
            members = [s for s in range(8) if s & mask == s]
            order = bin(mask).count("1")
            effects = []
            for values in (old, new):
                effect = sum(((-1) ** (order - bin(s).count("1"))) * values[s][metric] for s in members)
                effects.append(effect)
            outputs[f"{metric}_mask{mask}_old"] = effects[0]
            outputs[f"{metric}_mask{mask}_new"] = effects[1]
            delta = effects[1] - effects[0]
            summary[metric][str(mask)] = {"order": order, "max_abs_change": float(np.abs(delta).max()),
                                        "changed_coordinates": int(np.count_nonzero(delta)),
                                        "old_l1": float(np.abs(effects[0]).sum()),
                                        "new_l1": float(np.abs(effects[1]).sum())}
    return outputs, summary


def compare_task(source_path, out_path, tid):
    source, out = Path(source_path), Path(out_path)
    began = time.perf_counter()
    atomic_json(out / f"{tid}.progress.json", {"stage": "rebuilding_root"})
    task, root, plan, backend, runner, legacy = build_root(source, out, tid)
    fixed = RouteAwareRecorder(legacy.g, legacy.flows)
    fixed.observe(backend.engine)
    archive = backend.capture_archive()
    legacy_context, fixed_context, signal_context = legacy.context(), fixed.context(), runner.context()
    nodes = [legacy.g.node_index[x] for x in plan["panel_intersection_ids"][:3]]
    a0 = root["signal_context"]["current_phase"]
    by_action = {tuple(b["requests"][0]): (i, b) for i, b in enumerate(plan["branches"])}
    chosen = {}
    for mask in range(8):
        first = a0.copy()
        for bit, node in enumerate(nodes):
            if mask & (1 << bit):
                first[node] = (a0[node] + 1) % 4
        chosen[mask] = by_action[tuple(first)]
    branch_results, old_targets, new_targets = [], {}, {}
    for mask, (source_index, branch) in chosen.items():
        backend.restore_archive(archive)
        legacy.restore_context(legacy_context)
        fixed.restore_context(fixed_context)
        runner.restore_context(signal_context)
        corrected_rows, events = [], []

        class PairedReadout:
            def observe(self, engine):
                old_row = legacy.observe(engine)
                corrected_rows.append(fixed.observe(engine))
                events.extend(legacy.last_events)
                return old_row

        runner.recorder = PairedReadout()
        rows = []
        for request in branch["requests"]:
            rows.extend(runner.step(request))
        old, new = stack_rows(rows), stack_rows(corrected_rows)
        for k in ("phase_used", "stage_used", "phase_elapsed_start_s"):
            new[k] = old[k]
        label_changes = {k: float(np.max(np.abs(old[k].astype(np.float64) - new[k].astype(np.float64))))
                         for k in old if k != "unresolved_events"}
        stored_differences = {}
        src = source / "shards" / root["root_id"] / f"part_{source_index // 32:04d}.npz"
        with np.load(src, allow_pickle=False) as z:
            position = source_index % 32
            if str(z["branch_ids"][position]) != branch["branch_id"]:
                raise ValueError("Selected source branch ID does not match its file index")
            for k in old:
                stored_differences[k] = float(np.max(np.abs(old[k].astype(np.float64) - z[k][position].astype(np.float64))))
        directory = out / "comparisons" / tid / f"mask_{mask}"
        atomic_npz(directory / "legacy.npz", old)
        atomic_npz(directory / "route_aware.npz", new)
        atomic_json(directory / "events.json", events)
        old_targets[mask], new_targets[mask] = five_second_targets(old, False), five_second_targets(new, True)
        result = {"task_id": tid, "mask": mask, "root_id": root["root_id"], "branch_id": branch["branch_id"],
                  "legacy_unresolved": int(old["unresolved_events"].sum()),
                  "remaining_unresolved": int(new["unresolved_events"].sum()),
                  "internal_completions": int(new["internal_trip_completion_count"].sum()),
                  "boundary_exits": int(old["exit_count"].sum()),
                  "total_completions": int(new["trip_completion_count"].sum()),
                  "reasons": dict(Counter(x["reason"] for x in events)),
                  "legacy_label_max_abs_changes": label_changes,
                  "replay_vs_stored_max_abs_changes": stored_differences,
                  "events_matching_terminal_road": sum(x.get("before_road") == x["route_last_road"] for x in events)}
        atomic_json(directory / "summary.json", result)
        branch_results.append(result)
        atomic_json(out / f"{tid}.progress.json", {"stage": "comparing", "branches_done": len(branch_results),
                                                    "old_unresolved": sum(x["legacy_unresolved"] for x in branch_results),
                                                    "remaining_unresolved": sum(x["remaining_unresolved"] for x in branch_results)})
    effect_arrays, effects = compare_effects(old_targets, new_targets)
    atomic_npz(out / "comparisons" / tid / "effects.npz", effect_arrays)
    ranks = {}
    for metric in ("boundary_exit", "total_completion_vs_legacy_boundary_proxy"):
        old_scores = [float(old_targets[i][metric].sum()) for i in range(8)]
        new_scores = [float(new_targets[i][metric].sum()) for i in range(8)]
        comparisons_changed = sum(np.sign(old_scores[i]-old_scores[j]) != np.sign(new_scores[i]-new_scores[j])
                                  for i in range(8) for j in range(i+1, 8))
        ranks[metric] = {"old_scores": old_scores, "new_scores": new_scores,
                         "candidate_pair_order_or_tie_changes": int(comparisons_changed),
                         "old_best_masks": [i for i,x in enumerate(old_scores) if x==max(old_scores)],
                         "new_best_masks": [i for i,x in enumerate(new_scores) if x==max(new_scores)]}
    result = {"task_id": tid, "root_id": root["root_id"], "backlog": root["backlog"],
              "nodes": [legacy.g.intersections[i] for i in nodes], "branches": branch_results,
              "effects": effects, "ranking": ranks, "wall_time_s": time.perf_counter()-began}
    atomic_json(out / f"{tid}.result.json", result)
    atomic_json(out / f"{tid}.progress.json", {"stage": "complete", "branches_done": 8})
    backend.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    if not Path("/mnt/pan").is_mount() or not out.is_relative_to(Path("/mnt/pan")):
        raise ValueError("Diagnostic outputs must be under mounted /mnt/pan")
    if out == args.source_run.resolve():
        raise ValueError("Diagnostic must not overwrite the source run")
    out.mkdir(parents=True, exist_ok=True)
    if args.trace_only:
        trace_one(args.source_run, out)
    else:
        protocol = {"source_run": str(args.source_run.resolve()), "tasks": list(TASKS), "roots": 4,
                    "branches_per_root": 8, "branch_seconds": 180, "workers": args.workers,
                    "trace_stage_extra_branches": 1, "selection": "purposive training-only roots; not representative",
                    "candidates": "first three nodes in existing exact panel; each changed to next phase; all 8 subsets",
                    "comparison": "legacy and route-aware recorders observe the exact same Engine tick; no dynamics change",
                    "source_recorders": {str(p.name): sha256(p) for p in (
                        Path(__file__), Path(__file__).parent / "counterfactual/event_audit.py",
                        Path(__file__).parent / "counterfactual/route_aware_recorder.py")}}
        atomic_json(out / "diagnostic_protocol.json", protocol)
        results, errors = [], []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(compare_task, str(args.source_run), str(out), tid): tid for tid in TASKS}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(); results.append(result)
                    print(json.dumps({"finished": result["task_id"], "branches": len(result["branches"])}), flush=True)
                except Exception as exc:
                    import traceback
                    error = {"task_id": futures[future], "error": repr(exc), "traceback": traceback.format_exc()}
                    errors.append(error); atomic_json(out / f"{futures[future]}.error.json", error)
        branches = [b for r in results for b in r["branches"]]
        summary = {"stage": "complete" if not errors else "failed", "completed_roots": len(results),
                   "completed_branches": len(branches), "legacy_unresolved": sum(b["legacy_unresolved"] for b in branches),
                   "remaining_unresolved": sum(b["remaining_unresolved"] for b in branches),
                   "internal_completions": sum(b["internal_completions"] for b in branches),
                   "legacy_label_max_change": max((max(b["legacy_label_max_abs_changes"].values()) for b in branches), default=0),
                   "replay_vs_stored_max_change": max((max(b["replay_vs_stored_max_abs_changes"].values()) for b in branches), default=0),
                   "errors": errors, "results": results}
        atomic_json(out / "summary.json", summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='results'}), flush=True)


if __name__ == "__main__":
    main()
