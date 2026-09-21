"""Isolated end-to-end model versus objective-only CityFlow planning timing."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .collect_counterfactual import stamp
from .config import ControlConfig, ScenarioConfig
from .counterfactual.geometry import Geometry
from .counterfactual.recorder import SignalRunner
from .counterfactual.writer import atomic_json
from .effect_model.formal_model import group_mean, predict_root, root_metadata, tie_argmin
from .evaluate_formal_effects import MODES, verify_checkpoint_lock
from .simulator import CityFlowBackend


def interference():
    """Observe, never stop, other GPU workloads or active high-CPU processes."""
    issues = []
    parents = {os.getpid(), os.getppid()}
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True)
    if gpu.returncode:
        issues.append({"kind": "GPU_inventory_unavailable", "error": gpu.stderr.strip()})
    else:
        for line in gpu.stdout.splitlines():
            if line.strip():
                pid = int(line.split(",")[0])
                if pid not in parents:
                    issues.append({"kind": "other_GPU_process", "pid": pid})
    # Lifetime ps %CPU can stay high for an idle process; use a one-second delta.
    def ticks():
        result = {}
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                pid = int(path.parent.name)
                fields = path.read_text().rsplit(")", 1)[1].split()
                result[pid] = int(fields[11]) + int(fields[12])
            except (OSError, ValueError, IndexError):
                pass
        return result
    first = ticks(); started = time.perf_counter(); time.sleep(1); elapsed = time.perf_counter() - started
    hz = os.sysconf("SC_CLK_TCK")
    for pid, value in ticks().items():
        if pid not in parents and pid in first and (value - first[pid]) / hz / elapsed > .2:
            issues.append({"kind": "active_other_CPU_process", "pid": pid,
                           "CPU_cores": (value - first[pid]) / hz / elapsed})
    return issues


class ObjectiveRecorder:
    """Count each stopped active or generated-not-admitted vehicle once."""
    def observe(self, engine):
        speeds = engine.get_vehicle_speed()
        pending = len(set(engine.get_vehicles(include_waiting=True)) - speeds.keys())
        return {"waiting": sum(v < .1 for v in speeds.values()) + pending}


def cityflow_root(run, root, task):
    directory = run / "timing" / "cityflow" / root["root_id"]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        return json.loads((directory / "complete.json").read_text())
    m = task["manifest"]
    g = Geometry(Path(m["scenario"]["roadnet_path"]))
    with np.load(Path(task["manifest_path"]).parent / "trajectory.npz") as z:
        actions = z["actions"].copy()
    plan = json.loads((run / "roots" / root["root_id"] / "branches.json").read_text())
    lookup = {b["branch_id"]: b for b in plan["branches"]}
    candidates = [lookup[bid] for bid in plan["candidate_ids"]]
    backend = CityFlowBackend(ControlConfig())
    try:
        begin = time.perf_counter()
        backend.reset(ScenarioConfig(roadnet_path=Path(m["scenario"]["roadnet_path"]),
            flow_path=Path(m["scenario"]["flow_path"]), output_dir=directory / "runtime",
            duration_s=3600, seed=int(m["scenario"]["seed"]), thread_num=1, save_replay=False))
        runner = SignalRunner(backend, g, ObjectiveRecorder()); runner.initialize()
        initialization_s = time.perf_counter() - begin
        begin = time.perf_counter()
        for request in actions[:root["time_s"] // 30]:
            runner.step(request)
        archive = backend.capture_archive(); context = runner.context()
        replay_s = time.perf_counter() - begin
        if not (directory / "initialization.json").exists():
            atomic_json(directory / "initialization.json", {"initialization_s": initialization_s,
                                                           "source_replay_archive_s": replay_s})
        repeats = []
        for repetition in range(3):
            path = directory / f"repetition_{repetition}.json"
            if path.exists():
                repeats.append(json.loads(path.read_text())); continue
            inflight = directory / f"repetition_{repetition}.started.json"
            if inflight.exists():
                raise RuntimeError("Incomplete timing repetition retained; no silent extra rollout beyond budget")
            atomic_json(inflight, {"started_at": stamp(), "pid": os.getpid()})
            start = time.perf_counter(); scores = []
            for branch in candidates:
                backend.restore_archive(archive); runner.restore_context(context)
                total = 0
                for request in branch["requests"]:
                    total += sum(row["waiting"] for row in runner.step(request))
                scores.append(total)
            selected = tie_argmin(scores)
            duration = time.perf_counter() - start
            row = {"repetition": repetition, "decision_s": duration, "selected": selected,
                   "selected_requests": candidates[selected]["requests"], "scores": scores,
                   "branches": len(candidates), "finished_at": stamp()}
            atomic_json(path, row); repeats.append(row)
            atomic_json(run / "timing" / "progress.json", {"stage": "cityflow", "root_id": root["root_id"],
                                                           "repetition": repetition + 1, "updated_at": stamp()})
        durations = [r["decision_s"] for r in repeats]
        result = {**root_metadata(root), "stage": "complete", "repeats": repeats, "branches": 195,
                  "median_s": float(np.median(durations)), "p95_s": float(np.percentile(durations, 95)),
                  "initialization": json.loads((directory / "initialization.json").read_text())}
        atomic_json(directory / "complete.json", result)
        return result
    finally:
        backend.close()


def benchmark(run):
    from .effect_model.formal_data import load_data
    from .train_formal_effects import load_b, load_ranker
    run = Path(run); directory = run / "timing"; directory.mkdir(exist_ok=True)
    if not (run / "checkpoints_locked.json").exists():
        raise RuntimeError("Timing must follow checkpoint locking")
    verify_checkpoint_lock(run)
    if (directory / "summary.json").exists():
        return 0
    def isolated():
        issues = interference()
        if issues:
            atomic_json(directory / "status.json", {"stage": "waiting_for_isolation", "issues": issues, "updated_at": stamp()})
        return not issues
    if not isolated():
        return 75
    started = time.perf_counter()
    roots, _, stats = load_data(run, include_pairs=True, splits=("test",))
    disk_load_s = time.perf_counter() - started
    device = torch.device("cuda")
    records, loads = [], []
    for seed in (42, 43, 44):
        start = time.perf_counter()
        model, _ = load_b(run, seed, device); ranker, _ = load_ranker(run, seed, device)
        torch.cuda.synchronize()
        loads.append({"seed": seed, "checkpoint_load_s": time.perf_counter() - start})
        for root in roots:
            if not isolated():
                return 75
            for mode in MODES:
                path = directory / "model" / f'{root["root_id"]}_{mode}_{seed}.json'
                if path.exists():
                    records.append(json.loads(path.read_text())); continue
                samples = []
                for iteration in range(60):
                    torch.cuda.synchronize(); begin = time.perf_counter()
                    output = predict_root(model, root, stats, device, mode, ranker,
                                          rebuild_inputs=True, return_fields=False)
                    # Necessary selected-action/effect readback is inside the timed interval.
                    selected = output["selected"]
                    selected_plan = np.asarray(root["joint_plans"])[selected].copy()
                    selected_field = output["selected_field"].cpu().numpy()
                    selected_score = float(output["scores"][selected].cpu())
                    torch.cuda.synchronize(); seconds = time.perf_counter() - begin
                    if iteration >= 10:
                        samples.append(seconds)
                row = {**root_metadata(root), "seed": seed, "method": mode, "warmups": 10,
                       "samples_s": samples, "median_s": float(np.median(samples)),
                       "p95_s": float(np.percentile(samples, 95)), "selected": selected,
                       "selected_plan": selected_plan.tolist(), "selected_delta_J": selected_score,
                       "selected_effect_shape": list(selected_field.shape),
                       "pairs": len(output["selected_pairs"]), "detailed_pair_queries": len(output["pair_indices"])}
                # A late-arriving workload makes this block diagnostic only. Retain
                # it explicitly; do not silently repeat samples or simulator branches.
                issues = interference()
                if issues:
                    atomic_json(path.with_suffix(".interference.json"), {**row, "issues": issues})
                    raise RuntimeError("Workload interference during model timing; block retained, requires timing decision")
                atomic_json(path, row); records.append(row)
                atomic_json(directory / "progress.json", {"stage": "model", "completed_cases": len(records),
                                                           "total_cases": 216, "updated_at": stamp()})
        del model, ranker
        torch.cuda.empty_cache()
    tasks = json.loads((run / "selection.json").read_text())["tasks"]
    lookup = {(t["flow"]["flow_id"], t["policy"]): t for t in tasks}
    sim = []
    for root in roots:
        if not isolated():
            return 75
        row = cityflow_root(run, root, lookup[root["flow_id"], root["policy"]])
        issues = interference()
        if issues:
            atomic_json(directory / "cityflow" / root["root_id"] / "interference.json", {"issues": issues, "at": stamp()})
            raise RuntimeError("Workload interference during CityFlow timing; completed branches retained, not repeated")
        sim.append(row)
    sim_by = {r["root_id"]: r for r in sim}
    for row in records:
        row["matched_root_speedup"] = sim_by[row["root_id"]]["median_s"] / row["median_s"]
    groups = []
    for seed in (42, 43, 44):
        for mode in MODES:
            values = [r for r in records if r["seed"] == seed and r["method"] == mode]
            groups.append({"seed": seed, "method": mode, **{k: group_mean(values, k) for k in
                          ("median_s", "p95_s", "matched_root_speedup")}})
    hardware = {"platform": platform.platform(), "processor": platform.processor(), "logical_cpu": os.cpu_count(),
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "python": platform.python_version(), "engine_threads": 1, "model_CPU_threads": 2,
                "precision": "FP32 neural inference, FP64 physical scoring; AMP/TF32 disabled"}
    protocol = json.loads((run / "protocol.json").read_text())
    hardware.update({k: protocol[k] for k in ("engine_version", "engine_binary", "engine_binary_sha256")})
    result = {"stage": "complete", "finished_at": stamp(), "wall_s": time.perf_counter() - started,
              "hardware": hardware, "disk_load_s_excluded": disk_load_s, "checkpoint_loads_excluded": loads,
              "models": records, "cityflow": sim, "summaries": groups, "timing_branches": sum(r["branches"] for r in sim),
              "caution": "CPU CityFlow vs GPU model, offline collection/training excluded from online latency; explicitly reported separately"}
    atomic_json(directory / "summary.json", result)
    atomic_json(directory / "status.json", {"stage": "complete", "updated_at": stamp()})
    return 0


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    raise SystemExit(benchmark(args.run_dir))


if __name__ == "__main__":
    main()
