"""Launch only dependency-ready authorized stages, leaving blocked stages explicit."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .counterfactual.writer import atomic_json, sha256
from .collect_counterfactual import stamp
from .collect_timed_effects import inventory_demands, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--old-dataset", type=Path, required=True)
    parser.add_argument("--old-model", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not Path("/mnt/pan").is_mount() or not run.is_relative_to(Path("/mnt/pan")):
        raise ValueError("All outputs must remain on mounted /mnt/pan")
    if shutil.disk_usage(run).free < 120_000_000_000:
        raise OSError("Insufficient disk reserve for collection")
    import fcntl
    with (run / "launch.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "launcher.json").exists():
            raise ValueError("This run was already dispatched; inspect rather than relaunch")
        inventory = inventory_demands(args.old_dataset, run)
        tasks, config = prepare(run, args.old_dataset)
        source = Path(__file__).parent
        hashes = {str(p.relative_to(source)): sha256(p) for p in [Path(__file__), source / "train_effect_revision.py",
                  source / "collect_timed_effects.py", source / "effect_model/revision.py",
                  source / "effect_model/model.py", source / "effect_model/data.py", source / "effect_model/evaluation.py",
                  source / "train_effect_world_model.py", source / "collect_counterfactual.py", source / "simulator.py",
                  source / "config.py", *sorted((source / "counterfactual").glob("*.py"))]}
        atomic_json(run / "execution.json", {"created_at": stamp(), "old_dataset": str(args.old_dataset),
                    "old_model": str(args.old_model), "source_sha256": hashes,
                    "stages": {"old_A0_A1_A2": "launching_9_jobs", "timing_T0_T1_T2": "launching",
                               "formal_T1_collection": "needs_user_direction_on_demand_groups",
                               "new_A_B1_B2": "pending_new_data_and_selected_A",
                               "sparse_120_32_8": "pending_new_B_results_and_isolated_timing"},
                    "demand_groups_available": inventory["total_groups"], "demand_groups_required": 30,
                    "timing_coverage": config["coverage"], "no_new_demand_generation_authorized_yet": True})
        logs = run / "logs"; logs.mkdir(exist_ok=True)
        jobs = [
            ("old_training", "cityflow_tsc.train_effect_revision", ["--source", str(args.old_model)], 2),
            ("timing_collection", "cityflow_tsc.collect_timed_effects", ["--old-dataset", str(args.old_dataset)], 1),
        ]
        launches = {}
        for name, module, extra, threads in jobs:
            cmd = [sys.executable, "-u", "-m", module, "--run-dir", str(run), *extra]
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                       MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
            with (logs / (name + ".log")).open("ab") as stream:
                proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            launches[name] = {"pid": proc.pid, "command": cmd, "log": str(logs / (name + ".log"))}
        result = {"started_at": stamp(), "run_dir": str(run), "jobs": launches}
        atomic_json(run / "launcher.json", result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
