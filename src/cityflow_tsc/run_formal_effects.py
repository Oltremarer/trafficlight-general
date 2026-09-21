"""One persistent dependency scheduler for the approved v3 run, no agent polling."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .collect_counterfactual import stamp
from .counterfactual.writer import atomic_json, sha256


def read(path):
    return json.loads(Path(path).read_text())


def command_environment(run, threads):
    return dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
        PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        TMPDIR=str(run / "tmp"), XDG_CACHE_HOME=str(run / "cache"))


def prepare(run, dataset):
    from .collect_formal_effects import prepare as prepare_collection
    if not Path("/mnt/pan").is_mount() or not run.resolve().is_relative_to(Path("/mnt/pan")):
        raise ValueError("Mounted /mnt/pan is required; no output fallback permitted")
    for name in ("logs", "tmp", "cache", "protocol"):
        (run / name).mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(run).free < 100_000_000_000:
        raise OSError("Less than 100 GB free; cannot safely collect approved data")
    selection = prepare_collection(run, dataset)
    source = Path(__file__).parent
    hashes = {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*.py"))}
    atomic_json(run / "protocol" / "source_hashes.json", hashes)
    return selection


def supervise(run, dataset):
    import fcntl
    with (run / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "launcher.json").exists():
            raise RuntimeError("Run already launched; retained state requires inspection, not duplicate dispatch")
        selection = prepare(run, dataset)
        children, launches = {}, {}
        state = {"stage": "running", "run_dir": str(run), "dataset": str(dataset),
                 "supervisor_pid": os.getpid(), "created_at": stamp(),
                 "limits": {"cityflow_workers": 8, "GPU_jobs": 3},
                 "counts": {"roots": 60, "conditions": 30, "cohorts": 9, "collection_branches": 122940,
                            "timing_branches": 2340, "A_jobs": 6, "B_jobs": 3, "ranker_components": 3},
                 "stages": {"collection": "pending", "initial_data": "pending", "A": "pending_initial_data",
                            "pair_data": "pending_pairs", "B": "pending_A_and_pairs", "test_data": "pending_lock",
                            "evaluation": "pending_lock_and_test", "timing": "pending_no_collection_training"}}

        def save():
            state["updated_at"] = stamp()
            atomic_json(run / "execution.json", state)

        def launch(name, module, extras=(), threads=2):
            cmd = [sys.executable, "-u", "-m", module, "--run-dir", str(run), *extras]
            log = run / "logs" / (name + ".log")
            with log.open("ab") as stream:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                        env=command_environment(run, threads), start_new_session=True)
            children[name] = proc
            launches[name] = {"pid": proc.pid, "command": cmd, "log": str(log), "started_at": stamp()}
            state["stages"][name] = "running"
            atomic_json(run / "launcher.json", {"supervisor_pid": os.getpid(), "run_dir": str(run),
                                                "jobs": launches, "updated_at": stamp()})
            save()

        def materialize(name, pairs, test=False):
            from .effect_model.formal_data import prepare_dataset
            state["stages"][name] = "running"; save()
            prepare_dataset(run, include_pairs=pairs, splits=("test",) if test else ("train", "validation"))
            state["stages"][name] = "complete"; save()

        save()
        try:
            launch("collection", "cityflow_tsc.collect_formal_effects", ("--dataset", str(dataset), "--workers", "8"), 1)
            next_timing = 0.
            while True:
                # This server-side loop only advances dependencies, not agent monitoring.
                for name, proc in list(children.items()):
                    result = proc.poll()
                    if result is None or state["stages"][name] != "running":
                        continue
                    if name == "timing" and result == 75:
                        state["stages"][name] = "waiting_for_isolation"
                        next_timing = time.monotonic() + 600
                    elif result:
                        state["stages"][name] = "failed"
                        raise RuntimeError(f"{name} exited {result}; see {launches[name]['log']}")
                    else:
                        paths = {"collection": "summary.json", "A": "training_A/summary.json",
                                 "B": "training_B/summary.json", "evaluation": "evaluation/summary.json",
                                 "timing": "timing/summary.json"}
                        summary = read(run / paths[name])
                        if summary.get("stage") not in ("complete", "completed"):
                            raise RuntimeError(f"{name} exited without a successful completion summary")
                        state["stages"][name] = "complete"
                    save()
                stages = state["stages"]
                if (run / "initial_ready.json").exists() and stages["initial_data"] == "pending":
                    materialize("initial_data", False)
                    launch("A", "cityflow_tsc.train_formal_effects", ("--stage", "A"))
                if (run / "trainval_pairs_ready.json").exists() and stages["pair_data"] == "pending_pairs":
                    materialize("pair_data", True)
                if stages["A"] == "complete" and stages["pair_data"] == "complete" and stages["B"] == "pending_A_and_pairs":
                    launch("B", "cityflow_tsc.train_formal_effects", ("--stage", "B"))
                if stages["B"] == "complete" and stages["collection"] == "complete" and stages["test_data"] == "pending_lock":
                    if not (run / "checkpoints_locked.json").exists():
                        raise RuntimeError("B complete but immutable checkpoint lock is absent")
                    materialize("test_data", True, test=True)
                    launch("evaluation", "cityflow_tsc.evaluate_formal_effects")
                if stages["evaluation"] == "complete" and stages["timing"] in ("pending_no_collection_training", "waiting_for_isolation"):
                    if time.monotonic() >= next_timing:
                        launch("timing", "cityflow_tsc.benchmark_formal_effects")
                if stages["timing"] == "complete":
                    state["stage"] = "complete"; state["finished_at"] = stamp(); save()
                    atomic_json(run / "complete.json", {"stage": "complete", "finished_at": stamp(),
                        "collection": str(run / "summary.json"), "training_A": str(run / "training_A/summary.json"),
                        "training_B": str(run / "training_B/summary.json"), "evaluation": str(run / "evaluation/summary.json"),
                        "timing": str(run / "timing/summary.json"), "total_budget_branches": 125280})
                    return
                time.sleep(20)
        except BaseException as exc:
            state["stage"] = "attention_required"; state["error"] = repr(exc); save()
            atomic_json(run / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(), "at": stamp(),
                "children_left_running": {name: p.pid for name, p in children.items() if p.poll() is None},
                "policy": "Independent active work and all committed outputs retained; no blind restart"})
            raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(); run = args.run_dir.resolve()
    if args.prepare_only:
        result = prepare(run, args.dataset)
        print(json.dumps({"run_dir": str(run), "tasks": len(result["tasks"]), "prepared": True}))
    else:
        supervise(run, args.dataset)


if __name__ == "__main__":
    main()
