"""Run the pinned LLMLight PressLight components; preserve every round and seed."""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

FLOWS = {
    "Hangzhou1": "anon_4_4_hangzhou_real.json",
    "Hangzhou2": "anon_4_4_hangzhou_real_5816.json",
}
PAPER_ATT = {"Hangzhou1": 364.13, "Hangzhou2": 417.01}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda v: v.tolist() if hasattr(v, "tolist") else v.item()) + "\n")
    tmp.replace(path)


def phase_summary(actions, interval=30):
    """Actions are decision-major; count the elapsed time covered by equal choices."""
    if not actions or not actions[0]:
        raise ValueError("Missing phase trace")
    width = len(actions[0])
    if any(len(row) != width for row in actions):
        raise ValueError("Inconsistent intersection count")
    result = []
    for node in range(width):
        longest = streak = 0
        previous = None
        counts = [0, 0, 0, 0]
        for row in actions:
            action = int(row[node])
            if action not in range(4):
                raise ValueError("Invalid four-phase action")
            counts[action] += 1
            streak = streak + 1 if action == previous else 1
            longest = max(longest, streak)
            previous = action
        result.append({"node_index": node, "action_counts": counts,
                       "longest_same_action_decisions": longest,
                       "longest_same_action_s": longest * interval})
    return result


def author_data_path(root, flow, job):
    # CityFlowEnv.create_intersection_dict prefixes PATH_TO_DATA with './'.
    # A relative path is therefore required even when all artifacts are absolute.
    return os.path.relpath(root / "inputs" / flow, job)


def aggregate(root, tasks):
    rows = []
    for flow in FLOWS:
        summaries = []
        for task in tasks:
            if task["flow"] != flow or task["state"] != "completed":
                continue
            value = json.loads((root / task["job"] / "summary.json").read_text())
            if len(value["rounds"]) != 100:
                raise ValueError("An incomplete seed was marked completed")
            summaries.append(value)
        if len(summaries) != 5:
            continue
        means = [statistics.mean(r["evaluation"]["test_avg_travel_time_over"]
                                 for r in s["rounds"][-10:]) for s in summaries]
        rows.append({"flow": flow, "seeds": [s["seed"] for s in summaries],
                     "seed_last10_att_s": means, "att_mean_s": statistics.mean(means),
                     "att_sample_sd_s": statistics.stdev(means),
                     "paper_att_s": PAPER_ATT[flow],
                     "delta_vs_paper_s": statistics.mean(means) - PAPER_ATT[flow],
                     "completion_rate_mean": statistics.mean(
                         r["lifecycle"]["completion_rate"] for s in summaries for r in s["rounds"][-10:])})
    write_json(root / "comparison.json", {"complete_groups": len(rows), "groups": rows,
               "statistic": "Last 10 rounds per seed, then mean and sample SD across 5 seeds",
               "protocol": "Author code, explicit matched engine seeds; code timing 30/5/0"})


def worker(root, flow, seed):
    # These are set before importing TensorFlow or any author module.
    job = root / "jobs" / flow / ("seed_%d" % seed)
    job.mkdir(parents=True, exist_ok=True)
    lock = open(job / "worker.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (job / "summary.json").exists():
        raise FileExistsError("Refusing to overwrite an existing training")
    os.chdir(job)
    state = {"state": "starting", "pid": os.getpid(), "flow": flow, "seed": seed,
             "completed_rounds": 0, "started_at": time.time()}
    context = {"round": 0, "stage": "initializing", "engine": None, "actions": []}
    rounds = []

    def status(stage):
        context["stage"] = stage
        state.update(state="running", stage=stage, round=context["round"], updated_at=time.time())
        write_json(job / "status.json", state)

    try:
        assert os.path.ismount("/mnt/pan") and os.access("/mnt/pan", os.W_OK)
        sys.path.insert(0, str(root / "source"))
        sys.path.insert(0, str(root))
        import numpy as np
        import tensorflow as tf
        from lifecycle_metrics import LifecycleLedger
        from utils import config, cityflow_env, model_test
        from utils.generator import Generator
        from utils.construct_sample import ConstructSample
        from utils.updater import Updater

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        agent_conf = copy.deepcopy(config.DIC_BASE_AGENT_CONF)
        traffic_conf = copy.deepcopy(config.dic_traffic_env_conf)
        traffic_conf.update(NUM_ROUNDS=100, NUM_GENERATORS=1, NUM_AGENTS=1,
                            NUM_INTERSECTIONS=16, TOP_K_ADJACENCY=5, RUN_COUNTS=3600,
                            MODEL_NAME="EfficientPressLight", MODEL="PressLight",
                            PROJECT_NAME="presslight-author-hangzhou", NUM_ROW=4, NUM_COL=4,
                            TRAFFIC_FILE=FLOWS[flow], ROADNET_FILE="roadnet_4_4.json",
                            LIST_STATE_FEATURE=["cur_phase", "traffic_movement_pressure_queue"],
                            DIC_REWARD_INFO={"pressure": -0.25})
        paths = copy.deepcopy(config.DIC_PATH)
        paths.update(PATH_TO_MODEL=str(job / "model"), PATH_TO_WORK_DIRECTORY=str(job / "records"),
                     PATH_TO_DATA=author_data_path(root, flow, job), PATH_TO_ERROR=str(job / "errors"))
        for name in ("model", "records"):
            (job / name).mkdir(exist_ok=False)
        write_json(job / "records" / "agent.conf", agent_conf)
        write_json(job / "records" / "traffic_env.conf", traffic_conf)
        for name in (FLOWS[flow], "roadnet_4_4.json"):
            shutil.copy2(root / "inputs" / flow / name, job / "records" / name)
        write_json(job / "protocol.json", {
            "flow": flow, "seed": seed, "agent": agent_conf, "traffic": traffic_conf,
            "source_commit": "d5d4180f34edb843e1d1b462d5846c75d6d4533a",
            "runtime": {"tensorflow": tf.__version__, "keras": "tf-keras legacy", "device": "cpu"},
            "train_engine_seed": "100000 * training_seed + zero_based_round",
            "eval_engine_seed": "20000 + training_seed, fixed across all rounds",
            "adaptations": ["Explicit Python/NumPy/TensorFlow seeds and matched engine seeds",
                "Preserve author RNG draw when overriding engine seed; disable visual replay file only",
                "Direct Generator -> ConstructSample -> Updater -> model_test calls, as serial Pipeline",
                "Preserve shared agent config mutation, including cumulative epsilon decay",
                "Local logging replaces wandb and preserves each round instead of deleting outputs",
                "Read-only lifecycle and phase diagnostics; clear Keras session after completed round"],
        })

        original_engine = cityflow_env.engine.Engine

        class RecordedEngine:
            def __init__(self, filename, *args, **kwargs):
                engine_conf = json.loads(Path(filename).read_text())
                self.original_seed = engine_conf["seed"]
                engine_conf["seed"] = (20000 + seed if context["stage"] == "evaluation"
                                       else 100000 * seed + context["round"])
                engine_conf["saveReplay"] = False
                write_json(filename, engine_conf)
                self.config = engine_conf
                self.actual = original_engine(filename, *args, **kwargs)
                self.ledger = LifecycleLedger.from_scenario(SimpleNamespace(
                    flow_path=root / "inputs" / flow / FLOWS[flow], duration_s=3600), 1)
                self.ledger.active_count_source = "len(engine.get_vehicles(False)), excluding lane-change shadows"
                self.observe()
                context["engine"] = self

            def __getattr__(self, name):
                return getattr(self.actual, name)

            def observe(self):
                active = self.actual.get_vehicles(False)
                self.ledger.observe(self.actual.get_current_time(), self.actual.get_vehicles(True),
                                    active, len(active))

            def next_step(self):
                result = self.actual.next_step()
                self.observe()
                return result

        cityflow_env.engine.Engine = RecordedEngine
        original_step = cityflow_env.CityFlowEnv.step

        def recorded_step(env, actions):
            context["actions"].append([int(x) for x in actions])
            return original_step(env, actions)

        cityflow_env.CityFlowEnv.step = recorded_step

        class Logger:
            def log(self, metrics):
                with open(job / "author_metrics.jsonl", "a") as stream:
                    stream.write(json.dumps({"round": context["round"], "stage": context["stage"],
                                             "metrics": metrics}, default=float) + "\n")

        logger = Logger()
        for round_index in range(100):
            context.update(round=round_index, actions=[], engine=None)
            started = time.monotonic()
            status("training")
            generator = Generator(round_index, 0, paths, agent_conf, traffic_conf)
            epsilon = float(generator.agents[0].dic_agent_conf["EPSILON"])
            generator.generate(logger)
            train_engine_seed = context["engine"].config["seed"]
            train_lifecycle = context["engine"].ledger.summary()
            del generator
            context["engine"] = None
            status("construct_samples")
            samples = ConstructSample(str(job / "records" / "train_round"), round_index, traffic_conf)
            samples.make_reward_for_system()
            if any(x is None or len(x) != 120 for x in samples.samples_all_intersection):
                raise ValueError("Author sample construction did not produce 120 samples per intersection")
            del samples
            status("fit")
            updater = Updater(round_index, agent_conf, traffic_conf, paths)
            updater.load_sample_for_agents()
            updater.update_network_for_agents()
            agent = updater.agents[0]
            history = agent.q_network.history.history
            fit = {"epochs": len(history["loss"]), "loss": float(history["loss"][-1]),
                   "validation_loss": float(history["val_loss"][-1]),
                   "optimizer_iterations": int(agent.q_network.optimizer.iterations.numpy()),
                   "sample_size": int(len(agent.Y)), "shared_config_epsilon": float(agent_conf["EPSILON"])}
            if not all(math.isfinite(fit[k]) for k in ("loss", "validation_loss")):
                raise ValueError("Nonfinite fit loss")
            del updater, agent
            status("evaluation")
            context.update(actions=[], engine=None)
            result = model_test.test(paths["PATH_TO_MODEL"], paths["PATH_TO_DATA"], round_index,
                                     3600, traffic_conf, logger)
            if result is None:
                raise RuntimeError("Author evaluation returned None; inspect upstream error in log")
            metrics, source_trace = result
            if len(context["actions"]) != 120 or not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError("Incomplete or nonfinite evaluation")
            evidence = job / "rounds" / ("round_%04d" % round_index)
            evidence.mkdir(parents=True, exist_ok=False)
            lifecycle = context["engine"].ledger.write_evidence(evidence / "lifecycle.json")
            diagnostics = phase_summary(context["actions"])
            write_json(evidence / "actions.json", {"actions": context["actions"], "phase_summary": diagnostics})
            with gzip.open(evidence / "author_state_action.json.gz", "wt") as stream:
                json.dump(source_trace, stream, default=lambda v: v.tolist() if hasattr(v, "tolist") else v.item())
            row = {"round": round_index, "epsilon_training": epsilon, "fit": fit,
                   "training_lifecycle": train_lifecycle, "evaluation": metrics, "lifecycle": lifecycle,
                   "phase_summary": diagnostics, "train_engine_seed": train_engine_seed,
                   "eval_engine_seed": context["engine"].config["seed"],
                   "source_drawn_eval_seed": context["engine"].original_seed,
                   "wall_seconds": time.monotonic() - started}
            write_json(evidence / "result.json", row)
            rounds.append(row)
            write_json(job / "summary.json", {"flow": flow, "seed": seed, "rounds": rounds})
            state.update(completed_rounds=len(rounds), last_att_s=metrics["test_avg_travel_time_over"],
                         last_longest_same_action_s=max(x["longest_same_action_s"] for x in diagnostics))
            status("round_completed")
            print("AUTHOR_ROUND_COMPLETE " + json.dumps({k: state[k] for k in
                  ("flow", "seed", "completed_rounds", "last_att_s", "last_longest_same_action_s")}), flush=True)
            context.update(engine=None, actions=[])
            del source_trace, result
            tf.keras.backend.clear_session()
            gc.collect()
        state.update(state="completed", stage="completed", finished_at=time.time())
    except BaseException as error:
        state.update(state="failed", error=repr(error), traceback=traceback.format_exc(), finished_at=time.time())
        raise
    finally:
        write_json(job / "status.json", state)


def supervise(root, parallel):
    assert os.path.ismount("/mnt/pan") and os.access("/mnt/pan", os.W_OK)
    lock = open(root / "dispatch.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / "batch_status.json"
    if status_path.exists():
        raise FileExistsError("Existing batch: inspect worker state before restarting")
    tasks = [{"flow": flow, "seed": seed, "job": "jobs/%s/seed_%d" % (flow, seed), "state": "pending"}
             for seed in range(5) for flow in FLOWS]
    running = {}
    while any(t["state"] in ("pending", "running") for t in tasks):
        for index, (process, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            task = tasks[index]
            path = root / task["job"] / "status.json"
            worker_status = json.loads(path.read_text()) if path.exists() else {}
            task.update(exit_code=code, state="completed" if code == 0 and worker_status.get("state") == "completed"
                        and worker_status.get("completed_rounds") == 100 else "failed")
            del running[index]
        for index, task in enumerate(tasks):
            if task["state"] != "pending" or len(running) >= parallel:
                continue
            if shutil.disk_usage(root).free < 20 * 1024**3:
                break
            mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
            if mem['MemAvailable'] < 4 * 1024**2:
                break
            cache = root / "cache" / ("%s_%d" % (task["flow"], task["seed"]))
            cache.mkdir(exist_ok=True)
            env = os.environ.copy()
            env.update(TF_USE_LEGACY_KERAS="1", CUDA_VISIBLE_DEVICES="-1", TF_NUM_INTEROP_THREADS="1",
                       TF_NUM_INTRAOP_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2", PYTHONHASHSEED=str(task["seed"]),
                       PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(cache), XDG_CACHE_HOME=str(cache),
                       KERAS_HOME=str(cache / "keras"), MPLCONFIGDIR=str(cache / "matplotlib"),
                       WANDB_MODE="disabled", WANDB_DIR=str(cache))
            log_path = root / "logs" / ("%s_seed_%d.log" % (task["flow"], task["seed"]))
            log = open(log_path, "ab", buffering=0)
            argv = [sys.executable, "-u", "-B", str(root / Path(__file__).name), "worker", "--root", str(root),
                    "--flow", task["flow"], "--seed", str(task["seed"])]
            process = subprocess.Popen(argv, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            task.update(state="running", pid=process.pid, log=str(log_path), command=argv)
            running[index] = (process, log)
        state = "running" if running or any(t["state"] == "pending" for t in tasks) else (
            "completed" if all(t["state"] == "completed" for t in tasks) else "needs_attention")
        write_json(status_path, {"state": state, "pid": os.getpid(), "updated_at": time.time(),
                   "parallel": parallel, "completed": sum(t["state"] == "completed" for t in tasks),
                   "failed": sum(t["state"] == "failed" for t in tasks), "total": len(tasks), "tasks": tasks})
        aggregate(root, tasks)
        if running or any(t["state"] == "pending" for t in tasks):
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "supervise"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--flow", choices=tuple(FLOWS))
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if not args.root.is_relative_to(Path("/mnt/pan")):
        parser.error("All remote artifacts must be under /mnt/pan")
    if args.mode == "worker":
        if args.flow is None or args.seed is None:
            parser.error("worker requires --flow and --seed")
        worker(args.root, args.flow, args.seed)
    else:
        if args.parallel < 1:
            parser.error("parallel must be positive")
        supervise(args.root, args.parallel)
