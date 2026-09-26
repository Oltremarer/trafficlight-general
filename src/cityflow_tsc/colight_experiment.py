"""Train, evaluate and tabulate a standalone CoLight baseline.

Example: python -m cityflow_tsc.colight_experiment train --help
No simulations or training run at import time.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from importlib import metadata
import json
from pathlib import Path
import platform

from .baselines.colight import CoLightConfig, CoLightLearner, PROFILE
from .baselines.training import TrainingRunner
from .baselines.trajectory import BaselineTrajectoryWriter, write_json
from .colight_results import (CoLightMetrics, aggregate_seed_runs, identify_dataset,
                             write_comparison)
from .config import ControlConfig, ScenarioConfig
from .runner import EpisodeRunner
from .runtime import build_environment
from .simulator import CityFlowBackend
from .topology import load_network_spec
from .train_baseline import _curve_learner
from .trajectory import sha256_file


def make_environment(control, network, lane_change):
    env = build_environment(control, network, profile=PROFILE,
                            backend=CityFlowBackend(control, lane_change=lane_change))
    env.metrics = CoLightMetrics(network, lane_change=lane_change)
    return env


def new_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "numpy", "cityflow"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def evaluate_episode(control, network, learner, scenario, lane_change, *, initial=None, env=None):
    frozen, model_hash = _curve_learner(learner)
    if env is None:
        env = make_environment(control, network, lane_change)
    result = EpisodeRunner().run(
        env, frozen.policy, scenario, deterministic=True, initial=initial,
        policy_metadata={"mode": "frozen_evaluation", "profile": learner.profile.to_dict(),
                         "model_state_sha256": model_hash},
        writer_factory=BaselineTrajectoryWriter,
    )
    return result.metrics


def train(args, *, profile=PROFILE, config_class=CoLightConfig, learner_class=CoLightLearner,
          experiment_id="colight-round-fit-v1", baseline="CoLight",
          environment_factory=None, comparison_writer=write_comparison):
    import torch
    environment_factory = environment_factory or make_environment
    if args.rounds <= 0 or args.tail_rounds <= 0 or args.checkpoint_every <= 0:
        raise ValueError("rounds, tail_rounds and checkpoint_every must be positive")
    if args.torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(args.torch_threads)
    config = config_class(**{f.name: getattr(args, f.name) for f in fields(config_class)
                             if hasattr(args, f.name) and getattr(args, f.name) is not None})
    control = ControlConfig(decision_interval_s=args.decision_interval,
                            yellow_time_s=args.yellow_time, all_red_time_s=args.all_red_time,
                            all_red_phase_id=args.all_red_phase_id, green_phase_ids=args.green_phases)
    # Validate input paths and the roadnet before creating a run directory.
    source = ScenarioConfig(args.roadnet, args.flow, args.output, args.duration,
                            args.seeds[0], args.thread_num)
    network = load_network_spec(source.roadnet_path, control)
    road_hash, flow_hash = sha256_file(source.roadnet_path), sha256_file(source.flow_path)
    protocol = {
        "experiment": experiment_id, "baseline": baseline,
        "dataset": identify_dataset(road_hash, flow_hash),
        "roadnet_sha256": road_hash, "flow_sha256": flow_hash,
        "roadnet": str(source.roadnet_path), "flow": str(source.flow_path),
        "node_order": list(network.intersection_ids),
        "control": control.to_dict(), "lane_change": args.lane_change,
        "duration_s": args.duration, "thread_num": args.thread_num,
        "torch_threads": args.torch_threads, "device": args.device,
        "config": config.to_dict(), "profile": profile.to_dict(),
        "rounds": args.rounds, "training_seeds": list(args.seeds),
        "training_engine_seed_rule": "100000*training_seed + zero_based_round",
        "evaluation_engine_seed_rule": "eval_seed_base + training_seed (fixed across rounds)",
        "eval_seed_base": args.eval_seed_base, "tail_rounds": args.tail_rounds,
        "training_regime": "target-flow training; same flow evaluated without learning/exploration",
        "statistic": "last rounds mean per training seed, then mean/sample SD across seeds; no best selection",
        "metric_note": "engine ATT and LLMTSCS incoming-lane ATT are separate; other metrics use project definitions",
        "timing_note": "decision interval includes yellow/all-red, not additional to it",
        "runtime": versions(),
    }
    output = new_output(args.output)
    write_json(output / "protocol.json", protocol)
    runs = []
    for seed in args.seeds:
        root = output / f"seed_{seed}"
        root.mkdir()
        records = []
        env = environment_factory(control, network, args.lane_change)
        try:
            first = ScenarioConfig(source.roadnet_path, source.flow_path, root / "train" / "round_0000",
                                   args.duration, 100000 * seed, args.thread_num)
            initial = env.reset(first)
            learner = learner_class(network, initial[0], config, seed, args.device)
            run = {"status": "running", "seed": seed, "rounds": records}
            write_json(root / "protocol.json", {**protocol, "training_seed": seed})
            for r in range(args.rounds):
                if r:
                    env = environment_factory(control, network, args.lane_change)
                scenario = ScenarioConfig(source.roadnet_path, source.flow_path,
                                          root / "train" / f"round_{r:04d}", args.duration,
                                          100000 * seed + r, args.thread_num)
                epsilon = learner.epsilon
                result = TrainingRunner().run_episode(env, learner, scenario, initial=initial)
                initial = None
                eval_scenario = ScenarioConfig(source.roadnet_path, source.flow_path,
                                               root / "evaluation" / f"round_{r:04d}", args.duration,
                                               args.eval_seed_base + seed, args.thread_num)
                metrics = evaluate_episode(control, network, learner, eval_scenario, args.lane_change,
                                           env=environment_factory(control, network, args.lane_change))
                records.append({"round": r, "epsilon": epsilon, "train_seed": scenario.seed,
                                "eval_seed": eval_scenario.seed, "training": result.metrics,
                                "fit": dict(learner.last_fit), "evaluation": metrics})
                if (r + 1) % args.checkpoint_every == 0 or r == args.rounds - 1:
                    checkpoint = root / "latest.pt"
                    learner.save(checkpoint)
                    run["checkpoint"] = str(checkpoint)
                    run["checkpoint_sha256"] = sha256_file(checkpoint)
                write_json(root / "run.json", run)
                print(f'seed={seed} round={r + 1}/{args.rounds} epsilon={epsilon:.3f} '
                      f'ATT(engine)={metrics["average_travel_time_s"]:.2f} '
                      f'ATT(LLMTSCS)={metrics["llmtscs_att_s"]:.2f}', flush=True)
            run["status"] = "completed"
            write_json(root / "run.json", run)
            runs.append(run)
        finally:
            env.close()
    aggregate = aggregate_seed_runs(runs, args.tail_rounds)
    summary = {"status": "completed", "protocol": protocol, "runs": runs, "aggregate": aggregate}
    write_json(output / "summary.json", summary)
    comparison_writer(output, protocol, aggregate)
    return summary


def evaluate(args, *, config_class=CoLightConfig, learner_class=CoLightLearner,
             experiment_id="colight-round-fit-v1", environment_factory=None):
    environment_factory = environment_factory or make_environment
    protocol = json.loads((args.checkpoint.parent / "protocol.json").read_text())
    if protocol.get("experiment") != experiment_id:
        raise ValueError("checkpoint belongs to a different experiment")
    control_data = dict(protocol["control"])
    control_data["green_phase_ids"] = tuple(control_data["green_phase_ids"])
    control = ControlConfig(**control_data)
    source = ScenarioConfig(args.roadnet, args.flow, args.output, args.duration,
                            args.seed, protocol["thread_num"])
    if sha256_file(source.roadnet_path) != protocol["roadnet_sha256"]:
        raise ValueError("checkpoint is bound to a different roadnet")
    network = load_network_spec(source.roadnet_path, control)
    output = new_output(args.output)
    env = environment_factory(control, network, protocol["lane_change"])
    try:
        initial = env.reset(source)
        learner = learner_class(network, initial[0], config_class(**protocol["config"]),
                                protocol["training_seed"], args.device)
        learner.load(args.checkpoint)
        metrics = evaluate_episode(control, network, learner, source, protocol["lane_change"],
                                   initial=initial, env=env)
        result = {"checkpoint": str(args.checkpoint.resolve()),
                  "checkpoint_sha256": sha256_file(args.checkpoint),
                  "roadnet_sha256": sha256_file(source.roadnet_path),
                  "flow_sha256": sha256_file(source.flow_path),
                  "evaluation_seed": args.seed, "duration_s": args.duration, "metrics": metrics,
                  "training_protocol": protocol}
        write_json(output / "evaluation.json", result)
        return result
    finally:
        env.close()


def compare(args, *, comparison_writer=write_comparison):
    summaries = [json.loads((root / "summary.json").read_text()) for root in args.runs]
    if any(s.get("status") != "completed" for s in summaries):
        raise ValueError("comparison requires completed run summaries")
    protocol = summaries[0]["protocol"]
    for summary in summaries[1:]:
        # Only seed identities may differ when pooling independent training runs.
        for key in set(protocol) | set(summary["protocol"]):
            if key != "training_seeds" and summary["protocol"].get(key) != protocol.get(key):
                raise ValueError(f"cannot pool different experiment protocols: {key}")
    runs = [run for summary in summaries for run in summary["runs"]]
    aggregate = aggregate_seed_runs(runs, protocol["tail_rounds"])
    merged = {**protocol, "training_seeds": [r["seed"] for r in runs]}
    output = new_output(args.output)
    comparison_writer(output, merged, aggregate)
    return aggregate


def integers(value):
    try:
        items = tuple(int(v) for v in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated integers") from error
    if not items or min(items) < 0 or len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("use distinct nonnegative integers")
    return items


def build_parser(*, description=__doc__, name="CoLight", include_attention=True):
    parser = argparse.ArgumentParser(description=description)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help=f"Train {name} and evaluate after every round")
    evaluation = sub.add_parser("evaluate", help="Frozen checkpoint evaluation, no training")
    comparison = sub.add_parser("compare", help="Pool completed seeds and export paper references")
    for child in (training, evaluation):
        for name in ("roadnet", "flow", "output"):
            child.add_argument("--" + name, type=Path, required=True)
        child.add_argument("--duration", type=int, default=3600)
        child.add_argument("--device", default="cpu")
    training.add_argument("--seeds", type=integers, default=(0,))
    training.add_argument("--rounds", type=int, default=100)
    training.add_argument("--tail-rounds", type=int, default=10)
    training.add_argument("--checkpoint-every", type=int, default=10)
    training.add_argument("--eval-seed-base", type=int, default=20000)
    training.add_argument("--thread-num", type=int, default=1)
    training.add_argument("--torch-threads", type=int, default=1)
    training.add_argument("--lane-change", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--decision-interval", type=int, default=30)
    training.add_argument("--yellow-time", type=int, default=5)
    training.add_argument("--all-red-time", type=int, default=0)
    training.add_argument("--all-red-phase-id", type=int)
    training.add_argument("--green-phases", type=integers, default=(1, 2, 3, 4))
    for name in ("hidden_dim", "batch_size", "replay_capacity", "sample_size", "fit_epochs",
                 "patience", "target_lag_rounds", "attention_heads", "attention_head_dim"):
        if name.startswith("attention_") and not include_attention:
            continue
        training.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("learning_rate", "gamma", "reward_scale", "epsilon_start", "epsilon_end",
                 "epsilon_decay", "validation_fraction", "adam_epsilon"):
        training.add_argument("--" + name.replace("_", "-"), type=float)
    training.add_argument("--bootstrap-truncated", action=argparse.BooleanOptionalAction, default=None)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--seed", type=int, default=20000)
    comparison.add_argument("--runs", nargs="+", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = {"train": train, "evaluate": evaluate, "compare": compare}[args.command](args)
    if args.command == "train":
        result = {"output": str(args.output.resolve()), "aggregate": result["aggregate"]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
