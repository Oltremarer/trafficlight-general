from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .cli import _phase_ids
from .config import ControlConfig, ScenarioConfig
from .dqn import DQNConfig, SharedDQNPolicy
from .observations import QueuePressureObservationBuilder
from .runner import EpisodeRunner
from .runtime import build_environment
from .topology import load_network_spec
from .training import DQNTrainer
from .trajectory import sha256_file


def _seeds(value: str) -> Tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one evaluation seed is required")
    return result


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _aggregate(metrics: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not metrics:
        return {}
    shared_keys = set(metrics[0])
    for item in metrics[1:]:
        shared_keys.intersection_update(item)
    return {
        key: {
            "mean": float(np.mean([item[key] for item in metrics])),
            "std": float(np.std([item[key] for item in metrics])),
        }
        for key in sorted(shared_keys)
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cityflow-tsc-train-dqn",
        description="Train and evaluate the shared-parameter DQN baseline.",
    )
    parser.add_argument("--roadnet", required=True, type=Path)
    parser.add_argument("--flow", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seeds", type=_seeds, default=(100, 101, 102))
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--simulator-step", type=float, default=1.0)
    parser.add_argument("--decision-interval", type=int, default=30)
    parser.add_argument("--yellow-time", type=int, default=5)
    parser.add_argument("--all-red-time", type=int, default=0)
    parser.add_argument("--yellow-phase-id", type=int, default=0)
    parser.add_argument("--all-red-phase-id", type=int)
    parser.add_argument("--green-phases", type=_phase_ids, default=(1, 2, 3, 4))
    parser.add_argument("--waiting-speed-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--warmup-transitions", type=int, default=1_000)
    parser.add_argument("--target-update-steps", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    return parser


def run_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    output = _prepare_output(args.output)
    control = ControlConfig(
        decision_interval_s=args.decision_interval,
        simulator_step_s=args.simulator_step,
        yellow_time_s=args.yellow_time,
        all_red_time_s=args.all_red_time,
        yellow_phase_id=args.yellow_phase_id,
        all_red_phase_id=args.all_red_phase_id,
        green_phase_ids=tuple(args.green_phases),
    )
    first_scenario = ScenarioConfig(
        roadnet_path=args.roadnet,
        flow_path=args.flow,
        output_dir=output / "train" / "episode_0000",
        duration_s=args.duration,
        seed=args.seed,
        thread_num=args.thread_num,
    )
    network = load_network_spec(first_scenario.roadnet_path, control)
    roadnet_sha256 = sha256_file(first_scenario.roadnet_path)
    flow_sha256 = sha256_file(first_scenario.flow_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device!r}, but CUDA is unavailable")
    dqn_config = DQNConfig(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_transitions=args.warmup_transitions,
        target_update_steps=args.target_update_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        reward_scale=args.reward_scale,
    )
    policy = SharedDQNPolicy(
        network=network,
        feature_names=QueuePressureObservationBuilder.feature_names,
        observation_schema_id=QueuePressureObservationBuilder.schema_id,
        roadnet_sha256=roadnet_sha256,
        config=dqn_config,
        device=args.device,
        seed=args.seed,
    )
    trainer = DQNTrainer(policy=policy, config=dqn_config, seed=args.seed)
    checkpoint = output / "checkpoints" / "shared_dqn.pt"

    experiment_config = {
        "roadnet_path": str(first_scenario.roadnet_path),
        "roadnet_sha256": roadnet_sha256,
        "flow_path": str(first_scenario.flow_path),
        "flow_sha256": flow_sha256,
        "output_dir": str(output),
        "episodes": args.episodes,
        "duration_s": args.duration,
        "training_seed": args.seed,
        "evaluation_seeds": list(args.eval_seeds),
        "thread_num": args.thread_num,
        "device": str(policy.device),
        "control": control.to_dict(),
        "dqn": dqn_config.__dict__,
        "num_intersections": network.num_intersections,
        "max_movements": network.max_movements,
        "max_actions": network.max_actions,
    }
    _write_json(output / "experiment.config.json", experiment_config)

    training_results = []
    for episode in range(args.episodes):
        scenario = ScenarioConfig(
            roadnet_path=first_scenario.roadnet_path,
            flow_path=first_scenario.flow_path,
            output_dir=output / "train" / f"episode_{episode:04d}",
            duration_s=args.duration,
            seed=args.seed + episode,
            thread_num=args.thread_num,
        )
        result = trainer.run_episode(
            build_environment(control, network, args.waiting_speed_threshold),
            scenario,
        )
        trainer.save_checkpoint(checkpoint)
        training_results.append(
            {"episode": episode, "seed": scenario.seed, "metrics": result.metrics}
        )

    evaluation_results = []
    for seed in args.eval_seeds:
        scenario = ScenarioConfig(
            roadnet_path=first_scenario.roadnet_path,
            flow_path=first_scenario.flow_path,
            output_dir=output / "evaluation" / f"seed_{seed}",
            duration_s=args.duration,
            seed=seed,
            thread_num=args.thread_num,
        )
        result = EpisodeRunner().run(
            env=build_environment(control, network, args.waiting_speed_threshold),
            policy=policy,
            scenario=scenario,
            deterministic=True,
            policy_metadata={
                "mode": "deterministic_evaluation",
                "checkpoint_sha256": policy.checkpoint_sha256,
                "observation_schema_id": policy.observation_schema_id,
                "roadnet_sha256": policy.roadnet_sha256,
                "network_schema_sha256": policy.network_schema_sha256,
            },
        )
        evaluation_results.append({"seed": seed, "metrics": result.metrics})

    summary = {
        "checkpoint_path": str(checkpoint),
        "environment_steps": trainer.environment_steps,
        "gradient_steps": trainer.gradient_steps,
        "completed_episodes": trainer.completed_episodes,
        "training": training_results,
        "evaluation": {
            "runs": evaluation_results,
            "aggregate": _aggregate([item["metrics"] for item in evaluation_results]),
        },
    }
    _write_json(output / "training_summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
