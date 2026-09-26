from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .config import ControlConfig, ScenarioConfig
from .environment import TrafficEnv
from .metrics import MetricCollector
from .observations import QueuePressureObservationBuilder
from .policies import make_policy
from .rewards import QueueReward
from .runner import EpisodeRunner
from .simulator import CityFlowBackend
from .topology import load_network_spec
from .trajectory import sha256_file
from .experiment_tracking import ExperimentTracker, add_tracking_arguments


def _phase_ids(value: str) -> Tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("green phases must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one green phase is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cityflow-tsc",
        description="Run an algorithm-independent CityFlow traffic-control episode.",
    )
    add_tracking_arguments(parser)
    parser.add_argument("--roadnet", required=True, type=Path)
    parser.add_argument("--flow", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--policy",
        default="max-pressure",
        choices=("max-pressure", "fixed-time", "random"),
    )
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--simulator-step", type=float, default=1.0)
    parser.add_argument("--decision-interval", type=int, default=30)
    parser.add_argument("--yellow-time", type=int, default=5)
    parser.add_argument("--all-red-time", type=int, default=0)
    parser.add_argument("--yellow-phase-id", type=int, default=0)
    parser.add_argument("--all-red-phase-id", type=int)
    parser.add_argument("--green-phases", type=_phase_ids, default=(1, 2, 3, 4))
    parser.add_argument("--waiting-speed-threshold", type=float, default=0.1)
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument(
        "--sample-actions",
        action="store_true",
        help="Allow stochastic sampling for policies that support it.",
    )
    return parser


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_from_args(args: argparse.Namespace):
    output_dir = _prepare_output(args.output)
    control = ControlConfig(
        decision_interval_s=args.decision_interval,
        simulator_step_s=args.simulator_step,
        yellow_time_s=args.yellow_time,
        all_red_time_s=args.all_red_time,
        yellow_phase_id=args.yellow_phase_id,
        all_red_phase_id=args.all_red_phase_id,
        green_phase_ids=tuple(args.green_phases),
    )
    scenario = ScenarioConfig(
        roadnet_path=args.roadnet,
        flow_path=args.flow,
        output_dir=output_dir,
        duration_s=args.duration,
        seed=args.seed,
        thread_num=args.thread_num,
        save_replay=args.save_replay,
    )
    network = load_network_spec(scenario.roadnet_path, control)
    observation_builder = QueuePressureObservationBuilder(network)
    reward = QueueReward(network)
    backend = CityFlowBackend(control)
    env = TrafficEnv(
        network=network,
        control=control,
        backend=backend,
        observation_builder=observation_builder,
        reward_calculator=reward,
        metrics=MetricCollector(args.waiting_speed_threshold),
    )
    policy = make_policy(args.policy)

    run_config_path = output_dir / "run.config.json"
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scenario": scenario.to_dict(),
                "control": control.to_dict(),
                "policy": policy.name,
                "reward": reward.name,
                "feature_names": list(observation_builder.feature_names),
                "num_intersections": network.num_intersections,
                "max_movements": network.max_movements,
                "max_actions": network.max_actions,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    tracker = None
    exit_code = 1
    try:
        tracking_config = {**json.loads(run_config_path.read_text()),
                           'roadnet_sha256': sha256_file(scenario.roadnet_path),
                           'flow_sha256': sha256_file(scenario.flow_path)}
        tracker = ExperimentTracker(args, output_dir, tracking_config,
                                    baseline=policy.name, job_type='rule_evaluate')
        result = EpisodeRunner().run(
            env=env,
            policy=policy,
            scenario=scenario,
            deterministic=not args.sample_actions,
        )
        values = {'round': 0, 'eval/seed': args.seed,
                  **{f'eval/{key}': value for key, value in result.metrics.items()}}
        tracker.log(values)
        tracker.summarize(values)
        exit_code = 0
    finally:
        try:
            env.close()
        finally:
            if tracker is not None:
                tracker.finish(exit_code)
    print(
        json.dumps(
            {
                "steps": result.steps,
                "metrics": result.metrics,
                "trajectory_path": result.trajectory_path,
                "manifest_path": result.manifest_path,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
