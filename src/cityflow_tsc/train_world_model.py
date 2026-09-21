from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - covered on ML runtime
    raise RuntimeError(
        "The World Model requires the optional ML dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc

from .cli import _phase_ids
from .config import ControlConfig, ScenarioConfig
from .policies import FixedTimePolicy, MaxPressurePolicy, make_policy
from .runner import EpisodeRunner
from .runtime import build_environment
from .topology import load_network_spec
from .trajectory import sha256_file
from .world_model.checkpoint import (
    load_world_model_checkpoint,
    save_world_model_checkpoint,
)
from .world_model.data import (
    TrajectoryTransitionDataset,
    split_episode_manifests,
)
from .world_model.model import GraphWorldModel, WorldModelConfig
from .world_model.policy import PlannerConfig, WorldModelPolicy
from .world_model.training import WorldModelTrainConfig, train_world_model


def _seeds(value: str) -> Tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one evaluation seed is required")
    return result


def _policy_names(value: str) -> Tuple[str, ...]:
    names = tuple(
        item.strip().lower().replace("-", "_")
        for item in value.split(",")
        if item.strip()
    )
    allowed = {"random", "fixed_time", "max_pressure"}
    if not names or any(name not in allowed for name in names):
        raise argparse.ArgumentTypeError(
            f"collection policies must be chosen from {sorted(allowed)}"
        )
    return names


def _prepare_output(path: Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output}"
        )
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
    keys = set(metrics[0])
    for item in metrics[1:]:
        keys.intersection_update(item)
    return {
        key: {
            "mean": float(np.mean([item[key] for item in metrics])),
            "std": float(np.std([item[key] for item in metrics])),
        }
        for key in sorted(keys)
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cityflow-tsc-world-model",
        description=(
            "Collect CityFlow trajectories, train an action-conditioned graph "
            "World Model, and compare it with fixed-time and MaxPressure."
        ),
    )
    parser.add_argument("--roadnet", required=True, type=Path)
    parser.add_argument("--flow", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--trajectory-manifest",
        type=Path,
        action="append",
        default=[],
        help="Existing episode manifest; repeat to skip built-in collection.",
    )
    parser.add_argument("--collect-episodes", type=int, default=12)
    parser.add_argument(
        "--collect-policies",
        type=_policy_names,
        default=("random", "max_pressure", "fixed_time"),
    )
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--planning-discount", type=float, default=0.95)
    return parser


def _collect_trajectories(
    output: Path,
    args: argparse.Namespace,
    control: ControlConfig,
    network,
) -> Tuple[Path, ...]:
    if args.collect_episodes <= 0:
        raise ValueError("collect_episodes must be positive when collecting data")
    manifests = []
    for episode in range(args.collect_episodes):
        policy_name = args.collect_policies[episode % len(args.collect_policies)]
        policy = make_policy(policy_name)
        scenario = ScenarioConfig(
            roadnet_path=args.roadnet,
            flow_path=args.flow,
            output_dir=output / "dataset" / f"episode_{episode:04d}_{policy_name}",
            duration_s=args.duration,
            seed=args.seed + episode,
            thread_num=args.thread_num,
        )
        result = EpisodeRunner().run(
            env=build_environment(control, network, args.waiting_speed_threshold),
            policy=policy,
            scenario=scenario,
            deterministic=policy_name != "random",
            policy_metadata={"mode": "world_model_dataset_collection"},
        )
        manifests.append(Path(result.manifest_path))
    return tuple(manifests)


def _evaluate(
    output: Path,
    args: argparse.Namespace,
    control: ControlConfig,
    network,
    world_model_policy: WorldModelPolicy,
) -> Dict[str, Any]:
    policies = {
        "fixed_time": FixedTimePolicy(),
        "max_pressure": MaxPressurePolicy(),
        "world_model": world_model_policy,
    }
    summary: Dict[str, Any] = {}
    for name, policy in policies.items():
        runs = []
        for seed in args.eval_seeds:
            scenario = ScenarioConfig(
                roadnet_path=args.roadnet,
                flow_path=args.flow,
                output_dir=output / "evaluation" / name / f"seed_{seed}",
                duration_s=args.duration,
                seed=seed,
                thread_num=args.thread_num,
            )
            metadata = (
                world_model_policy.metadata() if name == "world_model" else None
            )
            result = EpisodeRunner().run(
                env=build_environment(control, network, args.waiting_speed_threshold),
                policy=policy,
                scenario=scenario,
                deterministic=True,
                policy_metadata=metadata,
            )
            runs.append({"seed": seed, "metrics": result.metrics})
        summary[name] = {
            "runs": runs,
            "aggregate": _aggregate([item["metrics"] for item in runs]),
        }
    return summary


def run_from_args(args: argparse.Namespace) -> Dict[str, Any]:
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
    roadnet_path = Path(args.roadnet).expanduser().resolve()
    network = load_network_spec(roadnet_path, control)
    roadnet_digest = sha256_file(roadnet_path)
    device = str(torch.device(args.device))
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {device!r}, but CUDA is unavailable")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifests = tuple(Path(path).expanduser().resolve() for path in args.trajectory_manifest)
    if not manifests:
        manifests = _collect_trajectories(output, args, control, network)
    train_manifests, validation_manifests = split_episode_manifests(
        manifests, args.validation_fraction, args.seed
    )
    train_dataset = TrajectoryTransitionDataset(train_manifests)
    reference = train_dataset.reference_manifest
    if reference["roadnet_sha256"] != roadnet_digest:
        raise ValueError("training trajectories do not match the evaluation roadnet")
    if reference["control"] != control.to_dict():
        raise ValueError("training trajectories do not match the control configuration")
    validation_dataset = None
    if validation_manifests:
        validation_dataset = TrajectoryTransitionDataset(
            validation_manifests, statistics=train_dataset.statistics
        )
        train_dataset.assert_compatible(validation_dataset)

    model_config = WorldModelConfig(
        hidden_dim=args.hidden_dim,
        max_actions=network.max_actions,
    )
    model = GraphWorldModel(
        model_config,
        neighbor_index=torch.as_tensor(network.neighbor_index, dtype=torch.long),
        neighbor_mask=torch.as_tensor(network.neighbor_mask, dtype=torch.bool),
    )
    train_config = WorldModelTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    training_summary = train_world_model(
        model,
        train_dataset,
        validation_dataset,
        train_config,
        device,
    )
    checkpoint_path = output / "checkpoints" / "graph_world_model.pt"
    checkpoint_digest = save_world_model_checkpoint(
        checkpoint_path,
        model,
        network,
        roadnet_digest,
        train_dataset.statistics,
        control.to_dict(),
        train_dataset.provenance(),
        validation_dataset.provenance() if validation_dataset is not None else [],
        training_summary,
    )
    restored_model, statistics, _, restored_digest = load_world_model_checkpoint(
        checkpoint_path,
        network,
        roadnet_digest,
        device,
        expected_control=control.to_dict(),
    )
    if restored_digest != checkpoint_digest:
        raise RuntimeError("checkpoint hash changed while restoring the World Model")
    planner = PlannerConfig(
        candidate_count=args.candidate_count,
        horizon=args.planning_horizon,
        discount=args.planning_discount,
    )
    policy = WorldModelPolicy(
        restored_model,
        statistics,
        network,
        control,
        planner,
        checkpoint_digest,
        device,
    )
    evaluation = _evaluate(output, args, control, network, policy)
    summary = {
        "output_dir": str(output),
        "roadnet_path": str(roadnet_path),
        "roadnet_sha256": roadnet_digest,
        "flow_path": str(Path(args.flow).expanduser().resolve()),
        "flow_sha256": sha256_file(Path(args.flow).expanduser().resolve()),
        "control": control.to_dict(),
        "device": device,
        "training_manifests": [str(path) for path in train_manifests],
        "validation_manifests": [str(path) for path in validation_manifests],
        "model": model_config.to_dict(),
        "planner": planner.__dict__,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "training": training_summary,
        "evaluation": evaluation,
    }
    _write_json(output / "experiment_summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
