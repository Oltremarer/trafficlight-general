from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - covered on ML runtime
    raise RuntimeError(
        "World Model prediction experiments require the optional RL dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc

from .cli import _phase_ids
from .config import ControlConfig, ScenarioConfig
from .dqn import DQNConfig, SharedDQNPolicy
from .observations import RichTrafficObservationBuilder
from .policies import MaxPressurePolicy, RandomPolicy
from .runner import EpisodeRunner
from .runtime import build_rich_environment
from .topology import load_network_spec
from .training import DQNTrainer
from .trajectory import sha256_file
from .world_model.data import discover_manifests, split_episode_manifests
from .world_model.latent_prediction import (
    DirectObservationModel,
    PretrainedLatentTrafficModel,
    TrafficPredictionConfig,
)
from .world_model.prediction_training import (
    PredictionTrainConfig,
    evaluate_prediction_model,
    save_prediction_checkpoint,
    train_prediction_model,
    write_json,
)
from .world_model.prediction_baselines import (
    ActionConditionedFlowBalanceBaseline,
    PersistencePredictionBaseline,
)
from .world_model.rich_data import RichTrajectorySequenceDataset


def _empty_output(path: Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _control(args: argparse.Namespace) -> ControlConfig:
    return ControlConfig(
        decision_interval_s=args.decision_interval,
        simulator_step_s=args.simulator_step,
        yellow_time_s=args.yellow_time,
        all_red_time_s=args.all_red_time,
        yellow_phase_id=args.yellow_phase_id,
        all_red_phase_id=args.all_red_phase_id,
        green_phase_ids=tuple(args.green_phases),
    )


def _add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--roadnet", required=True, type=Path)
    parser.add_argument("--flow", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cityflow-tsc-world-model-prediction",
        description=(
            "Collect rich CityFlow trajectories, fine-tune a pretrained latent "
            "World Model, and compare it with a direct observation predictor."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect one behavior family")
    _add_scenario_arguments(collect)
    collect.add_argument(
        "--policy",
        choices=("random", "max_pressure", "shared_dqn"),
        required=True,
    )
    collect.add_argument("--episodes", type=int, default=12)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--dqn-hidden-dim", type=int, default=128)
    collect.add_argument("--dqn-learning-rate", type=float, default=3e-4)
    collect.add_argument("--dqn-batch-size", type=int, default=128)
    collect.add_argument("--dqn-warmup-transitions", type=int, default=512)
    collect.add_argument("--dqn-epsilon-decay-steps", type=int, default=10000)

    train = subparsers.add_parser("train", help="train one prediction model")
    train.add_argument("--manifest-root", type=Path, action="append", default=[])
    train.add_argument("--trajectory-manifest", type=Path, action="append", default=[])
    train.add_argument("--dataset-index", type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--model", choices=("direct", "latent"), required=True)
    train.add_argument("--tdmpc2-checkpoint", type=Path)
    train.add_argument("--device", default="cuda")
    train.add_argument("--history-length", type=int, default=3)
    train.add_argument("--rollout-horizon", type=int, default=5)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--encoder-hidden-dim", type=int, default=128)
    train.add_argument("--movement-latent-dim", type=int, default=64)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--adapter-learning-rate", type=float, default=3e-4)
    train.add_argument("--pretrained-learning-rate", type=float, default=3e-5)
    train.add_argument("--frozen-pretrained-epochs", type=int, default=5)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--gradient-clip-norm", type=float, default=10.0)
    train.add_argument("--reward-loss-weight", type=float, default=0.1)
    train.add_argument(
        "--checkpoint-reward-selection-weight",
        type=float,
        default=0.0,
        help="Weight validation reward loss when retaining the best checkpoint.",
    )
    train.add_argument("--consistency-loss-weight", type=float, default=2.0)
    train.add_argument(
        "--movement-consistency-loss-weight", type=float, default=2.0
    )
    train.add_argument("--reconstruction-loss-weight", type=float, default=0.5)
    train.add_argument("--temporal-decay", type=float, default=0.8)
    train.add_argument("--direct-max-normalized-delta", type=float, default=3.0)
    train.add_argument("--direct-normalized-state-limit", type=float, default=12.0)
    train.add_argument("--direct-normalized-reward-limit", type=float, default=10.0)
    train.add_argument(
        "--sampling-strategy",
        choices=("shuffle", "flow_policy_balanced"),
        default="shuffle",
    )
    train.add_argument("--max-train-windows", type=int)
    train.add_argument("--max-train-trajectories", type=int)
    train.add_argument("--window-seed", type=int, default=0)
    train.add_argument("--evaluate-test", action="store_true")
    train.add_argument("--num-workers", type=int, default=0)

    compare = subparsers.add_parser("compare", help="merge two held-out reports")
    compare.add_argument("--direct-summary", required=True, type=Path)
    compare.add_argument("--latent-summary", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--split", choices=("validation", "test"), default="validation")

    baselines = subparsers.add_parser(
        "evaluate-baselines",
        help="evaluate non-neural state-prediction references against CityFlow truth",
    )
    baselines.add_argument("--dataset-index", required=True, type=Path)
    baselines.add_argument("--output", required=True, type=Path)
    baselines.add_argument("--device", default="cuda")
    baselines.add_argument("--history-length", type=int, default=3)
    baselines.add_argument("--rollout-horizon", type=int, default=5)
    baselines.add_argument("--batch-size", type=int, default=128)
    baselines.add_argument("--split", choices=("validation", "test"), default="validation")
    return parser


def _collect(args: argparse.Namespace) -> Dict[str, Any]:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    output = _empty_output(args.output)
    control = _control(args)
    roadnet = Path(args.roadnet).expanduser().resolve()
    flow = Path(args.flow).expanduser().resolve()
    network = load_network_spec(roadnet, control)
    roadnet_digest = sha256_file(roadnet)
    manifests = []
    episode_metrics = []

    dqn_trainer = None
    dqn_checkpoint = output / "checkpoints" / "collector_shared_dqn.pt"
    if args.policy == "shared_dqn":
        dqn_config = DQNConfig(
            hidden_dim=args.dqn_hidden_dim,
            learning_rate=args.dqn_learning_rate,
            batch_size=args.dqn_batch_size,
            warmup_transitions=args.dqn_warmup_transitions,
            epsilon_decay_steps=args.dqn_epsilon_decay_steps,
        )
        dqn_policy = SharedDQNPolicy(
            network=network,
            feature_names=RichTrafficObservationBuilder.feature_names,
            observation_schema_id=RichTrafficObservationBuilder.schema_id,
            roadnet_sha256=roadnet_digest,
            config=dqn_config,
            device=args.device,
            seed=args.seed,
        )
        dqn_trainer = DQNTrainer(dqn_policy, dqn_config, args.seed)

    for episode in range(args.episodes):
        scenario = ScenarioConfig(
            roadnet_path=roadnet,
            flow_path=flow,
            output_dir=output / "episodes" / f"episode_{episode:04d}",
            duration_s=args.duration,
            seed=args.seed + episode,
            thread_num=args.thread_num,
        )
        env = build_rich_environment(
            control, network, args.waiting_speed_threshold
        )
        if dqn_trainer is not None:
            result = dqn_trainer.run_episode(env, scenario)
            dqn_trainer.save_checkpoint(dqn_checkpoint)
        else:
            policy = RandomPolicy() if args.policy == "random" else MaxPressurePolicy()
            result = EpisodeRunner().run(
                env=env,
                policy=policy,
                scenario=scenario,
                deterministic=args.policy != "random",
                policy_metadata={
                    "mode": "rich_world_model_dataset_collection",
                    "behavior_family": args.policy,
                },
            )
        manifests.append(result.manifest_path)
        episode_metrics.append(
            {"episode": episode, "seed": scenario.seed, "metrics": result.metrics}
        )

    summary = {
        "mode": "trajectory_collection",
        "policy": args.policy,
        "episodes": args.episodes,
        "output_dir": str(output),
        "roadnet_path": str(roadnet),
        "roadnet_sha256": roadnet_digest,
        "flow_path": str(flow),
        "flow_sha256": sha256_file(flow),
        "feature_schema": RichTrafficObservationBuilder.schema_id,
        "feature_names": list(RichTrafficObservationBuilder.feature_names),
        "manifests": manifests,
        "episode_metrics": episode_metrics,
        "dqn_checkpoint": str(dqn_checkpoint) if dqn_trainer is not None else None,
    }
    write_json(output / "collection_summary.json", summary)
    return summary


def _manifest_paths(args: argparse.Namespace) -> Tuple[Path, ...]:
    paths = [Path(item).expanduser().resolve() for item in args.trajectory_manifest]
    for root in args.manifest_root:
        paths.extend(discover_manifests(root))
    unique = tuple(dict.fromkeys(paths))
    if not unique:
        raise ValueError("provide at least one manifest root or trajectory manifest")
    return unique


def _dataset_index_all_paths(
    path: Path,
) -> Tuple[Tuple[Path, ...], Tuple[Path, ...], Tuple[Path, ...], Dict[str, Any]]:
    index_path = Path(path).expanduser().resolve()
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("dataset index has no trajectory list")
    train = tuple(
        Path(item["manifest_path"]).expanduser().resolve()
        for item in trajectories
        if item.get("split") == "train"
    )
    validation = tuple(
        Path(item["manifest_path"]).expanduser().resolve()
        for item in trajectories
        if item.get("split") == "validation"
    )
    test = tuple(
        Path(item["manifest_path"]).expanduser().resolve()
        for item in trajectories
        if item.get("split") == "test"
    )
    if not train or not validation:
        raise ValueError("dataset index requires non-empty train and validation splits")
    if any(len(set(paths)) != len(paths) for paths in (train, validation, test)):
        raise ValueError("dataset index contains duplicate trajectory manifests")
    split_paths = {"train": set(train), "validation": set(validation), "test": set(test)}
    split_flows = {
        split: {
            item.get("flow_sha256")
            for item in trajectories
            if item.get("split") == split
        }
        for split in split_paths
    }
    for split, flows in split_flows.items():
        if None in flows:
            raise ValueError(f"dataset index {split} entries require flow_sha256")
    split_names = tuple(split_paths)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if split_paths[left] & split_paths[right]:
                raise ValueError(f"dataset index leaks manifests across {left} and {right}")
            if split_flows[left] & split_flows[right]:
                raise ValueError(f"dataset index leaks flows across {left} and {right}")
    return train, validation, test, {
        "path": str(index_path),
        "sha256": sha256_file(index_path),
        "city": payload.get("city"),
        "dataset_version": payload.get("dataset_version"),
        "split_trajectory_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
    }


def _dataset_index_paths(
    path: Path,
) -> Tuple[Tuple[Path, ...], Tuple[Path, ...], Dict[str, Any]]:
    train, validation, _, metadata = _dataset_index_all_paths(path)
    return train, validation, metadata


def _balanced_manifest_subset(
    manifest_paths: Tuple[Path, ...], maximum: Optional[int], seed: int
) -> Tuple[Path, ...]:
    if maximum is None or maximum >= len(manifest_paths):
        return manifest_paths
    if maximum <= 0:
        raise ValueError("max train trajectories must be positive")
    groups: Dict[Tuple[str, str], list] = {}
    for path in manifest_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        group = (str(manifest["flow_sha256"]), str(manifest["policy"]))
        groups.setdefault(group, []).append(path)
    generator = np.random.default_rng(seed)
    ordered_groups = sorted(groups)
    generator.shuffle(ordered_groups)
    shuffled = {}
    for group in ordered_groups:
        members = list(groups[group])
        generator.shuffle(members)
        shuffled[group] = members
    selected = []
    while len(selected) < maximum:
        added = False
        for group in ordered_groups:
            members = shuffled[group]
            if members and len(selected) < maximum:
                selected.append(members.pop())
                added = True
        if not added:
            break
    if len(selected) != maximum:
        raise ValueError("could not construct the requested balanced trajectory subset")
    return tuple(selected)


def _train(args: argparse.Namespace) -> Dict[str, Any]:
    output = _empty_output(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device!r}, but CUDA is unavailable")
    dataset_index = None
    test_manifests: Tuple[Path, ...] = ()
    if args.dataset_index is not None:
        if args.manifest_root or args.trajectory_manifest:
            raise ValueError(
                "--dataset-index cannot be combined with manifest roots or paths"
            )
        (
            train_manifests,
            validation_manifests,
            test_manifests,
            dataset_index,
        ) = _dataset_index_all_paths(args.dataset_index)
    else:
        manifests = _manifest_paths(args)
        train_manifests, validation_manifests = split_episode_manifests(
            manifests, args.validation_fraction, args.seed
        )
    if not validation_manifests:
        raise ValueError("prediction comparison requires held-out validation episodes")
    if args.evaluate_test and not test_manifests:
        raise ValueError("--evaluate-test requires a non-empty test split")
    full_train_manifest_count = len(train_manifests)
    train_manifests = _balanced_manifest_subset(
        train_manifests, args.max_train_trajectories, args.window_seed
    )
    train_dataset = RichTrajectorySequenceDataset(
        train_manifests,
        history_length=args.history_length,
        rollout_horizon=args.rollout_horizon,
        max_windows=args.max_train_windows,
        window_seed=args.window_seed,
    )
    validation_dataset = RichTrajectorySequenceDataset(
        validation_manifests,
        history_length=args.history_length,
        rollout_horizon=args.rollout_horizon,
        statistics=train_dataset.statistics,
    )
    train_dataset.assert_compatible(validation_dataset)
    test_dataset = None
    if args.evaluate_test:
        test_dataset = RichTrajectorySequenceDataset(
            test_manifests,
            history_length=args.history_length,
            rollout_horizon=args.rollout_horizon,
            statistics=train_dataset.statistics,
        )
        train_dataset.assert_compatible(test_dataset)
    sample = train_dataset[0]
    model_config = TrafficPredictionConfig(
        feature_count=int(sample["history_features"].shape[-1]),
        static_feature_count=int(sample["movement_static_features"].shape[-1]),
        demand_feature_count=int(sample["demand_features"].shape[-1]),
        max_actions=int(train_dataset.reference_manifest["max_actions"]),
        encoder_hidden_dim=args.encoder_hidden_dim,
        movement_latent_dim=args.movement_latent_dim,
        direct_max_normalized_delta=args.direct_max_normalized_delta,
        direct_normalized_state_limit=args.direct_normalized_state_limit,
        direct_normalized_reward_limit=args.direct_normalized_reward_limit,
    )
    if args.model == "direct":
        model = DirectObservationModel(model_config)
        pretrained_report = None
    else:
        if args.tdmpc2_checkpoint is None:
            raise ValueError("--tdmpc2-checkpoint is required for the latent model")
        model = PretrainedLatentTrafficModel(model_config)
        pretrained_report = model.load_tdmpc2_checkpoint(args.tdmpc2_checkpoint)

    train_config = PredictionTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        adapter_learning_rate=args.adapter_learning_rate,
        pretrained_learning_rate=args.pretrained_learning_rate,
        frozen_pretrained_epochs=args.frozen_pretrained_epochs,
        weight_decay=args.weight_decay,
        reward_loss_weight=args.reward_loss_weight,
        checkpoint_reward_loss_weight=args.checkpoint_reward_selection_weight,
        consistency_loss_weight=(
            args.consistency_loss_weight if args.model == "latent" else 0.0
        ),
        movement_consistency_loss_weight=(
            args.movement_consistency_loss_weight
            if args.model == "latent"
            else 0.0
        ),
        reconstruction_loss_weight=(
            args.reconstruction_loss_weight if args.model == "latent" else 0.0
        ),
        temporal_decay=args.temporal_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        num_workers=args.num_workers,
        sampling_strategy=args.sampling_strategy,
        seed=args.seed,
    )
    training = train_prediction_model(
        model,
        train_dataset,
        validation_dataset,
        train_config,
        str(device),
        progress_path=output / "training_progress.json",
    )
    evaluation = evaluate_prediction_model(
        model,
        validation_dataset,
        str(device),
        args.batch_size,
        train_dataset.reference_manifest["feature_names"],
    )
    test_evaluation = (
        evaluate_prediction_model(
            model,
            test_dataset,
            str(device),
            args.batch_size,
            train_dataset.reference_manifest["feature_names"],
        )
        if test_dataset is not None
        else None
    )
    checkpoint_path = output / "checkpoints" / f"{model.model_kind}.pt"
    checkpoint_digest = save_prediction_checkpoint(
        checkpoint_path,
        model,
        model_config,
        train_dataset.statistics,
        training,
        train_dataset.provenance(),
        validation_dataset.provenance(),
    )
    summary = {
        "mode": "prediction_training",
        "model_kind": model.model_kind,
        "output_dir": str(output),
        "device": str(device),
        "model_config": model_config.to_dict(),
        "training": training,
        "evaluation": evaluation,
        "test_evaluation": test_evaluation,
        "statistics": train_dataset.statistics.to_dict(),
        "training_provenance": train_dataset.provenance(),
        "validation_provenance": validation_dataset.provenance(),
        "test_provenance": test_dataset.provenance() if test_dataset is not None else [],
        "dataset_index": dataset_index,
        "train_manifest_selection": {
            "full_count": full_train_manifest_count,
            "selected_count": len(train_manifests),
            "strategy": (
                "flow_policy_round_robin"
                if args.max_train_trajectories is not None
                else "all"
            ),
            "seed": args.window_seed,
        },
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "pretrained_report": pretrained_report,
    }
    write_json(output / "prediction_summary.json", summary)
    return summary


def _compare(args: argparse.Namespace) -> Dict[str, Any]:
    output = _empty_output(args.output)
    with Path(args.direct_summary).open("r", encoding="utf-8") as handle:
        direct = json.load(handle)
    with Path(args.latent_summary).open("r", encoding="utf-8") as handle:
        latent = json.load(handle)
    if direct.get("model_kind") != "direct_observation":
        raise ValueError("direct summary has the wrong model kind")
    if latent.get("model_kind") != "pretrained_latent":
        raise ValueError("latent summary has the wrong model kind")
    provenance_key = f"{args.split}_provenance"
    evaluation_key = "evaluation" if args.split == "validation" else "test_evaluation"
    direct_manifests = [item["manifest_sha256"] for item in direct[provenance_key]]
    latent_manifests = [item["manifest_sha256"] for item in latent[provenance_key]]
    if direct_manifests != latent_manifests:
        raise ValueError(f"models were not evaluated on the same {args.split} trajectories")
    if direct.get(evaluation_key) is None or latent.get(evaluation_key) is None:
        raise ValueError(f"both summaries require {args.split} evaluation")

    direct_steps = direct[evaluation_key]["per_horizon"]
    latent_steps = latent[evaluation_key]["per_horizon"]
    if len(direct_steps) != len(latent_steps):
        raise ValueError("model summaries use different rollout horizons")
    table = []
    for direct_step, latent_step in zip(direct_steps, latent_steps):
        direct_rmse = float(direct_step["overall_feature_rmse"])
        latent_rmse = float(latent_step["overall_feature_rmse"])
        direct_normalized_rmse = float(
            direct_step["normalized_feature_rmse"]
        )
        latent_normalized_rmse = float(
            latent_step["normalized_feature_rmse"]
        )
        table.append(
            {
                "horizon": int(direct_step["horizon"]),
                "cityflow": "ground_truth",
                "direct_normalized_rmse": direct_normalized_rmse,
                "pretrained_latent_normalized_rmse": latent_normalized_rmse,
                "latent_normalized_relative_change_percent": (
                    100.0
                    * (latent_normalized_rmse - direct_normalized_rmse)
                    / max(direct_normalized_rmse, 1e-12)
                ),
                "direct_core_count_rmse": float(
                    direct_step["core_count_rmse"]
                ),
                "pretrained_latent_core_count_rmse": float(
                    latent_step["core_count_rmse"]
                ),
                "direct_observation_rmse": direct_rmse,
                "pretrained_latent_rmse": latent_rmse,
                "latent_raw_mixed_unit_relative_change_percent": (
                    100.0 * (latent_rmse - direct_rmse) / max(direct_rmse, 1e-12)
                ),
                "direct_reward_mae": float(direct_step["reward_mae"]),
                "pretrained_latent_reward_mae": float(latent_step["reward_mae"]),
            }
        )
    summary = {
        "comparison": "CityFlow truth vs direct observation vs pretrained latent state",
        "split": args.split,
        "manifest_sha256": direct_manifests,
        "table": table,
        "direct_summary": str(Path(args.direct_summary).resolve()),
        "latent_summary": str(Path(args.latent_summary).resolve()),
        "interpretation": (
            "Use latent_normalized_relative_change_percent as the cross-feature "
            "summary; negative means the pretrained latent World Model is more "
            "accurate. Raw mixed-unit RMSE is retained only for audit."
        ),
    }
    write_json(output / "prediction_comparison.json", summary)
    return summary


def _evaluate_baselines(args: argparse.Namespace) -> Dict[str, Any]:
    """Measure fixed, action-aware references on a held-out split.

    The train split is used only to fit normalization statistics, exactly as it
    is for every learned prediction model.  Neither baseline is fitted to the
    requested evaluation split.
    """
    output = _empty_output(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device!r}, but CUDA is unavailable")
    train_manifests, validation_manifests, test_manifests, dataset_index = (
        _dataset_index_all_paths(args.dataset_index)
    )
    target_manifests = validation_manifests if args.split == "validation" else test_manifests
    if not target_manifests:
        raise ValueError(f"dataset index has no {args.split} trajectories")
    train_dataset = RichTrajectorySequenceDataset(
        train_manifests,
        history_length=args.history_length,
        rollout_horizon=args.rollout_horizon,
    )
    target_dataset = RichTrajectorySequenceDataset(
        target_manifests,
        history_length=args.history_length,
        rollout_horizon=args.rollout_horizon,
        statistics=train_dataset.statistics,
    )
    train_dataset.assert_compatible(target_dataset)
    feature_names = train_dataset.reference_manifest["feature_names"]
    baselines = (
        PersistencePredictionBaseline(),
        ActionConditionedFlowBalanceBaseline(train_dataset.statistics, feature_names),
    )
    evaluations = {
        model.model_kind: evaluate_prediction_model(
            model, target_dataset, str(device), args.batch_size, feature_names
        )
        for model in baselines
    }
    summary = {
        "mode": "non_neural_prediction_baselines",
        "cityflow_role": "ground_truth_targets",
        "split": args.split,
        "output_dir": str(output),
        "dataset_index": dataset_index,
        "normalization_fitted_on": "train_split_only",
        "target_provenance": target_dataset.provenance(),
        "evaluations": evaluations,
        "interpretation": (
            "persistence is action-agnostic; action_conditioned_flow_balance uses "
            "the recorded action and local movement balance but is not CityFlow."
        ),
    }
    write_json(output / "prediction_baselines.json", summary)
    return summary


def run_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.command == "collect":
        return _collect(args)
    if args.command == "train":
        return _train(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "evaluate-baselines":
        return _evaluate_baselines(args)
    raise ValueError(f"unsupported command {args.command!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
