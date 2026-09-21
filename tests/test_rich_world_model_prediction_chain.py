import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cityflow_tsc.config import ControlConfig, ScenarioConfig
from cityflow_tsc.dqn import DQNConfig, SharedDQNPolicy
from cityflow_tsc.environment import TrafficEnv
from cityflow_tsc.metrics import MetricCollector
from cityflow_tsc.observations import RichTrafficObservationBuilder
from cityflow_tsc.policies import MaxPressurePolicy, RandomPolicy
from cityflow_tsc.rewards import QueueReward
from cityflow_tsc.runner import EpisodeRunner
from cityflow_tsc.topology import load_network_spec
from cityflow_tsc.training import DQNTrainer
from cityflow_tsc.trajectory import sha256_file
from cityflow_tsc.world_model.data import split_episode_manifests
from cityflow_tsc.world_model.latent_prediction import (
    ActionSemanticEncoder,
    DirectObservationModel,
    PretrainedLatentTrafficModel,
    TrafficPredictionConfig,
)
from cityflow_tsc.world_model.prediction_training import (
    PredictionTrainConfig,
    evaluate_prediction_model,
    train_prediction_model,
)
from cityflow_tsc.world_model.prediction_baselines import (
    ActionConditionedFlowBalanceBaseline,
    PersistencePredictionBaseline,
)
from cityflow_tsc.world_model.rich_data import RichTrajectorySequenceDataset

from .fakes import DeterministicBackend


FIXTURES = Path(__file__).parent / "fixtures"


def _structured_latent_batch():
    torch.manual_seed(11)
    batch = 1
    history = 3
    nodes = 2
    movements = 3
    features = 4
    static_features = 2
    movement_mask = torch.tensor(
        [[[[True, True, False], [True, True, False]]] * history]
    ).reshape(batch, history, nodes, movements)
    valid_mask = movement_mask[..., None].expand(
        batch, history, nodes, movements, features
    )
    return {
        "history_features": torch.randn(
            batch, history, nodes, movements, features
        ),
        "history_valid_mask": valid_mask,
        "history_movement_mask": movement_mask,
        "history_current_phase": torch.zeros(batch, history, nodes, dtype=torch.long),
        "history_signal_stage": torch.zeros(batch, history, nodes, dtype=torch.long),
        "history_phase_elapsed_s": torch.zeros(batch, history, nodes),
        "history_time_s": torch.arange(history).float()[None].expand(batch, -1),
        "movement_static_features": torch.randn(
            batch, nodes, movements, static_features
        ),
        "demand_features": torch.randn(batch, 3),
        "neighbor_index": torch.tensor([[[1], [0]]]),
        "neighbor_mask": torch.ones(batch, nodes, 1, dtype=torch.bool),
        "phase_movement_mask": torch.tensor(
            [
                [
                    [[True, False, False], [False, True, False]],
                    [[True, False, False], [False, True, False]],
                ]
            ]
        ),
        "actions": torch.zeros(batch, 2, nodes, dtype=torch.long),
        "target_valid_mask": valid_mask[:, :2].clone(),
        "target_signal_stage": torch.zeros(batch, 2, nodes, dtype=torch.long),
        "target_phase_elapsed_s": torch.zeros(batch, 2, nodes),
        "target_time_s": torch.arange(history, history + 2).float()[None].expand(
            batch, -1
        ),
    }


def test_movement_action_features_keep_exact_served_streams() -> None:
    batch = _structured_latent_batch()
    actions = torch.tensor([[0, 1]])
    features = ActionSemanticEncoder.movement_features(
        actions,
        torch.zeros_like(actions),
        batch["phase_movement_mask"],
        batch["history_movement_mask"][:, -1],
    )
    assert features.shape == (1, 2, 3, 2)
    assert features[0, 0, :, 0].tolist() == [1.0, 0.0, 0.0]
    assert features[0, 1, :, 0].tolist() == [0.0, 1.0, 0.0]
    assert features[0, 0, :, 1].tolist() == [0.0, 0.0, 0.0]
    assert features[0, 1, :, 1].tolist() == [1.0, 1.0, 0.0]


def test_structured_latent_keeps_movement_changes_in_their_own_slot() -> None:
    batch = _structured_latent_batch()
    config = TrafficPredictionConfig(
        feature_count=4,
        static_feature_count=2,
        demand_feature_count=3,
        max_actions=2,
        encoder_hidden_dim=16,
        latent_dim=16,
        movement_latent_dim=8,
        tdmpc_hidden_dim=16,
        task_dim=8,
    )
    model = PretrainedLatentTrafficModel(config).eval()
    _, original = model.encode(batch)
    modified_batch = {key: value.clone() for key, value in batch.items()}
    modified_batch["history_features"][:, :, 0, 0, 0] += 5.0
    _, modified = model.encode(modified_batch)

    assert not torch.allclose(original[:, 0, 0], modified[:, 0, 0])
    assert torch.allclose(original[:, 0, 1], modified[:, 0, 1])
    assert torch.allclose(original[:, 1], modified[:, 1])
    assert torch.count_nonzero(original[:, :, 2]) == 0


def test_direct_observation_rollout_stays_finite_under_recursive_ood_inputs() -> None:
    batch = _structured_latent_batch()
    batch["history_features"] = torch.full_like(batch["history_features"], 1_000.0)
    config = TrafficPredictionConfig(
        feature_count=4,
        static_feature_count=2,
        demand_feature_count=3,
        max_actions=2,
        encoder_hidden_dim=16,
        direct_max_normalized_delta=2.0,
        direct_normalized_state_limit=6.0,
        direct_normalized_reward_limit=4.0,
    )
    model = DirectObservationModel(config).eval()
    prediction = model.rollout(batch)

    assert torch.isfinite(prediction["features"]).all()
    assert torch.isfinite(prediction["rewards"]).all()
    assert prediction["features"].abs().max().item() <= 6.0
    assert prediction["rewards"].abs().max().item() <= 4.0


def _environment(control, network):
    return TrafficEnv(
        network,
        control,
        DeterministicBackend(control),
        RichTrafficObservationBuilder(network),
        QueueReward(network),
        MetricCollector(),
    )


def test_mixed_trajectory_to_pretrained_latent_prediction_chain(tmp_path: Path) -> None:
    control = ControlConfig(
        decision_interval_s=5,
        simulator_step_s=1,
        yellow_time_s=2,
        green_phase_ids=(1, 2),
    )
    roadnet = FIXTURES / "roadnet_two_intersections.json"
    flow = FIXTURES / "flow_empty.json"
    network = load_network_spec(roadnet, control)
    manifests = []

    for episode, policy in enumerate((RandomPolicy(), MaxPressurePolicy())):
        scenario = ScenarioConfig(
            roadnet_path=roadnet,
            flow_path=flow,
            output_dir=tmp_path / f"rule_{episode}",
            duration_s=30,
            seed=episode,
        )
        result = EpisodeRunner().run(
            _environment(control, network),
            policy,
            scenario,
            deterministic=episode == 1,
        )
        manifests.append(Path(result.manifest_path))

    dqn_config = DQNConfig(
        hidden_dim=16,
        batch_size=2,
        replay_capacity=100,
        warmup_transitions=2,
        epsilon_decay_steps=20,
    )
    dqn_policy = SharedDQNPolicy(
        network,
        RichTrafficObservationBuilder.feature_names,
        RichTrafficObservationBuilder.schema_id,
        sha256_file(roadnet),
        dqn_config,
        seed=7,
    )
    dqn_trainer = DQNTrainer(dqn_policy, dqn_config, seed=7)
    for episode in range(2):
        scenario = ScenarioConfig(
            roadnet_path=roadnet,
            flow_path=flow,
            output_dir=tmp_path / f"dqn_{episode}",
            duration_s=30,
            seed=10 + episode,
        )
        result = dqn_trainer.run_episode(
            _environment(control, network), scenario
        )
        manifests.append(Path(result.manifest_path))

    train_manifests, validation_manifests = split_episode_manifests(
        manifests, 0.25, seed=3
    )
    train_dataset = RichTrajectorySequenceDataset(
        train_manifests, history_length=3, rollout_horizon=2
    )
    validation_dataset = RichTrajectorySequenceDataset(
        validation_manifests,
        history_length=3,
        rollout_horizon=2,
        statistics=train_dataset.statistics,
    )
    assert {item["policy"] for item in train_dataset.provenance()} | {
        item["policy"] for item in validation_dataset.provenance()
    } == {"random", "max_pressure", "shared_dqn"}
    weights = train_dataset.sampling_weights("flow_policy_balanced")
    group_totals = {}
    for weight, (record_index, _) in zip(weights.tolist(), train_dataset._index):
        manifest = train_dataset.records[record_index]["manifest"]
        group = (manifest["flow_sha256"], manifest["policy"])
        group_totals[group] = group_totals.get(group, 0.0) + weight
    assert all(value == pytest.approx(1.0) for value in group_totals.values())

    sample = train_dataset[0]
    model_config = TrafficPredictionConfig(
        feature_count=sample["history_features"].shape[-1],
        static_feature_count=sample["movement_static_features"].shape[-1],
        demand_feature_count=sample["demand_features"].shape[-1],
        max_actions=network.max_actions,
        encoder_hidden_dim=16,
    )
    train_config = PredictionTrainConfig(
        epochs=1,
        batch_size=2,
        frozen_pretrained_epochs=0,
        sampling_strategy="flow_policy_balanced",
        seed=5,
    )

    direct = DirectObservationModel(model_config)
    direct_training = train_prediction_model(
        direct, train_dataset, validation_dataset, train_config, "cpu"
    )
    assert direct_training["train_windows"] > 0
    assert direct_training["best_epoch"] == 1

    latent = PretrainedLatentTrafficModel(model_config)
    source_state = {
        **{
            f"_dynamics.{key}": value.detach().clone()
            for key, value in latent.dynamics.state_dict().items()
        },
        **{
            f"_reward.{key}": value.detach().clone()
            for key, value in latent.reward_head.state_dict().items()
        },
        "_task_emb.weight": torch.randn(80, model_config.task_dim),
    }
    pretrained_checkpoint = tmp_path / "mt80-5M-compatible.pt"
    torch.save({"model": source_state, "metadata": {}}, pretrained_checkpoint)
    report = latent.load_tdmpc2_checkpoint(pretrained_checkpoint)
    assert report["source_task_count"] == 80
    assert report["loaded_parameters"] > 1_000_000

    latent_training = train_prediction_model(
        latent, train_dataset, validation_dataset, train_config, "cpu"
    )
    assert latent_training["history"][0]["pretrained_core_learning_rate"] > 0
    direct_metrics = evaluate_prediction_model(
        direct,
        validation_dataset,
        "cpu",
        2,
        RichTrafficObservationBuilder.feature_names,
    )
    latent_metrics = evaluate_prediction_model(
        latent,
        validation_dataset,
        "cpu",
        2,
        RichTrafficObservationBuilder.feature_names,
    )
    assert direct_metrics["cityflow_role"] == "ground_truth_targets"
    assert latent_metrics["cityflow_role"] == "ground_truth_targets"
    assert len(direct_metrics["per_horizon"]) == 2
    assert len(latent_metrics["per_horizon"]) == 2
    for step in direct_metrics["per_horizon"]:
        assert all(
            math.isfinite(step[name])
            for name in (
                "normalized_feature_mae",
                "normalized_feature_rmse",
                "overall_feature_mae",
                "overall_feature_rmse",
                "reward_mae",
                "reward_rmse",
            )
        )

    for baseline in (
        PersistencePredictionBaseline(),
        ActionConditionedFlowBalanceBaseline(
            train_dataset.statistics, RichTrafficObservationBuilder.feature_names
        ),
    ):
        metrics = evaluate_prediction_model(
            baseline,
            validation_dataset,
            "cpu",
            2,
            RichTrafficObservationBuilder.feature_names,
        )
        assert metrics["reward_available"] is False
        assert metrics["per_horizon"][0]["reward_rmse"] is None
        assert math.isfinite(metrics["per_horizon"][0]["normalized_feature_rmse"])
