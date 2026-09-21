from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cityflow_tsc.policies import RandomPolicy
from cityflow_tsc.environment import TrafficEnv
from cityflow_tsc.metrics import MetricCollector
from cityflow_tsc.observations import QueuePressureObservationBuilder
from cityflow_tsc.rewards import QueueReward
from cityflow_tsc.runner import EpisodeRunner
from cityflow_tsc.trajectory import sha256_file
from cityflow_tsc.world_model.checkpoint import (
    load_world_model_checkpoint,
    save_world_model_checkpoint,
)
from cityflow_tsc.world_model.data import (
    TrajectoryTransitionDataset,
    split_episode_manifests,
)
from cityflow_tsc.world_model.model import GraphWorldModel, WorldModelConfig
from cityflow_tsc.world_model.policy import PlannerConfig, WorldModelPolicy
from cityflow_tsc.world_model.training import (
    WorldModelTrainConfig,
    train_world_model,
)

from .test_core_chain import FIXTURES, make_stack
from .fakes import DeterministicBackend


def _collect(tmp_path: Path, episodes: int = 3):
    manifests = []
    for episode in range(episodes):
        control, scenario, network, _, env = make_stack(tmp_path / f"episode_{episode}")
        policy = RandomPolicy()
        result = EpisodeRunner().run(
            env,
            policy,
            scenario,
            deterministic=False,
            policy_metadata={"mode": "world_model_test_collection"},
        )
        manifests.append(Path(result.manifest_path))
    return control, network, tuple(manifests)


def test_episode_split_and_statistics_preserve_data_contract(tmp_path: Path) -> None:
    _, _, manifests = _collect(tmp_path)
    train, validation = split_episode_manifests(manifests, 0.34, seed=5)

    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(manifests)
    dataset = TrajectoryTransitionDataset(train)
    sample = dataset[0]
    assert sample["features"].shape == (2, 2, 4)
    assert sample["valid_mask"].dtype == torch.bool
    assert sample["actions"].shape == (2,)
    assert len(dataset.provenance()) == len(train)


def test_world_model_trains_restores_plans_and_steps_environment(tmp_path: Path) -> None:
    control, network, manifests = _collect(tmp_path / "data")
    train_manifests, validation_manifests = split_episode_manifests(
        manifests, 0.34, seed=3
    )
    train_dataset = TrajectoryTransitionDataset(train_manifests)
    validation_dataset = TrajectoryTransitionDataset(
        validation_manifests, statistics=train_dataset.statistics
    )
    model = GraphWorldModel(
        WorldModelConfig(hidden_dim=16, max_actions=network.max_actions),
        torch.as_tensor(network.neighbor_index),
        torch.as_tensor(network.neighbor_mask),
    )
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    training = train_world_model(
        model,
        train_dataset,
        validation_dataset,
        WorldModelTrainConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            seed=7,
        ),
        device="cpu",
    )
    assert training["train_transitions"] == 4
    assert training["validation_transitions"] == 2
    assert all(np.isfinite(item["train"]["loss"]) for item in training["history"])
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial, model.parameters())
    )

    checkpoint = tmp_path / "checkpoint" / "graph_world_model.pt"
    roadnet_digest = sha256_file(FIXTURES / "roadnet_two_intersections.json")
    digest = save_world_model_checkpoint(
        checkpoint,
        model,
        network,
        roadnet_digest,
        train_dataset.statistics,
        control.to_dict(),
        train_dataset.provenance(),
        validation_dataset.provenance(),
        training,
    )
    restored, statistics, metadata, restored_digest = load_world_model_checkpoint(
        checkpoint, network, roadnet_digest
    )
    assert restored_digest == digest
    assert metadata["training_summary"]["train_transitions"] == 4

    _, scenario, _, backend, env = make_stack(tmp_path / "evaluation")
    observation, _ = env.reset(scenario)
    policy = WorldModelPolicy(
        restored,
        statistics,
        network,
        control,
        PlannerConfig(candidate_count=8, horizon=2),
        digest,
    )
    policy.reset(scenario.seed, network)
    decision = policy.act(observation, deterministic=True)
    next_observation, rewards, _, _, _ = env.step(decision.actions)
    assert decision.actions.shape == (network.num_intersections,)
    assert np.all(
        observation.action_mask[
            np.arange(network.num_intersections), decision.actions
        ]
    )
    assert decision.diagnostics["candidate_count"] == 8
    assert next_observation.time_s == 5
    assert rewards.shape == (network.num_intersections,)
    env.close()
    assert backend.closed


def test_world_model_checkpoint_rejects_wrong_roadnet(tmp_path: Path) -> None:
    control, network, manifests = _collect(tmp_path / "data", episodes=1)
    dataset = TrajectoryTransitionDataset(manifests)
    model = GraphWorldModel(
        WorldModelConfig(hidden_dim=8, max_actions=network.max_actions),
        torch.as_tensor(network.neighbor_index),
        torch.as_tensor(network.neighbor_mask),
    )
    checkpoint = tmp_path / "model.pt"
    digest = sha256_file(FIXTURES / "roadnet_two_intersections.json")
    save_world_model_checkpoint(
        checkpoint,
        model,
        network,
        digest,
        dataset.statistics,
        control.to_dict(),
        dataset.provenance(),
        [],
        {},
    )
    with pytest.raises(ValueError, match="roadnet_sha256"):
        load_world_model_checkpoint(checkpoint, network, "0" * 64)
    wrong_control = control.to_dict()
    wrong_control["decision_interval_s"] = 99
    with pytest.raises(ValueError, match="control"):
        load_world_model_checkpoint(
            checkpoint,
            network,
            digest,
            expected_control=wrong_control,
        )


def test_experiment_entrypoint_runs_complete_core_chain(
    tmp_path: Path, monkeypatch
) -> None:
    import cityflow_tsc.train_world_model as experiment

    def fake_environment(control, network, waiting_speed_threshold=0.1):
        return TrafficEnv(
            network,
            control,
            DeterministicBackend(control),
            QueuePressureObservationBuilder(network),
            QueueReward(network),
            MetricCollector(waiting_speed_threshold),
        )

    monkeypatch.setattr(experiment, "build_environment", fake_environment)
    output = tmp_path / "experiment"
    args = experiment.build_parser().parse_args(
        [
            "--roadnet",
            str(FIXTURES / "roadnet_two_intersections.json"),
            "--flow",
            str(FIXTURES / "flow_empty.json"),
            "--output",
            str(output),
            "--collect-episodes",
            "3",
            "--duration",
            "10",
            "--green-phases",
            "1,2",
            "--decision-interval",
            "5",
            "--yellow-time",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-dim",
            "8",
            "--candidate-count",
            "4",
            "--planning-horizon",
            "2",
            "--eval-seeds",
            "100",
        ]
    )
    summary = experiment.run_from_args(args)

    assert (output / "experiment_summary.json").is_file()
    assert (output / "checkpoints" / "graph_world_model.pt").is_file()
    assert set(summary["evaluation"]) == {
        "fixed_time",
        "max_pressure",
        "world_model",
    }
    assert summary["training"]["train_transitions"] == 4
    assert summary["evaluation"]["world_model"]["runs"][0]["metrics"][
        "metric_ticks"
    ] == 10
