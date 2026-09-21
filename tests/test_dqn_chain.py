from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cityflow_tsc.dqn import (
    DQNConfig,
    ObservationVectorizer,
    ReplayBuffer,
    SharedDQNPolicy,
)
from cityflow_tsc.observations import QueuePressureObservationBuilder
from cityflow_tsc.training import DQNTrainer
from cityflow_tsc.trajectory import TrajectoryReader, sha256_file

from .test_core_chain import FIXTURES, make_stack


def make_policy(network, config, seed):
    return SharedDQNPolicy(
        network,
        feature_names=QueuePressureObservationBuilder.feature_names,
        observation_schema_id=QueuePressureObservationBuilder.schema_id,
        roadnet_sha256=sha256_file(FIXTURES / "roadnet_two_intersections.json"),
        config=config,
        seed=seed,
    )


def test_shared_dqn_trains_and_restores_checkpoint(tmp_path: Path) -> None:
    _, scenario, network, _, env = make_stack(tmp_path)
    config = DQNConfig(
        hidden_dim=16,
        learning_rate=1e-3,
        batch_size=2,
        replay_capacity=32,
        warmup_transitions=2,
        target_update_steps=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
    )
    policy = make_policy(network, config, seed=17)
    trainer = DQNTrainer(policy=policy, config=config, seed=17)
    result = trainer.run_episode(env, scenario)

    assert result.steps == 2
    assert trainer.replay.size == 4
    assert trainer.gradient_steps == 2
    assert result.metrics["training_updates"] == 2.0

    checkpoint = tmp_path / "checkpoint" / "shared_dqn.pt"
    trainer.save_checkpoint(checkpoint)
    restored_policy = make_policy(network, config, seed=999)
    restored = DQNTrainer(restored_policy, config=config, seed=999)
    restored.load_checkpoint(checkpoint)

    assert restored.environment_steps == trainer.environment_steps
    assert restored.gradient_steps == trainer.gradient_steps
    assert restored.completed_episodes == trainer.completed_episodes
    assert restored.replay.size == trainer.replay.size
    for expected, actual in zip(
        policy.online.parameters(), restored_policy.online.parameters()
    ):
        assert torch.equal(expected, actual)

    loaded = TrajectoryReader().load(Path(result.manifest_path))
    assert loaded["manifest"]["policy_metadata"]["mode"] == "online_training"
    assert loaded["manifest"]["policy_metadata"]["roadnet_sha256"] == sha256_file(
        FIXTURES / "roadnet_two_intersections.json"
    )

    expected_loss = trainer._learn_once()
    actual_loss = restored._learn_once()
    assert actual_loss == pytest.approx(expected_loss)
    for expected, actual in zip(
        policy.online.parameters(), restored_policy.online.parameters()
    ):
        assert torch.equal(expected, actual)


def test_dqn_rejects_unreachable_warmup() -> None:
    with pytest.raises(ValueError, match="warmup_transitions"):
        DQNConfig(batch_size=4, replay_capacity=8, warmup_transitions=9)


def test_vectorizer_preserves_per_feature_validity(tmp_path: Path) -> None:
    _, scenario, network, _, env = make_stack(tmp_path)
    observation, _ = env.reset(scenario)
    zero_features = np.zeros_like(observation.features)
    present = replace(observation, features=zero_features)
    missing_mask = observation.valid_mask.copy()
    missing_mask[0, 0, 0] = False
    missing = replace(
        observation,
        features=zero_features,
        valid_mask=missing_mask,
    )
    vectorizer = ObservationVectorizer(
        network, QueuePressureObservationBuilder.feature_names
    )

    present_state = vectorizer.transform(present)
    missing_state = vectorizer.transform(missing)

    assert not np.array_equal(present_state, missing_state)
    feature_values = network.max_movements * len(observation.feature_names)
    assert present_state[0, feature_values] == 1.0
    assert missing_state[0, feature_values] == 0.0


@pytest.mark.parametrize(
    ("feature_names", "roadnet_sha256", "message"),
    (
        (
            ("renamed",) + QueuePressureObservationBuilder.feature_names[1:],
            None,
            "feature_names",
        ),
        (
            QueuePressureObservationBuilder.feature_names,
            "0" * 64,
            "roadnet_sha256",
        ),
    ),
)
def test_checkpoint_rejects_semantic_mismatch(
    tmp_path: Path, feature_names, roadnet_sha256, message: str
) -> None:
    _, _, network, _, _ = make_stack(tmp_path)
    config = DQNConfig(batch_size=2, replay_capacity=8, warmup_transitions=2)
    policy = make_policy(network, config, seed=1)
    checkpoint = tmp_path / "semantic.pt"
    policy.save(checkpoint)
    incompatible = SharedDQNPolicy(
        network,
        feature_names=feature_names,
        observation_schema_id=QueuePressureObservationBuilder.schema_id,
        roadnet_sha256=roadnet_sha256
        or sha256_file(FIXTURES / "roadnet_two_intersections.json"),
        config=config,
        seed=1,
    )

    with pytest.raises(ValueError, match=message):
        incompatible.load(checkpoint)


def test_full_replay_buffer_restores_after_wraparound() -> None:
    replay = ReplayBuffer(capacity=3, state_dim=2, action_dim=2)
    for value in range(5):
        replay.append_batch(
            states=np.asarray([[value, value + 0.5]], dtype=np.float32),
            actions=np.asarray([value % 2], dtype=np.int64),
            rewards=np.asarray([float(value)], dtype=np.float32),
            next_states=np.asarray([[value + 1, value + 1.5]], dtype=np.float32),
            dones=np.asarray([False]),
            next_action_masks=np.asarray([[True, True]]),
        )
    restored = ReplayBuffer(capacity=3, state_dim=2, action_dim=2)
    restored.load_state_dict(replay.state_dict())

    assert restored.size == replay.size == 3
    assert restored.position == replay.position == 2
    assert np.array_equal(restored.states, replay.states)
    assert np.array_equal(restored.actions, replay.actions)
    assert np.array_equal(restored.rewards, replay.rewards)
