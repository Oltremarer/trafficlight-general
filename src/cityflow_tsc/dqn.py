from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - covered on RL runtime
    raise RuntimeError(
        "Shared DQN requires the optional RL dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc

from .types import NetworkObservation, NetworkSpec, PolicyOutput


CHECKPOINT_VERSION = "cityflow-tsc-shared-dqn-v2"
VECTORIZER_VERSION = "movement-features-with-valid-mask-v2"


def network_schema_sha256(network: NetworkSpec) -> str:
    """Hash every model-facing topology and action-mapping field."""

    payload = {
        "intersections": [
            {
                "index": item.index,
                "intersection_id": item.intersection_id,
                "point": list(item.point),
                "incoming_lanes": list(item.incoming_lanes),
                "outgoing_lanes": list(item.outgoing_lanes),
                "engine_phase_ids": list(item.engine_phase_ids),
                "phase_movement_mask": item.phase_movement_mask.tolist(),
                "movements": [
                    {
                        "index": movement.index,
                        "movement_type": movement.movement_type,
                        "start_road": movement.start_road,
                        "end_road": movement.end_road,
                        "incoming_lanes": list(movement.incoming_lanes),
                        "outgoing_lanes": list(movement.outgoing_lanes),
                        "static_features": list(movement.static_features),
                    }
                    for movement in item.movements
                ],
            }
            for item in network.intersections
        ],
        "neighbor_index": network.neighbor_index.tolist(),
        "neighbor_mask": network.neighbor_mask.tolist(),
        "max_movements": network.max_movements,
        "max_actions": network.max_actions,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DQNConfig:
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.95
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_transitions: int = 1_000
    train_every_steps: int = 1
    target_update_steps: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    reward_scale: float = 0.1
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.batch_size <= 0 or self.replay_capacity <= 0:
            raise ValueError("DQN dimensions and capacities must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.warmup_transitions > self.replay_capacity:
            raise ValueError("warmup_transitions cannot exceed replay_capacity")
        if (
            self.warmup_transitions < 0
            or self.epsilon_decay_steps <= 0
            or self.train_every_steps <= 0
            or self.target_update_steps <= 0
        ):
            raise ValueError("DQN schedule values are invalid")
        if self.learning_rate <= 0 or not 0 <= self.gamma <= 1:
            raise ValueError("DQN learning rate or discount factor is invalid")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("DQN epsilon schedule must satisfy 0 <= end <= start <= 1")


class ObservationVectorizer:
    """Deterministically flatten the shared per-intersection observation."""

    def __init__(self, network: NetworkSpec, feature_names: Sequence[str]) -> None:
        self.network = network
        self.feature_names = tuple(feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("feature_names must be non-empty and unique")
        self.feature_count = len(self.feature_names)
        self.output_dim = (
            2 * network.max_movements * self.feature_count
            + network.max_movements
            + network.max_actions
            + 1
            + 3
        )

    def transform(self, observation: NetworkObservation) -> np.ndarray:
        expected = (
            self.network.num_intersections,
            self.network.max_movements,
            self.feature_count,
        )
        if observation.features.shape != expected:
            raise ValueError(
                f"unexpected observation feature shape {observation.features.shape}; "
                f"expected {expected}"
            )
        if tuple(observation.feature_names) != self.feature_names:
            raise ValueError(
                "observation feature schema does not match the DQN vectorizer"
            )
        if observation.valid_mask.shape != expected:
            raise ValueError(
                f"unexpected valid-mask shape {observation.valid_mask.shape}; "
                f"expected {expected}"
            )
        values = observation.features.astype(np.float32, copy=False)
        values = np.sign(values) * np.log1p(np.abs(values))
        values = values * observation.valid_mask.astype(np.float32)
        flat_features = values.reshape(self.network.num_intersections, -1)
        flat_valid_mask = observation.valid_mask.reshape(
            self.network.num_intersections, -1
        ).astype(np.float32)
        movement_mask = observation.movement_mask.astype(np.float32)
        phase_one_hot = np.zeros(
            (self.network.num_intersections, self.network.max_actions),
            dtype=np.float32,
        )
        valid_phase = (
            (observation.current_phase >= 0)
            & (observation.current_phase < self.network.max_actions)
        )
        rows = np.flatnonzero(valid_phase)
        phase_one_hot[rows, observation.current_phase[rows]] = 1.0
        elapsed = np.log1p(observation.phase_elapsed_s.astype(np.float32))[:, None]
        stage_one_hot = np.zeros(
            (self.network.num_intersections, 3), dtype=np.float32
        )
        stages = np.clip(observation.signal_stage.astype(np.int64), 0, 2)
        stage_one_hot[np.arange(self.network.num_intersections), stages] = 1.0
        return np.concatenate(
            [
                flat_features,
                flat_valid_mask,
                movement_mask,
                phase_one_hot,
                elapsed,
                stage_one_hot,
            ],
            axis=1,
        )


class ReplayBuffer:
    """Fixed-size replay storage for per-intersection shared-policy samples."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.next_action_masks = np.empty((capacity, action_dim), dtype=np.bool_)
        self.size = 0
        self.position = 0

    def append_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        next_action_masks: np.ndarray,
    ) -> None:
        batch_size = int(states.shape[0])
        if states.shape != (batch_size, self.state_dim):
            raise ValueError("states have incompatible shape")
        if next_states.shape != states.shape:
            raise ValueError("next_states have incompatible shape")
        if actions.shape != (batch_size,) or rewards.shape != (batch_size,):
            raise ValueError("actions and rewards must be one-dimensional batches")
        if dones.shape != (batch_size,):
            raise ValueError("dones have incompatible shape")
        if next_action_masks.shape != (batch_size, self.action_dim):
            raise ValueError("next_action_masks have incompatible shape")
        for row in range(batch_size):
            index = self.position
            self.states[index] = states[row]
            self.actions[index] = actions[row]
            self.rewards[index] = rewards[row]
            self.next_states[index] = next_states[row]
            self.dones[index] = dones[row]
            self.next_action_masks[index] = next_action_masks[row]
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if self.size < batch_size:
            raise ValueError("not enough replay samples")
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "next_action_masks": self.next_action_masks[indices],
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "size": self.size,
            "position": self.position,
            "states": self.states[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "next_states": self.next_states[: self.size].copy(),
            "dones": self.dones[: self.size].copy(),
            "next_action_masks": self.next_action_masks[: self.size].copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        expected = (self.capacity, self.state_dim, self.action_dim)
        actual = (
            int(state["capacity"]),
            int(state["state_dim"]),
            int(state["action_dim"]),
        )
        if actual != expected:
            raise ValueError(
                f"replay checkpoint shape {actual} does not match runtime {expected}"
            )
        size = int(state["size"])
        position = int(state["position"])
        if size < 0 or size > self.capacity:
            raise ValueError("replay checkpoint has invalid size")
        if position < 0 or position >= self.capacity:
            raise ValueError("replay checkpoint has invalid position")
        expected_shapes = {
            "states": (size, self.state_dim),
            "actions": (size,),
            "rewards": (size,),
            "next_states": (size, self.state_dim),
            "dones": (size,),
            "next_action_masks": (size, self.action_dim),
        }
        for name, shape in expected_shapes.items():
            values = np.asarray(state[name])
            if values.shape != shape:
                raise ValueError(
                    f"replay checkpoint field {name} has shape {values.shape}; "
                    f"expected {shape}"
                )
            getattr(self, name)[:size] = values
        self.size = size
        self.position = position


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states):
        return self.layers(states)


class SharedDQNPolicy:
    """One shared Q-network applied independently to every intersection."""

    name = "shared_dqn"

    def __init__(
        self,
        network: NetworkSpec,
        feature_names: Sequence[str],
        observation_schema_id: str,
        roadnet_sha256: str,
        config: DQNConfig,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.network = network
        self.config = config
        self.device = torch.device(device)
        if not observation_schema_id:
            raise ValueError("observation_schema_id cannot be empty")
        if not isinstance(roadnet_sha256, str) or len(roadnet_sha256) != 64:
            raise ValueError("roadnet_sha256 must be a SHA-256 hexadecimal digest")
        try:
            roadnet_digest = bytes.fromhex(roadnet_sha256)
        except ValueError as exc:
            raise ValueError(
                "roadnet_sha256 must be a SHA-256 hexadecimal digest"
            ) from exc
        if len(roadnet_digest) != 32:
            raise ValueError("roadnet_sha256 must be a SHA-256 hexadecimal digest")
        self.observation_schema_id = observation_schema_id
        self.roadnet_sha256 = roadnet_sha256.lower()
        self.network_schema_sha256 = network_schema_sha256(network)
        self.vectorizer = ObservationVectorizer(network, feature_names)
        torch.manual_seed(seed)
        self.online = QNetwork(
            self.vectorizer.output_dim, config.hidden_dim, network.max_actions
        ).to(self.device)
        self._rng = np.random.default_rng(0)
        self._checkpoint_sha256: Optional[str] = None

    def reset(self, seed: int, network: NetworkSpec) -> None:
        if network_schema_sha256(network) != self.network_schema_sha256:
            raise ValueError("DQN policy is incompatible with this network schema")
        self._rng = np.random.default_rng(seed)

    @property
    def checkpoint_sha256(self) -> Optional[str]:
        return self._checkpoint_sha256

    def select_actions(
        self, observation: NetworkObservation, epsilon: float
    ) -> PolicyOutput:
        states = self.vectorizer.transform(observation)
        with torch.no_grad():
            q_values = self.online(
                torch.as_tensor(states, dtype=torch.float32, device=self.device)
            ).cpu().numpy()
        q_values = np.where(observation.action_mask, q_values, -np.inf)
        greedy = np.argmax(q_values, axis=1).astype(np.int64)
        actions = greedy.copy()
        explore = self._rng.random(self.network.num_intersections) < epsilon
        for index in np.flatnonzero(explore):
            valid = np.flatnonzero(observation.action_mask[int(index)])
            actions[int(index)] = int(self._rng.choice(valid))
        return PolicyOutput(
            actions=actions,
            diagnostics={"epsilon": float(epsilon), "q_values": q_values.tolist()},
        )

    def act(self, observation: NetworkObservation, deterministic: bool) -> PolicyOutput:
        return self.select_actions(observation, epsilon=0.0 if deterministic else 0.05)

    def save(self, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_state_dict": self.online.state_dict(),
            "config": asdict(self.config),
            "input_dim": self.vectorizer.output_dim,
            "feature_count": self.vectorizer.feature_count,
            "feature_names": list(self.vectorizer.feature_names),
            "vectorizer_version": VECTORIZER_VERSION,
            "observation_schema_id": self.observation_schema_id,
            "roadnet_sha256": self.roadnet_sha256,
            "network_schema_sha256": self.network_schema_sha256,
            "max_movements": self.network.max_movements,
            "max_actions": self.network.max_actions,
            "intersection_ids": list(self.network.intersection_ids),
            "engine_phase_ids": [
                list(item.engine_phase_ids) for item in self.network.intersections
            ],
            "extra": extra or {},
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        self._checkpoint_sha256 = self._sha256_file(target)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load(self, path: Path) -> Dict[str, Any]:
        try:
            payload = torch.load(
                Path(path), map_location=self.device, weights_only=False
            )
        except TypeError:  # PyTorch 2.0 compatibility
            payload = torch.load(Path(path), map_location=self.device)
        if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported DQN checkpoint version")
        if payload.get("config") != asdict(self.config):
            raise ValueError("checkpoint DQN configuration does not match runtime")
        semantic_expected = {
            "feature_names": list(self.vectorizer.feature_names),
            "vectorizer_version": VECTORIZER_VERSION,
            "observation_schema_id": self.observation_schema_id,
            "roadnet_sha256": self.roadnet_sha256,
            "network_schema_sha256": self.network_schema_sha256,
        }
        for key, expected_value in semantic_expected.items():
            if payload.get(key) != expected_value:
                raise ValueError(
                    f"checkpoint {key} does not match the runtime data contract"
                )
        expected = (
            self.vectorizer.output_dim,
            self.vectorizer.feature_count,
            self.network.max_movements,
            self.network.max_actions,
        )
        actual = (
            payload.get("input_dim"),
            payload.get("feature_count"),
            payload.get("max_movements"),
            payload.get("max_actions"),
        )
        if actual != expected:
            raise ValueError(f"checkpoint shape {actual} does not match runtime {expected}")
        expected_phase_ids = [
            list(item.engine_phase_ids) for item in self.network.intersections
        ]
        if payload.get("intersection_ids") != list(self.network.intersection_ids):
            raise ValueError("checkpoint intersection ordering does not match runtime")
        if payload.get("engine_phase_ids") != expected_phase_ids:
            raise ValueError("checkpoint action-to-engine phase mapping does not match runtime")
        self.online.load_state_dict(payload["model_state_dict"])
        self._checkpoint_sha256 = self._sha256_file(Path(path))
        return dict(payload.get("extra", {}))
