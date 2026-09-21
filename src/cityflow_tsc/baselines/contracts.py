from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Tuple
import numpy as np

from ..types import NetworkObservation, PolicyOutput


@dataclass(frozen=True)
class BaselineView:
    """Independent snapshot in model node order (the environment's stable order)."""

    features: np.ndarray                 # [N, F], actor input
    lane_features: np.ndarray            # [N, L, C], phase competition input
    lane_mask: np.ndarray                # [N, L]
    phase_lane_mask: np.ndarray          # [N, A, L]
    phase_encoding: np.ndarray           # [N, P], source phase representation
    neighbor_index: np.ndarray           # [N, K], -1 for padding
    neighbor_mask: np.ndarray            # [N, K]
    schema_id: str
    node_ids: Tuple[str, ...]
    feature_names: Tuple[str, ...] = ()
    source_action_to_local: Optional[np.ndarray] = None  # [N, S], -1 for unavailable source phases


@dataclass(frozen=True)
class BehaviorInfo:
    log_prob: Optional[np.ndarray] = None
    value: Optional[np.ndarray] = None
    policy_version: int = 0
    hidden_state: Optional[np.ndarray] = None   # state BEFORE sampling action
    action_vector: Optional[np.ndarray] = None


@dataclass(frozen=True)
class JointTransition:
    observation: NetworkObservation
    output: PolicyOutput
    reward: np.ndarray
    next_observation: NetworkObservation
    terminated: bool
    truncated: bool
    elapsed_s: float
    episode_id: str = ""
    scenario_id: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def actions(self) -> np.ndarray:
        return self.output.actions

    @property
    def behavior(self) -> Optional[BehaviorInfo]:
        return self.output.behavior


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.8
    reward_scale: float = 0.05
    batch_size: int = 32
    replay_capacity: int = 10000
    warmup_transitions: int = 32
    updates_per_round: int = 10
    target_update_steps: int = 100
    epsilon_start: float = 0.8
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    gradient_clip: float = 10.0
    bootstrap_truncated: bool = True
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    return_estimator: str = "nstep"
    n_steps: int = 5
    gae_lambda: float = 0.95
    tau: float = 0.01

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "batch_size", "replay_capacity", "updates_per_round",
                     "target_update_steps", "epsilon_decay_steps", "ppo_epochs", "n_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_transitions < 0 or self.warmup_transitions > self.replay_capacity:
            raise ValueError("warmup_transitions must fit replay_capacity")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size must fit replay_capacity")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("discount and GAE lambda must be in [0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("invalid epsilon schedule")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.return_estimator not in {"nstep", "gae"}:
            raise ValueError("return_estimator must be nstep or gae")
        if not 0 < self.ppo_clip < 1 or not 0 < self.tau <= 1:
            raise ValueError("invalid PPO clip or target tau")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_view(observation: NetworkObservation) -> BaselineView:
    if observation.baseline_view is None:
        raise ValueError("baseline requires an environment built with its profile")
    return observation.baseline_view
