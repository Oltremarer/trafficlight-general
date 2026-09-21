from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - covered on ML runtime
    raise RuntimeError(
        "The World Model requires the optional ML dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc

from ..config import ControlConfig
from ..dqn import network_schema_sha256
from ..policies import MaxPressurePolicy
from ..types import NetworkObservation, NetworkSpec, PolicyOutput
from .data import EXPECTED_FEATURE_NAMES, FeatureStatistics
from .model import GraphWorldModel


@dataclass(frozen=True)
class PlannerConfig:
    candidate_count: int = 64
    horizon: int = 3
    discount: float = 0.95

    def __post_init__(self) -> None:
        if self.candidate_count <= 0 or self.horizon <= 0:
            raise ValueError("planner candidate count and horizon must be positive")
        if not 0 < self.discount <= 1:
            raise ValueError("planner discount must be in (0, 1]")


class WorldModelPolicy:
    """Score valid local mutations of MaxPressure with imagined rollouts."""

    name = "world_model"

    def __init__(
        self,
        model: GraphWorldModel,
        statistics: FeatureStatistics,
        network: NetworkSpec,
        control: ControlConfig,
        planner: PlannerConfig,
        checkpoint_sha256: str,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.statistics = statistics
        self.network = network
        self.control = control
        self.planner = planner
        self.checkpoint_sha256 = checkpoint_sha256
        self.device = torch.device(device)
        self.network_schema_sha256 = network_schema_sha256(network)
        if model.config.max_actions != network.max_actions:
            raise ValueError("World Model action dimension does not match the network")
        if not torch.equal(
            model.neighbor_index.cpu(), torch.as_tensor(network.neighbor_index)
        ) or not torch.equal(
            model.neighbor_mask.cpu(), torch.as_tensor(network.neighbor_mask)
        ):
            raise ValueError("World Model graph does not match the network topology")
        self._base_policy = MaxPressurePolicy()
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int, network: NetworkSpec) -> None:
        if network_schema_sha256(network) != self.network_schema_sha256:
            raise ValueError("World Model policy is incompatible with this network")
        self._rng = np.random.default_rng(seed)
        self._base_policy.reset(seed, network)

    def _candidate_sequences(
        self,
        base_actions: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        candidates = np.tile(
            base_actions[None, None, :],
            (self.planner.candidate_count, self.planner.horizon, 1),
        ).astype(np.int64)
        positions = self.planner.horizon * self.network.num_intersections
        for candidate in range(1, self.planner.candidate_count):
            mutation_count = 1 + ((candidate - 1) // max(positions, 1))
            mutation_count = min(mutation_count, min(3, positions))
            for mutation in range(mutation_count):
                if deterministic:
                    flat = (candidate - 1 + mutation * 17) % positions
                else:
                    flat = int(self._rng.integers(positions))
                step = flat // self.network.num_intersections
                node = flat % self.network.num_intersections
                valid = np.flatnonzero(action_mask[node])
                if not len(valid):
                    raise ValueError(f"intersection {node} has no valid action")
                current = int(candidates[candidate, step, node])
                alternatives = valid[valid != current]
                if len(alternatives):
                    offset = candidate + mutation
                    selected = alternatives[offset % len(alternatives)]
                    candidates[candidate, step, node] = int(selected)
        return candidates

    def act(self, observation: NetworkObservation, deterministic: bool) -> PolicyOutput:
        if tuple(observation.feature_names) != EXPECTED_FEATURE_NAMES:
            raise ValueError("World Model observation feature schema does not match")
        base = self._base_policy.act(observation, deterministic=True).actions
        candidates = self._candidate_sequences(
            base, observation.action_mask, deterministic
        )
        batch = self.planner.candidate_count
        features = self.statistics.normalize_features(observation.features[..., :4])

        def repeated(values: np.ndarray) -> "torch.Tensor":
            return torch.as_tensor(
                np.repeat(values[None, ...], batch, axis=0), device=self.device
            )

        with torch.no_grad():
            scores = self.model.rollout_scores(
                features=repeated(features).float(),
                valid_mask=repeated(observation.valid_mask[..., :4]).bool(),
                movement_mask=repeated(observation.movement_mask).bool(),
                current_phase=repeated(observation.current_phase).long(),
                signal_stage=repeated(observation.signal_stage).long(),
                phase_elapsed_s=repeated(observation.phase_elapsed_s).float(),
                action_sequences=torch.as_tensor(
                    candidates, dtype=torch.long, device=self.device
                ),
                decision_interval_s=float(self.control.decision_interval_s),
                transition_time_s=float(
                    self.control.yellow_time_s + self.control.all_red_time_s
                ),
                discount=self.planner.discount,
            )
        score_values = scores.cpu().numpy()
        selected = int(np.argmax(score_values))
        actions = candidates[selected, 0].copy()
        if np.any(~observation.action_mask[np.arange(len(actions)), actions]):
            raise RuntimeError("World Model planner emitted a masked action")
        return PolicyOutput(
            actions=actions,
            diagnostics={
                "base_actions": base.tolist(),
                "selected_candidate": selected,
                "selected_score": float(score_values[selected]),
                "base_score": float(score_values[0]),
                "candidate_count": self.planner.candidate_count,
                "horizon": self.planner.horizon,
            },
        )

    def metadata(self):
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "network_schema_sha256": self.network_schema_sha256,
            "planner": asdict(self.planner),
        }
