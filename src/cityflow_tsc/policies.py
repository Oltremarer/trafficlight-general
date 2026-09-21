from __future__ import annotations

from typing import Optional

import numpy as np

from .types import NetworkObservation, NetworkSpec, PolicyOutput


class RandomPolicy:
    name = "random"

    def __init__(self) -> None:
        self._network: Optional[NetworkSpec] = None
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int, network: NetworkSpec) -> None:
        self._network = network
        self._rng = np.random.default_rng(seed)

    def act(self, observation: NetworkObservation, deterministic: bool) -> PolicyOutput:
        actions = np.zeros(observation.action_mask.shape[0], dtype=np.int64)
        for index, mask in enumerate(observation.action_mask):
            valid = np.flatnonzero(mask)
            if not len(valid):
                raise ValueError(f"intersection {index} has no valid action")
            actions[index] = valid[0] if deterministic else self._rng.choice(valid)
        return PolicyOutput(actions=actions)


class FixedTimePolicy:
    name = "fixed_time"

    def __init__(self, offset: int = 0) -> None:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        self._offset = int(offset)
        self._step = self._offset

    def reset(self, seed: int, network: NetworkSpec) -> None:
        self._step = self._offset

    def act(self, observation: NetworkObservation, deterministic: bool) -> PolicyOutput:
        actions = np.zeros(observation.action_mask.shape[0], dtype=np.int64)
        for index, mask in enumerate(observation.action_mask):
            valid = np.flatnonzero(mask)
            if not len(valid):
                raise ValueError(f"intersection {index} has no valid action")
            actions[index] = valid[self._step % len(valid)]
        self._step += 1
        return PolicyOutput(actions=actions)


class MaxPressurePolicy:
    """Choose the phase with maximum summed queue pressure."""

    name = "max_pressure"

    def __init__(self) -> None:
        self._network: Optional[NetworkSpec] = None
        self._phase_movement_mask: Optional[np.ndarray] = None

    def reset(self, seed: int, network: NetworkSpec) -> None:
        self._network = network
        self._phase_movement_mask = network.padded_phase_movement_mask()

    def _phase_scores(self, observation: NetworkObservation) -> np.ndarray:
        if self._phase_movement_mask is None:
            raise RuntimeError("policy must be reset before act")
        try:
            pressure_index = observation.feature_names.index("queue_pressure")
        except ValueError as exc:
            raise ValueError("MaxPressurePolicy requires queue_pressure") from exc

        queue_pressure = observation.features[:, :, pressure_index]
        scores = np.einsum(
            "nam,nm->na",
            self._phase_movement_mask.astype(np.float32),
            queue_pressure,
        )
        scores = np.where(observation.action_mask, scores, -np.inf)
        return scores

    def act(self, observation: NetworkObservation, deterministic: bool) -> PolicyOutput:
        scores = self._phase_scores(observation)
        if np.any(np.all(~observation.action_mask, axis=1)):
            raise ValueError("every intersection must expose at least one valid action")
        actions = np.argmax(scores, axis=1).astype(np.int64)
        return PolicyOutput(
            actions=actions,
            diagnostics={"phase_pressure": scores.tolist()},
        )


class SoftPressurePolicy(MaxPressurePolicy):
    """Sample valid phases from queue-pressure scores to diversify behavior."""

    name = "soft_pressure"

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._temperature = float(temperature)
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int, network: NetworkSpec) -> None:
        super().reset(seed, network)
        self._rng = np.random.default_rng(seed)

    def act(self, observation: NetworkObservation, deterministic: bool) -> PolicyOutput:
        scores = self._phase_scores(observation)
        if np.any(np.all(~observation.action_mask, axis=1)):
            raise ValueError("every intersection must expose at least one valid action")
        actions = np.zeros(observation.action_mask.shape[0], dtype=np.int64)
        for index, valid_mask in enumerate(observation.action_mask):
            valid = np.flatnonzero(valid_mask)
            values = scores[index, valid]
            if deterministic:
                actions[index] = int(valid[np.argmax(values)])
                continue
            logits = (values - np.max(values)) / self._temperature
            probabilities = np.exp(logits)
            probabilities /= np.sum(probabilities)
            actions[index] = int(self._rng.choice(valid, p=probabilities))
        return PolicyOutput(
            actions=actions,
            diagnostics={
                "phase_pressure": scores.tolist(),
                "temperature": self._temperature,
            },
        )


def make_policy(name: str):
    normalized = name.strip().lower().replace("-", "_")
    policies = {
        "random": RandomPolicy,
        "fixed_time": FixedTimePolicy,
        "max_pressure": MaxPressurePolicy,
        "soft_pressure": SoftPressurePolicy,
    }
    try:
        return policies[normalized]()
    except KeyError as exc:
        raise ValueError(
            f"unknown policy {name!r}; choose from {sorted(policies)}"
        ) from exc
