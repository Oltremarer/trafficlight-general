from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MovementSpec:
    index: int
    movement_type: str
    start_road: str
    end_road: str
    incoming_lanes: Tuple[str, ...]
    outgoing_lanes: Tuple[str, ...]
    static_features: Tuple[float, ...] = ()


@dataclass(frozen=True)
class IntersectionSpec:
    index: int
    intersection_id: str
    point: Tuple[float, float]
    incoming_lanes: Tuple[str, ...]
    outgoing_lanes: Tuple[str, ...]
    movements: Tuple[MovementSpec, ...]
    engine_phase_ids: Tuple[int, ...]
    phase_movement_mask: BoolArray

    @property
    def action_mask(self) -> BoolArray:
        return np.asarray([phase >= 0 for phase in self.engine_phase_ids], dtype=np.bool_)


@dataclass(frozen=True)
class NetworkSpec:
    intersections: Tuple[IntersectionSpec, ...]
    neighbor_index: IntArray
    neighbor_mask: BoolArray
    max_movements: int
    max_actions: int

    @property
    def num_intersections(self) -> int:
        return len(self.intersections)

    @property
    def intersection_ids(self) -> Tuple[str, ...]:
        return tuple(item.intersection_id for item in self.intersections)

    def action_mask(self) -> BoolArray:
        mask = np.zeros((self.num_intersections, self.max_actions), dtype=np.bool_)
        for item in self.intersections:
            mask[item.index, : len(item.engine_phase_ids)] = item.action_mask
        return mask

    def padded_phase_movement_mask(self) -> BoolArray:
        mask = np.zeros(
            (self.num_intersections, self.max_actions, self.max_movements),
            dtype=np.bool_,
        )
        for item in self.intersections:
            actions, movements = item.phase_movement_mask.shape
            mask[item.index, :actions, :movements] = item.phase_movement_mask
        return mask

    def padded_movement_static_features(self) -> FloatArray:
        feature_count = max(
            (
                len(movement.static_features)
                for item in self.intersections
                for movement in item.movements
            ),
            default=0,
        )
        values = np.zeros(
            (self.num_intersections, self.max_movements, feature_count),
            dtype=np.float32,
        )
        for item in self.intersections:
            for movement in item.movements:
                values[item.index, movement.index, : len(movement.static_features)] = (
                    movement.static_features
                )
        return values

    def engine_phase(self, intersection_index: int, action_index: int) -> int:
        item = self.intersections[intersection_index]
        try:
            phase = item.engine_phase_ids[action_index]
        except IndexError as exc:
            raise ValueError(
                f"invalid action {action_index} for {item.intersection_id}"
            ) from exc
        if phase < 0:
            raise ValueError(f"action {action_index} is masked for {item.intersection_id}")
        return phase


@dataclass(frozen=True)
class NetworkSnapshot:
    time_s: float
    lane_vehicle_count: Mapping[str, int]
    lane_waiting_count: Mapping[str, int]
    vehicle_speeds: Mapping[str, float]
    lane_vehicles: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    vehicle_distances: Mapping[str, float] = field(default_factory=dict)

    @property
    def vehicle_ids(self) -> Tuple[str, ...]:
        return tuple(self.vehicle_speeds.keys())


@dataclass(frozen=True)
class NetworkObservation:
    features: FloatArray
    movement_mask: BoolArray
    valid_mask: BoolArray
    current_phase: IntArray
    signal_stage: IntArray
    phase_elapsed_s: FloatArray
    neighbor_index: IntArray
    neighbor_mask: BoolArray
    action_mask: BoolArray
    time_s: float
    feature_names: Tuple[str, ...]
    baseline_view: Optional[Any] = None


@dataclass(frozen=True)
class PolicyOutput:
    actions: IntArray
    recurrent_state: Optional[Any] = None
    diagnostics: Optional[Dict[str, Any]] = None
    behavior: Optional[Any] = None


@dataclass(frozen=True)
class EpisodeResult:
    metrics: Dict[str, float]
    steps: int
    trajectory_path: str
    manifest_path: str


class Policy(Protocol):
    name: str

    def reset(self, seed: int, network: NetworkSpec) -> None:
        ...

    def act(self, observation: NetworkObservation, deterministic: bool) -> PolicyOutput:
        ...


class ObservationBuilder(Protocol):
    feature_names: Tuple[str, ...]

    def build(
        self,
        snapshot: NetworkSnapshot,
        current_phase: IntArray,
        signal_stage: IntArray,
        phase_elapsed_s: FloatArray,
    ) -> NetworkObservation:
        ...


class RewardCalculator(Protocol):
    name: str

    def compute(self, snapshot: NetworkSnapshot) -> FloatArray:
        ...
