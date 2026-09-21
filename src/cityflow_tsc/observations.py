from __future__ import annotations

from typing import Optional

import numpy as np

from .types import NetworkObservation, NetworkSnapshot, NetworkSpec


class QueuePressureObservationBuilder:
    """Build dense movement-level queue and pressure observations."""

    schema_id = "queue-pressure-movement-v1"
    feature_names = (
        "incoming_vehicle_count",
        "incoming_queue_count",
        "outgoing_vehicle_count",
        "outgoing_queue_count",
        "vehicle_pressure",
        "queue_pressure",
    )

    def __init__(self, network: NetworkSpec) -> None:
        self.network = network

    @staticmethod
    def _sum(mapping, keys) -> float:
        return float(sum(mapping.get(key, 0) for key in keys))

    def build(
        self,
        snapshot: NetworkSnapshot,
        current_phase: np.ndarray,
        signal_stage: np.ndarray,
        phase_elapsed_s: np.ndarray,
    ) -> NetworkObservation:
        num_features = len(self.feature_names)
        features = np.zeros(
            (
                self.network.num_intersections,
                self.network.max_movements,
                num_features,
            ),
            dtype=np.float32,
        )
        movement_mask = np.zeros(
            (self.network.num_intersections, self.network.max_movements),
            dtype=np.bool_,
        )
        valid_mask = np.zeros_like(features, dtype=np.bool_)

        for intersection in self.network.intersections:
            for movement in intersection.movements:
                incoming_vehicle = self._sum(
                    snapshot.lane_vehicle_count, movement.incoming_lanes
                )
                incoming_queue = self._sum(
                    snapshot.lane_waiting_count, movement.incoming_lanes
                )
                outgoing_vehicle = self._sum(
                    snapshot.lane_vehicle_count, movement.outgoing_lanes
                )
                outgoing_queue = self._sum(
                    snapshot.lane_waiting_count, movement.outgoing_lanes
                )
                features[intersection.index, movement.index] = np.asarray(
                    [
                        incoming_vehicle,
                        incoming_queue,
                        outgoing_vehicle,
                        outgoing_queue,
                        incoming_vehicle - outgoing_vehicle,
                        incoming_queue - outgoing_queue,
                    ],
                    dtype=np.float32,
                )
                movement_mask[intersection.index, movement.index] = True
                valid_mask[intersection.index, movement.index, :] = True

        return NetworkObservation(
            features=features,
            movement_mask=movement_mask,
            valid_mask=valid_mask,
            current_phase=np.asarray(current_phase, dtype=np.int64).copy(),
            signal_stage=np.asarray(signal_stage, dtype=np.int8).copy(),
            phase_elapsed_s=np.asarray(phase_elapsed_s, dtype=np.float32).copy(),
            neighbor_index=self.network.neighbor_index.copy(),
            neighbor_mask=self.network.neighbor_mask.copy(),
            action_mask=self.network.action_mask(),
            time_s=float(snapshot.time_s),
            feature_names=self.feature_names,
        )


class RichTrafficObservationBuilder:
    """Movement observations for learning action-conditioned traffic dynamics.

    Raw vehicle records remain variable length in CityFlow.  This builder turns
    them into stable movement-level statistics while preserving masks and the
    static road semantics separately in ``NetworkSpec``.
    """

    schema_id = "rich-traffic-movement-v1"
    feature_names = (
        "incoming_vehicle_count",
        "incoming_queue_count",
        "outgoing_vehicle_count",
        "outgoing_queue_count",
        "incoming_mean_speed_mps",
        "outgoing_mean_speed_mps",
        "incoming_speed_std_mps",
        "incoming_stopped_ratio",
        "incoming_progress_mean",
        "incoming_progress_std",
        "incoming_headway_mean_m",
        "incoming_near_stopline_50m_count",
        "incoming_near_stopline_100m_count",
        "recent_arrival_rate_veh_s",
        "recent_departure_rate_veh_s",
        "incoming_occupancy_ratio",
        "downstream_occupancy_ratio",
        "vehicle_pressure",
        "queue_pressure",
    )

    def __init__(self, network: NetworkSpec, waiting_speed_threshold: float = 0.1) -> None:
        if waiting_speed_threshold < 0:
            raise ValueError("waiting_speed_threshold cannot be negative")
        self.network = network
        self.waiting_speed_threshold = float(waiting_speed_threshold)
        self._previous_time_s: Optional[float] = None
        self._previous_vehicle_ids: dict[tuple[int, int], set[str]] = {}
        self._previous_vehicle_count: dict[tuple[int, int], float] = {}

    def reset(self) -> None:
        self._previous_time_s = None
        self._previous_vehicle_ids.clear()
        self._previous_vehicle_count.clear()

    @staticmethod
    def _sum(mapping, keys) -> float:
        return float(sum(mapping.get(key, 0) for key in keys))

    @staticmethod
    def _vehicle_ids(snapshot: NetworkSnapshot, lanes) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                vehicle_id
                for lane_id in lanes
                for vehicle_id in snapshot.lane_vehicles.get(lane_id, ())
            )
        )

    def _micro_statistics(
        self,
        snapshot: NetworkSnapshot,
        lanes,
        lane_length_m: float,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        vehicle_ids = self._vehicle_ids(snapshot, lanes)
        speeds = np.asarray(
            [snapshot.vehicle_speeds.get(vehicle_id, 0.0) for vehicle_id in vehicle_ids],
            dtype=np.float32,
        )
        distances = np.asarray(
            [snapshot.vehicle_distances.get(vehicle_id, 0.0) for vehicle_id in vehicle_ids],
            dtype=np.float32,
        )
        if not len(vehicle_ids):
            return (0.0,) * 8
        safe_length = max(float(lane_length_m), 1.0)
        progress = np.clip(distances / safe_length, 0.0, 1.0)
        headways = []
        for lane_id in lanes:
            lane_distances = sorted(
                (
                    float(snapshot.vehicle_distances.get(vehicle_id, 0.0))
                    for vehicle_id in snapshot.lane_vehicles.get(lane_id, ())
                ),
                reverse=True,
            )
            headways.extend(
                max(0.0, leader - follower)
                for leader, follower in zip(lane_distances, lane_distances[1:])
            )
        remaining = np.clip(safe_length - distances, 0.0, None)
        return (
            float(speeds.mean()),
            float(speeds.std()),
            float(np.mean(speeds <= self.waiting_speed_threshold)),
            float(progress.mean()),
            float(progress.std()),
            float(np.mean(headways)) if headways else 0.0,
            float(np.count_nonzero(remaining <= 50.0)),
            float(np.count_nonzero(remaining <= 100.0)),
        )

    def build(
        self,
        snapshot: NetworkSnapshot,
        current_phase: np.ndarray,
        signal_stage: np.ndarray,
        phase_elapsed_s: np.ndarray,
    ) -> NetworkObservation:
        features = np.zeros(
            (
                self.network.num_intersections,
                self.network.max_movements,
                len(self.feature_names),
            ),
            dtype=np.float32,
        )
        movement_mask = np.zeros(
            (self.network.num_intersections, self.network.max_movements),
            dtype=np.bool_,
        )
        valid_mask = np.zeros_like(features, dtype=np.bool_)
        dt = None
        if self._previous_time_s is not None:
            dt = max(float(snapshot.time_s) - self._previous_time_s, 1e-6)

        for intersection in self.network.intersections:
            for movement in intersection.movements:
                incoming_vehicle = self._sum(
                    snapshot.lane_vehicle_count, movement.incoming_lanes
                )
                incoming_queue = self._sum(
                    snapshot.lane_waiting_count, movement.incoming_lanes
                )
                outgoing_vehicle = self._sum(
                    snapshot.lane_vehicle_count, movement.outgoing_lanes
                )
                outgoing_queue = self._sum(
                    snapshot.lane_waiting_count, movement.outgoing_lanes
                )
                static = movement.static_features
                incoming_length = static[2] if len(static) > 2 else 1.0
                outgoing_length = static[3] if len(static) > 3 else 1.0
                incoming_capacity = static[4] if len(static) > 4 else 1.0
                outgoing_capacity = static[5] if len(static) > 5 else 1.0
                (
                    incoming_mean_speed,
                    incoming_speed_std,
                    incoming_stopped_ratio,
                    incoming_progress_mean,
                    incoming_progress_std,
                    incoming_headway_mean,
                    near_50m,
                    near_100m,
                ) = self._micro_statistics(
                    snapshot, movement.incoming_lanes, incoming_length
                )
                outgoing_mean_speed = self._micro_statistics(
                    snapshot, movement.outgoing_lanes, outgoing_length
                )[0]

                key = (intersection.index, movement.index)
                current_ids = set(self._vehicle_ids(snapshot, movement.incoming_lanes))
                previous_ids = self._previous_vehicle_ids.get(key, set())
                if dt is None:
                    arrival_rate = 0.0
                    departure_rate = 0.0
                    rate_valid = False
                elif current_ids or previous_ids:
                    arrival_rate = len(current_ids - previous_ids) / dt
                    departure_rate = len(previous_ids - current_ids) / dt
                    rate_valid = True
                else:
                    previous_count = self._previous_vehicle_count.get(key, incoming_vehicle)
                    arrival_rate = max(incoming_vehicle - previous_count, 0.0) / dt
                    departure_rate = max(previous_count - incoming_vehicle, 0.0) / dt
                    rate_valid = True
                self._previous_vehicle_ids[key] = current_ids
                self._previous_vehicle_count[key] = incoming_vehicle

                features[intersection.index, movement.index] = np.asarray(
                    [
                        incoming_vehicle,
                        incoming_queue,
                        outgoing_vehicle,
                        outgoing_queue,
                        incoming_mean_speed,
                        outgoing_mean_speed,
                        incoming_speed_std,
                        incoming_stopped_ratio,
                        incoming_progress_mean,
                        incoming_progress_std,
                        incoming_headway_mean,
                        near_50m,
                        near_100m,
                        arrival_rate,
                        departure_rate,
                        incoming_vehicle / max(float(incoming_capacity), 1.0),
                        outgoing_vehicle / max(float(outgoing_capacity), 1.0),
                        incoming_vehicle - outgoing_vehicle,
                        incoming_queue - outgoing_queue,
                    ],
                    dtype=np.float32,
                )
                movement_mask[intersection.index, movement.index] = True
                valid_mask[intersection.index, movement.index, :] = True
                valid_mask[intersection.index, movement.index, 13:15] = rate_valid

        self._previous_time_s = float(snapshot.time_s)
        return NetworkObservation(
            features=features,
            movement_mask=movement_mask,
            valid_mask=valid_mask,
            current_phase=np.asarray(current_phase, dtype=np.int64).copy(),
            signal_stage=np.asarray(signal_stage, dtype=np.int8).copy(),
            phase_elapsed_s=np.asarray(phase_elapsed_s, dtype=np.float32).copy(),
            neighbor_index=self.network.neighbor_index.copy(),
            neighbor_mask=self.network.neighbor_mask.copy(),
            action_mask=self.network.action_mask(),
            time_s=float(snapshot.time_s),
            feature_names=self.feature_names,
        )
