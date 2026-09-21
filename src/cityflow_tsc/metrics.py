from __future__ import annotations

from typing import Dict, Set

import numpy as np

from .types import NetworkSnapshot


class MetricCollector:
    """Collect metrics from every simulator tick, independently of reward."""

    def __init__(self, waiting_speed_threshold: float = 0.1) -> None:
        self.waiting_speed_threshold = waiting_speed_threshold
        self.reset()

    def reset(self) -> None:
        self._previous_active: Set[str] = set()
        self._seen: Set[str] = set()
        self._completed: Set[str] = set()
        self._waiting_time: Dict[str, float] = {}
        self._queue_samples = []
        self._active_samples = []
        self._tick_count = 0

    def observe(self, snapshot: NetworkSnapshot, elapsed_s: float) -> None:
        active = set(snapshot.vehicle_ids)
        self._seen.update(active)
        self._completed.update(self._previous_active - active)
        if elapsed_s > 0:
            for vehicle_id, speed in snapshot.vehicle_speeds.items():
                if speed <= self.waiting_speed_threshold:
                    self._waiting_time[vehicle_id] = (
                        self._waiting_time.get(vehicle_id, 0.0) + elapsed_s
                    )
            self._queue_samples.append(
                float(sum(snapshot.lane_waiting_count.values()))
            )
            self._active_samples.append(float(len(active)))
            self._tick_count += 1
        self._previous_active = active

    def summary(self, engine_average_travel_time_s: float) -> Dict[str, float]:
        waiting_values = [self._waiting_time.get(vehicle, 0.0) for vehicle in self._seen]
        return {
            "average_travel_time_s": float(engine_average_travel_time_s),
            "average_queue_vehicles": float(np.mean(self._queue_samples))
            if self._queue_samples
            else 0.0,
            "average_waiting_time_s": float(np.mean(waiting_values))
            if waiting_values
            else 0.0,
            "average_active_vehicles": float(np.mean(self._active_samples))
            if self._active_samples
            else 0.0,
            "throughput_vehicles": float(len(self._completed)),
            "observed_vehicles": float(len(self._seen)),
            "metric_ticks": float(self._tick_count),
        }
