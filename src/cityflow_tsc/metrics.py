from __future__ import annotations

from typing import Dict, Set
import json

import numpy as np

from .types import NetworkSnapshot
from .lifecycle_metrics import LifecycleLedger, REVISION, UnsupportedLifecycleFlow


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
        self._lifecycle = None
        self._lifecycle_output = None
        self._lifecycle_unavailable_reason = "collector has not been configured for an episode"

    def configure(self, scenario, step_s: float, initial: NetworkSnapshot) -> None:
        self._lifecycle = None
        self._lifecycle_output = scenario.output_dir / "lifecycle.manifest.json"
        if (initial.vehicle_pool_ids is None or initial.active_vehicle_ids is None
                or initial.active_vehicle_count is None):
            self._lifecycle_unavailable_reason = "backend does not expose complete pool and active vehicle IDs"
            return
        try:
            self._lifecycle = LifecycleLedger.from_scenario(scenario, step_s)
        except UnsupportedLifecycleFlow as error:
            self._lifecycle_unavailable_reason = str(error)
        else:
            self._lifecycle_unavailable_reason = None

    def observe(self, snapshot: NetworkSnapshot, elapsed_s: float) -> None:
        if self._lifecycle is not None:
            if (snapshot.vehicle_pool_ids is None or snapshot.active_vehicle_ids is None
                    or snapshot.active_vehicle_count is None):
                raise ValueError("backend lifecycle membership disappeared during the episode")
            self._lifecycle.observe(snapshot.time_s, snapshot.vehicle_pool_ids,
                                    snapshot.active_vehicle_ids, snapshot.active_vehicle_count)
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
        result = {
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
        result["lifecycle_metrics_available"] = float(self._lifecycle is not None)
        if self._lifecycle_output is not None:
            self._lifecycle_output.parent.mkdir(parents=True, exist_ok=True)
        if self._lifecycle is not None:
            result.update(self._lifecycle.write_evidence(self._lifecycle_output))
        elif self._lifecycle_output is not None:
            evidence = {"metric_schema_revision": REVISION, "available": False,
                        "reason": self._lifecycle_unavailable_reason}
            temporary = self._lifecycle_output.with_suffix(".tmp")
            temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
            temporary.replace(self._lifecycle_output)
        return result
