"""Opt-in event semantics v2; legacy collection and its files remain unchanged."""
from __future__ import annotations

import numpy as np

from .recorder import PhysicalRecorder, counts16


class RouteAwareRecorder(PhysicalRecorder):
    """Separate an internal trip destination from a geographical boundary exit.

    Only a vanished vehicle on its declared final road, within one observed
    tick's speed-bound reach of the lane end, is reclassified. This is specific
    to the pinned CityFlow routes and ordinary vehicle lifecycle; unknown
    deletions and other transition anomalies remain unresolved.
    """

    EVENT_SEMANTICS = "route-completion-v2"

    def __init__(self, geometry, flows):
        super().__init__(geometry, flows)
        self.previous_distances = {}
        self.previous_time = None

    def context(self):
        return {**super().context(), "previous_distances": self.previous_distances.copy(),
                "previous_time": self.previous_time}

    def restore_context(self, values):
        super().restore_context(values)
        self.previous_distances = values.get("previous_distances", {}).copy()
        self.previous_time = values.get("previous_time")

    def observe(self, engine):
        previous = self.previous_locations.copy()
        distances = self.previous_distances
        now_time = float(engine.get_current_time())
        dt = 0. if self.previous_time is None else now_time - self.previous_time
        row = super().observe(engine)
        internal = np.zeros(len(self.g.lanes), dtype=np.uint32)
        if dt > 0:
            for vid in previous.keys() - self.previous_locations.keys():
                li = self.g.lane_index.get(previous[vid])
                if li is None or vid in self.previous_all or vid not in distances:
                    continue
                road = self.g.lane_roads[li]
                if road in self.g.exit_roads:
                    continue
                flow = self.flows[int(vid.split("_")[1])]
                reach = min(float(flow["vehicle"]["maxSpeed"]), self.g.speeds[li]) * dt
                remaining = self.g.lengths[li] - distances[vid]
                if road == flow["route"][-1] and -1e-6 <= remaining <= reach + 1e-6:
                    internal[li] += 1
        resolved = int(internal.sum(dtype=np.uint64))
        old_unresolved = int(row["unresolved_events"])
        if resolved > old_unresolved:
            raise RuntimeError("Cannot reclassify more events than the legacy unresolved count")
        row["unresolved_events"] = np.asarray(old_unresolved - resolved, dtype=np.uint32)
        row["internal_trip_completion_by_lane"] = counts16(internal)
        row["internal_trip_completion_count"] = counts16(resolved)
        # exit_count is deliberately unchanged: it means geographical boundary
        # exits, not all completed trips. Do not silently relabel existing data.
        row["trip_completion_count"] = counts16(int(row["exit_count"]) + resolved)
        self.previous_distances = engine.get_vehicle_distance()
        self.previous_time = now_time
        return row
