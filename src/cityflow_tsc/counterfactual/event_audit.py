"""Read-only event tracing; does not change the original recorder's labels."""
from __future__ import annotations

from .recorder import PhysicalRecorder


class EventAuditRecorder(PhysicalRecorder):
    def __init__(self, geometry, flows):
        super().__init__(geometry, flows)
        self.previous_speeds = {}
        self.previous_distances = {}
        self.last_events = []

    def context(self):
        return {**super().context(), "previous_speeds": self.previous_speeds.copy(),
                "previous_distances": self.previous_distances.copy()}

    def restore_context(self, values):
        super().restore_context(values)
        self.previous_speeds = values.get("previous_speeds", {}).copy()
        self.previous_distances = values.get("previous_distances", {}).copy()

    def observe(self, engine):
        previous = self.previous_locations.copy()
        row = super().observe(engine)
        now = self.previous_locations
        self.last_events = []
        g = self.g

        def add(vid, reason):
            old, new = previous.get(vid), now.get(vid)
            li = g.lane_index.get(old)
            flow = self.flows[int(vid.split("_")[1])]
            event = {"time_s": int(row["time_s"]), "vehicle_id": vid, "reason": reason,
                     "before": old, "after": new,
                     "before_speed": self.previous_speeds.get(vid),
                     "before_distance": self.previous_distances.get(vid),
                     "route_first_road": flow["route"][0], "route_last_road": flow["route"][-1],
                     "still_exists": vid in self.previous_all,
                     "vehicle_max_speed": float(flow["vehicle"]["maxSpeed"])}
            if li is not None:
                event.update(before_road=g.lane_roads[li], lane_length_m=float(g.lengths[li]),
                             remaining_lane_m=float(g.lengths[li]) - self.previous_distances.get(vid, 0.))
            if new is not None:
                event["current_info"] = engine.get_vehicle_info(vid)
            self.last_events.append(event)

        if int(row["unresolved_events"]):
            for vid, new in now.items():
                old = previous.get(vid)
                if old == new:
                    continue
                ni, oi = g.lane_index.get(new), g.lane_index.get(old)
                if old is None:
                    if ni is None:
                        add(vid, "appeared_on_lanelink")
                elif oi is not None and new in g.links:
                    if g.links[new][2] != oi:
                        add(vid, "release_lane_mismatch")
                elif old in g.links and ni is not None:
                    if g.links[old][3] != ni:
                        add(vid, "receive_lane_mismatch")
                elif oi is not None and ni is not None:
                    if (g.lane_roads[oi], g.lane_roads[ni]) not in g.road_pair_movement:
                        add(vid, "unmapped_road_transition")
                else:
                    add(vid, "unsupported_drivable_transition")
            for vid in previous.keys() - now.keys():
                li = g.lane_index.get(previous[vid])
                if li is None:
                    add(vid, "disappeared_from_lanelink")
                elif g.lane_roads[li] not in g.exit_roads:
                    add(vid, "disappeared_from_internal_lane")
            if len(self.last_events) != int(row["unresolved_events"]):
                raise RuntimeError("Diagnostic reasons do not cover the original counter")
        self.previous_speeds = engine.get_vehicle_speed()
        self.previous_distances = engine.get_vehicle_distance()
        return row
