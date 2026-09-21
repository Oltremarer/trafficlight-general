from __future__ import annotations

import numpy as np

from .geometry import Geometry


def counts16(values):
    values = np.asarray(values)
    if values.size and (values.min() < 0 or values.max() > 65535):
        raise OverflowError("Physical count does not fit uint16; refusing silent overflow")
    return values.astype(np.uint16)


class PhysicalRecorder:
    """One readout per simulator second. No paired replay or conservation gate."""

    def __init__(self, geometry: Geometry, flows):
        self.g = geometry
        self.flows = flows
        self.metadata = {}
        self.previous_locations = {}
        self.previous_all = set()

    def meta(self, vehicle_id):
        if vehicle_id not in self.metadata:
            parts = vehicle_id.split("_")
            if len(parts) != 3 or parts[0] != "flow":
                raise ValueError(f"Unsupported generated vehicle identity: {vehicle_id}")
            f = self.flows[int(parts[1])]
            self.metadata[vehicle_id] = (
                self.g.boundary_index[f["route"][0]],
                float(f["vehicle"]["maxSpeed"]), float(f["vehicle"]["length"]),
            )
        return self.metadata[vehicle_id]

    def context(self):
        return {"previous_locations": dict(self.previous_locations),
                "previous_all": sorted(self.previous_all)}

    def restore_context(self, context):
        self.previous_locations = dict(context["previous_locations"])
        self.previous_all = set(context["previous_all"])

    def observe(self, engine):
        g = self.g
        lane_lists = engine.get_lane_vehicles()
        speeds = engine.get_vehicle_speed()
        distances = engine.get_vehicle_distance()
        all_ids = set(engine.get_vehicles(include_waiting=True))
        active = set(speeds)
        locations = {}
        nl, nn, nm, nb = len(g.lanes), len(g.intersections), len(g.movements), len(g.boundaries)
        n = np.zeros((nl, 3), dtype=np.uint32)
        q = np.zeros((nl, 3), dtype=np.uint32)
        velocity_sum = np.zeros((nl, 3), dtype=np.float32)
        loss = np.zeros(nl, dtype=np.float32)
        tail = np.zeros((nl, 2), dtype=np.float32)
        unavailable = np.zeros(nl, dtype=np.uint8)
        node_nq = np.zeros((nn, 2), dtype=np.uint32)
        node_loss = np.zeros(nn, dtype=np.float64)
        pending = np.zeros(nb, dtype=np.uint32)
        births = np.zeros(nb, dtype=np.uint32)
        admitted = np.zeros(nb, dtype=np.uint32)
        lane_events = np.zeros((nl, 2), dtype=np.uint32)
        movement_events = np.zeros((nm, 2), dtype=np.uint32)
        unresolved = 0
        exited = 0

        for li, lid in enumerate(g.lanes):
            ids = lane_lists.get(lid, ())
            if not ids:
                continue
            locations.update((vid, lid) for vid in ids)
            x = np.fromiter((distances[v] for v in ids), dtype=np.float64, count=len(ids))
            v = np.fromiter((speeds[i] for i in ids), dtype=np.float64, count=len(ids))
            refs = np.fromiter((min(self.meta(i)[1], g.speeds[li]) for i in ids), dtype=np.float64, count=len(ids))
            bins = np.where(x < g.ends[li], 0, np.where(x < g.lengths[li] - g.ends[li], 1, 2))
            n[li] = np.bincount(bins, minlength=3)
            q[li] = np.bincount(bins[v < 0.1], minlength=3)
            velocity_sum[li] = np.bincount(bins, weights=v, minlength=3)
            loss[li] = np.maximum(0.0, 1.0 - v / refs).sum(dtype=np.float64)
            ti = int(np.argmin(x))
            tail[li] = x[ti], v[ti]
            tail_length = self.meta(ids[ti])[2]
            unavailable[li] = not (x[ti] > tail_length + 5.0 or v[ti] >= 2.0)

        for vid in active - locations.keys():
            info = engine.get_vehicle_info(vid)
            lid = info["drivable"]
            if lid not in g.links:
                raise ValueError(f"Unmapped drivable {lid}")
            locations[vid] = lid
            mi, ni, src, dst = g.links[lid]
            node_nq[ni, 0] += 1
            node_nq[ni, 1] += speeds[vid] < 0.1
            ref = min(self.meta(vid)[1], g.speeds[src], g.speeds[dst])
            node_loss[ni] += max(0.0, 1.0 - speeds[vid] / ref)
        for vid in all_ids - active:
            pending[self.meta(vid)[0]] += 1
        for vid in all_ids - self.previous_all:
            births[self.meta(vid)[0]] += 1

        previous = self.previous_locations
        for vid, now in locations.items():
            before = previous.get(vid)
            if before == now:
                continue
            now_lane = g.lane_index.get(now)
            before_lane = g.lane_index.get(before)
            if now_lane is not None:
                lane_events[now_lane, 0] += 1
            if before is None:
                admitted[self.meta(vid)[0]] += 1
                if now_lane is None:
                    unresolved += 1
                continue
            if before_lane is not None:
                lane_events[before_lane, 1] += 1
            if before_lane is not None and now in g.links:
                mi, _, src, _ = g.links[now]
                movement_events[mi, 0] += 1
                unresolved += src != before_lane
            elif before in g.links and now_lane is not None:
                mi, _, _, dst = g.links[before]
                movement_events[mi, 1] += 1
                unresolved += dst != now_lane
            elif before_lane is not None and now_lane is not None:
                mi = g.road_pair_movement.get((g.lane_roads[before_lane], g.lane_roads[now_lane]))
                if mi is None:
                    unresolved += 1
                else:
                    movement_events[mi] += 1
            else:
                unresolved += 1
        for vid in previous.keys() - locations.keys():
            lid = previous[vid]
            li = g.lane_index.get(lid)
            if li is not None:
                lane_events[li, 1] += 1
                if g.lane_roads[li] in g.exit_roads:
                    exited += 1
                else:
                    unresolved += 1
            else:
                unresolved += 1
        # Generated-and-deleted-within-one-tick vehicles are not visible in the
        # public ID readout. Preserve this observation boundary in protocol.json.
        self.previous_locations = locations
        self.previous_all = all_ids
        return {
            "time_s": np.asarray(engine.get_current_time(), dtype=np.int32),
            "lane_n": counts16(n), "lane_q": counts16(q),
            "lane_speed_sum": velocity_sum, "lane_speed_loss": loss,
            "lane_tail": tail, "receiving_unavailable": unavailable,
            "lane_events": counts16(lane_events),
            "movement_events": counts16(movement_events),
            "intersection_nq": counts16(node_nq),
            "intersection_speed_loss": node_loss.astype(np.float32),
            "boundary_pending": counts16(pending), "boundary_generated": counts16(births),
            "boundary_admitted": counts16(admitted),
            "active_vehicles": np.asarray(len(active), dtype=np.uint32),
            "exit_count": np.asarray(exited, dtype=np.uint16),
            "unresolved_events": np.asarray(unresolved, dtype=np.uint32),
        }


class SignalRunner:
    """Execute the existing 30 s request / 5 s phase-0 switch contract."""

    def __init__(self, backend, geometry, recorder):
        self.backend, self.g, self.recorder = backend, geometry, recorder
        self.current = np.zeros(len(geometry.intersections), dtype=np.int64)
        self.elapsed = np.zeros(len(geometry.intersections), dtype=np.float64)

    def initialize(self):
        for iid in self.g.intersections:
            self.backend.set_phase(iid, 1)

    def context(self):
        return {"current_phase": self.current.tolist(), "phase_elapsed_s": self.elapsed.tolist()}

    def restore_context(self, values):
        self.current = np.asarray(values["current_phase"], dtype=np.int64)
        self.elapsed = np.asarray(values["phase_elapsed_s"], dtype=np.float64)

    def step(self, requested):
        target = np.asarray(requested, dtype=np.int64)
        if target.shape != self.current.shape or np.any((target < 0) | (target > 3)):
            raise ValueError("Expected one action index in 0..3 per intersection")
        changed = target != self.current
        phase = self.current.astype(np.uint8) + 1
        stage = np.zeros(len(target), dtype=np.uint8)
        stage_elapsed = self.elapsed.copy()
        for i in np.flatnonzero(changed):
            self.backend.set_phase(self.g.intersections[i], 0)
            phase[i] = 0; stage[i] = 1; stage_elapsed[i] = 0
        rows = []
        for tick in range(30):
            if tick == 5:
                for i in np.flatnonzero(changed):
                    self.backend.set_phase(self.g.intersections[i], int(target[i] + 1))
                    phase[i] = target[i] + 1; stage[i] = 0; stage_elapsed[i] = 0
            used_phase, used_stage = phase.copy(), stage.copy()
            used_elapsed = stage_elapsed.astype(np.float32)
            self.backend.next_step()
            row = self.recorder.observe(self.backend.engine)
            row.update(phase_used=used_phase, stage_used=used_stage, phase_elapsed_start_s=used_elapsed)
            rows.append(row)
            stage_elapsed += 1.0
        self.elapsed = np.where(changed, 25.0, self.elapsed + 30.0)
        self.current = target.copy()
        return rows


def stack_rows(rows):
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def window_view(arrays, width=5):
    """Derive exact stored-tick sums; never interpolate missing microstates."""
    length = len(arrays["time_s"])
    if length % width:
        raise ValueError("Window must divide stored tick count")

    def summed(key):
        a = arrays[key]
        dtype = np.float64 if a.dtype.kind == "f" else np.int64
        return a.reshape(length // width, width, *a.shape[1:]).sum(axis=1, dtype=dtype)

    result = {"time_s": arrays["time_s"][width - 1::width],
              "lane_n": arrays["lane_n"][width - 1::width],
              "lane_q": arrays["lane_q"][width - 1::width],
              "lane_wait_vehicle_s": summed("lane_q").sum(axis=-1),
              "intersection_wait_vehicle_s": summed("intersection_nq")[:, :, 1],
              "boundary_wait_vehicle_s": summed("boundary_pending")}
    for key in ("lane_speed_loss", "intersection_speed_loss", "lane_events",
                "movement_events", "receiving_unavailable", "exit_count", "unresolved_events"):
        result[key] = summed(key)
    for key in ("internal_trip_completion_by_lane", "internal_trip_completion_count",
                "trip_completion_count"):
        if key in arrays:  # Legacy raw data has no route-completion fields.
            result[key] = summed(key)
    return result
