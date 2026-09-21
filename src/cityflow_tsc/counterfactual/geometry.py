from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..topology import _natural_key


class Geometry:
    """Physical lanes and laneLinks for the pinned straight-road CityFlow maps."""

    def __init__(self, path: Path):
        raw = json.loads(Path(path).read_text())
        nodes = {n["id"]: n for n in raw["intersections"]}
        self.intersections = sorted(
            (k for k, v in nodes.items() if not v.get("virtual", False)), key=_natural_key
        )
        self.node_index = {k: i for i, k in enumerate(self.intersections)}
        self.roads = {r["id"]: r for r in raw["roads"]}
        self.lanes = []
        self.lengths = []
        self.speeds = []
        self.lane_roads = []
        self.boundaries = sorted(
            (k for k, r in self.roads.items()
             if nodes[r["startIntersection"]].get("virtual", False)), key=_natural_key
        )
        self.boundary_index = {k: i for i, k in enumerate(self.boundaries)}
        self.exit_roads = {k for k, r in self.roads.items()
                           if nodes[r["endIntersection"]].get("virtual", False)}
        for rid in sorted(self.roads, key=_natural_key):
            road = self.roads[rid]
            pts = road["points"]
            if len(pts) != 2:
                raise ValueError("Physical collector currently supports straight roads only")
            length = math.hypot(pts[1]["x"] - pts[0]["x"], pts[1]["y"] - pts[0]["y"])
            for end in ("startIntersection", "endIntersection"):
                node = nodes[road[end]]
                if not node.get("virtual", False):
                    length -= float(node["width"])
            if length <= 0:
                raise ValueError(f"Nonpositive drivable length: {rid}")
            for j, lane in enumerate(road["lanes"]):
                self.lanes.append(f"{rid}_{j}")
                self.lengths.append(length)
                self.speeds.append(float(lane["maxSpeed"]))
                self.lane_roads.append(rid)
        self.lane_index = {k: i for i, k in enumerate(self.lanes)}
        self.lengths = np.asarray(self.lengths, dtype=np.float64)
        self.speeds = np.asarray(self.speeds, dtype=np.float64)
        self.ends = np.minimum(100.0, self.lengths / 3.0)
        self.links = {}
        self.movements = []
        self.road_pair_movement = {}
        self.movement_node = []
        self.phase_masks = []
        self.adjacency = [set() for _ in self.intersections]
        for ni, node_id in enumerate(self.intersections):
            node = nodes[node_id]
            if len(node["trafficLight"]["lightphases"]) < 5:
                raise ValueError("Expected phases 0 through 4")
            local_masks = []
            for phase in node["trafficLight"]["lightphases"][:5]:
                local_masks.append(phase["availableRoadLinks"])
            self.phase_masks.append(local_masks)
            for ri, movement in enumerate(node["roadLinks"]):
                mi = len(self.movements)
                self.movements.append(f"{node_id}:roadLink:{ri}")
                self.movement_node.append(ni)
                self.road_pair_movement[(movement["startRoad"], movement["endRoad"])] = mi
                for link in movement["laneLinks"]:
                    src = self.lane_index[f'{movement["startRoad"]}_{link["startLaneIndex"]}']
                    dst = self.lane_index[f'{movement["endRoad"]}_{link["endLaneIndex"]}']
                    lid = self.lanes[src] + "_TO_" + self.lanes[dst]
                    self.links[lid] = (mi, ni, src, dst)
        for r in self.roads.values():
            a, b = r["startIntersection"], r["endIntersection"]
            if a in self.node_index and b in self.node_index:
                i, j = self.node_index[a], self.node_index[b]
                self.adjacency[i].add(j)
                self.adjacency[j].add(i)

    def to_dict(self):
        return {
            "intersection_ids": self.intersections, "lane_ids": self.lanes,
            "lane_road_ids": self.lane_roads, "lane_lengths_m": self.lengths.tolist(),
            "lane_speed_limits_mps": self.speeds.tolist(),
            "spatial_end_m": self.ends.tolist(), "boundary_road_ids": self.boundaries,
            "movement_ids": self.movements, "movement_node": self.movement_node,
            "lanelinks": {k: list(v) for k, v in self.links.items()},
            "phase_available_local_roadlinks_0_to_4": self.phase_masks,
            "adjacency": [sorted(x) for x in self.adjacency],
        }
