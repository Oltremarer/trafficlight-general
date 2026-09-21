"""Static lane, phase and graph mappings for explicit baseline profiles.

Canonical order follows LLMTSCS d5d4180 utils/cityflow_env.py: W/E/N/S,
left/through/right. Lane indices themselves are not assumed to encode turns.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..types import NetworkSpec


CANONICAL_LANES = tuple(f"{side}{turn}" for side in ("W", "E", "N", "S") for turn in ("L", "T", "R"))
PHASE_LANE_INDICES = (0, 1, 3, 4, 6, 7, 9, 10)
SOURCE_PHASES = np.asarray([
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 0],
], dtype=np.bool_)


def _turn(value: str) -> str:
    value = value.lower()
    if "left" in value:
        return "L"
    if "straight" in value or value == "through":
        return "T"
    if "right" in value:
        return "R"
    raise ValueError(f"canonical12 does not support movement type {value!r}")


def _lane_ids(road: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{road['id']}_{i}" for i in range(len(road.get("lanes", []))))


def _road_points(road: dict[str, Any], nodes: dict[str, Any]) -> list[tuple[float, float]]:
    points = road.get("points", [])
    if len(points) < 2:
        points = [nodes[road[key]]["point"] for key in ("startIntersection", "endIntersection")]
    return [(float(p["x"]), float(p["y"])) for p in points]


def _incoming_side(road: dict[str, Any], nodes: dict[str, Any]) -> str:
    points = _road_points(road, nodes)
    end = points[-1]
    start = next((p for p in reversed(points[:-1]) if p != end), None)
    if start is None:
        raise ValueError(f"road {road['id']} has no incoming direction")
    dx, dy = start[0] - end[0], start[1] - end[1]
    if math.isclose(abs(dx), abs(dy), rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"road {road['id']} has ambiguous cardinal direction")
    if abs(dx) > abs(dy):
        return "W" if dx < 0 else "E"
    return "S" if dy < 0 else "N"


@dataclass(frozen=True)
class LaneCodec:
    lane_ids: tuple[tuple[str, ...], ...]
    downstream_lanes: tuple[tuple[tuple[str, ...], ...], ...]
    lane_lengths: dict[str, float]
    lane_mask: np.ndarray
    phase_lane_mask: np.ndarray
    phase_encodings: np.ndarray
    source_action_to_local: np.ndarray
    road_neighbors: tuple[tuple[int, ...], ...]
    layout: str

    @property
    def max_lanes(self) -> int:
        return self.lane_mask.shape[1]

    @classmethod
    def build(cls, network: NetworkSpec, layout: str, roadnet_path: Path | str | None = None,
              *, phase_pairs: bool = False) -> "LaneCodec":
        if layout not in {"canonical12", "generic"}:
            raise ValueError(f"unknown lane layout {layout!r}")
        raw: dict[str, Any] | None = None
        roads: dict[str, Any] = {}
        nodes: dict[str, Any] = {}
        lengths: dict[str, float] = {}
        if roadnet_path is not None:
            raw = json.loads(Path(roadnet_path).read_text(encoding="utf-8"))
            roads = {str(r["id"]): r for r in raw["roads"]}
            nodes = {str(n["id"]): n for n in raw["intersections"]}
            for road in roads.values():
                points = _road_points(road, nodes)
                length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
                lengths.update({lane: length for lane in _lane_ids(road)})
        if layout == "canonical12" and raw is None:
            raise ValueError("canonical12 requires roadnet_path; call configure(scenario) before build")

        ordered: list[tuple[str, ...]] = []
        downstream: list[tuple[tuple[str, ...], ...]] = []
        for inter in network.intersections:
            if layout == "canonical12":
                incoming = [r for r in roads.values() if r["endIntersection"] == inter.intersection_id]
                outgoing = [r for r in roads.values() if r["startIntersection"] == inter.intersection_id]
                if len(incoming) != 4 or len(outgoing) != 4 or any(len(r.get("lanes", [])) != 3 for r in incoming + outgoing):
                    raise ValueError(f"canonical12 requires four incoming/outgoing three-lane roads at {inter.intersection_id}")
                by_side = {_incoming_side(r, nodes): r for r in incoming}
                if set(by_side) != {"W", "E", "N", "S"}:
                    raise ValueError(f"canonical12 requires distinct W/E/N/S approaches at {inter.intersection_id}")
                lane_order = []
                for side in ("W", "E", "N", "S"):
                    road = by_side[side]
                    lane_turns: dict[str, set[str]] = {lane: set() for lane in _lane_ids(road)}
                    for movement in inter.movements:
                        if movement.start_road == road["id"]:
                            turn = _turn(movement.movement_type)
                            for lane in movement.incoming_lanes:
                                if lane not in lane_turns:
                                    raise ValueError(f"invalid incoming lane {lane}")
                                lane_turns[lane].add(turn)
                    if any(len(turns) != 1 for turns in lane_turns.values()):
                        raise ValueError(f"canonical12 requires dedicated L/T/R lanes on {road['id']}")
                    for turn in ("L", "T", "R"):
                        matches = [lane for lane, turns in lane_turns.items() if turns == {turn}]
                        if len(matches) != 1:
                            raise ValueError(f"canonical12 requires one {turn} lane on {road['id']}")
                        lane_order.append(matches[0])
                lanes = tuple(lane_order)
                if set(lanes) != set(inter.incoming_lanes):
                    raise ValueError(f"roadnet and NetworkSpec incoming lanes differ at {inter.intersection_id}")
            else:
                lanes = tuple(inter.incoming_lanes)
            if not lanes:
                raise ValueError(f"baseline requires incoming lanes at {inter.intersection_id}")
            ordered.append(lanes)
            targets = []
            for lane in lanes:
                target_roads = {m.end_road for m in inter.movements if lane in m.incoming_lanes}
                if layout == "canonical12" and len(target_roads) != 1:
                    raise ValueError(f"canonical12 needs one destination road for lane {lane}")
                if roads:
                    target_lanes = tuple(dict.fromkeys(l for rid in sorted(target_roads) for l in _lane_ids(roads[rid])))
                else:
                    target_lanes = tuple(dict.fromkeys(l for m in inter.movements if lane in m.incoming_lanes for l in m.outgoing_lanes))
                    # Length fallback is sufficient for generic count-only views.
                    for m in inter.movements:
                        if lane in m.incoming_lanes and len(m.static_features) > 2:
                            lengths.setdefault(lane, float(m.static_features[2]))
                targets.append(target_lanes)
            downstream.append(tuple(targets))

        max_lanes = max(map(len, ordered))
        lane_mask = np.zeros((network.num_intersections, max_lanes), dtype=np.bool_)
        phase_lane_mask = np.zeros((network.num_intersections, network.max_actions, max_lanes), dtype=np.bool_)
        for inter, lanes in zip(network.intersections, ordered):
            lane_mask[inter.index, :len(lanes)] = True
            lane_index = {lane: i for i, lane in enumerate(lanes)}
            for action, phase_movements in enumerate(inter.phase_movement_mask):
                if not inter.action_mask[action]:
                    continue
                for m in inter.movements:
                    if phase_pairs and _turn(m.movement_type) == "R":
                        continue
                    if phase_movements[m.index]:
                        for lane in m.incoming_lanes:
                            phase_lane_mask[inter.index, action, lane_index[lane]] = True
            if phase_pairs:
                pairs = phase_lane_mask[inter.index, inter.action_mask]
                if np.any(pairs.sum(axis=-1) != 2):
                    raise ValueError(f"LibSignal phase_pairs require exactly two dedicated L/T demand lanes "
                                     f"per green phase at {inter.intersection_id}")
                for j in np.flatnonzero(pairs.any(axis=0)):
                    turns = {_turn(m.movement_type) for m in inter.movements if lanes[j] in m.incoming_lanes}
                    if len(turns) != 1 or not turns <= {"L", "T"}:
                        raise ValueError(f"LibSignal phase_pairs require dedicated L/T demand lanes: {lanes[j]}")
                if len(np.unique(pairs, axis=0)) != len(pairs):
                    raise ValueError(f"LibSignal phase_pairs contain duplicate demand pairs at {inter.intersection_id}")
        source_to_local = np.full((network.num_intersections, len(SOURCE_PHASES)), -1, dtype=np.int64)
        if layout == "canonical12":
            phase_encodings = phase_lane_mask[:, :, PHASE_LANE_INDICES].copy()
            for inter in network.intersections:
                for action in np.flatnonzero(inter.action_mask):
                    matches = np.flatnonzero(np.all(SOURCE_PHASES == phase_encodings[inter.index, action], axis=1))
                    if len(matches) != 1:
                        raise ValueError(f"canonical12 phase at {inter.intersection_id} action {action} is not a supported source phase")
                    source_action = int(matches[0])
                    if source_to_local[inter.index, source_action] >= 0:
                        raise ValueError(f"canonical12 has duplicate phase encoding at {inter.intersection_id}")
                    source_to_local[inter.index, source_action] = action
        else:
            phase_encodings = np.broadcast_to(np.eye(network.max_actions, dtype=np.bool_),
                                              (network.num_intersections, network.max_actions, network.max_actions)).copy()
            phase_encodings &= network.action_mask()[..., None]
            source_to_local = np.broadcast_to(np.arange(network.max_actions),
                                              (network.num_intersections, network.max_actions)).copy()
            source_to_local[~network.action_mask()] = -1

        ids = {inter.intersection_id: inter.index for inter in network.intersections}
        road_neighbors: list[set[int]] = [{i} for i in range(network.num_intersections)]
        if roads:
            for road in roads.values():
                start, end = road["startIntersection"], road["endIntersection"]
                if start in ids and end in ids:
                    road_neighbors[ids[end]].add(ids[start])
        else:
            # Directed source -> target edges can be recovered from shared road IDs.
            starts = {m.end_road: inter.index for inter in network.intersections for m in inter.movements}
            for inter in network.intersections:
                for m in inter.movements:
                    if m.start_road in starts:
                        road_neighbors[inter.index].add(starts[m.start_road])
        return cls(tuple(ordered), tuple(downstream), lengths, lane_mask, phase_lane_mask,
                   phase_encodings, source_to_local,
                   tuple((i,) + tuple(sorted(neighbors - {i})) for i, neighbors in enumerate(road_neighbors)), layout)

    def neighbors(self, network: NetworkSpec, mode: str, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if mode == "knn":
            if top_k < 1:
                raise ValueError("top_k must be positive")
            points = np.asarray([inter.point for inter in network.intersections])
            rows = []
            for i in range(network.num_intersections):
                distance = np.square(points - points[i]).sum(axis=1)
                others = sorted((j for j in range(len(points)) if j != i), key=lambda j: (distance[j], j))
                rows.append((i,) + tuple(others[:top_k - 1]))
        elif mode == "road":
            rows = list(self.road_neighbors)
        elif mode == "none":
            rows = [(i,) for i in range(network.num_intersections)]
        else:
            raise ValueError(f"unknown graph {mode!r}")
        width = max(map(len, rows))
        index = np.full((network.num_intersections, width), -1, dtype=np.int64)
        mask = np.zeros_like(index, dtype=np.bool_)
        for i, row in enumerate(rows):
            index[i, :len(row)] = row
            mask[i, :len(row)] = True
        return index, mask
