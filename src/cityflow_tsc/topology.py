from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .config import ControlConfig
from .types import IntersectionSpec, MovementSpec, NetworkSpec


NEIGHBOR_DIRECTIONS = ("north", "south", "east", "west")
MOVEMENT_STATIC_FEATURE_NAMES = (
    "incoming_lane_count",
    "outgoing_lane_count",
    "incoming_mean_lane_length_m",
    "outgoing_mean_lane_length_m",
    "incoming_capacity_vehicles",
    "outgoing_capacity_vehicles",
    "incoming_mean_speed_limit_mps",
    "outgoing_mean_speed_limit_mps",
    "is_left_turn",
    "is_straight",
    "is_right_turn",
    "incoming_boundary",
    "outgoing_boundary",
)


def _point(value: Mapping[str, Any]) -> Tuple[float, float]:
    return float(value.get("x", 0.0)), float(value.get("y", 0.0))


def _road_length_m(
    road: Mapping[str, Any], intersection_by_id: Mapping[str, Mapping[str, Any]]
) -> float:
    points = road.get("points", [])
    coordinates = [_point(item) for item in points if isinstance(item, Mapping)]
    if len(coordinates) < 2:
        start = intersection_by_id.get(str(road.get("startIntersection")), {})
        end = intersection_by_id.get(str(road.get("endIntersection")), {})
        coordinates = [_point(start.get("point", {})), _point(end.get("point", {}))]
    length = sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(coordinates, coordinates[1:])
    )
    return max(float(length), 1.0)


def _road_static(
    road: Mapping[str, Any], intersection_by_id: Mapping[str, Mapping[str, Any]]
) -> Tuple[float, float, float]:
    lanes = road.get("lanes", [])
    lane_count = float(len(lanes))
    length = _road_length_m(road, intersection_by_id)
    speed_limits = [float(lane.get("maxSpeed", 0.0)) for lane in lanes]
    mean_speed = float(np.mean(speed_limits)) if speed_limits else 0.0
    capacity = lane_count * length / 7.5
    return length, capacity, mean_speed


def _natural_key(value: str) -> Tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def _lane_ids(road: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(f"{road['id']}_{index}" for index, _ in enumerate(road.get("lanes", [])))


def _movement_lanes(
    road_link: Mapping[str, Any],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    incoming = {
        f"{road_link['startRoad']}_{lane_link['startLaneIndex']}"
        for lane_link in road_link.get("laneLinks", [])
    }
    outgoing = {
        f"{road_link['endRoad']}_{lane_link['endLaneIndex']}"
        for lane_link in road_link.get("laneLinks", [])
    }
    return tuple(sorted(incoming, key=_natural_key)), tuple(sorted(outgoing, key=_natural_key))


def _neighbor_direction(
    source: Tuple[float, float], target: Tuple[float, float]
) -> str:
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    if dx == 0 and dy == 0:
        raise ValueError("connected intersections cannot share the same coordinates")
    if abs(dx) > abs(dy):
        return "east" if dx > 0 else "west"
    return "north" if dy > 0 else "south"


def load_network_spec(roadnet_path: Path, control: ControlConfig) -> NetworkSpec:
    """Parse a CityFlow roadnet into stable model-facing topology metadata."""

    path = Path(roadnet_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        roadnet = json.load(handle)

    raw_intersections = roadnet.get("intersections")
    raw_roads = roadnet.get("roads")
    if not isinstance(raw_intersections, list) or not isinstance(raw_roads, list):
        raise ValueError("roadnet must contain intersections and roads lists")

    intersection_by_id = {item["id"]: item for item in raw_intersections}
    road_by_id = {item["id"]: item for item in raw_roads}
    real_ids = sorted(
        (
            item["id"]
            for item in raw_intersections
            if not item.get("virtual", False)
        ),
        key=_natural_key,
    )
    if not real_ids:
        raise ValueError("roadnet contains no controllable intersections")

    real_index = {intersection_id: index for index, intersection_id in enumerate(real_ids)}
    specs: List[IntersectionSpec] = []

    for index, intersection_id in enumerate(real_ids):
        raw = intersection_by_id[intersection_id]
        connected_roads = [road_by_id[road_id] for road_id in raw.get("roads", [])]
        incoming_roads = [
            road for road in connected_roads if road.get("endIntersection") == intersection_id
        ]
        outgoing_roads = [
            road for road in connected_roads if road.get("startIntersection") == intersection_id
        ]
        incoming_lanes = tuple(
            lane
            for road in sorted(incoming_roads, key=lambda item: _natural_key(item["id"]))
            for lane in _lane_ids(road)
        )
        outgoing_lanes = tuple(
            lane
            for road in sorted(outgoing_roads, key=lambda item: _natural_key(item["id"]))
            for lane in _lane_ids(road)
        )

        movements: List[MovementSpec] = []
        for movement_index, road_link in enumerate(raw.get("roadLinks", [])):
            movement_in, movement_out = _movement_lanes(road_link)
            start_road = road_by_id[road_link["startRoad"]]
            end_road = road_by_id[road_link["endRoad"]]
            incoming_length, incoming_capacity, incoming_speed = _road_static(
                start_road, intersection_by_id
            )
            outgoing_length, outgoing_capacity, outgoing_speed = _road_static(
                end_road, intersection_by_id
            )
            movement_type = str(road_link.get("type", "unknown"))
            normalized_type = movement_type.lower()
            incoming_boundary = float(
                intersection_by_id[start_road["startIntersection"]].get("virtual", False)
            )
            outgoing_boundary = float(
                intersection_by_id[end_road["endIntersection"]].get("virtual", False)
            )
            movements.append(
                MovementSpec(
                    index=movement_index,
                    movement_type=movement_type,
                    start_road=road_link["startRoad"],
                    end_road=road_link["endRoad"],
                    incoming_lanes=movement_in,
                    outgoing_lanes=movement_out,
                    static_features=(
                        float(len(movement_in)),
                        float(len(movement_out)),
                        incoming_length,
                        outgoing_length,
                        incoming_capacity,
                        outgoing_capacity,
                        incoming_speed,
                        outgoing_speed,
                        float("left" in normalized_type),
                        float("straight" in normalized_type),
                        float("right" in normalized_type),
                        incoming_boundary,
                        outgoing_boundary,
                    ),
                )
            )

        light_phases = raw.get("trafficLight", {}).get("lightphases", [])
        if control.yellow_time_s and control.yellow_phase_id >= len(light_phases):
            raise ValueError(
                f"yellow phase {control.yellow_phase_id} does not exist at "
                f"{intersection_id}"
            )
        if (
            control.all_red_time_s
            and control.all_red_phase_id is not None
            and control.all_red_phase_id >= len(light_phases)
        ):
            raise ValueError(
                f"all-red phase {control.all_red_phase_id} does not exist at "
                f"{intersection_id}"
            )
        phase_ids: List[int] = []
        phase_mask = np.zeros(
            (len(control.green_phase_ids), len(movements)), dtype=np.bool_
        )
        for action_index, engine_phase_id in enumerate(control.green_phase_ids):
            if engine_phase_id < 0 or engine_phase_id >= len(light_phases):
                phase_ids.append(-1)
                continue
            phase_ids.append(engine_phase_id)
            for movement_index in light_phases[engine_phase_id].get(
                "availableRoadLinks", []
            ):
                if 0 <= movement_index < len(movements):
                    phase_mask[action_index, movement_index] = True

        if not any(phase >= 0 for phase in phase_ids):
            raise ValueError(
                f"none of the configured green phases exist at {intersection_id}"
            )

        point = raw.get("point", {})
        specs.append(
            IntersectionSpec(
                index=index,
                intersection_id=intersection_id,
                point=(float(point.get("x", 0.0)), float(point.get("y", 0.0))),
                incoming_lanes=incoming_lanes,
                outgoing_lanes=outgoing_lanes,
                movements=tuple(movements),
                engine_phase_ids=tuple(phase_ids),
                phase_movement_mask=phase_mask,
            )
        )

    neighbor_index = np.full((len(specs), 4), -1, dtype=np.int64)
    neighbor_mask = np.zeros((len(specs), 4), dtype=np.bool_)
    for spec in specs:
        raw = intersection_by_id[spec.intersection_id]
        for road_id in raw.get("roads", []):
            road = road_by_id[road_id]
            start = road.get("startIntersection")
            end = road.get("endIntersection")
            other_id = end if start == spec.intersection_id else start
            if other_id not in real_index:
                continue
            other_raw = intersection_by_id[other_id]
            other_point = other_raw.get("point", {})
            direction = _neighbor_direction(
                spec.point,
                (float(other_point.get("x", 0.0)), float(other_point.get("y", 0.0))),
            )
            direction_index = NEIGHBOR_DIRECTIONS.index(direction)
            previous = int(neighbor_index[spec.index, direction_index])
            candidate = real_index[other_id]
            if previous >= 0 and previous != candidate:
                raise ValueError(
                    f"multiple {direction} neighbors found for {spec.intersection_id}"
                )
            neighbor_index[spec.index, direction_index] = candidate
            neighbor_mask[spec.index, direction_index] = True

    max_movements = max(len(item.movements) for item in specs)
    return NetworkSpec(
        intersections=tuple(specs),
        neighbor_index=neighbor_index,
        neighbor_mask=neighbor_mask,
        max_movements=max_movements,
        max_actions=len(control.green_phase_ids),
    )
