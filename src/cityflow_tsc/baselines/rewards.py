from __future__ import annotations

import numpy as np

from ..types import NetworkSnapshot, NetworkSpec
from .profiles import BaselineProfile, get_profile


class BaselineReward:
    def __init__(self, network: NetworkSpec, profile: str | BaselineProfile) -> None:
        self.network = network
        self.profile = get_profile(profile)
        self.name = f"{self.profile.reward_kind}:{self.profile.reward_factor}:local"
        if self.profile.reward_kind not in {"queue", "queue_mean", "absolute_queue_pressure"}:
            raise ValueError(f"unknown baseline reward {self.profile.reward_kind!r}")
        self._reward_component = {
            "queue": "incoming_queue",
            "queue_mean": "queue_mean_road",
            "absolute_queue_pressure": "absolute_queue_pressure",
        }[self.profile.reward_kind]
        self._incoming_roads = []
        for inter in network.intersections:
            by_road: dict[str, list[str]] = {}
            for lane in inter.incoming_lanes:
                # CityFlow lane IDs are road_id + "_" + lane_index. Split only
                # the last separator: road IDs themselves can contain underscores.
                road_id, separator, lane_index = lane.rpartition("_")
                if not separator or not road_id or not lane_index.isdigit():
                    raise ValueError(f"invalid CityFlow lane identifier: {lane!r}")
                by_road.setdefault(road_id, []).append(lane)
            self._incoming_roads.append(tuple(tuple(lanes) for lanes in by_road.values()))

    def components(self, snapshot: NetworkSnapshot) -> dict[str, np.ndarray]:
        """Unscaled, local reward quantities, before decision-window reduction."""
        result = {name: np.zeros(self.network.num_intersections, dtype=np.float32)
                  for name in ("incoming_queue", "outgoing_queue",
                               "absolute_queue_pressure", "queue_mean_road")}
        for inter in self.network.intersections:
            incoming = sum(float(snapshot.lane_waiting_count.get(lane, 0)) for lane in inter.incoming_lanes)
            outgoing = sum(float(snapshot.lane_waiting_count.get(lane, 0)) for lane in inter.outgoing_lanes)
            result["incoming_queue"][inter.index] = incoming
            result["outgoing_queue"][inter.index] = outgoing
            result["absolute_queue_pressure"][inter.index] = abs(incoming - outgoing)
            # LibSignal LaneVehicleGenerator(average="all") averages
            # within each road first, then gives each incoming road equal weight.
            roads = self._incoming_roads[inter.index]
            if roads:
                result["queue_mean_road"][inter.index] = sum(
                    sum(float(snapshot.lane_waiting_count.get(lane, 0)) for lane in lanes) / len(lanes)
                    for lanes in roads
                ) / len(roads)
            elif self.profile.reward_kind == "queue_mean":
                raise ValueError("queue_mean requires at least one incoming lane")
        return result

    def compute(self, snapshot: NetworkSnapshot) -> np.ndarray:
        return self.components(snapshot)[self._reward_component] * self.profile.reward_factor


class RewardAccumulator:
    """Decision-window reward: means are time weighted; sum counts post-tick samples."""

    def __init__(self, reward: BaselineReward, window: str | None = None) -> None:
        self.reward = reward
        self.window = window or reward.profile.reward_window
        if self.window not in {"last", "pre_mean", "post_mean", "sum", "integral"}:
            raise ValueError(f"unknown reward window {self.window!r}")
        self.reset()

    def reset(self) -> None:
        self._component_totals: dict[str, np.ndarray] = {}
        self.last_components: dict[str, np.ndarray] = {}
        self._elapsed = 0.0
        self._ticks = 0

    def begin(self) -> None:
        self.reset()

    def observe_tick(self, before: NetworkSnapshot, after: NetworkSnapshot, elapsed_s: float) -> None:
        if not np.isfinite(elapsed_s) or elapsed_s <= 0:
            raise ValueError("reward tick elapsed_s must be finite and positive")
        components = self.reward.components(before if self.window == "pre_mean" else after)
        weight = 1.0 if self.window == "sum" else elapsed_s
        for name, sample in components.items():
            if name not in self._component_totals:
                self._component_totals[name] = np.zeros_like(sample, dtype=np.float64)
            self._component_totals[name] += sample.astype(np.float64) * weight
        self._elapsed += elapsed_s
        self._ticks += 1

    def finish(self, final_snapshot: NetworkSnapshot) -> np.ndarray:
        if self.window == "last" or not self._ticks:
            self.last_components = self.reward.components(final_snapshot)
        else:
            divisor = self._elapsed if self.window in {"pre_mean", "post_mean"} else 1.0
            self.last_components = {name: (total / divisor).astype(np.float32)
                                    for name, total in self._component_totals.items()}
        return self.last_components[self.reward._reward_component] * self.reward.profile.reward_factor
