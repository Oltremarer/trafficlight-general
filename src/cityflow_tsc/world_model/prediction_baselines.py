"""Non-neural prediction references for rich CityFlow trajectories.

These are deliberately small and auditable references.  They are not a second
simulator: the action-conditioned baseline only applies a local conservation
update to each movement and cannot reproduce CityFlow's routing or car-following
rules.  Its purpose is to tell us whether a learned model beats an explicit,
action-aware traffic prior.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from ..observations import RichTrafficObservationBuilder
from ..topology import MOVEMENT_STATIC_FEATURE_NAMES
from .rich_data import RichFeatureStatistics


def _feature_index(names: Sequence[str], name: str) -> int:
    try:
        return tuple(names).index(name)
    except ValueError as exc:
        raise ValueError(f"rich feature schema is missing {name!r}") from exc


class PersistencePredictionBaseline(torch.nn.Module):
    """Repeat the latest observed movement state for every future step."""

    model_kind = "persistence"

    def rollout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        horizon = batch["actions"].shape[1]
        latest = batch["history_features"][:, -1]
        return {"features": latest[:, None].expand(-1, horizon, -1, -1, -1)}


class ActionConditionedFlowBalanceBaseline(torch.nn.Module):
    """Roll out a local action-conditioned, flow-balance approximation.

    The model persists measured arrivals, estimates departure capacity only for
    movements served by the selected signal phase, and lets downstream vehicles
    leave according to observed speed and road length.  It intentionally does
    not learn from validation data.
    """

    model_kind = "action_conditioned_flow_balance"

    def __init__(
        self, statistics: RichFeatureStatistics, feature_names: Sequence[str]
    ) -> None:
        super().__init__()
        self.statistics = statistics
        names = tuple(feature_names)
        if names != RichTrafficObservationBuilder.feature_names:
            raise ValueError("flow-balance baseline requires rich-traffic-movement-v1")
        self._indices = {
            name: _feature_index(names, name)
            for name in (
                "incoming_vehicle_count",
                "incoming_queue_count",
                "outgoing_vehicle_count",
                "outgoing_queue_count",
                "outgoing_mean_speed_mps",
                "incoming_stopped_ratio",
                "incoming_near_stopline_50m_count",
                "incoming_near_stopline_100m_count",
                "recent_arrival_rate_veh_s",
                "recent_departure_rate_veh_s",
                "incoming_occupancy_ratio",
                "downstream_occupancy_ratio",
                "vehicle_pressure",
                "queue_pressure",
            )
        }
        self._static_indices = {
            name: _feature_index(MOVEMENT_STATIC_FEATURE_NAMES, name)
            for name in (
                "incoming_capacity_vehicles",
                "outgoing_capacity_vehicles",
                "outgoing_mean_lane_length_m",
            )
        }

    @staticmethod
    def _served_mask(batch: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        phase_mask = batch["phase_movement_mask"]
        actions = batch["actions"][:, step]
        movement_count = phase_mask.shape[-1]
        return torch.gather(
            phase_mask,
            2,
            actions[..., None, None].expand(-1, -1, 1, movement_count),
        ).squeeze(2)

    def rollout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw = self.statistics.denormalize_features_tensor(batch["history_features"][:, -1])
        static = batch["movement_static_features"]
        # Static roads were normalized independently; inverse-transform them
        # inline because this baseline only needs the three physical columns.
        static_mean = torch.as_tensor(
            self.statistics.static_mean, dtype=raw.dtype, device=raw.device
        )
        static_std = torch.as_tensor(
            self.statistics.static_std, dtype=raw.dtype, device=raw.device
        )
        restored = static * static_std + static_mean
        static_raw = torch.sign(restored) * torch.expm1(torch.abs(restored))
        valid = batch["history_movement_mask"][:, -1]
        previous_time = batch["history_time_s"][:, -1]
        predictions = []
        idx = self._indices
        static_idx = self._static_indices

        for step in range(batch["actions"].shape[1]):
            dt = (batch["target_time_s"][:, step] - previous_time).clamp(min=1.0)
            dt = dt[:, None, None]
            served = self._served_mask(batch, step).to(raw.dtype) * valid.to(raw.dtype)
            incoming = raw[..., idx["incoming_vehicle_count"]].clamp(min=0.0)
            incoming_queue = raw[..., idx["incoming_queue_count"]].clamp(min=0.0)
            outgoing = raw[..., idx["outgoing_vehicle_count"]].clamp(min=0.0)
            outgoing_queue = raw[..., idx["outgoing_queue_count"]].clamp(min=0.0)
            arrivals = raw[..., idx["recent_arrival_rate_veh_s"]].clamp(min=0.0)
            measured_departures = raw[..., idx["recent_departure_rate_veh_s"]].clamp(min=0.0)
            speed = raw[..., idx["outgoing_mean_speed_mps"]].clamp(min=0.0)
            out_length = static_raw[..., static_idx["outgoing_mean_lane_length_m"]].clamp(min=1.0)
            in_capacity = static_raw[..., static_idx["incoming_capacity_vehicles"]].clamp(min=1.0)
            out_capacity = static_raw[..., static_idx["outgoing_capacity_vehicles"]].clamp(min=1.0)

            discharge = served * torch.minimum(measured_departures, incoming / dt)
            downstream_exit = torch.minimum(outgoing / dt, outgoing * speed / out_length)
            next_incoming = (incoming + dt * (arrivals - discharge)).clamp(min=0.0)
            next_outgoing = (outgoing + dt * (discharge - downstream_exit)).clamp(min=0.0)
            next_incoming_queue = torch.minimum(
                next_incoming,
                (incoming_queue + dt * (arrivals - discharge)).clamp(min=0.0),
            )
            next_outgoing_queue = torch.minimum(
                next_outgoing,
                (outgoing_queue + dt * (discharge - downstream_exit)).clamp(min=0.0),
            )

            raw = raw.clone()
            raw[..., idx["incoming_vehicle_count"]] = next_incoming
            raw[..., idx["incoming_queue_count"]] = next_incoming_queue
            raw[..., idx["outgoing_vehicle_count"]] = next_outgoing
            raw[..., idx["outgoing_queue_count"]] = next_outgoing_queue
            raw[..., idx["incoming_stopped_ratio"]] = (
                next_incoming_queue / next_incoming.clamp(min=1.0)
            ).clamp(0.0, 1.0)
            raw[..., idx["incoming_near_stopline_50m_count"]] = torch.minimum(
                raw[..., idx["incoming_near_stopline_50m_count"]].clamp(min=0.0),
                next_incoming,
            )
            raw[..., idx["incoming_near_stopline_100m_count"]] = torch.minimum(
                raw[..., idx["incoming_near_stopline_100m_count"]].clamp(min=0.0),
                next_incoming,
            )
            raw[..., idx["recent_departure_rate_veh_s"]] = discharge
            raw[..., idx["incoming_occupancy_ratio"]] = (
                next_incoming / in_capacity
            ).clamp(0.0, 1.0)
            raw[..., idx["downstream_occupancy_ratio"]] = (
                next_outgoing / out_capacity
            ).clamp(0.0, 1.0)
            raw[..., idx["vehicle_pressure"]] = next_incoming - next_outgoing
            raw[..., idx["queue_pressure"]] = next_incoming_queue - next_outgoing_queue
            predictions.append(self.statistics.normalize_features_tensor(raw))
            previous_time = batch["target_time_s"][:, step]
        return {"features": torch.stack(predictions, dim=1)}
