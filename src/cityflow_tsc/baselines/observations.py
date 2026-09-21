from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..observations import QueuePressureObservationBuilder
from ..types import NetworkObservation, NetworkSnapshot, NetworkSpec
from .codecs import CANONICAL_LANES, LaneCodec
from .contracts import BaselineView
from .profiles import BaselineProfile, get_profile


class BaselineObservationBuilder:
    """Retain public movement observations and attach an algorithm-specific view."""

    def __init__(self, network: NetworkSpec, profile: str | BaselineProfile,
                 roadnet_path: Path | str | None = None) -> None:
        self.network = network
        self.profile = get_profile(profile)
        self.base = QueuePressureObservationBuilder(network)
        self.feature_names = self.base.feature_names
        self.schema_id = self.base.schema_id
        self.view_schema_id = f"baseline-{self.profile.schema_hash}"
        self._roadnet_path: Path | None = None
        self.codec: LaneCodec | None = None
        if roadnet_path is not None:
            self._configure_path(Path(roadnet_path))
        elif self.profile.layout == "generic":
            self._install_codec(self._build_codec())

    def _build_codec(self, path=None) -> LaneCodec:
        return LaneCodec.build(self.network, self.profile.layout, path,
                               phase_pairs=self.profile.algorithm in {"frap", "mplight"}
                               and self.profile.layout == "generic")

    def _install_codec(self, codec: LaneCodec) -> None:
        self.codec = codec
        self._neighbor_index, self._neighbor_mask = codec.neighbors(self.network, self.profile.graph)

    def _configure_path(self, path: Path) -> None:
        path = path.resolve()
        if path != self._roadnet_path:
            self._install_codec(self._build_codec(path))
            self._roadnet_path = path

    def configure(self, scenario) -> None:
        self._configure_path(Path(scenario.roadnet_path))

    def reset(self) -> None:
        reset_base = getattr(self.base, "reset", None)
        if callable(reset_base):
            reset_base()

    def build(self, snapshot: NetworkSnapshot, current_phase: np.ndarray,
              signal_stage: np.ndarray, phase_elapsed_s: np.ndarray) -> NetworkObservation:
        if self.codec is None:
            raise ValueError("canonical12 needs configure(scenario) or roadnet_path before building observations")
        codec = self.codec
        kind = self.profile.feature_kind
        channels = 2 if kind in {"advanced", "counts_queue"} else 1
        values = np.zeros((*codec.lane_mask.shape, channels), dtype=np.float32)
        for node, lanes in enumerate(codec.lane_ids):
            for j, lane in enumerate(lanes):
                count = float(snapshot.lane_vehicle_count.get(lane, 0))
                queue = float(snapshot.lane_waiting_count.get(lane, 0))
                target = codec.downstream_lanes[node][j]
                if kind == "vehicle_count":
                    values[node, j, 0] = count
                elif kind == "counts_queue":
                    values[node, j] = (count, queue)
                elif kind in {"general_vehicle_pressure", "general_queue_pressure", "efficient_queue_pressure", "advanced"}:
                    source = snapshot.lane_vehicle_count if kind == "general_vehicle_pressure" else snapshot.lane_waiting_count
                    incoming = count if kind == "general_vehicle_pressure" else queue
                    outgoing = sum(float(source.get(k, 0)) for k in target)
                    if kind in {"efficient_queue_pressure", "advanced"}:
                        if not target:
                            raise ValueError(f"lane {lane} has no downstream lanes for efficient pressure")
                        outgoing /= len(target)
                    values[node, j, 0] = incoming - outgoing
                    if kind == "advanced":
                        if lane not in codec.lane_lengths:
                            raise ValueError(f"missing lane length for advanced observation: {lane}")
                        if lane not in snapshot.lane_vehicles:
                            raise ValueError(f"advanced observation requires per-lane vehicle IDs for {lane}")
                        near = codec.lane_lengths[lane] - 167.0
                        running = 0
                        for vehicle in snapshot.lane_vehicles[lane]:
                            if vehicle.endswith("_shadow"):
                                vehicle = vehicle[:-7]
                            if vehicle not in snapshot.vehicle_distances or vehicle not in snapshot.vehicle_speeds:
                                raise ValueError(f"advanced observation lacks position/speed for {vehicle}")
                            running += snapshot.vehicle_distances[vehicle] >= near and snapshot.vehicle_speeds[vehicle] > 0.1
                        values[node, j, 1] = running
                else:
                    raise ValueError(f"unknown baseline feature kind {kind!r}")
        phases = np.asarray(current_phase)
        if phases.shape != (self.network.num_intersections,) or not np.issubdtype(phases.dtype, np.integer):
            raise ValueError("current_phase must contain one local integer action per intersection")
        encodings = np.zeros((self.network.num_intersections, codec.phase_encodings.shape[-1]), dtype=np.float32)
        for i, action in enumerate(phases):
            if action < 0 or action >= self.network.max_actions or not self.network.action_mask()[i, action]:
                raise ValueError(f"invalid current local action {action} at node {i}")
            if signal_stage[i] == 0:
                encodings[i] = codec.phase_encodings[i, action]
        # Channel blocks match source input: phase8, pressure12, running12.
        features = np.concatenate([encodings] + [values[..., c] for c in range(channels)], axis=1)
        neighbors, neighbor_mask = self._neighbor_index.copy(), self._neighbor_mask.copy()
        lane_names = CANONICAL_LANES if self.profile.layout == "canonical12" else tuple(f"lane_{i}" for i in range(codec.max_lanes))
        channel_names = ("pressure", "running_167m") if kind == "advanced" else (("vehicle_count", "queue_count") if kind == "counts_queue" else (kind,))
        names = tuple(f"phase_{i}" for i in range(encodings.shape[1])) + tuple(f"{channel}:{lane}" for channel in channel_names for lane in lane_names)
        view = BaselineView(features=features, lane_features=values, lane_mask=codec.lane_mask.copy(),
                            phase_lane_mask=codec.phase_lane_mask.copy(), phase_encoding=encodings,
                            neighbor_index=neighbors, neighbor_mask=neighbor_mask,
                            schema_id=self.view_schema_id, node_ids=self.network.intersection_ids,
                            feature_names=names, source_action_to_local=codec.source_action_to_local.copy())
        return replace(self.base.build(snapshot, current_phase, signal_stage, phase_elapsed_s), baseline_view=view)
