from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .config import ControlConfig, ScenarioConfig
from .topology import MOVEMENT_STATIC_FEATURE_NAMES
from .types import NetworkObservation, NetworkSpec


SCHEMA_VERSION = "cityflow-tsc-trajectory-v1"
DEMAND_FEATURE_NAMES = (
    "flow_definition_count",
    "estimated_vehicle_count",
    "mean_generation_interval_s",
    "first_departure_time_s",
    "last_departure_time_s",
)


def demand_features(flow_path: Path) -> np.ndarray:
    with Path(flow_path).open("r", encoding="utf-8") as handle:
        flows = json.load(handle)
    if not isinstance(flows, list):
        raise ValueError("CityFlow flow file must contain a list")
    estimated = 0.0
    intervals = []
    starts = []
    ends = []
    for flow in flows:
        if not isinstance(flow, Mapping):
            continue
        start = float(flow.get("startTime", 0.0))
        end = float(flow.get("endTime", start))
        interval = float(flow.get("interval", 1.0))
        starts.append(start)
        ends.append(end)
        if interval > 0:
            intervals.append(interval)
            estimated += max(0.0, np.floor((end - start) / interval) + 1.0)
    return np.asarray(
        [
            float(len(flows)),
            estimated,
            float(np.mean(intervals)) if intervals else 0.0,
            float(min(starts)) if starts else 0.0,
            float(max(ends)) if ends else 0.0,
        ],
        dtype=np.float32,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TrajectoryWriter:
    """Collect one episode and persist a numeric, mask-preserving trajectory."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        control: ControlConfig,
        network: NetworkSpec,
        policy_name: str,
        reward_name: str,
        policy_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.scenario = scenario
        self.control = control
        self.network = network
        self.policy_name = policy_name
        self.reward_name = reward_name
        self.policy_metadata = dict(policy_metadata or {})
        self._records: List[Dict[str, Any]] = []

    def append(
        self,
        observation: NetworkObservation,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observation: NetworkObservation,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._records.append(
            {
                "observation": observation,
                "actions": np.asarray(actions, dtype=np.int64).copy(),
                "rewards": np.asarray(rewards, dtype=np.float32).copy(),
                "next_observation": next_observation,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    @staticmethod
    def _observation_arrays(prefix: str, observations: List[NetworkObservation]):
        return {
            f"{prefix}_features": np.stack([item.features for item in observations]),
            f"{prefix}_movement_mask": np.stack(
                [item.movement_mask for item in observations]
            ),
            f"{prefix}_valid_mask": np.stack([item.valid_mask for item in observations]),
            f"{prefix}_current_phase": np.stack(
                [item.current_phase for item in observations]
            ),
            f"{prefix}_signal_stage": np.stack(
                [item.signal_stage for item in observations]
            ),
            f"{prefix}_phase_elapsed_s": np.stack(
                [item.phase_elapsed_s for item in observations]
            ),
            f"{prefix}_action_mask": np.stack(
                [item.action_mask for item in observations]
            ),
            f"{prefix}_time_s": np.asarray(
                [item.time_s for item in observations], dtype=np.float32
            ),
        }

    def write(self, output_dir: Path) -> Dict[str, str]:
        if not self._records:
            raise ValueError("cannot write an empty trajectory")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        trajectory_path = output / "trajectory.npz"
        manifest_path = output / "trajectory.manifest.json"

        observations = [item["observation"] for item in self._records]
        next_observations = [item["next_observation"] for item in self._records]
        arrays = self._observation_arrays("observation", observations)
        arrays.update(self._observation_arrays("next_observation", next_observations))
        arrays.update(
            {
                "actions": np.stack([item["actions"] for item in self._records]),
                "rewards": np.stack([item["rewards"] for item in self._records]),
                "terminated": np.asarray(
                    [item["terminated"] for item in self._records], dtype=np.bool_
                ),
                "truncated": np.asarray(
                    [item["truncated"] for item in self._records], dtype=np.bool_
                ),
                "neighbor_index": self.network.neighbor_index,
                "neighbor_mask": self.network.neighbor_mask,
                "phase_movement_mask": self.network.padded_phase_movement_mask(),
                "movement_static_features": self.network.padded_movement_static_features(),
                "demand_features": demand_features(self.scenario.flow_path),
            }
        )

        temporary_npz = output / ".trajectory.npz.tmp"
        with temporary_npz.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_npz, trajectory_path)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "policy": self.policy_name,
            "policy_metadata": self.policy_metadata,
            "reward": self.reward_name,
            "feature_names": list(observations[0].feature_names),
            "movement_static_feature_names": list(MOVEMENT_STATIC_FEATURE_NAMES),
            "demand_feature_names": list(DEMAND_FEATURE_NAMES),
            "steps": len(self._records),
            "num_intersections": self.network.num_intersections,
            "max_movements": self.network.max_movements,
            "max_actions": self.network.max_actions,
            "intersection_ids": list(self.network.intersection_ids),
            "scenario": self.scenario.to_dict(),
            "control": self.control.to_dict(),
            "roadnet_sha256": sha256_file(self.scenario.roadnet_path),
            "flow_sha256": sha256_file(self.scenario.flow_path),
            "trajectory_sha256": sha256_file(trajectory_path),
        }
        temporary_manifest = output / ".trajectory.manifest.json.tmp"
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        os.replace(temporary_manifest, manifest_path)
        return {
            "trajectory_path": str(trajectory_path),
            "manifest_path": str(manifest_path),
        }


class TrajectoryReader:
    """Load and validate the public trajectory contract without pickle."""

    def load(self, manifest_path: Path) -> Dict[str, Any]:
        manifest_file = Path(manifest_path)
        with manifest_file.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported trajectory schema: {manifest.get('schema_version')!r}"
            )
        trajectory_path = manifest_file.parent / "trajectory.npz"
        if sha256_file(trajectory_path) != manifest.get("trajectory_sha256"):
            raise ValueError("trajectory checksum does not match manifest")
        with np.load(trajectory_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        steps = int(manifest["steps"])
        for key in ("actions", "rewards", "observation_features", "next_observation_features"):
            if arrays[key].shape[0] != steps:
                raise ValueError(f"{key} has inconsistent step dimension")
        if arrays["actions"].shape[1] != int(manifest["num_intersections"]):
            raise ValueError("actions have inconsistent intersection dimension")
        return {"manifest": manifest, "arrays": arrays}
