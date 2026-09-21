from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..trajectory import TrajectoryWriter, sha256_file
from .contracts import require_view


SCHEMA_VERSION = "rl-trafficlight-baseline-trajectory-v2"


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=json_value, allow_nan=False)
    os.replace(temporary, path)


class BaselineTrajectoryWriter(TrajectoryWriter):
    """Explicit v2 writer; does not change the legacy writer/reader contract."""

    def append(self, observation, actions, rewards, next_observation, terminated, truncated,
               *, output=None, info=None) -> None:
        require_view(observation)
        require_view(next_observation)
        super().append(observation, actions, rewards, next_observation, terminated, truncated)
        self._records[-1]["output"] = output
        self._records[-1]["info"] = info or {}

    def write(self, output_dir: Path) -> dict[str, str]:
        if not self._records:
            raise ValueError("cannot write an empty baseline trajectory")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        observations = [r["observation"] for r in self._records]
        next_observations = [r["next_observation"] for r in self._records]
        arrays = self._observation_arrays("observation", observations)
        arrays.update(self._observation_arrays("next_observation", next_observations))
        first = require_view(observations[0])
        for prefix, values in (("observation", observations), ("next_observation", next_observations)):
            views = [require_view(obs) for obs in values]
            if any(v.schema_id != first.schema_id or v.node_ids != first.node_ids for v in views):
                raise ValueError("baseline schema/node order changed inside an episode")
            for name in ("features", "lane_features", "lane_mask", "phase_lane_mask",
                         "phase_encoding", "neighbor_index", "neighbor_mask"):
                arrays[f"{prefix}_baseline_{name}"] = np.stack([getattr(v, name) for v in views])
            mappings = [v.source_action_to_local for v in views]
            if any(mapping is not None for mapping in mappings):
                if not all(mapping is not None for mapping in mappings):
                    raise ValueError("source action mapping missing from some observations")
                arrays[f"{prefix}_baseline_source_action_to_local"] = np.stack(mappings)
        arrays.update(
            actions=np.stack([r["actions"] for r in self._records]),
            rewards=np.stack([r["rewards"] for r in self._records]),
            terminated=np.asarray([r["terminated"] for r in self._records], dtype=bool),
            truncated=np.asarray([r["truncated"] for r in self._records], dtype=bool),
            elapsed_s=np.asarray([r["next_observation"].time_s - r["observation"].time_s
                                  for r in self._records], dtype=np.float64),
            executed_actions=np.stack([
                r["info"].get("baseline", {}).get("executed_actions", r["next_observation"].current_phase)
                for r in self._records
            ]),
            executed_signal_stage=np.stack([r["next_observation"].signal_stage for r in self._records]),
        )
        behaviors = [getattr(r["output"], "behavior", None) for r in self._records]
        components = [r["info"].get("baseline", {}).get("reward_components", {}) for r in self._records]
        component_names = set(components[0])
        if any(set(c) != component_names for c in components):
            raise ValueError("reward components changed inside an episode")
        for name in sorted(component_names):
            arrays[f"reward_component_{name}"] = np.stack([c[name] for c in components])
        arrays["behavior_present"] = np.asarray([b is not None for b in behaviors], dtype=bool)
        if any(b is not None for b in behaviors):
            if not all(b is not None for b in behaviors):
                raise ValueError("behavior information missing from some decisions")
            arrays["behavior_policy_version"] = np.asarray([b.policy_version for b in behaviors])
            for name in ("log_prob", "value", "hidden_state", "action_vector"):
                values = [getattr(b, name) for b in behaviors]
                if all(value is not None for value in values):
                    arrays[f"behavior_{name}"] = np.stack(values)
                elif any(value is not None for value in values):
                    raise ValueError(f"behavior field {name} missing from some decisions")
        path = output_dir / "baseline.trajectory.npz"
        temporary = path.with_name('.' + path.name + '.tmp')
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
        manifest_path = output_dir / "baseline.trajectory.manifest.json"
        write_json(manifest_path, {
            "schema_version": SCHEMA_VERSION,
            "trajectory_file": path.name,
            "trajectory_sha256": sha256_file(path),
            "policy": self.policy_name,
            "policy_metadata": self.policy_metadata,
            "reward": self.reward_name,
            "scenario": self.scenario.to_dict(),
            "control": self.control.to_dict(),
            "roadnet_sha256": sha256_file(self.scenario.roadnet_path),
            "flow_sha256": sha256_file(self.scenario.flow_path),
            "steps": len(self._records),
            "num_intersections": self.network.num_intersections,
            "intersection_ids": list(first.node_ids),
            "baseline_schema_id": first.schema_id,
            "baseline_feature_names": list(first.feature_names),
            "feature_names": list(observations[0].feature_names),
            "reward_semantics": "local environment reward; learner aggregation/scaling in profile/config",
            "reward_components": sorted(component_names),
        })
        return {"trajectory_path": str(path), "manifest_path": str(manifest_path)}


class BaselineTrajectoryReader:
    def load(self, manifest_path: Path) -> dict:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("not a baseline v2 trajectory; v1 cannot reconstruct lane or behavior data")
        filename = manifest.get("trajectory_file")
        if filename != "baseline.trajectory.npz":
            raise ValueError("invalid baseline trajectory filename")
        path = manifest_path.parent / filename
        if sha256_file(path) != manifest.get("trajectory_sha256"):
            raise ValueError("baseline trajectory checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        steps, nodes = int(manifest["steps"]), int(manifest["num_intersections"])
        for name, values in arrays.items():
            if values.shape[0] != steps:
                raise ValueError(f"inconsistent time dimension: {name}")
        if arrays["actions"].shape != (steps, nodes) or arrays["rewards"].shape != (steps, nodes):
            raise ValueError("invalid action/reward node alignment")
        if np.any(arrays["elapsed_s"] <= 0):
            raise ValueError("non-positive transition duration")
        return {"manifest": manifest, "arrays": arrays}
