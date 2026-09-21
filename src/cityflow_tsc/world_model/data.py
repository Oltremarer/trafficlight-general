from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - covered on ML runtime
    raise RuntimeError(
        "The World Model requires the optional ML dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc

from ..observations import QueuePressureObservationBuilder
from ..trajectory import TrajectoryReader, sha256_file


PRIMITIVE_FEATURE_NAMES = (
    "incoming_vehicle_count",
    "incoming_queue_count",
    "outgoing_vehicle_count",
    "outgoing_queue_count",
)
EXPECTED_FEATURE_NAMES = QueuePressureObservationBuilder.feature_names


@dataclass(frozen=True)
class FeatureStatistics:
    """Normalization fitted only from the training episodes."""

    feature_mean: Tuple[float, ...]
    feature_std: Tuple[float, ...]
    reward_mean: float
    reward_std: float

    def __post_init__(self) -> None:
        if len(self.feature_mean) != len(PRIMITIVE_FEATURE_NAMES):
            raise ValueError("feature statistics have the wrong dimension")
        if len(self.feature_std) != len(PRIMITIVE_FEATURE_NAMES):
            raise ValueError("feature statistics have the wrong dimension")
        if any(value <= 0 for value in self.feature_std) or self.reward_std <= 0:
            raise ValueError("normalization standard deviations must be positive")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["feature_mean"] = list(self.feature_mean)
        value["feature_std"] = list(self.feature_std)
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "FeatureStatistics":
        return cls(
            feature_mean=tuple(float(item) for item in value["feature_mean"]),
            feature_std=tuple(float(item) for item in value["feature_std"]),
            reward_mean=float(value["reward_mean"]),
            reward_std=float(value["reward_std"]),
        )

    def normalize_features(self, values: np.ndarray) -> np.ndarray:
        transformed = np.log1p(np.clip(values, 0.0, None)).astype(np.float32)
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        std = np.asarray(self.feature_std, dtype=np.float32)
        return (transformed - mean) / std

    def normalize_rewards(self, values: np.ndarray) -> np.ndarray:
        return (values.astype(np.float32) - self.reward_mean) / self.reward_std

    def denormalize_features_tensor(self, values: "torch.Tensor") -> "torch.Tensor":
        mean = torch.as_tensor(
            self.feature_mean, dtype=values.dtype, device=values.device
        )
        std = torch.as_tensor(
            self.feature_std, dtype=values.dtype, device=values.device
        )
        return torch.clamp(torch.expm1(values * std + mean), min=0.0)


def _validate_manifest_set(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("at least one trajectory manifest is required")
    first = records[0]["manifest"]
    if tuple(first.get("feature_names", ())) != EXPECTED_FEATURE_NAMES:
        raise ValueError("trajectory feature schema is not queue-pressure-movement-v1")
    semantic_fields = (
        "roadnet_sha256",
        "num_intersections",
        "max_movements",
        "max_actions",
        "intersection_ids",
        "feature_names",
        "control",
    )
    first_arrays = records[0]["arrays"]
    for record in records[1:]:
        manifest = record["manifest"]
        for field in semantic_fields:
            if manifest.get(field) != first.get(field):
                raise ValueError(
                    f"trajectory manifests disagree on semantic field {field}"
                )
        arrays = record["arrays"]
        for field in ("neighbor_index", "neighbor_mask", "phase_movement_mask"):
            if not np.array_equal(arrays[field], first_arrays[field]):
                raise ValueError(f"trajectory manifests disagree on {field}")
    return first


def load_trajectory_records(manifest_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    reader = TrajectoryReader()
    records = []
    for path in manifest_paths:
        manifest_path = Path(path).expanduser().resolve()
        record = reader.load(manifest_path)
        record["manifest_path"] = manifest_path
        records.append(record)
    _validate_manifest_set(records)
    return records


def fit_statistics(records: Sequence[Dict[str, Any]]) -> FeatureStatistics:
    _validate_manifest_set(records)
    feature_rows: List[List[np.ndarray]] = [[], [], [], []]
    reward_rows: List[np.ndarray] = []
    for record in records:
        arrays = record["arrays"]
        values = arrays["observation_features"][..., :4]
        valid = arrays["observation_valid_mask"][..., :4]
        next_values = arrays["next_observation_features"][..., :4]
        next_valid = arrays["next_observation_valid_mask"][..., :4]
        for index in range(4):
            feature_rows[index].append(values[..., index][valid[..., index]])
            feature_rows[index].append(
                next_values[..., index][next_valid[..., index]]
            )
        reward_rows.append(arrays["rewards"].reshape(-1))
    means = []
    stds = []
    for rows in feature_rows:
        joined = np.concatenate(rows).astype(np.float32)
        transformed = np.log1p(np.clip(joined, 0.0, None))
        means.append(float(transformed.mean()))
        stds.append(float(max(transformed.std(), 1e-6)))
    rewards = np.concatenate(reward_rows).astype(np.float32)
    return FeatureStatistics(
        feature_mean=tuple(means),
        feature_std=tuple(stds),
        reward_mean=float(rewards.mean()),
        reward_std=float(max(rewards.std(), 1e-6)),
    )


class TrajectoryTransitionDataset(Dataset):
    """Transition dataset that keeps complete episodes as the split unit."""

    def __init__(
        self,
        manifest_paths: Sequence[Path],
        statistics: Optional[FeatureStatistics] = None,
    ) -> None:
        self.manifest_paths = tuple(Path(path).expanduser().resolve() for path in manifest_paths)
        self.records = load_trajectory_records(self.manifest_paths)
        self.reference_manifest = self.records[0]["manifest"]
        self.statistics = statistics or fit_statistics(self.records)
        self._index: List[Tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            steps = int(record["manifest"]["steps"])
            self._index.extend((record_index, step) for step in range(steps))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Dict[str, "torch.Tensor"]:
        record_index, step = self._index[index]
        arrays = self.records[record_index]["arrays"]
        features = arrays["observation_features"][step, ..., :4]
        next_features = arrays["next_observation_features"][step, ..., :4]
        return {
            "features": torch.from_numpy(
                self.statistics.normalize_features(features)
            ).float(),
            "next_features": torch.from_numpy(
                self.statistics.normalize_features(next_features)
            ).float(),
            "valid_mask": torch.from_numpy(
                arrays["observation_valid_mask"][step, ..., :4]
            ).bool(),
            "next_valid_mask": torch.from_numpy(
                arrays["next_observation_valid_mask"][step, ..., :4]
            ).bool(),
            "movement_mask": torch.from_numpy(
                arrays["observation_movement_mask"][step]
            ).bool(),
            "current_phase": torch.from_numpy(
                arrays["observation_current_phase"][step]
            ).long(),
            "signal_stage": torch.from_numpy(
                arrays["observation_signal_stage"][step]
            ).long(),
            "phase_elapsed_s": torch.from_numpy(
                arrays["observation_phase_elapsed_s"][step]
            ).float(),
            "actions": torch.from_numpy(arrays["actions"][step]).long(),
            "rewards": torch.from_numpy(
                self.statistics.normalize_rewards(arrays["rewards"][step])
            ).float(),
        }

    @property
    def neighbor_index(self) -> np.ndarray:
        return self.records[0]["arrays"]["neighbor_index"].copy()

    @property
    def neighbor_mask(self) -> np.ndarray:
        return self.records[0]["arrays"]["neighbor_mask"].copy()

    def provenance(self) -> List[Dict[str, Any]]:
        values = []
        for record in self.records:
            manifest = record["manifest"]
            path = record["manifest_path"]
            values.append(
                {
                    "manifest_path": str(path),
                    "manifest_sha256": sha256_file(path),
                    "trajectory_sha256": manifest["trajectory_sha256"],
                    "flow_sha256": manifest["flow_sha256"],
                    "policy": manifest["policy"],
                    "steps": manifest["steps"],
                }
            )
        return values

    def assert_compatible(self, other: "TrajectoryTransitionDataset") -> None:
        fields = (
            "roadnet_sha256",
            "num_intersections",
            "max_movements",
            "max_actions",
            "intersection_ids",
            "feature_names",
            "control",
        )
        for field in fields:
            if self.reference_manifest.get(field) != other.reference_manifest.get(field):
                raise ValueError(
                    f"trajectory dataset splits disagree on semantic field {field}"
                )
        for field in ("neighbor_index", "neighbor_mask", "phase_movement_mask"):
            if not np.array_equal(
                self.records[0]["arrays"][field], other.records[0]["arrays"][field]
            ):
                raise ValueError(f"trajectory dataset splits disagree on {field}")


def discover_manifests(root: Path) -> Tuple[Path, ...]:
    paths = tuple(sorted(Path(root).expanduser().resolve().rglob("trajectory.manifest.json")))
    if not paths:
        raise FileNotFoundError(f"no trajectory manifests found under {root}")
    return paths


def split_episode_manifests(
    manifest_paths: Sequence[Path], validation_fraction: float, seed: int
) -> Tuple[Tuple[Path, ...], Tuple[Path, ...]]:
    paths = tuple(Path(path).expanduser().resolve() for path in manifest_paths)
    if not paths:
        raise ValueError("at least one trajectory manifest is required")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if len(paths) == 1 or validation_fraction == 0:
        return paths, ()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    validation_count = max(1, int(round(len(paths) * validation_fraction)))
    validation_count = min(validation_count, len(paths) - 1)
    validation_indices = set(int(item) for item in order[:validation_count])
    train = tuple(path for index, path in enumerate(paths) if index not in validation_indices)
    validation = tuple(path for index, path in enumerate(paths) if index in validation_indices)
    return train, validation
