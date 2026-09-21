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
        "Rich World Model datasets require the optional RL dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc

from ..observations import RichTrafficObservationBuilder
from ..trajectory import TrajectoryReader, sha256_file


def _signed_log(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    return np.sign(values) * np.log1p(np.abs(values))


@dataclass(frozen=True)
class RichFeatureStatistics:
    feature_mean: Tuple[float, ...]
    feature_std: Tuple[float, ...]
    static_mean: Tuple[float, ...]
    static_std: Tuple[float, ...]
    demand_mean: Tuple[float, ...]
    demand_std: Tuple[float, ...]
    reward_mean: float
    reward_std: float

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        for name in (
            "feature_mean",
            "feature_std",
            "static_mean",
            "static_std",
            "demand_mean",
            "demand_std",
        ):
            value[name] = list(value[name])
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RichFeatureStatistics":
        return cls(
            feature_mean=tuple(float(item) for item in value["feature_mean"]),
            feature_std=tuple(float(item) for item in value["feature_std"]),
            static_mean=tuple(float(item) for item in value["static_mean"]),
            static_std=tuple(float(item) for item in value["static_std"]),
            demand_mean=tuple(float(item) for item in value["demand_mean"]),
            demand_std=tuple(float(item) for item in value["demand_std"]),
            reward_mean=float(value["reward_mean"]),
            reward_std=float(value["reward_std"]),
        )

    @staticmethod
    def _normalize(values: np.ndarray, mean, std) -> np.ndarray:
        return (
            _signed_log(values)
            - np.asarray(mean, dtype=np.float32)
        ) / np.asarray(std, dtype=np.float32)

    def normalize_features(self, values: np.ndarray) -> np.ndarray:
        return self._normalize(values, self.feature_mean, self.feature_std)

    def normalize_features_tensor(self, values: "torch.Tensor") -> "torch.Tensor":
        """Apply the dataset's signed-log normalization to a tensor."""
        mean = torch.as_tensor(
            self.feature_mean, dtype=values.dtype, device=values.device
        )
        std = torch.as_tensor(
            self.feature_std, dtype=values.dtype, device=values.device
        )
        transformed = torch.sign(values) * torch.log1p(torch.abs(values))
        return (transformed - mean) / std

    def normalize_static(self, values: np.ndarray) -> np.ndarray:
        return self._normalize(values, self.static_mean, self.static_std)

    def normalize_demand(self, values: np.ndarray) -> np.ndarray:
        return self._normalize(values, self.demand_mean, self.demand_std)

    def normalize_rewards(self, values: np.ndarray) -> np.ndarray:
        return (values.astype(np.float32) - self.reward_mean) / self.reward_std

    def denormalize_features_tensor(self, values: "torch.Tensor") -> "torch.Tensor":
        mean = torch.as_tensor(
            self.feature_mean, dtype=values.dtype, device=values.device
        )
        std = torch.as_tensor(
            self.feature_std, dtype=values.dtype, device=values.device
        )
        transformed = values * std + mean
        return torch.sign(transformed) * torch.expm1(torch.abs(transformed))

    def denormalize_rewards_tensor(self, values: "torch.Tensor") -> "torch.Tensor":
        return values * self.reward_std + self.reward_mean


def load_rich_trajectory_records(
    manifest_paths: Sequence[Path],
) -> List[Dict[str, Any]]:
    if not manifest_paths:
        raise ValueError("at least one trajectory manifest is required")
    reader = TrajectoryReader()
    records = []
    for path in manifest_paths:
        manifest_path = Path(path).expanduser().resolve()
        record = reader.load(manifest_path)
        record["manifest_path"] = manifest_path
        manifest = record["manifest"]
        arrays = record["arrays"]
        if tuple(manifest.get("feature_names", ())) != tuple(
            RichTrafficObservationBuilder.feature_names
        ):
            raise ValueError(
                f"trajectory {manifest_path} is not rich-traffic-movement-v1"
            )
        for key in (
            "movement_static_features",
            "demand_features",
            "phase_movement_mask",
            "neighbor_index",
            "neighbor_mask",
        ):
            if key not in arrays:
                raise ValueError(f"trajectory {manifest_path} is missing {key}")
        records.append(record)

    reference = records[0]
    semantic_fields = (
        "roadnet_sha256",
        "num_intersections",
        "max_movements",
        "max_actions",
        "intersection_ids",
        "feature_names",
        "movement_static_feature_names",
        "demand_feature_names",
        "control",
    )
    for record in records[1:]:
        for field in semantic_fields:
            if record["manifest"].get(field) != reference["manifest"].get(field):
                raise ValueError(f"trajectory manifests disagree on {field}")
        for key in (
            "movement_static_features",
            "phase_movement_mask",
            "neighbor_index",
            "neighbor_mask",
        ):
            if not np.array_equal(record["arrays"][key], reference["arrays"][key]):
                raise ValueError(f"trajectory manifests disagree on {key}")
    return records


def _column_statistics(rows: Sequence[np.ndarray]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    joined = np.concatenate([row.reshape(-1, row.shape[-1]) for row in rows], axis=0)
    transformed = _signed_log(joined)
    return (
        tuple(float(item) for item in transformed.mean(axis=0)),
        tuple(float(max(item, 1e-6)) for item in transformed.std(axis=0)),
    )


def fit_rich_statistics(records: Sequence[Dict[str, Any]]) -> RichFeatureStatistics:
    feature_count = len(RichTrafficObservationBuilder.feature_names)
    feature_columns: List[List[np.ndarray]] = [[] for _ in range(feature_count)]
    static_rows = []
    demand_rows = []
    reward_rows = []
    for record in records:
        arrays = record["arrays"]
        values = np.concatenate(
            [arrays["observation_features"], arrays["next_observation_features"]],
            axis=0,
        )
        masks = np.concatenate(
            [
                arrays["observation_valid_mask"],
                arrays["next_observation_valid_mask"],
            ],
            axis=0,
        )
        for feature_index in range(values.shape[-1]):
            selected = values[..., feature_index][masks[..., feature_index]]
            if not len(selected):
                selected = np.asarray([0.0], dtype=np.float32)
            feature_columns[feature_index].append(selected.astype(np.float32))

        static = arrays["movement_static_features"]
        movement_mask = arrays["observation_movement_mask"][0]
        static_rows.append(static[movement_mask])
        demand_rows.append(arrays["demand_features"][None, :])
        reward_rows.append(arrays["rewards"].reshape(-1))

    feature_mean_values = []
    feature_std_values = []
    for rows in feature_columns:
        transformed = _signed_log(np.concatenate(rows))
        feature_mean_values.append(float(transformed.mean()))
        feature_std_values.append(float(max(transformed.std(), 1e-6)))
    feature_mean = tuple(feature_mean_values)
    feature_std = tuple(feature_std_values)
    static_mean, static_std = _column_statistics(static_rows)
    demand_mean, demand_std = _column_statistics(demand_rows)
    rewards = np.concatenate(reward_rows).astype(np.float32)
    return RichFeatureStatistics(
        feature_mean=feature_mean,
        feature_std=feature_std,
        static_mean=static_mean,
        static_std=static_std,
        demand_mean=demand_mean,
        demand_std=demand_std,
        reward_mean=float(rewards.mean()),
        reward_std=float(max(rewards.std(), 1e-6)),
    )


class RichTrajectorySequenceDataset(Dataset):
    """History windows with multi-step CityFlow targets and executed actions."""

    def __init__(
        self,
        manifest_paths: Sequence[Path],
        history_length: int = 3,
        rollout_horizon: int = 5,
        statistics: Optional[RichFeatureStatistics] = None,
        max_windows: Optional[int] = None,
        window_seed: int = 0,
    ) -> None:
        if history_length <= 0 or rollout_horizon <= 0:
            raise ValueError("history_length and rollout_horizon must be positive")
        self.manifest_paths = tuple(
            Path(path).expanduser().resolve() for path in manifest_paths
        )
        self.records = load_rich_trajectory_records(self.manifest_paths)
        self.reference_manifest = self.records[0]["manifest"]
        self.history_length = int(history_length)
        self.rollout_horizon = int(rollout_horizon)
        self.statistics = statistics or fit_rich_statistics(self.records)
        self._index: List[Tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            steps = int(record["manifest"]["steps"])
            first_step = self.history_length - 1
            last_step = steps - self.rollout_horizon
            self._index.extend(
                (record_index, step)
                for step in range(first_step, last_step + 1)
            )
        if not self._index:
            raise ValueError(
                "trajectories are too short for the requested history and rollout"
            )
        if max_windows is not None:
            if max_windows <= 0:
                raise ValueError("max_windows must be positive when provided")
            if max_windows < len(self._index):
                generator = np.random.default_rng(window_seed)
                selected = np.sort(
                    generator.choice(len(self._index), max_windows, replace=False)
                )
                self._index = [self._index[int(item)] for item in selected]

    def __len__(self) -> int:
        return len(self._index)

    def sampling_weights(self, strategy: str) -> "torch.Tensor":
        """Return per-window weights without exposing behavior policy to the model."""
        if strategy != "flow_policy_balanced":
            raise ValueError(f"unsupported sampling strategy: {strategy}")
        group_counts: Dict[Tuple[str, str], int] = {}
        groups: List[Tuple[str, str]] = []
        for record_index, _ in self._index:
            manifest = self.records[record_index]["manifest"]
            group = (str(manifest["flow_sha256"]), str(manifest["policy"]))
            groups.append(group)
            group_counts[group] = group_counts.get(group, 0) + 1
        return torch.as_tensor(
            [1.0 / group_counts[group] for group in groups], dtype=torch.double
        )

    @staticmethod
    def _state_sequence(arrays: Dict[str, np.ndarray], name: str) -> np.ndarray:
        return np.concatenate(
            [arrays[f"observation_{name}"][:1], arrays[f"next_observation_{name}"]],
            axis=0,
        )

    def __getitem__(self, index: int) -> Dict[str, "torch.Tensor"]:
        record_index, step = self._index[index]
        arrays = self.records[record_index]["arrays"]
        start = step - self.history_length + 1
        stop = step + 1
        future_stop = step + self.rollout_horizon + 1

        state_features = self._state_sequence(arrays, "features")
        state_valid = self._state_sequence(arrays, "valid_mask")
        state_movement = self._state_sequence(arrays, "movement_mask")
        state_phase = self._state_sequence(arrays, "current_phase")
        state_stage = self._state_sequence(arrays, "signal_stage")
        state_elapsed = self._state_sequence(arrays, "phase_elapsed_s")
        state_time = self._state_sequence(arrays, "time_s")

        target_histories = np.stack(
            [
                state_features[end - self.history_length : end]
                for end in range(stop + 1, future_stop + 1)
            ]
        )
        target_history_valid = np.stack(
            [
                state_valid[end - self.history_length : end]
                for end in range(stop + 1, future_stop + 1)
            ]
        )
        target_history_movement = np.stack(
            [
                state_movement[end - self.history_length : end]
                for end in range(stop + 1, future_stop + 1)
            ]
        )
        target_history_phase = np.stack(
            [
                state_phase[end - self.history_length : end]
                for end in range(stop + 1, future_stop + 1)
            ]
        )
        target_history_stage = np.stack(
            [
                state_stage[end - self.history_length : end]
                for end in range(stop + 1, future_stop + 1)
            ]
        )
        target_history_elapsed = np.stack(
            [
                state_elapsed[end - self.history_length : end]
                for end in range(stop + 1, future_stop + 1)
            ]
        )
        target_history_time = np.stack(
            [
                state_time[end - self.history_length : end]
                for end in range(stop + 1, future_stop + 1)
            ]
        )

        return {
            "history_features": torch.from_numpy(
                self.statistics.normalize_features(state_features[start:stop])
            ).float(),
            "history_valid_mask": torch.from_numpy(state_valid[start:stop]).bool(),
            "history_movement_mask": torch.from_numpy(
                state_movement[start:stop]
            ).bool(),
            "history_current_phase": torch.from_numpy(state_phase[start:stop]).long(),
            "history_signal_stage": torch.from_numpy(state_stage[start:stop]).long(),
            "history_phase_elapsed_s": torch.from_numpy(
                state_elapsed[start:stop]
            ).float(),
            "history_time_s": torch.from_numpy(state_time[start:stop]).float(),
            "target_features": torch.from_numpy(
                self.statistics.normalize_features(state_features[stop:future_stop])
            ).float(),
            "target_valid_mask": torch.from_numpy(
                state_valid[stop:future_stop]
            ).bool(),
            "target_histories": torch.from_numpy(
                self.statistics.normalize_features(target_histories)
            ).float(),
            "target_history_valid_mask": torch.from_numpy(
                target_history_valid
            ).bool(),
            "target_history_movement_mask": torch.from_numpy(
                target_history_movement
            ).bool(),
            "target_history_current_phase": torch.from_numpy(
                target_history_phase
            ).long(),
            "target_history_signal_stage": torch.from_numpy(
                target_history_stage
            ).long(),
            "target_history_phase_elapsed_s": torch.from_numpy(
                target_history_elapsed
            ).float(),
            "target_history_time_s": torch.from_numpy(
                target_history_time
            ).float(),
            "target_current_phase": torch.from_numpy(
                state_phase[stop:future_stop]
            ).long(),
            "target_signal_stage": torch.from_numpy(
                state_stage[stop:future_stop]
            ).long(),
            "target_phase_elapsed_s": torch.from_numpy(
                state_elapsed[stop:future_stop]
            ).float(),
            "target_time_s": torch.from_numpy(state_time[stop:future_stop]).float(),
            "actions": torch.from_numpy(
                arrays["actions"][step : step + self.rollout_horizon]
            ).long(),
            "rewards": torch.from_numpy(
                self.statistics.normalize_rewards(
                    arrays["rewards"][step : step + self.rollout_horizon]
                )
            ).float(),
            "movement_static_features": torch.from_numpy(
                self.statistics.normalize_static(arrays["movement_static_features"])
            ).float(),
            "phase_movement_mask": torch.from_numpy(
                arrays["phase_movement_mask"]
            ).bool(),
            "neighbor_index": torch.from_numpy(arrays["neighbor_index"]).long(),
            "neighbor_mask": torch.from_numpy(arrays["neighbor_mask"]).bool(),
            "demand_features": torch.from_numpy(
                self.statistics.normalize_demand(arrays["demand_features"])
            ).float(),
        }

    def provenance(self) -> List[Dict[str, Any]]:
        return [
            {
                "manifest_path": str(record["manifest_path"]),
                "manifest_sha256": sha256_file(record["manifest_path"]),
                "trajectory_sha256": record["manifest"]["trajectory_sha256"],
                "flow_sha256": record["manifest"]["flow_sha256"],
                "policy": record["manifest"]["policy"],
                "steps": record["manifest"]["steps"],
            }
            for record in self.records
        ]

    def assert_compatible(self, other: "RichTrajectorySequenceDataset") -> None:
        if self.history_length != other.history_length:
            raise ValueError("dataset splits use different history lengths")
        if self.rollout_horizon != other.rollout_horizon:
            raise ValueError("dataset splits use different rollout horizons")
        for field in (
            "roadnet_sha256",
            "num_intersections",
            "max_movements",
            "max_actions",
            "intersection_ids",
            "feature_names",
            "control",
        ):
            if self.reference_manifest.get(field) != other.reference_manifest.get(field):
                raise ValueError(f"dataset splits disagree on {field}")
