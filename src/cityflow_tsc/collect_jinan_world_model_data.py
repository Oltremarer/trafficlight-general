from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch

from .config import ControlConfig, ScenarioConfig
from .dqn import DQNConfig, SharedDQNPolicy
from .observations import RichTrafficObservationBuilder
from .policies import (
    FixedTimePolicy,
    MaxPressurePolicy,
    RandomPolicy,
    SoftPressurePolicy,
)
from .runner import EpisodeRunner
from .runtime import build_rich_environment
from .topology import load_network_spec
from .training import DQNTrainer
from .trajectory import TrajectoryReader, sha256_file


PLAN_VERSION = "jinan-world-model-dataset-v1"
GENERIC_PLAN_VERSION = "traffic-world-model-dataset-v1"
LARGE_PLAN_VERSION = "traffic-world-model-diverse-2000-v1"
SELECTED_24H_HOURS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22)
TEST_SLICE_HOURS = frozenset((8, 18, 22))
VALIDATION_SLICE_HOURS = frozenset((12,))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, target)


def _read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _empty_directory(path: Path) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {target}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def slice_hour(source_path: Path, hour: int, output_path: Path) -> Dict[str, Any]:
    if hour < 0 or hour >= 24:
        raise ValueError("hour must be in [0, 24)")
    source = Path(source_path).expanduser().resolve()
    records = _read_json(source)
    if not isinstance(records, list):
        raise ValueError("CityFlow flow file must contain a list")
    start = hour * 3600
    stop = start + 3600
    selected = []
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        departure = float(raw.get("startTime", 0.0))
        if not start <= departure < stop:
            continue
        shifted = copy.deepcopy(raw)
        shifted["startTime"] = float(raw.get("startTime", 0.0)) - start
        shifted["endTime"] = float(raw.get("endTime", departure)) - start
        selected.append(shifted)
    if not selected:
        raise ValueError(f"24-hour flow has no vehicles in hour {hour}")
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, separators=(",", ":"))
    os.replace(temporary, target)
    starts = [float(item["startTime"]) for item in selected]
    ends = [float(item.get("endTime", item["startTime"])) for item in selected]
    return {
        "vehicle_definitions": len(selected),
        "first_departure_s": min(starts),
        "last_departure_s": max(ends),
    }


def _base_condition(
    flow_id: str,
    split: str,
    source: Path,
    flow_dir: Path,
) -> Dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"traffic flow does not exist: {source_path}")
    records = _read_json(source_path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"traffic flow must contain a non-empty list: {source_path}")
    target = flow_dir / f"{flow_id}.json"
    shutil.copy2(source_path, target)
    return {
        "flow_id": flow_id,
        "split": split,
        "path": str(target.resolve()),
        "flow_sha256": sha256_file(target),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_kind": "existing_60min",
        "source_hour": None,
        "vehicle_definitions": len(records),
    }


DIVERSE_FLOW_SPLITS = {"train": 110, "validation": 13, "test": 13}
DIVERSE_RULE_COUNTS = {
    "train": {"random": 4, "fixed_time": 2, "max_pressure": 1, "soft_pressure": 2},
    "validation": {
        "random": 1,
        "fixed_time": 1,
        "max_pressure": 1,
        "soft_pressure": 1,
    },
    "test": {
        "random": 1,
        "fixed_time": 1,
        "max_pressure": 1,
        "soft_pressure": 1,
    },
}
DIVERSE_DQN_TRAINING_ROUNDS = 8
DIVERSE_DQN_HELDOUT_PER_FLOW = 1


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_000_000_000


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_flow_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, separators=(",", ":"))
    os.replace(temporary, target)


def _source_cohort(
    records: Sequence[Mapping[str, Any]], source_sha256: str, split: str
) -> tuple[List[int], str]:
    if split not in DIVERSE_FLOW_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    indices = list(range(len(records)))
    random.Random(_stable_seed("cohort", source_sha256)).shuffle(indices)
    train_stop = max(1, int(len(indices) * 0.8))
    validation_stop = max(train_stop + 1, int(len(indices) * 0.9))
    validation_stop = min(validation_stop, len(indices) - 1)
    groups = {
        "train": indices[:train_stop],
        "validation": indices[train_stop:validation_stop],
        "test": indices[validation_stop:],
    }
    cohort = groups[split]
    if not cohort:
        raise ValueError(f"source flow is too small for {split} cohort")
    return cohort, _json_digest(
        {"source_sha256": source_sha256, "split": split, "indices": sorted(cohort)}
    )


def _demand_profile(value: float, profile: str, parity: int) -> float:
    clipped = min(1.0, max(0.0, value))
    if profile == "flat":
        return clipped
    if profile in ("early", "sharp_early"):
        exponent = 1.7 if profile == "early" else 2.4
        return clipped**exponent
    if profile in ("late", "sharp_late"):
        exponent = 1.7 if profile == "late" else 2.4
        return 1.0 - (1.0 - clipped) ** exponent
    if profile == "bimodal":
        if parity % 2:
            return 0.5 + 0.5 * (1.0 - (1.0 - clipped) ** 1.7)
        return 0.5 * clipped**1.7
    if profile == "stress_peak":
        return 0.35 + 0.3 * clipped
    if profile == "stress_late":
        return 0.55 + 0.45 * (1.0 - (1.0 - clipped) ** 2.0)
    raise ValueError(f"unknown demand profile: {profile}")


def _variant_parameters(split: str, ordinal: int) -> Dict[str, Any]:
    profiles = {
        "train": ("flat", "early", "late", "bimodal"),
        "validation": ("sharp_early", "sharp_late"),
        "test": ("stress_peak", "stress_late"),
    }
    scales = {
        "train": (0.65, 0.8, 1.0, 1.2, 1.4),
        "validation": (0.75, 1.15, 1.35),
        "test": (0.9, 1.3, 1.5),
    }
    return {
        "profile": profiles[split][ordinal % len(profiles[split])],
        "demand_scale": scales[split][ordinal % len(scales[split])],
        "departure_jitter_s": 30 if split == "train" else 20,
    }


def _materialize_variant(
    records: Sequence[Mapping[str, Any]],
    cohort: Sequence[int],
    output_path: Path,
    seed: int,
    parameters: Mapping[str, Any],
) -> int:
    rng = random.Random(seed)
    target_count = max(1, round(len(records) * float(parameters["demand_scale"])))
    if target_count <= len(cohort):
        selected = rng.sample(list(cohort), target_count)
    else:
        selected = [rng.choice(list(cohort)) for _ in range(target_count)]
    generated: List[Mapping[str, Any]] = []
    for position, record_index in enumerate(selected):
        item = copy.deepcopy(records[record_index])
        original_start = float(item.get("startTime", 0.0))
        normalized = min(1.0, max(0.0, original_start / 3600.0))
        shifted = 3600.0 * _demand_profile(
            normalized, str(parameters["profile"]), position
        )
        jitter = rng.randint(
            -int(parameters["departure_jitter_s"]),
            int(parameters["departure_jitter_s"]),
        )
        departure = int(round(min(3599.0, max(0.0, shifted + jitter))))
        item["startTime"] = departure
        item["endTime"] = departure
        generated.append(item)
    _write_flow_records(output_path, generated)
    return len(generated)


def prepare_diverse_plan(
    output: Path,
    city: str,
    roadnet: Path,
    source_flows: Sequence[Path],
    source_24h_flow: Optional[Path] = None,
) -> Dict[str, Any]:
    city_key = city.strip().lower().replace("_", "").replace(" ", "")
    city_names = {"jinan": "Jinan", "hangzhou": "Hangzhou", "newyork": "NewYork"}
    try:
        normalized_city = city_names[city_key]
    except KeyError as exc:
        raise ValueError("city must be Jinan, Hangzhou, or NewYork") from exc
    root = _empty_directory(output)
    roadnet_path = Path(roadnet).expanduser().resolve()
    if not roadnet_path.is_file():
        raise FileNotFoundError(f"roadnet does not exist: {roadnet_path}")
    source_dir = root / "source_flows"
    flow_dir = root / "flows"
    source_dir.mkdir(parents=True)
    flow_dir.mkdir(parents=True)

    sources: List[Dict[str, Any]] = []
    for index, raw_path in enumerate(source_flows):
        path = Path(raw_path).expanduser().resolve()
        records = _read_json(path)
        if not path.is_file() or not isinstance(records, list) or not records:
            raise ValueError(f"source flow must be a non-empty JSON list: {path}")
        sources.append(
            {
                "source_id": f"base_{index:02d}",
                "path": path,
                "records": records,
                "source_hour": None,
            }
        )
    if source_24h_flow is not None:
        source_24h = Path(source_24h_flow).expanduser().resolve()
        if not source_24h.is_file():
            raise FileNotFoundError(f"24-hour source flow does not exist: {source_24h}")
        for hour in range(24):
            slice_path = source_dir / f"hour_{hour:02d}.json"
            slice_hour(source_24h, hour, slice_path)
            sources.append(
                {
                    "source_id": f"hour_{hour:02d}",
                    "path": slice_path,
                    "records": _read_json(slice_path),
                    "source_hour": hour,
                }
            )
    if not sources:
        raise ValueError("at least one 60-minute source flow is required")

    conditions: List[Dict[str, Any]] = []
    ordinal = 0
    for split, count in DIVERSE_FLOW_SPLITS.items():
        for split_ordinal in range(count):
            source = sources[ordinal % len(sources)]
            source_path = Path(source["path"])
            source_sha256 = sha256_file(source_path)
            cohort, cohort_sha256 = _source_cohort(
                source["records"], source_sha256, split
            )
            parameters = _variant_parameters(split, split_ordinal)
            seed = _stable_seed(normalized_city, split, split_ordinal, source_sha256)
            flow_id = f"{split}_{split_ordinal:03d}"
            flow_path = flow_dir / f"{flow_id}.json"
            vehicle_definitions = _materialize_variant(
                source["records"], cohort, flow_path, seed, parameters
            )
            generator_config = {
                "city": normalized_city,
                "split": split,
                "source_id": source["source_id"],
                "source_sha256": source_sha256,
                "source_hour": source["source_hour"],
                "cohort_sha256": cohort_sha256,
                "seed": seed,
                **parameters,
            }
            conditions.append(
                {
                    "flow_id": flow_id,
                    "split": split,
                    "path": str(flow_path.resolve()),
                    "flow_sha256": sha256_file(flow_path),
                    "source_path": str(source_path),
                    "source_sha256": source_sha256,
                    "source_kind": "demand_variant",
                    "source_hour": source["source_hour"],
                    "cohort_sha256": cohort_sha256,
                    "generator_config": generator_config,
                    "generator_config_sha256": _json_digest(generator_config),
                    "vehicle_definitions": vehicle_definitions,
                }
            )
            ordinal += 1

    expected_policy_counts = {
        "random": 466,
        "fixed_time": 246,
        "max_pressure": 136,
        "soft_pressure": 246,
        "shared_dqn": 906,
    }
    expected = {
        "rule_trajectories": 1_094,
        "dqn_training_trajectories": 880,
        "dqn_heldout_trajectories": 26,
        "total_trajectories": 2_000,
        "steps_per_trajectory": 120,
        "network_decision_steps": 240_000,
        "policy_counts": expected_policy_counts,
    }
    if len({item["flow_sha256"] for item in conditions}) != len(conditions):
        raise RuntimeError("generated flows are not unique")
    plan = {
        "plan_version": LARGE_PLAN_VERSION,
        "city": normalized_city,
        "roadnet_path": str(roadnet_path),
        "roadnet_sha256": sha256_file(roadnet_path),
        "duration_s": 3600,
        "control": ControlConfig().to_dict(),
        "split_counts": dict(DIVERSE_FLOW_SPLITS),
        "policy_recipe": {
            "rule_counts": DIVERSE_RULE_COUNTS,
            "dqn_training_rounds": DIVERSE_DQN_TRAINING_ROUNDS,
            "dqn_heldout_per_flow": DIVERSE_DQN_HELDOUT_PER_FLOW,
        },
        "expected": expected,
        "conditions": conditions,
    }
    _write_json(root / "dataset_plan.json", plan)
    return plan


def prepare_plan(
    output: Path,
    roadnet: Path,
    real_flow: Path,
    real_2000_flow: Path,
    real_2500_flow: Path,
    synthetic_60min_flow: Path,
    synthetic_24h_flow: Path,
) -> Dict[str, Any]:
    root = _empty_directory(output)
    roadnet_path = Path(roadnet).expanduser().resolve()
    if not roadnet_path.is_file():
        raise FileNotFoundError(f"Jinan roadnet does not exist: {roadnet_path}")
    flow_dir = root / "flows"
    flow_dir.mkdir(parents=True)
    conditions = [
        _base_condition("base_real", "test", real_flow, flow_dir),
        _base_condition("base_real_2000", "train", real_2000_flow, flow_dir),
        _base_condition(
            "base_real_2500", "validation", real_2500_flow, flow_dir
        ),
        _base_condition(
            "base_synthetic_24000",
            "train",
            synthetic_60min_flow,
            flow_dir,
        ),
    ]
    source_24h = Path(synthetic_24h_flow).expanduser().resolve()
    source_24h_sha256 = sha256_file(source_24h)
    for hour in SELECTED_24H_HOURS:
        split = (
            "test"
            if hour in TEST_SLICE_HOURS
            else "validation"
            if hour in VALIDATION_SLICE_HOURS
            else "train"
        )
        flow_id = f"slice_hour_{hour:02d}"
        target = flow_dir / f"{flow_id}.json"
        summary = slice_hour(source_24h, hour, target)
        conditions.append(
            {
                "flow_id": flow_id,
                "split": split,
                "path": str(target.resolve()),
                "flow_sha256": sha256_file(target),
                "source_path": str(source_24h),
                "source_sha256": source_24h_sha256,
                "source_kind": "shifted_24h_hour",
                "source_hour": hour,
                **summary,
            }
        )

    split_counts = {
        split: sum(item["split"] == split for item in conditions)
        for split in ("train", "validation", "test")
    }
    if split_counts != {"train": 16, "validation": 2, "test": 4}:
        raise RuntimeError(f"unexpected Jinan split counts: {split_counts}")
    expected = {
        "rule_trajectories": 130,
        "dqn_training_trajectories": 160,
        "dqn_heldout_trajectories": 12,
        "total_trajectories": 302,
        "steps_per_trajectory": 120,
        "network_decision_steps": 36_240,
    }
    plan = {
        "plan_version": PLAN_VERSION,
        "city": "Jinan",
        "roadnet_path": str(roadnet_path),
        "roadnet_sha256": sha256_file(roadnet_path),
        "duration_s": 3600,
        "control": ControlConfig().to_dict(),
        "split_counts": split_counts,
        "expected": expected,
        "conditions": conditions,
    }
    _write_json(root / "dataset_plan.json", plan)
    return plan


def prepare_hangzhou_plan(
    output: Path,
    roadnet: Path,
    real_flow: Path,
    real_5816_flow: Path,
    synthetic_60min_flow: Path,
) -> Dict[str, Any]:
    root = _empty_directory(output)
    roadnet_path = Path(roadnet).expanduser().resolve()
    if not roadnet_path.is_file():
        raise FileNotFoundError(f"Hangzhou roadnet does not exist: {roadnet_path}")
    flow_dir = root / "flows"
    flow_dir.mkdir(parents=True)
    conditions = [
        _base_condition("base_hangzhou_real", "validation", real_flow, flow_dir),
        _base_condition(
            "base_hangzhou_real_5816", "train", real_5816_flow, flow_dir
        ),
        _base_condition(
            "base_hangzhou_synthetic_24000",
            "train",
            synthetic_60min_flow,
            flow_dir,
        ),
    ]
    expected = {
        "rule_trajectories": 17,
        "dqn_training_trajectories": 20,
        "dqn_heldout_trajectories": 2,
        "total_trajectories": 39,
        "steps_per_trajectory": 120,
        "network_decision_steps": 4_680,
    }
    plan = {
        "plan_version": GENERIC_PLAN_VERSION,
        "city": "Hangzhou",
        "roadnet_path": str(roadnet_path),
        "roadnet_sha256": sha256_file(roadnet_path),
        "duration_s": 3600,
        "control": ControlConfig().to_dict(),
        "split_counts": {"train": 2, "validation": 1, "test": 0},
        "expected": expected,
        "conditions": conditions,
    }
    _write_json(root / "dataset_plan.json", plan)
    return plan


def load_plan(path: Path) -> Dict[str, Any]:
    plan_path = Path(path).expanduser().resolve()
    plan = _read_json(plan_path)
    if plan.get("plan_version") not in (
        PLAN_VERSION,
        GENERIC_PLAN_VERSION,
        LARGE_PLAN_VERSION,
    ):
        raise ValueError("unsupported traffic dataset plan")
    if plan.get("city") not in ("Jinan", "Hangzhou", "NewYork"):
        raise ValueError("dataset city must be Jinan, Hangzhou, or NewYork")
    if sha256_file(Path(plan["roadnet_path"])) != plan["roadnet_sha256"]:
        raise ValueError("roadnet hash no longer matches the plan")
    for condition in plan["conditions"]:
        if sha256_file(Path(condition["path"])) != condition["flow_sha256"]:
            raise ValueError(f"flow hash changed for {condition['flow_id']}")
    if plan.get("plan_version") == LARGE_PLAN_VERSION:
        hashes_by_split: Dict[str, set[str]] = {}
        for condition in plan["conditions"]:
            hashes_by_split.setdefault(condition["split"], set()).add(
                condition["flow_sha256"]
            )
        for left, left_hashes in hashes_by_split.items():
            for right, right_hashes in hashes_by_split.items():
                if left < right and left_hashes & right_hashes:
                    raise ValueError(f"generated flow hashes leak between {left} and {right}")
    return plan


def _control_from_plan(plan: Mapping[str, Any]) -> ControlConfig:
    values = dict(plan["control"])
    values["green_phase_ids"] = tuple(values["green_phase_ids"])
    return ControlConfig(**values)


def _trajectory_entry(
    result,
    condition: Mapping[str, Any],
    policy: str,
    mode: str,
    episode: int,
    seed: int,
) -> Dict[str, Any]:
    return {
        "flow_id": condition["flow_id"],
        "split": condition["split"],
        "flow_sha256": condition["flow_sha256"],
        "policy": policy,
        "mode": mode,
        "episode": episode,
        "seed": seed,
        "steps": result.steps,
        "manifest_path": result.manifest_path,
        "manifest_sha256": sha256_file(Path(result.manifest_path)),
        "trajectory_path": result.trajectory_path,
        "trajectory_sha256": sha256_file(Path(result.trajectory_path)),
        "metrics": result.metrics,
    }


def _rule_jobs(
    plan: Mapping[str, Any], condition: Mapping[str, Any]
) -> List[tuple[str, int]]:
    recipe = plan.get("policy_recipe")
    if recipe is None:
        count_random = 6 if condition["split"] == "train" else 2
        return [("random", episode) for episode in range(count_random)] + [
            ("max_pressure", 0)
        ]
    counts = recipe["rule_counts"][condition["split"]]
    jobs: List[tuple[str, int]] = []
    for policy_name in ("random", "fixed_time", "max_pressure", "soft_pressure"):
        jobs.extend(
            (policy_name, episode) for episode in range(int(counts[policy_name]))
        )
    return jobs


def _rule_policy(policy_name: str, episode: int):
    if policy_name == "random":
        return RandomPolicy(), False, {"policy_variant": "uniform_valid_action"}
    if policy_name == "fixed_time":
        return FixedTimePolicy(offset=episode), True, {"phase_offset": episode}
    if policy_name == "max_pressure":
        return MaxPressurePolicy(), True, {"policy_variant": "deterministic"}
    if policy_name == "soft_pressure":
        temperature = 0.35 if episode == 0 else 0.8
        return SoftPressurePolicy(temperature=temperature), False, {
            "temperature": temperature
        }
    raise ValueError(f"unsupported rule policy: {policy_name}")


def _existing_rule_entry(
    condition: Mapping[str, Any],
    policy_name: str,
    mode: str,
    episode: int,
    seed: int,
    episode_dir: Path,
) -> Optional[Dict[str, Any]]:
    manifest_path = episode_dir / "trajectory.manifest.json"
    trajectory_path = episode_dir / "trajectory.npz"
    metrics_path = episode_dir / "metrics.json"
    if not (manifest_path.is_file() and trajectory_path.is_file() and metrics_path.is_file()):
        return None
    manifest = _read_json(manifest_path)
    if (
        manifest.get("policy") != policy_name
        or manifest.get("flow_sha256") != condition["flow_sha256"]
        or int(manifest.get("steps", -1)) <= 0
        or manifest.get("trajectory_sha256") != sha256_file(trajectory_path)
    ):
        return None
    metrics_payload = _read_json(metrics_path)
    if int(metrics_payload.get("steps", -1)) != int(manifest["steps"]):
        return None
    return {
        "flow_id": condition["flow_id"],
        "split": condition["split"],
        "flow_sha256": condition["flow_sha256"],
        "policy": policy_name,
        "mode": mode,
        "episode": episode,
        "seed": seed,
        "steps": int(manifest["steps"]),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_sha256": sha256_file(trajectory_path),
        "metrics": dict(metrics_payload.get("metrics", {})),
    }


def _rule_condition(
    plan: Mapping[str, Any],
    condition: Mapping[str, Any],
    output_root: str,
    thread_num: int,
) -> List[Dict[str, Any]]:
    output = Path(output_root)
    control = _control_from_plan(plan)
    network = load_network_spec(Path(plan["roadnet_path"]), control)
    jobs = _rule_jobs(plan, condition)
    entries = []
    condition_index = next(
        index
        for index, item in enumerate(plan["conditions"])
        if item["flow_id"] == condition["flow_id"]
    )
    for policy_name, episode in jobs:
        seed = _stable_seed("rule", plan["city"], policy_name, condition_index, episode)
        episode_dir = (
            output
            / condition["split"]
            / policy_name
            / condition["flow_id"]
            / f"episode_{episode:02d}"
        )
        existing = _existing_rule_entry(
            condition,
            policy_name,
            "rule_collection",
            episode,
            seed,
            episode_dir,
        )
        if existing is not None:
            entries.append(existing)
            continue
        policy, deterministic, policy_details = _rule_policy(policy_name, episode)
        scenario = ScenarioConfig(
            roadnet_path=Path(plan["roadnet_path"]),
            flow_path=Path(condition["path"]),
            output_dir=episode_dir,
            duration_s=int(plan["duration_s"]),
            seed=seed,
            thread_num=thread_num,
        )
        result = EpisodeRunner().run(
            build_rich_environment(control, network, 0.1),
            policy,
            scenario,
            deterministic=deterministic,
            policy_metadata={
                "mode": "traffic_world_model_dataset",
                "city": plan["city"],
                "flow_id": condition["flow_id"],
                "split": condition["split"],
                **policy_details,
            },
        )
        entries.append(
            _trajectory_entry(
                result,
                condition,
                policy_name,
                "rule_collection",
                episode,
                seed,
            )
        )
    return entries


def collect_rules(
    plan_path: Path,
    output: Path,
    workers: int = 8,
    thread_num: int = 1,
    resume: bool = False,
) -> Dict[str, Any]:
    if workers <= 0 or thread_num <= 0:
        raise ValueError("workers and thread_num must be positive")
    plan = load_plan(plan_path)
    root = Path(output).expanduser().resolve()
    if resume:
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = _empty_directory(root)
    entries: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _rule_condition,
                plan,
                condition,
                str(root),
                thread_num,
            ): condition["flow_id"]
            for condition in plan["conditions"]
        }
        for future in as_completed(futures):
            entries.extend(future.result())
            _write_json(
                root / "rules_progress.json",
                {
                    "completed_trajectories": len(entries),
                    "latest_flow_id": futures[future],
                },
            )
    entries.sort(
        key=lambda item: (
            item["split"],
            item["policy"],
            item["flow_id"],
            item["episode"],
        )
    )
    result = {
        "mode": "traffic_rule_collection",
        "city": plan["city"],
        "expected": int(plan["expected"]["rule_trajectories"]),
        "completed": len(entries),
        "resumed": bool(resume),
        "trajectories": entries,
    }
    if result["completed"] != result["expected"]:
        raise RuntimeError(f"rule collection count mismatch: {result}")
    _write_json(root / "rules_index.json", result)
    return result


def collect_shared_dqn(
    plan_path: Path,
    output: Path,
    device: str = "cuda",
    thread_num: int = 1,
    environment_factory: Callable[..., Any] = build_rich_environment,
    training_rounds: int = 10,
) -> Dict[str, Any]:
    if training_rounds <= 0 or thread_num <= 0:
        raise ValueError("training_rounds and thread_num must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {device!r}, but CUDA is unavailable")
    plan = load_plan(plan_path)
    recipe = plan.get("policy_recipe", {})
    plan_rounds = int(recipe.get("dqn_training_rounds", training_rounds))
    if training_rounds != 10 and training_rounds != plan_rounds:
        raise ValueError(
            f"requested {training_rounds} DQN rounds but plan requires {plan_rounds}"
        )
    training_rounds = plan_rounds
    heldout_per_flow = int(recipe.get("dqn_heldout_per_flow", 2))
    if heldout_per_flow not in (1, 2):
        raise ValueError("dqn_heldout_per_flow must be one or two")
    root = _empty_directory(output)
    control = _control_from_plan(plan)
    network = load_network_spec(Path(plan["roadnet_path"]), control)
    config = DQNConfig(
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.95,
        batch_size=128,
        replay_capacity=100_000,
        warmup_transitions=512,
        target_update_steps=250,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20_000,
        reward_scale=0.1,
    )
    policy = SharedDQNPolicy(
        network,
        RichTrafficObservationBuilder.feature_names,
        RichTrafficObservationBuilder.schema_id,
        plan["roadnet_sha256"],
        config,
        device=device,
        seed=200_000,
    )
    trainer = DQNTrainer(policy, config, seed=200_000)
    training_conditions = [
        item for item in plan["conditions"] if item["split"] == "train"
    ]
    training_entries = []
    for round_index in range(training_rounds):
        rotated = (
            training_conditions[round_index % len(training_conditions) :]
            + training_conditions[: round_index % len(training_conditions)]
        )
        for position, condition in enumerate(rotated):
            seed = 200_000 + round_index * 1000 + position
            scenario = ScenarioConfig(
                roadnet_path=Path(plan["roadnet_path"]),
                flow_path=Path(condition["path"]),
                output_dir=(
                    root
                    / "train"
                    / condition["flow_id"]
                    / f"round_{round_index + 1:02d}"
                ),
                duration_s=int(plan["duration_s"]),
                seed=seed,
                thread_num=thread_num,
            )
            result = trainer.run_episode(
                environment_factory(control, network, 0.1), scenario
            )
            training_entries.append(
                _trajectory_entry(
                    result,
                    condition,
                    "shared_dqn",
                    "online_training",
                    round_index,
                    seed,
                )
            )
        if round_index + 1 in (2, 5, 8, training_rounds):
            trainer.save_checkpoint(
                root
                / "checkpoints"
                / f"shared_dqn_round_{round_index + 1:02d}.pt"
            )
        _write_json(
            root / "dqn_progress.json",
            {
                "completed_training_rounds": round_index + 1,
                "completed_training_trajectories": len(training_entries),
                "environment_steps": trainer.environment_steps,
                "gradient_steps": trainer.gradient_steps,
                "epsilon": trainer.epsilon(),
            },
        )

    final_policy_path = root / "checkpoints" / "shared_dqn_final_policy.pt"
    policy.save(
        final_policy_path,
        extra={
            "mode": "frozen_traffic_world_model_collector",
            "city": plan["city"],
            "environment_steps": trainer.environment_steps,
            "gradient_steps": trainer.gradient_steps,
        },
    )
    heldout_entries = []
    heldout_conditions = [
        item for item in plan["conditions"] if item["split"] != "train"
    ]
    heldout_modes = (False,) if heldout_per_flow == 1 else (False, True)
    for condition_index, condition in enumerate(heldout_conditions):
        for exploratory in heldout_modes:
            seed = 300_000 + condition_index * 10 + int(exploratory)
            mode = "epsilon_0.05" if exploratory else "greedy"
            scenario = ScenarioConfig(
                roadnet_path=Path(plan["roadnet_path"]),
                flow_path=Path(condition["path"]),
                output_dir=(
                    root
                    / condition["split"]
                    / condition["flow_id"]
                    / mode
                ),
                duration_s=int(plan["duration_s"]),
                seed=seed,
                thread_num=thread_num,
            )
            result = EpisodeRunner().run(
                environment_factory(control, network, 0.1),
                policy,
                scenario,
                deterministic=not exploratory,
                policy_metadata={
                    "mode": mode,
                    "flow_id": condition["flow_id"],
                    "split": condition["split"],
                    "checkpoint_path": str(final_policy_path),
                    "checkpoint_sha256": policy.checkpoint_sha256,
                },
            )
            heldout_entries.append(
                _trajectory_entry(
                    result,
                    condition,
                    "shared_dqn",
                    mode,
                    int(exploratory),
                    seed,
                )
            )

    expected_training = len(training_conditions) * training_rounds
    expected_heldout = len(heldout_conditions) * heldout_per_flow
    if (
        len(training_entries) != expected_training
        or len(heldout_entries) != expected_heldout
    ):
        raise RuntimeError("DQN trajectory count does not match the collection plan")
    result = {
        "mode": "jinan_shared_dqn_collection",
        "training_rounds": training_rounds,
        "environment_steps": trainer.environment_steps,
        "gradient_steps": trainer.gradient_steps,
        "final_epsilon": trainer.epsilon(),
        "final_policy_path": str(final_policy_path),
        "final_policy_sha256": policy.checkpoint_sha256,
        "training_trajectories": training_entries,
        "heldout_trajectories": heldout_entries,
    }
    _write_json(root / "dqn_index.json", result)
    return result


def finalize_dataset(
    plan_path: Path,
    rules_index_path: Path,
    dqn_index_path: Path,
    output: Path,
) -> Dict[str, Any]:
    plan = load_plan(plan_path)
    rules = _read_json(rules_index_path)
    dqn = _read_json(dqn_index_path)
    entries = list(rules["trajectories"])
    entries.extend(dqn["training_trajectories"])
    entries.extend(dqn["heldout_trajectories"])
    expected = int(plan["expected"]["total_trajectories"])
    if len(entries) != expected:
        raise ValueError(f"expected {expected} trajectories, found {len(entries)}")
    conditions = {item["flow_id"]: item for item in plan["conditions"]}
    reader = TrajectoryReader()
    policy_counts: Dict[str, int] = {}
    split_counts: Dict[str, int] = {}
    manifest_paths = set()
    for entry in entries:
        manifest_path = Path(entry["manifest_path"])
        if str(manifest_path) in manifest_paths:
            raise ValueError(f"duplicate trajectory manifest: {manifest_path}")
        manifest_paths.add(str(manifest_path))
        record = reader.load(manifest_path)
        manifest = record["manifest"]
        condition = conditions[entry["flow_id"]]
        if manifest["flow_sha256"] != condition["flow_sha256"]:
            raise ValueError(f"flow hash mismatch for {entry['flow_id']}")
        if manifest["roadnet_sha256"] != plan["roadnet_sha256"]:
            raise ValueError("trajectory roadnet hash does not match the plan")
        if int(manifest["steps"]) != int(plan["expected"]["steps_per_trajectory"]):
            raise ValueError(f"trajectory has wrong length: {manifest_path}")
        if sha256_file(manifest_path) != entry["manifest_sha256"]:
            raise ValueError(f"manifest hash changed: {manifest_path}")
        policy_counts[entry["policy"]] = policy_counts.get(entry["policy"], 0) + 1
        split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + 1
    expected_policy_counts = plan["expected"].get("policy_counts")
    if expected_policy_counts is not None and policy_counts != expected_policy_counts:
        raise ValueError(
            f"policy trajectory counts do not match plan: {policy_counts}"
        )
    result = {
        "dataset_version": plan["plan_version"],
        "city": plan["city"],
        "roadnet_path": plan["roadnet_path"],
        "roadnet_sha256": plan["roadnet_sha256"],
        "trajectory_count": len(entries),
        "network_decision_steps": sum(int(item["steps"]) for item in entries),
        "policy_counts": policy_counts,
        "split_trajectory_counts": split_counts,
        "flow_split_counts": plan["split_counts"],
        "plan_path": str(Path(plan_path).expanduser().resolve()),
        "plan_sha256": sha256_file(Path(plan_path)),
        "trajectories": sorted(
            entries,
            key=lambda item: (
                item["split"], item["policy"], item["flow_id"], item["seed"]
            ),
        ),
    }
    _write_json(Path(output), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cityflow-tsc-collect-jinan-world-model-data"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--roadnet", required=True, type=Path)
    prepare.add_argument("--real-flow", required=True, type=Path)
    prepare.add_argument("--real-2000-flow", required=True, type=Path)
    prepare.add_argument("--real-2500-flow", required=True, type=Path)
    prepare.add_argument("--synthetic-60min-flow", required=True, type=Path)
    prepare.add_argument("--synthetic-24h-flow", required=True, type=Path)

    prepare_hangzhou = subparsers.add_parser("prepare-hangzhou")
    prepare_hangzhou.add_argument("--output", required=True, type=Path)
    prepare_hangzhou.add_argument("--roadnet", required=True, type=Path)
    prepare_hangzhou.add_argument("--real-flow", required=True, type=Path)
    prepare_hangzhou.add_argument("--real-5816-flow", required=True, type=Path)
    prepare_hangzhou.add_argument(
        "--synthetic-60min-flow", required=True, type=Path
    )

    prepare_diverse = subparsers.add_parser("prepare-diverse")
    prepare_diverse.add_argument("--output", required=True, type=Path)
    prepare_diverse.add_argument(
        "--city", required=True, choices=("Jinan", "Hangzhou", "NewYork")
    )
    prepare_diverse.add_argument("--roadnet", required=True, type=Path)
    prepare_diverse.add_argument("--flow", required=True, action="append", type=Path)
    prepare_diverse.add_argument("--source-24h-flow", type=Path)

    rules = subparsers.add_parser("collect-rules")
    rules.add_argument("--plan", required=True, type=Path)
    rules.add_argument("--output", required=True, type=Path)
    rules.add_argument("--workers", type=int, default=8)
    rules.add_argument("--thread-num", type=int, default=1)
    rules.add_argument("--resume", action="store_true")

    dqn = subparsers.add_parser("collect-dqn")
    dqn.add_argument("--plan", required=True, type=Path)
    dqn.add_argument("--output", required=True, type=Path)
    dqn.add_argument("--device", default="cuda")
    dqn.add_argument("--thread-num", type=int, default=1)
    dqn.add_argument("--training-rounds", type=int, default=10)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--plan", required=True, type=Path)
    finalize.add_argument("--rules-index", required=True, type=Path)
    finalize.add_argument("--dqn-index", required=True, type=Path)
    finalize.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_plan(
            args.output,
            args.roadnet,
            args.real_flow,
            args.real_2000_flow,
            args.real_2500_flow,
            args.synthetic_60min_flow,
            args.synthetic_24h_flow,
        )
    elif args.command == "prepare-hangzhou":
        result = prepare_hangzhou_plan(
            args.output,
            args.roadnet,
            args.real_flow,
            args.real_5816_flow,
            args.synthetic_60min_flow,
        )
    elif args.command == "prepare-diverse":
        result = prepare_diverse_plan(
            args.output,
            args.city,
            args.roadnet,
            args.flow,
            args.source_24h_flow,
        )
    elif args.command == "collect-rules":
        result = collect_rules(
            args.plan, args.output, args.workers, args.thread_num, args.resume
        )
    elif args.command == "collect-dqn":
        result = collect_shared_dqn(
            args.plan,
            args.output,
            args.device,
            args.thread_num,
            training_rounds=args.training_rounds,
        )
    else:
        result = finalize_dataset(
            args.plan, args.rules_index, args.dqn_index, args.output
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
