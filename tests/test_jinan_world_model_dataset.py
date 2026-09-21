import json
from pathlib import Path

import pytest

from cityflow_tsc.collect_jinan_world_model_data import (
    SELECTED_24H_HOURS,
    _existing_rule_entry,
    _rule_jobs,
    collect_shared_dqn,
    load_plan,
    prepare_diverse_plan,
    prepare_hangzhou_plan,
    prepare_plan,
)
from cityflow_tsc.trajectory import sha256_file
from cityflow_tsc.config import ControlConfig
from cityflow_tsc.environment import TrafficEnv
from cityflow_tsc.metrics import MetricCollector
from cityflow_tsc.observations import RichTrafficObservationBuilder
from cityflow_tsc.rewards import QueueReward
from cityflow_tsc.train_world_model_prediction import (
    _balanced_manifest_subset,
    _dataset_index_all_paths,
    _dataset_index_paths,
)

from .fakes import DeterministicBackend


FIXTURES = Path(__file__).parent / "fixtures"


def _flow_record(start_time: int):
    return {
        "vehicle": {
            "length": 5.0,
            "width": 2.0,
            "maxPosAcc": 2.0,
            "maxNegAcc": 4.5,
            "usualPosAcc": 2.0,
            "usualNegAcc": 4.5,
            "minGap": 2.5,
            "maxSpeed": 11.111,
            "headwayTime": 2.0,
        },
        "route": ["road_0_0_0"],
        "interval": 1.0,
        "startTime": start_time,
        "endTime": start_time,
    }


def _write_flow(path: Path, records) -> Path:
    path.write_text(json.dumps(list(records)), encoding="utf-8")
    return path


def _prepared_plan(tmp_path: Path) -> Path:
    base = [_flow_record(10)]
    source_24h = [
        _flow_record(hour * 3600 + 10) for hour in SELECTED_24H_HOURS
    ]
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    plan_root = tmp_path / "plan"
    prepare_plan(
        plan_root,
        FIXTURES / "roadnet_two_intersections.json",
        _write_flow(inputs / "real.json", base),
        _write_flow(inputs / "real_2000.json", base),
        _write_flow(inputs / "real_2500.json", base),
        _write_flow(inputs / "synthetic_60min.json", base),
        _write_flow(inputs / "synthetic_24h.json", source_24h),
    )
    return plan_root / "dataset_plan.json"


def test_prepare_jinan_plan_creates_flow_disjoint_22_condition_split(
    tmp_path: Path,
) -> None:
    plan_path = _prepared_plan(tmp_path)
    plan = load_plan(plan_path)
    assert plan["split_counts"] == {"train": 16, "validation": 2, "test": 4}
    assert plan["expected"]["total_trajectories"] == 302
    assert len(plan["conditions"]) == 22
    shifted = next(
        item for item in plan["conditions"] if item["flow_id"] == "slice_hour_22"
    )
    records = json.loads(Path(shifted["path"]).read_text(encoding="utf-8"))
    assert records[0]["startTime"] == 10.0
    assert records[0]["endTime"] == 10.0
    assert shifted["split"] == "test"


def test_prepare_hangzhou_trial_uses_two_train_and_one_validation_flow(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "hangzhou_inputs"
    inputs.mkdir()
    base = [_flow_record(10)]
    plan = prepare_hangzhou_plan(
        tmp_path / "hangzhou_plan",
        FIXTURES / "roadnet_two_intersections.json",
        _write_flow(inputs / "real.json", base),
        _write_flow(inputs / "real_5816.json", base),
        _write_flow(inputs / "synthetic.json", base),
    )
    assert plan["city"] == "Hangzhou"
    assert plan["split_counts"] == {"train": 2, "validation": 1, "test": 0}
    assert plan["expected"]["total_trajectories"] == 39


def test_prepare_diverse_plan_has_exact_2000_trajectory_recipe(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "diverse_inputs"
    inputs.mkdir()
    source = [_flow_record(index * 20) for index in range(180)]
    plan = prepare_diverse_plan(
        tmp_path / "diverse_plan",
        "Jinan",
        FIXTURES / "roadnet_two_intersections.json",
        [_write_flow(inputs / "first.json", source), _write_flow(inputs / "second.json", source)],
    )
    assert plan["split_counts"] == {
        "train": 110,
        "validation": 13,
        "test": 13,
    }
    assert len(plan["conditions"]) == 136
    assert len({item["flow_sha256"] for item in plan["conditions"]}) == 136
    assert plan["expected"]["total_trajectories"] == 2_000
    assert plan["expected"]["policy_counts"] == {
        "random": 466,
        "fixed_time": 246,
        "max_pressure": 136,
        "soft_pressure": 246,
        "shared_dqn": 906,
    }
    train = next(item for item in plan["conditions"] if item["split"] == "train")
    validation = next(
        item for item in plan["conditions"] if item["split"] == "validation"
    )
    assert len(_rule_jobs(plan, train)) == 9
    assert len(_rule_jobs(plan, validation)) == 4
    assert load_plan(tmp_path / "diverse_plan" / "dataset_plan.json")["city"] == "Jinan"
    new_york = prepare_diverse_plan(
        tmp_path / "new_york_plan",
        "NewYork",
        FIXTURES / "roadnet_two_intersections.json",
        [_write_flow(inputs / "new_york.json", source)],
    )
    assert new_york["city"] == "NewYork"


def test_existing_rule_entry_reconstructs_a_completed_rule_trajectory(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    trajectory = episode_dir / "trajectory.npz"
    trajectory.write_bytes(b"complete trajectory")
    manifest = {
        "policy": "fixed_time",
        "flow_sha256": "f" * 64,
        "steps": 3,
        "trajectory_sha256": sha256_file(trajectory),
    }
    (episode_dir / "trajectory.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (episode_dir / "metrics.json").write_text(
        json.dumps({"steps": 3, "metrics": {"average_travel_time": 1.0}}),
        encoding="utf-8",
    )
    entry = _existing_rule_entry(
        {"flow_id": "train_000", "split": "train", "flow_sha256": "f" * 64},
        "fixed_time",
        "rule_collection",
        0,
        7,
        episode_dir,
    )
    assert entry is not None
    assert entry["steps"] == 3
    assert entry["metrics"]["average_travel_time"] == 1.0


def test_prediction_dataset_index_enforces_flow_disjoint_validation(
    tmp_path: Path,
) -> None:
    train_manifest = tmp_path / "train.manifest.json"
    validation_manifest = tmp_path / "validation.manifest.json"
    test_manifest = tmp_path / "test.manifest.json"
    train_manifest.touch()
    validation_manifest.touch()
    test_manifest.touch()
    index_path = tmp_path / "dataset.index.json"
    payload = {
        "city": "Hangzhou",
        "dataset_version": "test",
        "trajectories": [
            {
                "split": "train",
                "manifest_path": str(train_manifest),
                "flow_sha256": "a" * 64,
            },
            {
                "split": "validation",
                "manifest_path": str(validation_manifest),
                "flow_sha256": "b" * 64,
            },
            {
                "split": "test",
                "manifest_path": str(test_manifest),
                "flow_sha256": "c" * 64,
            },
        ],
    }
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    train, validation, metadata = _dataset_index_paths(index_path)
    assert train == (train_manifest.resolve(),)
    assert validation == (validation_manifest.resolve(),)
    assert metadata["city"] == "Hangzhou"
    train_all, validation_all, test_all, metadata_all = _dataset_index_all_paths(
        index_path
    )
    assert train_all == train
    assert validation_all == validation
    assert test_all == (test_manifest.resolve(),)
    assert metadata_all["split_trajectory_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }

    payload["trajectories"][1]["flow_sha256"] = "a" * 64
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="leaks flows"):
        _dataset_index_paths(index_path)


def test_balanced_manifest_subset_round_robins_flow_and_policy(
    tmp_path: Path,
) -> None:
    manifests = []
    for flow in ("a", "b"):
        for policy in ("random", "max_pressure"):
            for episode in range(2):
                path = tmp_path / f"{flow}_{policy}_{episode}.json"
                path.write_text(
                    json.dumps(
                        {"flow_sha256": flow * 64, "policy": policy}
                    ),
                    encoding="utf-8",
                )
                manifests.append(path)
    selected = _balanced_manifest_subset(tuple(manifests), 4, seed=17)
    groups = set()
    for path in selected:
        payload = json.loads(path.read_text(encoding="utf-8"))
        groups.add((payload["flow_sha256"], payload["policy"]))
    assert len(groups) == 4


def _fake_rich_environment(control, network, waiting_speed_threshold):
    return TrafficEnv(
        network,
        control,
        DeterministicBackend(control),
        RichTrafficObservationBuilder(network, waiting_speed_threshold),
        QueueReward(network),
        MetricCollector(),
    )


def test_shared_dqn_cycles_across_flows_and_freezes_for_heldout(
    tmp_path: Path,
) -> None:
    plan_path = _prepared_plan(tmp_path)
    plan = load_plan(plan_path)
    plan["conditions"] = [
        next(item for item in plan["conditions"] if item["flow_id"] == flow_id)
        for flow_id in (
            "base_real_2000",
            "base_synthetic_24000",
            "base_real_2500",
            "base_real",
        )
    ]
    plan["split_counts"] = {"train": 2, "validation": 1, "test": 1}
    plan["duration_s"] = 30
    plan["control"] = ControlConfig(
        decision_interval_s=5,
        simulator_step_s=1,
        yellow_time_s=2,
        green_phase_ids=(1, 2),
    ).to_dict()
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = collect_shared_dqn(
        plan_path,
        tmp_path / "dqn",
        device="cpu",
        environment_factory=_fake_rich_environment,
        training_rounds=1,
    )
    assert len(result["training_trajectories"]) == 2
    assert len(result["heldout_trajectories"]) == 4
    assert {item["split"] for item in result["heldout_trajectories"]} == {
        "validation",
        "test",
    }
    assert {item["mode"] for item in result["heldout_trajectories"]} == {
        "greedy",
        "epsilon_0.05",
    }
    assert Path(result["final_policy_path"]).is_file()
