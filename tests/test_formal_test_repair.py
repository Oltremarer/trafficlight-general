import copy
import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from cityflow_tsc import collect_formal_effects as collector
from cityflow_tsc import repair_formal_test as repair
from cityflow_tsc.collect_counterfactual import root_identity
from cityflow_tsc.counterfactual.writer import atomic_json, sha256
from cityflow_tsc.effect_model import formal_data
from cityflow_tsc.evaluate_formal_effects import action_opportunity, verify_checkpoint_lock


@pytest.fixture
def source_run(tmp_path):
    parent = tmp_path / "original"
    parent.mkdir()
    roadnet = parent / "roadnet.json"
    roadnet.write_text("synthetic network, no simulator used")
    tasks = []
    for row in collector.fixed_conditions():
        flow = parent / "sources" / (row["flow_id"] + ".json")
        first = {"stress_peak": 1240, "stress_late": 1960}.get(row["profile"], 0)
        atomic_json(flow, [{"startTime": first}])
        for policy in collector.POLICIES:
            tid = row["flow_id"] + "_" + policy
            directory = parent / "sources" / tid
            directory.mkdir()
            trajectory = directory / "trajectory.npz"
            features = np.zeros((2, 16, 12, 2), dtype=np.float32)
            features[:, 0, 0] = (10, 3)
            # Invalid padded movements must not contribute to coverage descriptors.
            features[:, 0, 1] = (900, 900)
            valid = np.ones_like(features, dtype=bool)
            valid[:, 0, 1] = False
            np.savez(trajectory, observation_time_s=[1800, 2700],
                     observation_features=features, observation_valid_mask=valid)
            manifest = {"scenario": {"roadnet_path": str(roadnet), "flow_path": str(flow), "seed": 7},
                        "feature_names": ["incoming_vehicle_count", "incoming_queue_count"],
                        "roadnet_sha256": sha256(roadnet), "flow_sha256": sha256(flow),
                        "trajectory_sha256": sha256(trajectory)}
            mp = directory / "trajectory.manifest.json"
            atomic_json(mp, manifest)
            task = {**row, "policy": policy, "task_id": tid, "manifest": manifest,
                    "manifest_path": str(mp), "manifest_file_sha256": sha256(mp),
                    "source_sha256": "source-" + row["source_id"],
                    "cohort_id": row["split"] + "-" + row["source_id"],
                    "collection": {"root_times": [row["time_s"]], "full_all_roots": True},
                    "flow": {"flow_id": row["flow_id"], "split": row["split"]}}
            task["root_id"] = root_identity(task, row["time_s"])
            tasks.append(task)
            atomic_json(parent / "roots" / task["root_id"] / "root.json",
                        {"root_id": task["root_id"], "time_s": row["time_s"],
                         "active_vehicles": 0 if row["split"] == "test" else 100,
                         "backlog": 0 if row["split"] == "test" else 10})
    atomic_json(parent / "selection.json", {"schema": collector.SCHEMA, "dataset": "/synthetic-source", "tasks": tasks})
    atomic_json(parent / "complete.json", {"stage": "complete"})
    atomic_json(parent / "summary.json", {"stage": "complete", "old_test_results": True})
    atomic_json(parent / "selected_A.json", {"variant": "A_ref"})
    atomic_json(parent / "static/geometry.json", {"fixture": True})
    (parent / "derived").mkdir()
    for name in ("static", "normalization", "pair_normalization"):
        np.savez(parent / "derived" / (name + ".npz"), fixture=[1])
    checkpoints = {}
    for model in ("A_ref", "A_MS", "B1", "ranker"):
        for seed in (42, 43, 44):
            name = f"{model}/seed_{seed}/best.pt"
            path = parent / name
            path.parent.mkdir(parents=True)
            path.write_bytes(name.encode())
            checkpoints[name] = sha256(path)
    atomic_json(parent / "checkpoints_locked.json", {
        "checkpoints": checkpoints, "selected_A_sha256": sha256(parent / "selected_A.json"),
        "normalization_sha256": sha256(parent / "derived/normalization.npz"),
        "pair_normalization_sha256": sha256(parent / "derived/pair_normalization.npz")})
    atomic_json(parent / "protocol.json", {"engine_binary": str(roadnet),
                "engine_binary_sha256": sha256(roadnet), "branch_seconds": 240, "history_seconds": 150,
                "master_seed": 20260909, "plan_generator": "formal_t1_joint_v2"})
    return parent


def test_corrected_selection_changes_only_test_time_and_identity(source_run):
    source = repair.read(source_run / "selection.json")
    before = copy.deepcopy(source)
    selected = repair.corrected_selection(source, source_run)
    assert source == before
    assert len(selected["tasks"]) == 12
    old = {t["task_id"]: t for t in source["tasks"]}
    for task in selected["tasks"]:
        prior = old[task["task_id"]]
        assert task["time_s"] == repair.TEST_ROOT_TIMES[task["profile"]]
        assert task["root_id"] != prior["root_id"]
        assert task["previous_root_id"] == prior["root_id"]
        assert task["previous_time_s"] == prior["time_s"]
        assert task["collection"]["root_times"] == [task["time_s"]]
        for key in ("manifest", "flow", "policy", "cohort_id", "source_sha256"):
            assert task[key] == prior[key]
    assert repair.COLLECTION_BRANCHES == 24588 and repair.TIMING_BRANCHES == 2340
    for splits in (("train",), ("validation",), ("train", "validation"), ("test", "train")):
        with pytest.raises(ValueError, match="test-only"):
            formal_data._selected_tasks(selected, splits)
    with pytest.raises(ValueError, match="original sixty-root"):
        repair.corrected_selection(selected, source_run)


def test_prepare_copies_only_frozen_assets_and_never_mutates_parent(source_run, tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "require_destination", lambda *_: None)
    before = {str(p.relative_to(source_run)): sha256(p) for p in source_run.rglob("*") if p.is_file()}
    run = tmp_path / "corrected"
    selection = repair.prepare(run, source_run)
    assert repair.prepare(run, source_run) == selection
    assert not any((run / name).exists() for name in ("complete.json", "summary.json", "roots", "derived/roots", "evaluation", "timing"))
    manifest = repair.read(run / "test_repair.json")
    assert manifest["training_jobs"] == 0
    for name, expected in manifest["reused_files"].items():
        assert sha256(run / name) == expected == sha256(source_run / name)
        assert not (run / name).is_symlink()
    coverage = repair.read(run / "protocol/temporal_coverage.json")
    assert len(coverage["original_roots"]) == 60
    assert len(coverage["corrected_source_observations"]) == 12
    assert all(r["incoming_vehicle_movement_sum"] == 10 and r["incoming_queue_movement_sum"] == 3
               for r in coverage["corrected_source_observations"])
    after = {str(p.relative_to(source_run)): sha256(p) for p in source_run.rglob("*") if p.is_file()}
    assert after == before
    # A missing copied scale must fail instead of being re-fit from twelve tests.
    (run / "derived/normalization.npz").unlink()
    with pytest.raises(OSError):
        verify_checkpoint_lock(run)
    with pytest.raises(ValueError, match="training roots"):
        formal_data._fit_statistics(run, selection["tasks"], False)


def test_repair_rejects_stale_caches_tampered_time_and_lost_provenance(source_run, tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "require_destination", lambda *_: None)
    stale = tmp_path / "stale"
    atomic_json(stale / "summary.json", {"stage": "complete"})
    with pytest.raises(ValueError, match="fresh directory"):
        repair.prepare(stale, source_run)
    run = tmp_path / "corrected"
    selection = repair.prepare(run, source_run)
    changed = copy.deepcopy(selection)
    changed["tasks"][0]["time_s"] = 600
    atomic_json(run / "selection.json", changed)
    with pytest.raises(OSError, match="repair input changed"):
        verify_checkpoint_lock(run)
    atomic_json(run / "selection.json", selection)
    (run / "test_repair.json").unlink()
    with pytest.raises(OSError, match="provenance is missing"):
        verify_checkpoint_lock(run)


def test_zero_source_observations_are_reported_not_filtered(source_run):
    source = repair.read(source_run / "selection.json")
    selection = repair.corrected_selection(source, source_run)
    task = selection["tasks"][0]
    # Coverage fixture with no queues must not change the predetermined roots.
    path = Path(task["manifest_path"]).parent / "trajectory.npz"
    with np.load(path) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["observation_features"][:] = 0
    np.savez(path, **arrays)
    task["manifest"]["trajectory_sha256"] = sha256(path)
    atomic_json(task["manifest_path"], task["manifest"])
    task["manifest_file_sha256"] = sha256(task["manifest_path"])
    report = repair.temporal_coverage(source_run, source, selection)
    assert len(report["corrected_source_observations"]) == 12
    assert report["corrected_source_observations"][0]["incoming_vehicle_movement_sum"] == 0


@pytest.mark.parametrize("workers", [8, 12])
def test_collector_dispatches_only_twelve_corrected_roots(source_run, tmp_path, monkeypatch, workers):
    selection = repair.corrected_selection(repair.read(source_run / "selection.json"), source_run)
    calls = []
    class InlinePool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == workers
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def submit(self, fn, run, task, stage):
            assert fn is repair.collect_root and stage == "remaining" and task["split"] == "test"
            calls.append(task["root_id"])
            f = Future()
            f.set_result({"root_id": task["root_id"], "split": "test", "stage": "complete",
                          "branch_count": 2049, "unresolved_events": 0})
            return f
    monkeypatch.setattr(repair.concurrent.futures, "ProcessPoolExecutor", InlinePool)
    run = tmp_path / "collection"
    summary = repair.collect_tests(run, selection, workers)
    assert summary["branch_count"] == 24588 and summary["root_count"] == 12
    assert len(calls) == len(set(calls)) == 12
    assert repair.collect_tests(run, selection, workers) == summary
    assert len(calls) == 12


def test_action_opportunity_separates_ties_cancellation_and_reference_optimum():
    root = {"root_id": "r", "split": "test", "cohort_id": "c", "flow_id": "f", "policy": "fixed_time", "time_s": 1800,
            "joint": np.zeros((3, 272, 48), dtype=np.float32)}
    empty = action_opportunity(root)
    assert empty["all_scores_tied"] and empty["all_joint_fields_zero"]
    root["joint"][1, 0, 0] = 2
    root["joint"][1, 1, 0] = -2
    cancellation = action_opportunity(root)
    assert cancellation["all_scores_tied"] and not cancellation["all_joint_fields_zero"]
    root["joint"][2, 0, 0] = 5
    reference_best = action_opportunity(root)
    assert reference_best["oracle_benefit_vehicle_seconds"] == 0
    assert reference_best["score_range_vehicle_seconds"] == 5
    assert not reference_best["all_scores_tied"]


def test_cli_defaults_to_prepare_without_launching(monkeypatch):
    calls = []
    monkeypatch.setattr(repair, "execute", lambda *args: calls.append(args) or 0)
    assert repair.main(["--source-run", "/source", "--run-dir", "/destination"]) == 0
    assert calls[0][2:] == ("prepare", 8)


def test_destination_requires_mounted_storage_and_separate_directory(monkeypatch):
    parent = Path("/mnt/pan/experiments/original")
    monkeypatch.setattr(Path, "is_mount", lambda _: False)
    with pytest.raises(ValueError, match="mounted /mnt/pan"):
        repair.require_destination(Path("/mnt/pan/experiments/new"), parent)
    monkeypatch.setattr(Path, "is_mount", lambda _: True)
    with pytest.raises(ValueError, match="system-disk fallback"):
        repair.require_destination(Path("/tmp/new"), parent)
    for path in (parent, parent / "repair", parent.parent):
        with pytest.raises(ValueError, match="separate"):
            repair.require_destination(path, parent)
    repair.require_destination(Path("/mnt/pan/experiments/new"), parent)


def test_prelaunch_parallelism_amendment_preserves_scientific_inputs(source_run, tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "require_destination", lambda *_: None)
    run = tmp_path / "corrected"
    repair.prepare(run, source_run)
    before = repair.read(run / "test_repair.json")
    protocol = repair.read(run / "protocol.json")
    repair.configure_workers(run, 12)
    after = repair.read(run / "test_repair.json")
    assert after["selection_sha256"] == before["selection_sha256"]
    assert after["reused_files"] == before["reused_files"]
    assert repair.read(run / "protocol.json") == {**protocol, "workers_max": 12}
    verify_checkpoint_lock(run)
    assert repair.prepare(run, source_run)["root_count"] == 12
    with pytest.raises(ValueError, match="1..12"):
        repair.configure_workers(run, 13)
    other = tmp_path / "started"
    repair.prepare(other, source_run)
    atomic_json(other / "status.json", {"stage": "running"})
    with pytest.raises(ValueError, match="after test collection starts"):
        repair.configure_workers(other, 12)
