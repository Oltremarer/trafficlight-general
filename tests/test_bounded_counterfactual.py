import json
import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from cityflow_tsc import collect_counterfactual as collector
from cityflow_tsc.counterfactual import SCHEMA
from cityflow_tsc.counterfactual.writer import atomic_json, sha256


def test_expired_collection_stops_without_starting_simulator(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Expired budget must not launch a simulator")

    monkeypatch.setattr(collector, "CityFlowBackend", forbidden)
    atomic_json(tmp_path / "roots/train_004_fixed_time_t0600_example/root.json", {"time_s": 600})
    atomic_json(tmp_path / "shards/train_004_fixed_time_t0600_example/part_0000.npz.json",
                {"branch_count": 4, "unresolved_events": 2, "internal_trip_completion_count": 3,
                 "trip_completion_count": 10, "boundary_exit_count": 7, "bytes": 100})
    result = collector.collect_task(str(tmp_path), {"task_id": "train_004_fixed_time"}, 4,
                                    "route-completion-v2", time.monotonic() - 1)
    assert result["stage"] == "budget_exhausted"
    assert result["root_count"] == 1 and result["branch_count"] == 4
    assert result["unresolved_events"] == 2 and result["internal_trip_completion_count"] == 3
    assert not (tmp_path / "indexes/train_004_fixed_time.complete.json").exists()
    assert not (tmp_path / "logs/train_004_fixed_time.error.json").exists()
    assert json.loads((tmp_path / "logs/train_004_fixed_time.progress.json").read_text())["stage"] == "budget_exhausted"


def test_cannot_resume_legacy_run_with_route_aware_labels(tmp_path):
    atomic_json(tmp_path / "protocol.json", {"schema": SCHEMA, "consistency_checks": False})
    atomic_json(tmp_path / "selection.json", {"tasks": []})
    with pytest.raises(ValueError, match="mix recorder event semantics"):
        collector.prepare(tmp_path, tmp_path, "route-completion-v2")


def test_budget_disabled_or_not_yet_expired():
    collector.check_budget(None)
    collector.check_budget(time.monotonic() + 10)


def test_four_hour_scope_counts_order_and_source_ordinals():
    settings = collector.collection_config([600, 1800, 3000], [1800, 600, 3000], True)
    tasks = [{"task_id": str(i), "ordinal": i + 10} for i in range(6)]
    ordered = collector.ordered_tasks(tasks, ["5", "3", "0", "2", "1", "4"])
    assert [t["ordinal"] for t in ordered] == [15, 13, 10, 12, 11, 14]
    assert collector.collection_counts(ordered, settings) == {
        "root_count": 18, "full_root_count": 18, "branch_upper_bound": 24876}
    roots = [{"time_s": t, "full_pair_root": True} for t in settings["root_times"]]
    assert [r["time_s"] for r in collector.order_roots(roots, settings)] == [1800, 600, 3000]
    assert collector.ROOT_TIMES == (600, 1200, 1800, 2400, 3000)
    with pytest.raises(ValueError, match="Duplicate task"):
        collector.ordered_tasks(tasks, ["0", "0"])
    with pytest.raises(ValueError, match="Unknown task"):
        collector.ordered_tasks(tasks, ["unknown"])


@pytest.mark.parametrize("times,order,full", [
    ([600, 600], None, True), ([601], None, True), ([120], None, True),
    ([3600], None, True), ([600, 1800, 3000], [600, 1800, 1800], True),
    ([600, 1800, 3000], None, False),
])
def test_reject_invalid_root_configuration(times, order, full):
    with pytest.raises(ValueError):
        collector.collection_config(times, order, full)


def test_selected_root_capture_marks_every_root_full(tmp_path, monkeypatch):
    settings = collector.collection_config([600, 1800, 3000], [1800, 600, 3000], True)
    state = {"time": 0}
    captures = []

    def step(_):
        rows = []
        for _ in range(30):
            state["time"] += 1
            rows.append({"time_s": np.asarray(state["time"]), "lane_q": np.zeros((1, 3), dtype=np.uint16),
                         "intersection_nq": np.zeros((1, 2), dtype=np.uint16),
                         "boundary_pending": np.zeros(1, dtype=np.uint16),
                         "active_vehicles": np.asarray(0), "receiving_unavailable": np.zeros(1)})
        return rows

    def capture():
        captures.append(state["time"])
        return object()

    backend = SimpleNamespace(engine=object(), capture_archive=capture)
    recorder = SimpleNamespace(observe=lambda _: None, context=lambda: {})
    runner = SimpleNamespace(step=step, context=lambda: {"current_phase": [0], "phase_elapsed_s": [state["time"]]})
    monkeypatch.setattr(collector, "atomic_npz", lambda *args: "history-test-hash")
    task = {"task_id": "flow_fixed_time", "ordinal": 0, "policy": "fixed_time", "collection": settings,
            "flow": {"flow_id": "flow", "split": "train"}, "manifest_path": "test",
            "manifest": {"flow_sha256": "flow", "roadnet_sha256": "road", "scenario": {"seed": 0}}}
    roots, archives = collector.create_roots(tmp_path, task, backend, runner, recorder, np.zeros((100, 1)))
    assert captures == [600, 1800, 3000]
    assert len(roots) == len(archives) == 3
    assert all(r["full_pair_root"] for r in roots)
    saved = json.loads((tmp_path / "indexes/flow_fixed_time.roots.json").read_text())["roots"]
    assert saved == roots


def test_root_scope_is_part_of_immutable_contract():
    current = {"collection": collector.collection_config([600, 1800, 3000], [1800, 600, 3000], True)}
    changed = deepcopy(current)
    changed["collection"]["root_order"] = [600, 1800, 3000]
    with pytest.raises(ValueError, match="Immutable collection contract changed"):
        collector.verify_resume_contract(current, changed, {}, {})


@pytest.mark.parametrize("key", ["source_hashes", "engine_binary_sha256", "dataset",
                                "dataset_plan_sha256", "dataset_index_sha256", "branch_seconds"])
def test_resume_rejects_changed_contract(key):
    frozen = {"source_hashes": {"recorder.py": "old"}, "engine_binary_sha256": "engine",
              "dataset": "/data", "dataset_plan_sha256": "plan", "dataset_index_sha256": "index",
              "branch_seconds": 180}
    current = deepcopy(frozen)
    current[key] = "changed"
    with pytest.raises(ValueError, match="Immutable collection contract changed"):
        collector.verify_resume_contract(frozen, current, {}, {})


def test_resume_rejects_changed_manifest_and_accepts_timestamp_only():
    collector.verify_resume_contract({"created_at": "old", "execution": {}},
                                     {"created_at": "new"}, {"tasks": []}, {"tasks": []})
    with pytest.raises(ValueError, match="source manifests changed"):
        collector.verify_resume_contract({}, {}, {"tasks": ["old"]}, {"tasks": ["new"]})


def test_execution_contract_is_immutable(tmp_path):
    protocol = {}
    tasks = [{"task_id": "task"}]
    collector.freeze_execution(tmp_path, protocol, tasks, 4)
    original = (tmp_path / "protocol.json").read_bytes()
    collector.freeze_execution(tmp_path, protocol, tasks, 4)
    for other_tasks, size in [(tasks, 16), ([{"task_id": "other"}], 4)]:
        with pytest.raises(ValueError, match="Frozen task selection or shard-size"):
            collector.freeze_execution(tmp_path, protocol, other_tasks, size)
    assert (tmp_path / "protocol.json").read_bytes() == original


def completed_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "ROOT_TIMES", (600,))
    monkeypatch.setattr(collector, "root_identity", lambda task, t: "saved_task_t0600")
    rid = "saved_task_t0600"
    directory = tmp_path / "roots" / rid
    directory.mkdir(parents=True)
    np.savez(directory / "history.npz", time_s=[600])
    root = {"root_id": rid, "history_sha256": sha256(directory / "history.npz")}
    atomic_json(directory / "root.json", root)
    atomic_json(tmp_path / "indexes/saved_task.roots.json", {"roots": [root]})
    atomic_json(directory / "branches.json", {"root_id": rid, "branch_count": 1,
                                               "branches": [{"branch_id": "b0"}]})
    atomic_json(tmp_path / "indexes" / f"{rid}.complete.json",
                {"root_id": rid, "branch_count": 1,
                 "branch_plan_sha256": sha256(directory / "branches.json")})
    shard = tmp_path / "shards" / rid / "part_0000.npz"
    shard.parent.mkdir(parents=True)
    np.savez(shard, branch_ids=["b0"])
    atomic_json(str(shard) + ".json", {"branch_ids": ["b0"], "branch_count": 1, "sha256": sha256(shard)})
    result = {"task_id": "saved_task", "root_count": 1, "branch_count": 1}
    atomic_json(tmp_path / "indexes/saved_task.complete.json", result)
    def forbidden(*args):
        raise AssertionError("Validation must not launch a simulator")
    monkeypatch.setattr(collector, "CityFlowBackend", forbidden)
    return shard, result


def test_completed_task_checks_real_files_without_recollecting(tmp_path, monkeypatch):
    _, result = completed_fixture(tmp_path, monkeypatch)
    assert collector.collect_task(str(tmp_path), {"task_id": "saved_task"}, 4) == result


@pytest.mark.parametrize("problem", ["missing", "corrupt", "wrong_ids", "extra", "plan"])
def test_completed_marker_cannot_hide_bad_payload(tmp_path, monkeypatch, problem):
    shard, _ = completed_fixture(tmp_path, monkeypatch)
    if problem == "missing":
        shard.unlink()
    elif problem == "corrupt":
        shard.write_bytes(b"corrupted test payload")
    elif problem == "wrong_ids":
        np.savez(shard, branch_ids=["wrong"])
        atomic_json(str(shard) + ".json", {"branch_ids": ["b0"], "branch_count": 1, "sha256": sha256(shard)})
    elif problem == "extra":
        np.savez(shard.parent / "part_9999.npz", branch_ids=["extra"])
    else:
        atomic_json(tmp_path / "roots/saved_task_t0600/branches.json", {"changed": True})
    with pytest.raises((OSError, ValueError, KeyError)):
        collector.collect_task(str(tmp_path), {"task_id": "saved_task"}, 4)
    progress = json.loads((tmp_path / "logs/saved_task.progress.json").read_text())
    assert progress["stage"] == "failed"
    assert (tmp_path / "indexes/saved_task.complete.json").exists()  # Preserve evidence.
