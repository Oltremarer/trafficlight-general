import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from cityflow_tsc import collect_formal_effects as collector
from cityflow_tsc.counterfactual.timed_plan import formal_t1_plan
from cityflow_tsc.counterfactual.writer import atomic_json, sha256


def grid():
    return [[j for j in range(16) if abs(i // 4 - j // 4) + abs(i % 4 - j % 4) == 1]
            for i in range(16)]


def test_fixed_conditions_cover_approved_sixty_roots():
    rows = collector.fixed_conditions()
    assert len(rows) == len({r["flow_id"] for r in rows}) == 30
    assert not {"train_012", "train_036", "train_069"}.intersection(r["flow_id"] for r in rows)
    assert Counter(r["split"] for r in rows) == {"train": 18, "validation": 6, "test": 6}
    for split, total in (("train", 18), ("validation", 6), ("test", 6)):
        selected = [r for r in rows if r["split"] == split]
        assert Counter(r["time_s"] for r in selected) == {600: total // 2, 1800: total // 2}
        assert Counter(r["source_id"] for r in selected) == {f"base_{i:02d}": total // 3 for i in range(3)}
    assert next(r for r in rows if r["flow_id"] == "validation_002")["demand_scale"] == 1.35
    assert next(r for r in rows if r["flow_id"] == "test_005")["time_s"] == 1800


def fixture_sources(tmp_path, monkeypatch):
    """Metadata-only source fixture, never starts CityFlow or extra rollouts."""
    roadnet = tmp_path / "roadnet.json"
    roadnet.write_text("source-roadnet", encoding="utf-8")
    true_sha = sha256
    monkeypatch.setattr(collector, "sha256", lambda p: collector.ROADNET_SHA256
                        if Path(p) == roadnet else true_sha(p))
    conditions, trajectories = [], []
    for row in collector.fixed_conditions():
        flow_path = tmp_path / (row["flow_id"] + ".json")
        flow_path.write_text(row["flow_id"], encoding="utf-8")
        source, cohort = "source-" + row["source_id"], row["source_id"] + "-" + row["split"]
        flow = {"flow_id": row["flow_id"], "split": row["split"], "path": str(flow_path),
                "source_sha256": source, "cohort_sha256": cohort, "flow_sha256": true_sha(flow_path),
                "generator_config": {k: row[k] for k in ("source_id", "profile", "demand_scale")}}
        conditions.append(flow)
        for policy in collector.POLICIES:
            directory = tmp_path / (row["flow_id"] + "_" + policy)
            directory.mkdir()
            trajectory = directory / "trajectory.npz"
            trajectory.write_bytes(b"source-trajectory")
            manifest = {"control": {"decision_interval_s": 30, "simulator_step_s": 1., "yellow_time_s": 5,
                                    "green_phase_ids": [1, 2, 3, 4], "all_red_time_s": 0},
                        "scenario": {"roadnet_path": str(roadnet), "flow_path": str(flow_path), "seed": 7},
                        "roadnet_sha256": collector.ROADNET_SHA256, "flow_sha256": flow["flow_sha256"],
                        "trajectory_sha256": true_sha(trajectory)}
            manifest_path = directory / "trajectory.manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            trajectories.append({"flow_id": row["flow_id"], "policy": policy, "episode": 0,
                                 "manifest_path": str(manifest_path), "manifest_sha256": true_sha(manifest_path)})
    return {"conditions": conditions}, {"trajectories": trajectories}


def test_exact_source_selection_and_group_metadata(tmp_path, monkeypatch):
    plan, index = fixture_sources(tmp_path, monkeypatch)
    tasks = collector.select_tasks(plan, index, tmp_path)
    assert len(tasks) == len({t["root_id"] for t in tasks}) == 60
    assert Counter(t["split"] for t in tasks) == {"train": 36, "validation": 12, "test": 12}
    assert Counter(t["policy"] for t in tasks) == {"fixed_time": 30, "max_pressure": 30}
    assert len({t["group_id"] for t in tasks}) == 9
    assert all(t["source_id"].startswith("base_") and t["cohort_id"] == t["cohort_sha256"] for t in tasks)
    assert all(t["time_s"] == t["collection"]["root_times"][0] for t in tasks)
    assert all(t["collection"]["full_all_roots"] for t in tasks)
    assert {t["time_s"] for t in tasks[:8]} == {600, 1800}
    assert len([t for t in tasks if t["split"] != "test"]) * collector.INITIAL_PER_ROOT == 6192
    assert len(tasks) * collector.BRANCHES_PER_ROOT == 122940


def test_condition_or_episode_drift_is_rejected(tmp_path, monkeypatch):
    plan, index = fixture_sources(tmp_path, monkeypatch)
    changed = copy.deepcopy(plan)
    changed["conditions"][0]["generator_config"]["demand_scale"] = 1.5
    with pytest.raises(ValueError, match="approved v3 row"):
        collector.select_tasks(changed, index, tmp_path)
    index["trajectories"].append(copy.deepcopy(index["trajectories"][0]))
    with pytest.raises(ValueError, match="unique episode 0"):
        collector.select_tasks(plan, index, tmp_path)


def test_noncontiguous_initial_shards_leave_factor_order_unchanged():
    plan = formal_t1_plan("test-formal", np.arange(16) % 4, grid())
    original = copy.deepcopy(plan)
    initial, pairs = collector.stage_chunks(plan, "initial"), collector.stage_chunks(plan, "pairs")
    first_ids = [i for _, indices in initial for i in indices]
    pair_ids = [i for _, indices in pairs for i in indices]
    assert len(initial) == 33 and len(pairs) == 480
    assert initial[-1] == (32, [2048])
    assert pairs[0] == (33, [65, 66, 67, 68])
    assert first_ids == list(range(65)) + list(range(1985, 2049))
    assert pair_ids == list(range(65, 1985))
    assert len(first_ids + pair_ids) == len(set(first_ids + pair_ids)) == 2049
    assert set(first_ids + pair_ids) == set(range(2049))
    assert plan == original
    by_id = {b["branch_id"]: i for i, b in enumerate(plan["branches"])}
    assert all(by_id[c] in first_ids for c in plan["candidate_ids"])
    assert all(by_id[sid] in first_ids for i in pair_ids for sid in plan["branches"][i]["single_ids"])


def test_initial_stage_commit_and_resume_do_not_repeat_rollout(tmp_path, monkeypatch):
    plan = formal_t1_plan("test-formal", np.arange(16) % 4, grid())
    root = {"root_id": "test-formal", "signal_context": {"context": "root"}}
    task = {"task_id": "test-task", "split": "train"}
    atomic_json(tmp_path / "roots/test-formal/branches.json", plan)

    def save_npz(path, arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return sha256(path)

    monkeypatch.setattr(collector, "atomic_npz", save_npz)

    class Backend:
        restores = 0

        def restore_archive(self, archive):
            assert archive == "actual-owned-snapshot"
            self.restores += 1

    class Runner:
        def restore_context(self, context):
            assert context == root["signal_context"]

        def step(self, request):
            return [{"unresolved_events": np.asarray(0, dtype=np.uint32),
                     "example_field": np.asarray(request, dtype=np.uint8)} for _ in range(30)]

    class Recorder:
        def restore_context(self, context):
            assert context == {"tick": 600}

    backend = Backend()
    args = (tmp_path, task, root, "actual-owned-snapshot", backend, Runner(), Recorder(),
            plan, {"tick": 600}, "initial", 0.)
    result = collector._collect_stage(*args)
    assert result["branch_count"] == 129 and result["unresolved_events"] == 0
    assert backend.restores == 129
    assert (tmp_path / "indexes/test-formal.initial.complete.json").exists()
    metadata = json.loads((tmp_path / "shards/test-formal/part_0016.npz.json").read_text())
    assert metadata["branch_indices"] == [64, 1985, 1986, 1987]
    with np.load(tmp_path / "shards/test-formal/part_0016.npz", allow_pickle=False) as data:
        assert data["example_field"].shape == (4, 240, 16)
        assert data["branch_ids"].tolist() == metadata["branch_ids"]
    assert collector._collect_stage(*args) == result
    assert backend.restores == 129


def test_frozen_metadata_cannot_adopt_another_plan(tmp_path):
    path = tmp_path / "selection.json"
    collector._freeze_json(path, {"schema": collector.SCHEMA, "tasks": ["one"]})
    collector._freeze_json(path, {"schema": collector.SCHEMA, "tasks": ["one"]})
    with pytest.raises(ValueError, match="frozen formal run"):
        collector._freeze_json(path, {"schema": collector.SCHEMA, "tasks": ["two"]})


def test_second_phase_rebuild_preserves_frozen_root_files(tmp_path, monkeypatch):
    task = {"root_id": "saved-root", "task_id": "source-fixed", "time_s": 600,
            "source_id": "base_00", "cohort_id": "cohort", "cohort_sha256": "cohort",
            "source_sha256": "source", "group_id": "source:cohort", "profile": "flat",
            "demand_scale": .65, "split": "train", "policy": "fixed_time",
            "manifest": {"flow_sha256": "flow", "scenario": {"seed": 17}}}
    directory = tmp_path / "roots/saved-root"
    directory.mkdir(parents=True)
    history = directory / "history.npz"
    history.write_bytes(b"frozen-history")
    root = {"root_id": "saved-root", "time_s": 600, "flow_sha256": "flow", "policy": "fixed_time",
            "roadnet_sha256": collector.ROADNET_SHA256, "simulator_seed": 17,
            "history_sha256": sha256(history), **collector._root_groups(task)}
    atomic_json(directory / "root.json", root)
    atomic_json(directory / "recorder_context.json", {"tick": 600})
    before = {p.name: p.read_bytes() for p in directory.iterdir()}

    class Backend:
        engine = "engine"

        def capture_archive(self):
            return "live-engine-archive"

    class Recorder:
        calls = 0

        def observe(self, engine):
            assert engine == "engine"
            self.calls += 1

    class Runner:
        calls = 0

        def step(self, request):
            assert len(request) == 16
            self.calls += 1

    def no_root_rewrite(*args, **kwargs):
        pytest.fail("Saved root metadata/history must not be rewritten for the pair stage")

    monkeypatch.setattr(collector, "create_roots", no_root_rewrite)
    recorder, runner = Recorder(), Runner()
    actual, archive = collector._build_root(tmp_path, task, Backend(), runner, recorder,
                                            np.zeros((120, 16), dtype=np.uint8))
    assert actual == root and archive == "live-engine-archive"
    assert runner.calls == 20 and recorder.calls == 1
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
