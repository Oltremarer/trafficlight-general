import json
from pathlib import Path

import numpy as np
import pytest

from cityflow_tsc.counterfactual.timed_plan import formal_t1_plan
from cityflow_tsc.effect_model import GROUPS
from cityflow_tsc.effect_model import formal_data as data


def _grid():
    return [[j for j in range(16) if abs(i // 4 - j // 4) + abs(i % 4 - j % 4) == 1] for i in range(16)]


def _p95(values):
    values = np.abs(values).ravel()
    return max(1., np.percentile(values[values != 0], 95)) if np.any(values) else 1.


def _selection():
    tasks = []
    for split, count in (("train", 36), ("validation", 12), ("test", 12)):
        for i in range(count):
            tasks.append({"root_id": f"{split}_{i}", "split": split,
                          "source_id": f"base_{i % 3}", "cohort_id": f"{split}_{i % 3}"})
    return {"tasks": tasks}


def test_formal_query_ids_reference_incidence_and_joint_groups():
    plan = formal_t1_plan("data_fixture", np.arange(16) % 4, _grid())
    arrays = data.plan_arrays(plan)
    assert arrays["single_queries"].shape == (64, 2)
    assert arrays["pair_queries"].shape == (1920, 4)
    assert arrays["joint_plans"].shape == (65, 16)
    assert arrays["single_actions"].min() == 1 and arrays["single_actions"].max() == 4
    assert arrays["joint_sizes"][0] == 0 and arrays["joint_groups"][0] == "reference"
    assert not arrays["joint_plans"][0].any()
    assert not arrays["s_incidence"][0].any() and not arrays["p_incidence"][0].any()
    np.testing.assert_array_equal(arrays["s_incidence"].sum(1), arrays["joint_sizes"])
    np.testing.assert_array_equal(arrays["p_incidence"].sum(1), arrays["joint_sizes"] * (arrays["joint_sizes"] - 1) / 2)
    np.testing.assert_array_equal(np.bincount(arrays["pair_query_pair_index"]), np.full(120, 16))
    np.testing.assert_array_equal(arrays["pair_unique_nodes"][arrays["pair_query_pair_index"]], arrays["pair_nodes"])
    # A hold of the current phase is a nonreference full request sequence.
    assert tuple(arrays["single_queries"][0]) == (0, 1)
    assert len(set(arrays["joint_ids"])) == 65
    assert dict(zip(*np.unique(arrays["joint_groups"], return_counts=True))) == {
        "all": 16, "connected": 24, "reference": 1, "uniform": 24}


def test_waiting_windows_int64_and_disjoint_240_second_fields():
    raw = {"lane_q": np.full((1, 240, 240, 3), 65535, dtype=np.uint16),
           "intersection_nq": np.full((1, 240, 16, 2), 2, dtype=np.uint16),
           "boundary_pending": np.full((1, 240, 16), 3, dtype=np.uint16)}
    result = data.waiting_windows(raw)
    assert result.shape == (1, 272, 48) and result.dtype == np.int64
    assert np.all(result[:, :240] == 65535 * 15)
    assert np.all(result[:, 240:256] == 10)
    assert np.all(result[:, 256:] == 15)


def test_streaming_p95_matches_dense_and_signed_cumulative_definition():
    rng = np.random.default_rng(4)
    values = rng.integers(-10, 11, size=(5, 272, 48), dtype=np.int64)
    values[:, 240:256] *= 4
    values[:, 256:] = 0
    # Spatial cancellation must occur before taking absolute total magnitude.
    values[:, 1] = -values[:, 0]
    accumulator = data.ScaleAccumulator()
    for chunk in (values[:2], values[2:]):
        part = data.ScaleAccumulator()
        part.update(chunk)
        accumulator.merge(part.to_dict())
    scales = accumulator.arrays("single")
    cumulative = values.cumsum(-1)[:, :, np.asarray(data.HORIZONS) // 5 - 1]
    for a, b, _ in GROUPS:
        np.testing.assert_allclose(scales["single_s5"][a:b], _p95(values[:, a:b]), rtol=1e-6)
        for h in range(4):
            np.testing.assert_allclose(scales["single_sp"][a:b, h], _p95(cumulative[:, a:b, h]), rtol=1e-6)
    np.testing.assert_allclose(scales["single_sj"], [_p95(cumulative.sum(1)[:, h]) for h in range(4)], rtol=1e-6)
    assert np.all(scales["single_s5"][256:] == 1)
    histogram = data._Histogram()
    histogram.update(np.asarray([0, 1, 1, 2, 10000000000], dtype=np.int64))
    assert histogram.percentile() == pytest.approx(_p95(np.asarray([0, 1, 1, 2, 10000000000])))


def _committed(path, root_id, ids, outcomes):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, branch_ids=np.asarray(ids), outcomes=outcomes)
    Path(str(path) + ".json").write_text(json.dumps({"root_id": root_id, "branch_ids": ids,
                                                  "branch_count": len(ids), "sha256": "fixture-checksum"}))


def test_only_required_committed_shards_open_and_duplicate_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "waiting_windows", lambda raw: raw["outcomes"])
    directory = tmp_path / "shards/r"
    _committed(directory / "part_0000.npz", "r", ["a", "b"], np.asarray([[[7]], [[8]]], dtype=np.int64))
    # Both an uncommitted payload and an irrelevant committed corrupt payload
    # must not be opened when preparing the requested initial factors.
    (directory / "part_0001.npz").write_bytes(b"not committed")
    (directory / "part_0002.npz").write_bytes(b"not a numpy payload")
    (directory / "part_0002.npz.json").write_text(json.dumps({"root_id": "r", "branch_ids": ["pair"],
                                                            "branch_count": 1, "sha256": "fixture"}))
    rows = list(data.committed_outcomes(tmp_path, "r", ["a"]))
    assert len(rows) == 1 and rows[0][0] == ["a"] and rows[0][1][0, 0, 0] == 7
    with pytest.raises(FileNotFoundError, match="not committed"):
        list(data.committed_outcomes(tmp_path, "r", ["missing"]))
    _committed(directory / "part_0003.npz", "r", ["a"], np.asarray([[[7]]], dtype=np.int64))
    with pytest.raises(ValueError, match="Duplicate"):
        list(data.committed_outcomes(tmp_path, "r", ["a"]))


def test_staged_pair_derivation_subtracts_integer_truth_and_keeps_memmap(tmp_path, monkeypatch):
    plan = data.plan_arrays(formal_t1_plan("r", np.zeros(16, dtype=np.int64), _grid()))
    target = tmp_path / "derived/roots/r"
    target.mkdir(parents=True)
    data._write_numpy(target / "inputs.npz", plan, compressed=True)
    baseline = np.full((272, 48), 10000000000, dtype=np.int64)
    single = np.zeros((64, 272, 48), dtype=np.int64)
    single[:, 0, 0] = -np.arange(1, 65)
    data._write_numpy(target / "baseline.npy", baseline)
    data._write_numpy(target / "single_int64.npy", single)

    def source(*_):
        for i, bid in enumerate(plan["pair_ids"]):
            left, right = plan["pair_single_indices"][i]
            outcome = baseline + single[left] + single[right]
            outcome[0, 0] += 7
            yield [str(bid)], outcome[None], {"path": "synthetic", "sha256": "fixture"}

    monkeypatch.setattr(data, "committed_outcomes", source)
    data._derive_pairs(tmp_path, {"root_id": "r"})
    pair = np.load(target / "pair.npy", mmap_mode="r")
    assert isinstance(pair, np.memmap) and not pair.flags.writeable
    assert pair.shape == (1920, 272, 48)
    assert np.all(pair[:, 0, 0] == 7) and not pair[:, 1:].any()
    np.testing.assert_array_equal(np.load(target / "rank_targets.npy"), np.full(120, 7))
    stored_scales = data.ScaleAccumulator()
    stored_scales.merge(json.loads((target / "pair_scale_counts.json").read_text()))
    assert np.all(stored_scales.arrays("pair")["pair_sj"] == 7)


def test_initial_stage_exact_deltas_and_reference_zero(tmp_path, monkeypatch):
    plan = formal_t1_plan("r", np.zeros(16, dtype=np.int64), _grid())
    source = tmp_path / "roots/r"
    source.mkdir(parents=True)
    task = {"root_id": "r", "split": "train", "source_id": "base_00", "cohort_id": "train_0",
            "flow_id": "train_fixture", "policy": "fixed_time", "time_s": 600}
    root = {**task, "signal_context": {"current_phase": [0] * 16, "phase_elapsed_s": [0] * 16}}
    (source / "root.json").write_text(json.dumps(root))
    (source / "branches.json").write_text(json.dumps(plan))
    np.savez(source / "history.npz", **{key: np.asarray([0]) for key in data.HISTORY_KEYS})
    monkeypatch.setattr(data, "history_features", lambda *_: np.zeros((30, 272, 20), dtype=np.float32))
    lookup = {row["branch_id"]: row for row in plan["branches"]}

    def outcomes(_, rid, required):
        assert rid == "r" and len(required) == 129
        for bid in required:
            row = lookup[bid]
            assert len(row["changed_intersections"]) != 2
            effect = -sum((node + 1) * row["plan_ids"][node] for node in row["changed_intersections"])
            value = np.full((1, 272, 48), 10000000000, dtype=np.int64)
            value[0, 0, 0] += effect
            yield [bid], value, {"path": "synthetic", "sha256": "fixture"}

    monkeypatch.setattr(data, "committed_outcomes", outcomes)
    static = {"phase_masks": np.ones((16, 5, 12), dtype=np.float32)}
    data._derive_initial(tmp_path, task, static)
    target = tmp_path / "derived/roots/r"
    single = np.load(target / "single.npy", mmap_mode="r")
    joint = np.load(target / "joint.npy", mmap_mode="r")
    assert single.shape == (64, 272, 48) and joint.shape == (65, 272, 48)
    assert not joint[0].any()
    np.testing.assert_array_equal(single[:4, 0, 0], [-1, -2, -3, -4])
    assert np.load(target / "single_int64.npy").dtype == np.int64
    with np.load(target / "inputs.npz") as arrays:
        assert arrays["action_bank"].shape == (16, 5, 288)
        np.testing.assert_array_equal(arrays["s_incidence"] @ single[:, 0, 0], joint[:, 0, 0])
    assert not (target / "pair.npy").exists()


def test_explicit_test_lock_split_boundaries_and_immutable_initial_scales(tmp_path):
    selection = _selection()
    tasks = data._selected_tasks(selection, ("train", "validation"))
    assert len(tasks) == 48 and all(task["split"] != "test" for task in tasks)
    selection["tasks"][-1]["cohort_id"] = "train_2"
    with pytest.raises(ValueError, match="cannot cross splits"):
        data._selected_tasks(selection, ("train",))
    with pytest.raises(ValueError, match="checkpoints_locked"):
        data.prepare_dataset(tmp_path, splits=("test",))
    with pytest.raises(ValueError, match="checkpoints_locked"):
        data.load_data(tmp_path, splits=("test",))
    # Pair augmentation must never replace the already frozen input/single file.
    derived = tmp_path / "derived"
    derived.mkdir()
    data._write_numpy(derived / "normalization.npz", {"mean": np.asarray([123]), "std": np.asarray([456])}, compressed=True)
    before = (derived / "normalization.npz").read_bytes()
    all_zero = data.ScaleAccumulator().to_dict()
    train_tasks = data._selected_tasks(_selection(), ("train",))
    for task in train_tasks:
        path = derived / "roots" / task["root_id"]
        path.mkdir(parents=True)
        (path / "pair_scale_counts.json").write_text(json.dumps(all_zero))
        data._write_numpy(path / "rank_targets.npy", np.asarray([0, 3]))
    # No validation files exist: fitting must not attempt to read their labels.
    all_non_test = data._selected_tasks(_selection(), ("train", "validation"))
    data._fit_statistics(tmp_path, all_non_test, include_pairs=True)
    assert (derived / "normalization.npz").read_bytes() == before
    with np.load(derived / "pair_normalization.npz") as stats:
        assert stats["rank_scale"] == 3
        assert np.all(stats["pair_s5"] == 1)
