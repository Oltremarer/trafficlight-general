import json

import numpy as np
import pytest
import torch

from cityflow_tsc import train_temporal_effects as temporal
from cityflow_tsc.train_effect_world_model import learning_rate


def test_file_limit_reserves_memmap_capacity(monkeypatch):
    import resource
    calls = []
    monkeypatch.setattr(resource, 'getrlimit', lambda _: (1024, 1048576))
    monkeypatch.setattr(resource, 'setrlimit', lambda kind, limits: calls.append(limits))
    temporal.ensure_file_limit(738)
    assert calls == [(1732, 1048576)]
    calls.clear()
    temporal.ensure_file_limit(298)
    assert not calls


def test_update_schedule_matches_v4_at_epoch_boundaries():
    for epoch in (1, 5, 6, 50, 100, 299, 300):
        assert temporal.update_learning_rate(epoch * 440) == pytest.approx(learning_rate(epoch, 300))
    assert temporal.update_learning_rate(1) == pytest.approx(3e-4 / 2200)
    assert temporal.update_learning_rate(2200) == pytest.approx(3e-4)
    assert temporal.update_learning_rate(132000) == pytest.approx(3e-5)
    for update in (0, 132001):
        with pytest.raises(ValueError):
            temporal.update_learning_rate(update)


@pytest.mark.parametrize("roots,cycles", [(220, 300), (660, 100)])
def test_exact_equal_update_budget_and_complete_groups(roots, cycles):
    groups = np.arange(roots * 64).reshape(-1, 4)
    seen = set()
    for update, cycle, indices in temporal.grouped_batches(groups, 42):
        assert indices.shape == (32,)
        assert np.all(indices.reshape(8, 4) % 4 == np.arange(4))
        if cycle == 1:
            seen.update(indices.tolist())
    assert update == 132000
    assert cycle == cycles
    assert len(seen) == roots * 64


def test_sampler_reproducible_and_partial_cycle_exact_stop():
    groups = np.arange(64).reshape(16, 4)
    first = list(temporal.grouped_batches(groups, 42, 3))
    again = list(temporal.grouped_batches(groups, 42, 3))
    assert [(u, c) for u, c, _ in first] == [(1, 1), (2, 1), (3, 2)]
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(first, again))
    assert len(set(np.concatenate([first[0][2], first[1][2]]))) == 64
    with pytest.raises(ValueError, match="divisible by eight"):
        list(temporal.grouped_batches(groups[:9], 42, 1))


def test_arm_identity_and_approved_counts(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"temporal_arm": "multi", "split_roots": {"train": 660, "validation": 78, "test": 12}}))
    assert temporal.resolve_arm(tmp_path) == "multi"
    with pytest.raises(ValueError, match="disagrees"):
        temporal.resolve_arm(tmp_path, "early")
    path.write_text(json.dumps({"arm": "early", "split_roots": {"train": 220, "validation": 26, "test": 12}}))
    with pytest.raises(ValueError, match="counts"):
        temporal.resolve_arm(tmp_path)


def test_temporal_coverage_rejects_early_only_validation(monkeypatch):
    monkeypatch.setattr(temporal, "check_roots", lambda roots: None)
    roots = [{"split": "train", "time_s": 600} for _ in range(220)]
    roots += [{"split": "validation", "time_s": t} for t in temporal.TIME_POINTS for _ in range(26)]
    temporal.check_temporal_roots(roots, "early")
    roots[-1]["time_s"] = 600
    with pytest.raises(ValueError, match="temporal coverage"):
        temporal.check_temporal_roots(roots, "early")


def test_validation_aggregates_each_time_without_new_prediction():
    rows = []
    for time, regret in ((600, 0), (1800, 3), (2700, 6)):
        rows.append({"time_s": time, "cohort_id": "v", "flow_id": "f", "regret": regret,
                     "joint_mae": regret, "optimal_choice": regret == 0,
                     "worse_than_reference": regret > 0, "excess_wait_vs_reference": regret,
                     "benefit_vs_reference": -regret})
    metrics = temporal.temporal_metrics({"records": rows})
    assert metrics["regret"] == 3
    assert metrics["by_time"]["600"]["regret"] == 0
    assert metrics["by_time"]["2700"]["regret"] == 6
    assert metrics["worse_than_reference"] == pytest.approx(2 / 3)


def test_checkpoint_selection_records_updates_and_prefers_earlier_tie(tmp_path):
    model = torch.nn.Linear(1, 1)
    metrics = {"regret": 10., "joint_mae": 2., "by_time": {}}
    best = temporal.select_update_checkpoint(tmp_path, model, metrics, 2200, 2, "multi", 42, (float("inf"),) * 3)
    assert best == (10., 2., 2200)
    best = temporal.select_update_checkpoint(tmp_path, model, metrics, 4400, 4, "multi", 42, best)
    assert best[2] == 2200
    saved = torch.load(tmp_path / "best.pt", weights_only=False)
    assert saved["updates"] == 2200
    assert saved["variant"] == "A_ref"
    assert saved["test_used"] is False
    assert "epoch" not in saved
