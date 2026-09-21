import copy
from collections import Counter

import numpy as np
import pytest
import torch

from cityflow_tsc.counterfactual.plan import make_branches
from cityflow_tsc.effect_model.data import (
    action_bank, effect_labels, fit_input_stats, fit_output_scales,
    history_features, normalize_history, root_split, waiting_windows,
)
from cityflow_tsc.effect_model.evaluation import predict_factors
from cityflow_tsc.effect_model.model import EffectModel, effect_loss
from cityflow_tsc.train_effect_world_model import learning_rate


@pytest.fixture
def inputs():
    torch.set_num_threads(2)
    rng = np.random.default_rng(1)
    static = {
        "adjacency": np.zeros((8, 272, 272), dtype=np.float32),
        "relative": rng.normal(size=(16, 272, 11)).astype(np.float32),
        "permission": rng.integers(0, 2, (16, 5, 272)).astype(np.float32),
        "phase_masks": rng.integers(0, 2, (16, 5, 12)).astype(np.float32),
        "object_static": np.ones((272, 2), dtype=np.float32),
    }
    raw = {"time_s": np.arange(451, 601),
           "lane_n": np.zeros((150, 240, 3), dtype=np.uint16),
           "lane_q": np.zeros((150, 240, 3), dtype=np.uint16),
           "lane_speed_sum": np.zeros((150, 240, 3), dtype=np.float32),
           "lane_tail": np.zeros((150, 240, 2), dtype=np.float32),
           "receiving_unavailable": np.zeros((150, 240), dtype=np.uint8),
           "lane_events": np.ones((150, 240, 2), dtype=np.uint16),
           "intersection_nq": np.zeros((150, 16, 2), dtype=np.uint16),
           "phase_used": np.ones((150, 16), dtype=np.uint8),
           "phase_elapsed_start_s": np.zeros((150, 16), dtype=np.float32),
           "boundary_pending": np.zeros((150, 16), dtype=np.uint16),
           "boundary_generated": np.ones((150, 16), dtype=np.uint16),
           "boundary_admitted": np.ones((150, 16), dtype=np.uint16)}
    root = {"time_s": 600, "signal_context": {"current_phase": [3] * 16, "phase_elapsed_s": [25.] * 16}}
    return raw, root, static


def test_integer_labels_before_scaling_and_joint_references():
    adjacency = [[j for j in range(16) if abs(i // 4 - j // 4) + abs(i % 4 - j % 4) == 1]
                 for i in range(16)]
    rows, _ = make_branches("test", [0] * 16, True, adjacency, [0, 1, 4, 5])
    expected_single, expected_pair, values = [], [], []
    for row in rows:
        ns = row["changed_intersections"]
        effects = [(i + 1) * row["requests"][0][i] for i in ns]
        value = 10_000_000_000 - sum(effects) + 7 * len(ns) * (len(ns) - 1) // 2
        values.append(value)
        if len(ns) == 1:
            expected_single.append(-sum(effects))
        if len(ns) == 2:
            expected_pair.append(7)
    labels = effect_labels(np.asarray(values, dtype=np.int64)[:, None, None], {"branches": rows})
    np.testing.assert_array_equal(labels["single"][:, 0, 0], expected_single)
    np.testing.assert_array_equal(labels["pair"][:, 0, 0], expected_pair)
    reconstructed = labels["s_incidence"] @ labels["single"][:, 0, 0] + labels["p_incidence"] @ labels["pair"][:, 0, 0]
    np.testing.assert_array_equal(reconstructed, labels["joint"][:, 0, 0])


def test_waiting_windows_keep_disjoint_location_classes():
    raw = {"lane_q": np.ones((2, 180, 240, 3), dtype=np.uint16),
           "intersection_nq": np.full((2, 180, 16, 2), 2, dtype=np.uint16),
           "boundary_pending": np.full((2, 180, 16), 3, dtype=np.uint16)}
    y = waiting_windows(raw)
    assert y.shape == (2, 272, 36) and y.dtype == np.int64
    assert np.all(y[:, :240] == 15)
    assert np.all(y[:, 240:256] == 10)
    assert np.all(y[:, 256:] == 15)


def test_fixed_root_split():
    counts = Counter(root_split({"flow_id": flow, "time_s": t})
                     for flow in ("train_012", "train_036", "train_069")
                     for t in (600, 1800, 3000) for _ in range(2))
    assert counts == {"train": 8, "validation": 4, "development_holdout": 6}
    with pytest.raises(ValueError):
        root_split({"flow_id": "train_999", "time_s": 600})


def test_history_endpoint_masks_and_training_statistics(inputs):
    raw, root, static = inputs
    raw["lane_n"][:, 0, 0] = 2
    raw["lane_speed_sum"][:, 0, 0] = 12
    raw["lane_tail"][:, 0] = [40, 2]
    x = history_features(raw, root, static)
    assert x.shape == (30, 272, 20)
    assert np.all(x[:, 0, 6] == 6)
    assert np.all(x[:, 1, 9:15] == 0)
    assert np.all(x[:, :240, 16:18] == 5)
    assert np.all(x[-1, 240:256, 6] == 1)
    assert np.all(x[-1, 240:256, 7] == 25)
    assert np.all(x[:, 256:, 1:3] == 5)
    stats = fit_input_stats([x])
    assert stats["mean"][0, 6] == 6  # Empty segments do not dilute speed stats.
    assert stats["mean"][0, 12] == 40
    normalized = normalize_history(x, stats)
    assert np.all(normalized[:, 1, 6:9] == 0)
    assert np.all(normalized[:, 1, 12:14] == 0)
    assert np.all(normalized[:, 0, 9] == 1)
    raw["time_s"][-1] = 601
    with pytest.raises(ValueError, match="150 recorded ticks"):
        history_features(raw, root, static)


def test_action_schedule_exposes_actual_transition_permissions(inputs):
    _, _, static = inputs
    masks = static["phase_masks"]
    bank = action_bank(np.zeros(16, dtype=np.int64), masks)
    assert bank.shape == (16, 4, 218)
    held, changed = bank[0, 0, 8:43], bank[0, 1, 8:43]
    np.testing.assert_array_equal(held[:5], np.eye(5)[1])
    np.testing.assert_array_equal(changed[:5], np.eye(5)[0])
    np.testing.assert_array_equal(changed[10:22], masks[0, 0])
    np.testing.assert_array_equal(changed[22:34], masks[0, 2])
    assert held[-1] == 0 and changed[-1] == 1
    # At the next interval target 1 is held for this candidate, but baseline switches.
    assert bank[0, 1, 77] == 0 and bank[0, 0, 77] == 1


def test_separate_physical_output_scales_and_zero_loss():
    single = np.ones((2, 272, 36), dtype=np.float32)
    single[:, 240:256] *= 2
    single[:, 256:] *= 3
    pair = single * 4
    scales, totals = fit_output_scales([{"single": single, "pair": pair}])
    np.testing.assert_array_equal(scales[:, [0, 240, 256], 0], [[1, 2, 3], [4, 8, 12]])
    assert totals[1] == totals[0] * 4
    prediction = torch.zeros((2, 272, 36), requires_grad=True)
    loss = effect_loss(prediction, torch.zeros_like(prediction), torch.tensor(scales[0]), torch.tensor(totals[0]))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_pair_symmetry_identity_and_frozen_stage(inputs):
    _, _, static = inputs
    torch.manual_seed(42)
    model = EffectModel(static).eval()
    history = torch.randn(1, 30, 272, 20)
    actions = torch.randn(1, 16, 4, 218)
    base = torch.zeros((1, 16), dtype=torch.long)
    roots, nodes, targets = torch.tensor([0]), torch.tensor([[0, 15]]), torch.tensor([[1, 2]])
    with torch.no_grad():
        encoded = model.encode(history, actions)
        forward = model.pair(encoded, roots, nodes, targets, base)
        reverse = model.pair(encoded, roots, nodes.flip(1), targets.flip(1), base)
        assert forward.shape == (1, 272, 36)
        assert torch.equal(forward, reverse)
        assert torch.count_nonzero(forward) > 0
        assert not model.single(encoded, roots, nodes[:, 0], base[0, :1], base).any()
        assert not model.pair(encoded, roots, nodes, torch.tensor([[0, 2]]), base).any()
    frozen = copy.deepcopy(model.state_dict())
    model.start_pair_stage()
    model.train(True)
    assert model.pair_head.training and not model.single_head.training and not model.gru.training
    assert all(p.requires_grad == name.startswith("pair_head.") for name, p in model.named_parameters())
    zero_pair = model.pair(encoded, roots, nodes, targets, base)
    assert not zero_pair.any()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    effect_loss(zero_pair, torch.ones_like(zero_pair), torch.ones(272, 1), torch.tensor(100.)).backward()
    optimizer.step()
    for name, value in model.state_dict().items():
        if not name.startswith("pair_head."):
            assert torch.equal(value, frozen[name])
    model.eval()
    assert model.pair(encoded, roots, nodes, targets, base).any()


def test_online_prediction_does_not_require_outcomes_or_future_baseline(inputs):
    raw, context, static = inputs
    history = history_features(raw, context, static)
    # Deliberately exclude all future labels, baseline, IDs, flows and seeds.
    root = {"raw_history": raw, "root_context": context, "input_static": static,
            "input_stats": fit_input_stats([history]), "base_phase": np.asarray([3] * 16),
            "phase_masks": static["phase_masks"], "single_nodes": np.asarray([[0], [15]]),
            "single_actions": np.asarray([[0], [1]]), "pair_nodes": np.asarray([[0, 15]]),
            "pair_actions": np.asarray([[0, 1]])}
    single, pair, timing = predict_factors(EffectModel(static), root, torch.ones(2, 272, 1),
                                          torch.device("cpu"), True)
    assert single.shape == (2, 272, 36) and pair.shape == (1, 272, 36)
    assert torch.isfinite(single).all() and torch.isfinite(pair).all()
    assert all(v >= 0 for v in timing.values())


def test_fixed_learning_rate_schedule():
    assert learning_rate(1, 300) == pytest.approx(6e-5)
    assert learning_rate(5, 300) == pytest.approx(3e-4)
    assert learning_rate(300, 300) == pytest.approx(3e-5)
    assert learning_rate(100, 100) == pytest.approx(3e-5)
