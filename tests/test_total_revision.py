import numpy as np
import pytest
import torch

from cityflow_tsc.effect_model import total_revision as v14
from cityflow_tsc.effect_model.formal_model import FormalEffectModel
from cityflow_tsc.effect_model.local_metrics import predict_single4
from cityflow_tsc.effect_model.local_revision import composed_contrast_loss
from cityflow_tsc.effect_model.revision import sequence_bank
from tests.test_local_revision import static_inputs


def test_zero_head_exactly_preserves_frozen_totals_and_reference():
    torch.manual_seed(42)
    head = v14.TotalCalibrator()
    features, reference = torch.randn(64, 257), torch.randn(64, 257)
    changed = torch.ones(64, dtype=torch.bool)
    changed[0] = False
    base = torch.randn(64, dtype=torch.float64)
    result = head(features, reference, changed, base, 13.)
    torch.testing.assert_close(result, base, rtol=0, atol=0)
    with torch.no_grad():
        head.head[-1].weight.normal_()
    result = head(features, reference, changed, base, 13.)
    assert result[0] == base[0]
    assert torch.any(result[1:] != base[1:])
    torch.testing.assert_close(head(reference, reference, changed, base, 13.), base, rtol=0, atol=0)


def test_scalar_loss_is_physical_total_plus_original_contrast():
    torch.manual_seed(1)
    predicted = torch.randn(64, dtype=torch.float64, requires_grad=True)
    truth = torch.randn(64, dtype=torch.float64)
    incidence = torch.cat((torch.zeros(1, 64), torch.eye(64))).double()
    expected = torch.nn.functional.smooth_l1_loss(predicted / 5, truth / 5)
    expected += .1 * composed_contrast_loss(predicted[:, None, None], truth[:, None, None], incidence, 7)
    actual = v14.total_loss(predicted, truth, incidence, 5, 7)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() > 0
    assert v14.total_loss(truth, truth, incidence, 5, 7) == 0
    with pytest.raises(ValueError):
        v14.total_loss(predicted[:63], truth[:63], incidence, 5, 7)


def test_cached_features_need_no_future_truth_and_preserve_frozen_fields():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    backbone = FormalEffectModel(static_inputs()).eval()
    before = {name: value.clone() for name, value in backbone.state_dict().items()}
    bank, _ = sequence_bank(np.zeros(16, np.int64), np.zeros((16, 5, 12)))
    root = {'normalized_history': np.zeros((30, 272, 20), np.float32),
        'base_phase': np.zeros(16, np.int64), 'action_bank': bank,
        'single_nodes': np.repeat(np.arange(16), 4)[:, None],
        'single_actions': np.tile(np.arange(1, 5), 16)[:, None]}
    scales = {'s5': torch.ones(272, 48), 'sj': torch.ones(4) * 3}
    f, r, changed, base, fields = v14.inputs_for_root(backbone, root, scales, 'cpu')
    assert f.shape == r.shape == (64, 257) and changed.shape == base.shape == (64,)
    assert f.dtype == r.dtype == np.float32
    np.testing.assert_array_equal(fields, predict_single4(backbone, root, scales, 'cpu'))
    np.testing.assert_array_equal(base, fields[:, :, -1].sum(1))
    head = v14.TotalCalibrator()
    optimizer = torch.optim.AdamW(head.parameters())
    prediction = head(torch.from_numpy(f), torch.from_numpy(r), torch.from_numpy(changed), torch.from_numpy(base), 3.)
    (prediction - 1).square().mean().backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in backbone.parameters())
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_calibrated_decision_uses_real_joint_truth_but_fields_remain_separate():
    incidence = np.r_[np.zeros((1, 64)), np.eye(64)]
    truth = np.zeros((738, 64))
    truth[:, 0], truth[:, 1] = -10, -5
    joint_truth = np.zeros((78, 65))
    joint_truth[:, 1], joint_truth[:, 2] = 20, -3
    arrays = {'incidence': np.repeat(incidence[None], 738, axis=0), 'truth': truth,
        'joint_truth': joint_truth, 'groups': np.repeat(np.arange(64).reshape(1, 16, 4), 78, axis=0)}
    metadata = {'validation_fields': [{'root_id': str(i), 'split': 'validation', 'cohort_id': 'c',
        'flow_id': str(i // 6), 'time_s': 240, 'policy': 'fixed', 'single4_mae': 4.,
        'joint4_mae': 25., 'single_total240_mae': 550., 'joint_total240_mae': 1200.} for i in range(78)]}
    predicted = np.zeros((78, 64))
    predicted[:, 1] = -7
    rows = v14.decision_records(predicted, arrays, metadata)
    assert all(row['selected'] == 2 and row['regret'] == 0 and row['additive_regret'] == 5 for row in rows)
    result = v14.summarize_records(rows)
    assert result['single4_mae'] == 4 and result['joint4_mae'] == 25
    assert result['single_total240_mae'] == 550
    assert result['calibrated_single_total240_mae'] == pytest.approx(12 / 64)
    assert result['single_action_regret'] == pytest.approx(5 / 16)
    assert result['independent_flow_count'] == 13
    assert result['by_flow']['0']['calibrated_single_total240_mae'] == pytest.approx(12 / 64)
    predicted[0, 0] = np.nan
    with pytest.raises(ValueError):
        v14.decision_records(predicted, arrays, metadata)


def test_incremental_schedule_and_module_dispatch_are_explicit():
    from cityflow_tsc import run_balanced_revision
    assert run_balanced_revision.v14 is v14
    assert v14.SCHEDULES[v14.FAMILIES[0]] == {'updates': 66000, 'batch': 64,
        'warmup': 1100, 'validate': 1100, 'log': 220}
    assert v14.TRAIN_MODULE == 'cityflow_tsc.train_total_revision'
    assert v14.EVALUATION_MODULE == 'cityflow_tsc.evaluate_total_revision'
