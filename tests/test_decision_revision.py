import json

import numpy as np
import pytest
import torch

from cityflow_tsc.counterfactual.writer import atomic_json, sha256
from cityflow_tsc.effect_model import decision_revision as v9, local_metrics
from cityflow_tsc.effect_model.formal_model import FormalEffectModel, multiscale_loss
from cityflow_tsc.effect_model.local_revision import local_loss, composed_contrast_loss
from cityflow_tsc.evaluate_decision_revision import lock_checkpoints, verify_lock, LOCKED_FILES, split_results
from cityflow_tsc.run_local_revision import training_order
from cityflow_tsc.train_local_revision import select_checkpoint
from tests.test_local_revision import static_inputs, pair_root


def test_top8_stable_ties_484_edges_and_weighted_means():
    torch.manual_seed(7)
    p = torch.randn(64, 2, 3, requires_grad=True)
    y = torch.zeros_like(p)
    incidence = torch.cat((torch.zeros(1, 64), torch.eye(64)))
    scores = incidence @ p.sum((1, 2))
    i, j = torch.triu_indices(65, 65, 1)
    mask = (i < 8) | (j < 8)
    assert len(i) == 2080 and mask.sum() == 484
    terms = torch.nn.functional.smooth_l1_loss((scores[i] - scores[j]) / 10,
                                              torch.zeros(2080), reduction='none')
    actual = v9.top8_contrast_loss(p, y, incidence, 10)
    torch.testing.assert_close(actual, .05 * terms.mean() + .05 * terms[mask].mean())
    actual.backward()
    assert p.grad.abs().sum() > 0
    assert v9.top8_contrast_loss(y, y, incidence, 10) == 0


def test_dense_local_preserves_cancellation_fp32_natural_mean():
    p = torch.zeros(2, 272, 48, requires_grad=True)
    y = torch.zeros_like(p)
    y[0, 0, 0], y[0, 0, 6], y[0, 0, 18] = 10, -12, 3
    scale = torch.full((272, 4), 2.)
    loss = v9.dense_local_loss(p, y, scale)
    assert loss.dtype == torch.float32
    assert float(loss.detach()) == pytest.approx((10 + 2 + 1 + 1) / 2 / (2 * 272 * 4))
    loss.backward()
    assert p.grad.abs().sum() > 0


def test_pair_contrast_eight_groups_136_differences_and_zero_reference():
    torch.manual_seed(8)
    p = torch.randn(128, 272, 4, requires_grad=True)
    y = torch.randn_like(p)
    ps, ys = p[:, :, -1].sum(1).reshape(8, 16), y[:, :, -1].sum(1).reshape(8, 16)
    ps = torch.cat((ps, torch.zeros(8, 1)), 1)
    ys = torch.cat((ys, torch.zeros(8, 1)), 1)
    i, j = torch.triu_indices(17, 17, 1)
    assert len(i) == 136 and (j == 16).sum() == 16
    expected = torch.nn.functional.smooth_l1_loss((ps[:, i] - ps[:, j]) / 11, (ys[:, i] - ys[:, j]) / 11)
    actual = v9.pair_action_contrast_loss(p, y, 11)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert p.grad[:, :, -1].abs().sum() > 0 and p.grad[:, :, :-1].count_nonzero() == 0
    with pytest.raises(ValueError):
        v9.pair_action_contrast_loss(p[:-1], y[:-1], 11)


def test_pair_scale_train_only_nonzero_p95_and_per_pair_grouping():
    root = {**pair_root(), 'split': 'train', 'pair4': np.zeros((1920, 272, 4), np.float32)}
    totals = np.tile(np.arange(-8, 8), 120).reshape(120, 16)
    root['pair4'][:, 0, -1] = totals.ravel()
    values = np.c_[totals, np.zeros(120)]
    i, j = np.triu_indices(17, 1)
    differences = abs(values[:, i] - values[:, j])
    assert v9.fit_pair_contrast_scale([root]) == np.percentile(differences[differences > 0], 95)
    with pytest.raises(ValueError, match='training roots only'):
        v9.fit_pair_contrast_scale([{**root, 'split': 'validation'}])
    root['pair4'].fill(0)
    assert v9.fit_pair_contrast_scale([root]) == 1


def test_B_match_loss_identical_and_contrast_is_only_addition():
    torch.manual_seed(10)
    p = torch.randn(128, 272, 4)
    y, logits = torch.randn_like(p), torch.randn_like(p)
    scales = {'sp': torch.full((272, 4), 2.), 'sj': torch.full((4,), 3.)}
    original = local_loss(p, y, scales['sp'], scales['sj'], logits)
    actual = v9.training_loss('B_Match', p, y, scales, logits, None, 5, 7)
    torch.testing.assert_close(original, actual, rtol=0, atol=0)
    contrast = v9.training_loss('B_Contrast', p, y, scales, logits, None, 5, 7)
    torch.testing.assert_close(contrast, original + .1 * v9.pair_action_contrast_loss(p * 2, y, 7))


def test_A_local_keeps_dense_base_and_candidate_loss():
    torch.manual_seed(11)
    p = torch.randn(64, 272, 48)
    y, logits = torch.randn_like(p), torch.randn_like(p)
    scales = {'sp': torch.full((272, 4), 2.), 's5': torch.full((272, 48), 3.), 'sj': torch.full((4,), 5.)}
    incidence = torch.cat((torch.zeros(1, 64), torch.eye(64)))
    expected = multiscale_loss(p, y, scales, 'A_ref', logits)
    expected += .1 * composed_contrast_loss(p * 3, y, incidence, 7)
    expected += .1 * v9.dense_local_loss(p * 3, y, scales['sp'])
    torch.testing.assert_close(v9.training_loss('A_D_Local', p, y, scales, logits, incidence, 7, 9), expected)


def test_B_validation_uses_fixed_predicted_A_but_true_pair_errors(monkeypatch):
    root = {'root_id': 'r', 'split': 'validation', 'cohort_id': 'c', 'flow_id': 'f', 'policy': 'p', 'time_s': 600,
        's_incidence': np.r_[np.zeros((1, 64)), np.eye(64)], 'p_incidence': np.zeros((65, 1)),
        'single4': np.zeros((64, 272, 4)), 'fixed_A_D4': np.zeros((64, 272, 4)),
        'joint4': np.zeros((65, 272, 4)), 'pair4': np.ones((1, 272, 4))}
    root['single4'][0, 0, -1] = -10
    root['fixed_A_D4'][1, 0, -1] = -20
    root['joint4'][1, 0, -1], root['joint4'][2, 0, -1] = -10, 5
    monkeypatch.setattr(local_metrics, 'predict_pair4', lambda *args: (np.array([0]), np.zeros((1, 272, 4))))
    old = local_metrics.validate_model(None, [root], {}, {}, 'cpu', 'B_L4')
    new = v9.validate_model(None, [root], {}, {}, 'cpu', 'B_Match')
    assert old['regret'] == 0 and new['regret'] == 15
    assert new['records'][0]['selected'] == 2
    assert new['pair4_mae'] == old['pair4_mae'] == 1


def test_new_B_constructor_matches_v8_weights_and_rng(tmp_path):
    static = static_inputs()
    np.savez(tmp_path / 'static.npz', **static)
    source = tmp_path / 'source.pt'
    torch.save({'state_dict': FormalEffectModel(static).state_dict()}, source)
    atomic_json(tmp_path / 'protocol.json', {'source_A': {'42': {'path': str(source), 'sha256': sha256(source)}}})
    torch.manual_seed(42)
    old = local_metrics.make_model(tmp_path, 'B_L4', 42, 'cpu')
    state = torch.get_rng_state()
    for family in ('B_Match', 'B_Contrast'):
        torch.manual_seed(42)
        model = v9.make_model(tmp_path, family, 42, 'cpu')
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        for key, value in old.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_frozen_predictions_do_not_consume_rng_and_diagnostic_is_locked(tmp_path):
    root = {'root_id': 'r', 'split': 'validation'}
    path = tmp_path / 'fixed_A_D/seed_42/r.npy'
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((64, 272, 4)))
    atomic_json(tmp_path / 'fixed_A_D.ready.json', {'stage': 'complete', 'roots': 18,
        'files': {'fixed_A_D/seed_42/r.npy': sha256(path)}})
    before = torch.get_rng_state()
    v9.attach_fixed_predictions(tmp_path, [root], 42)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert not root['fixed_A_D4'].flags.writeable
    with pytest.raises(ValueError, match='locked'):
        v9.reuse_labels(tmp_path, test=True)
    with pytest.raises(ValueError, match='locked'):
        v9.load_roots(tmp_path, 'A_D_Top8', ('test',))


def test_v9_schedules_queue_order_and_zero_B_selection(tmp_path):
    jobs = training_order(True)
    assert len(jobs) == len(set(jobs)) == 12
    assert all(f.startswith('A_') for f, _ in jobs[:6])
    assert jobs[6:8] == [('B_Match', 42), ('B_Contrast', 42)]
    assert {v9.SCHEDULES[f]['updates'] * v9.SCHEDULES[f]['batch'] for f in v9.FAMILIES[:2]} == {4224000}
    assert {v9.SCHEDULES[f]['updates'] * v9.SCHEDULES[f]['batch'] for f in v9.FAMILIES[2:]} == {3840000}
    atomic_json(tmp_path / 'protocol.json', {'source_A': {'42': {'sha256': 'frozen-v5'}}})
    metrics = {'regret': 1., 'pair_total240_mae': 3., 'joint4_mae': 8.}
    model = torch.nn.Linear(1, 1)
    best = select_checkpoint(tmp_path, tmp_path, model, metrics, 'B_Match', 42, 0, 0, (float('inf'),) * 3)
    assert best == (1, 3, 0)
    saved = torch.load(tmp_path / 'best.pt', weights_only=False)
    assert saved['source_A_sha256'] == 'frozen-v5' and saved['family'] == 'B_Match'


def test_all_twelve_v9_checkpoints_locked_before_evaluation(tmp_path):
    for name in LOCKED_FILES:
        atomic_json(tmp_path / name, {'files': {}})
    for family in v9.FAMILIES:
        for seed in (42, 43, 44):
            path = tmp_path / family / f'seed_{seed}'
            path.mkdir(parents=True)
            torch.save({'seed': seed}, path / 'best.pt')
            atomic_json(path / 'complete.json', {'stage': 'complete', 'family': family, 'seed': seed,
                'updates': v9.SCHEDULES[family]['updates'], 'test_used': False,
                'checkpoint_sha256': sha256(path / 'best.pt')})
    lock_checkpoints(tmp_path)
    verify_lock(tmp_path)
    locked = json.loads((tmp_path / 'checkpoints_locked.json').read_text())
    assert len(locked['checkpoints']) == 12
    with pytest.raises(RuntimeError, match='retained'):
        lock_checkpoints(tmp_path)


def test_final_population_separates_selection18_remaining60_and_diagnostic12():
    rows = []
    for i in range(90):
        rows.append({'root_id': str(i), 'split': 'validation' if i < 78 else 'test',
            'cohort_id': 'c', 'flow_id': str(i // 6), 'policy': 'p', 'time_s': 600,
            'regret': i, 'joint4_mae': 1, 'joint_total240_mae': 2, 'optimal_choice': False,
            'worse_than_reference': False, 'excess_wait_vs_reference': 0, 'benefit_vs_reference': 1})
    result = split_results(rows, {str(i) for i in range(18)})
    assert [result[s]['root_count'] for s in ('validation', 'test', 'B_selection18', 'B_remaining60')] == [78, 12, 18, 60]
    assert result['validation']['independent_flow_count'] == 13
