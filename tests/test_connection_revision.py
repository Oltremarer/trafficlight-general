import numpy as np
import pytest
import torch

from cityflow_tsc.effect_model import connection_revision as v10
from cityflow_tsc.effect_model.formal_model import FormalEffectModel
from cityflow_tsc.effect_model.local_revision import LocalPairModel, local_loss
from cityflow_tsc.effect_model.revision import sequence_bank
from cityflow_tsc.evaluate_connection_revision import (chosen_metrics, factor_attribution,
    LOCKED_FILES, lock_checkpoints, verify_lock)
from cityflow_tsc.counterfactual.writer import atomic_json, sha256
from tests.test_local_revision import static_inputs, encoded_inputs


def connected_static():
    static = static_inputs()
    static['relative'][0, :2, 5] = 1
    static['relative'][1, :2, 4] = 1
    static['relative'][1, 2, 5] = 1
    static['relative'][0, 2, 4] = 1
    return static


def test_directed_lane_pool_and_missing_links_are_exact_zero():
    weights, present = v10.connection_pool(connected_static()['relative'])
    np.testing.assert_array_equal(weights[0, 1, :3], [.5, .5, 0])
    np.testing.assert_array_equal(weights[1, 0, :3], [0, 0, 1])
    assert present[0, 1, 0] == present[1, 0, 0] == 1
    assert weights[0, 2].sum() == present[0, 2, 0] == 0


def test_connection_context_uses_only_observed_history_and_correct_direction():
    static = connected_static()
    source = FormalEffectModel(static)
    model = v10.ConnectionPairModel(static, source.state_dict()).eval()
    history = torch.randn(1, 30, 272, 20)
    bank, _ = sequence_bank(np.zeros(16, dtype=np.int64), np.zeros((16, 5, 12)))
    with torch.no_grad():
        encoded = model.encode(history, torch.tensor(bank[None]))
    context = encoded['connection_context']
    assert context.shape == (1, 16, 16, 105)
    torch.testing.assert_close(context[0, 0, 1, :64], encoded['objects'][0, :2].mean(0))
    torch.testing.assert_close(context[0, 0, 1, 64:84], history[0, -1, :2].mean(0))
    torch.testing.assert_close(context[0, 0, 1, 84:104], history[0, :, :2].mean((0, 1)))
    assert context[0, 0, 1, -1] == 1 and context[0, 0, 2].count_nonzero() == 0


def test_adapter_symmetry_reference_mask_frozen_A_and_gradient():
    torch.set_num_threads(1)
    static = connected_static()
    source = FormalEffectModel(static)
    model = v10.ConnectionPairModel(static, source.state_dict())
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any(n.startswith('connection_adapter.') for n in trainable)
    assert all(n.startswith(('pair_head.', 'pair_gate.', 'connection_adapter.')) for n in trainable)
    model.train()
    assert not model.gru.training and not model.single_gate.training and not model.action_encoder.training
    encoded = encoded_inputs()
    encoded['connection_context'] = torch.randn(1, 16, 16, 105)
    roots, nodes = torch.tensor([0, 0]), torch.tensor([[0, 1], [0, 2]])
    plans, base = torch.tensor([[1, 4], [0, 2]]), torch.zeros(1, 16, dtype=torch.long)
    with torch.no_grad():
        model.pair_head[-1].weight.normal_()
        model.connection_adapter[-1].weight.normal_()
    model.eval()
    forward, logits = model.pair(encoded, roots, nodes, plans, base, True)
    reverse = model.pair(encoded, roots, nodes.flip(1), plans.flip(1), base)
    torch.testing.assert_close(forward, reverse)
    assert forward[1].count_nonzero() == 0
    (forward.square().mean() + .01 * logits.square().mean()).backward()
    assert model.connection_adapter[-1].weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    for name, value in source.state_dict().items():
        if not name.startswith('pair_head.'):
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_zero_adapter_starts_as_same_seed_B_heads_and_loss_unchanged():
    static = connected_static()
    state = FormalEffectModel(static).state_dict()
    torch.manual_seed(42)
    old = LocalPairModel(static, state).eval()
    torch.manual_seed(42)
    new = v10.ConnectionPairModel(static, state).eval()
    for name, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
    assert new.connection_adapter(torch.randn(3, 466)).count_nonzero() == 0
    p, y, logits = (torch.randn(2, 272, 4) for _ in range(3))
    scales = {'sp': torch.ones(272, 4), 'sj': torch.ones(4)}
    torch.testing.assert_close(v10.training_loss('B_Adapter', p, y, scales, logits),
                              local_loss(p, y, scales['sp'], scales['sj'], logits), rtol=0, atol=0)


def test_consensus_accepts_only_unanimous_improvement_and_keeps_reference_ties():
    singles = np.array([[0, -20, -10]] * 3, dtype=np.float64)
    joint = np.array([[0, -10, -30], [0, -10, -20], [0, -10, -11]], dtype=np.float64)
    result = v10.consensus_choice(singles, joint)
    assert result['selected'] == 2 and result['accepted_switch']
    joint[2, 2] = -9
    result = v10.consensus_choice(singles, joint)
    assert result['proposed_selected'] == 2 and result['selected'] == 1 and not result['accepted_switch']
    assert v10.consensus_choice(np.zeros((3, 4)), np.zeros((3, 4)))['selected'] == 0
    joint[2, 2] = np.nan
    with pytest.raises(FloatingPointError):
        v10.consensus_choice(singles, joint)


def test_consensus_reports_actual_choice_without_rewriting_predicted_scores():
    predicted, truth = np.array([0, -10, -30]), np.array([0, -40, -20])
    result = chosen_metrics(predicted, truth, 1)
    assert result['selected'] == 1 and result['regret'] == 0 and result['optimal_choice']
    assert result['predicted_delta_J'] == -10 and result['true_delta_J'] == -40
    assert result['benefit_vs_reference'] == 40 and result['benefit_capture'] == 1


def test_factor_attribution_signed_identity_and_true_factor_substitutions():
    root = {k: 'r' for k in ('root_id', 'split', 'flow_id', 'cohort_id', 'policy')}
    root.update(time_s=600, s_incidence=np.eye(3), p_incidence=np.eye(3),
                single4=np.zeros((3, 1, 4)), pair4=np.zeros((3, 1, 4)), joint4=np.zeros((3, 1, 4)))
    root['single4'][:, 0, -1] = [0, -20, -10]
    root['pair4'][:, 0, -1] = [0, 5, -15]
    root['joint4'][:, 0, -1] = [0, -12, -30]
    predicted_s, predicted_p = root['single4'].copy(), np.zeros((3, 1, 4))
    result = factor_attribution(root, predicted_s, predicted_p)
    assert result['true_regret'] == 18 and result['True_S']['regret'] == 18
    assert result['True_SP']['regret'] == 0
    assert result['predicted_gap'] - result['true_regret'] == pytest.approx(
        result['single_gap_error'] + result['pair_gap_error'] - result['higher_order_gap'])


def test_v10_locks_all_three_checkpoints_and_retains_artifacts(tmp_path):
    for name in LOCKED_FILES:
        atomic_json(tmp_path / name, {'files': {}})
    for seed in (42, 43, 44):
        directory = tmp_path / 'B_Adapter' / f'seed_{seed}'
        directory.mkdir(parents=True)
        torch.save({'seed': seed}, directory / 'best.pt')
        atomic_json(directory / 'complete.json', {'stage': 'complete', 'family': 'B_Adapter', 'seed': seed,
            'updates': 30000, 'test_used': False, 'checkpoint_sha256': sha256(directory / 'best.pt')})
    lock_checkpoints(tmp_path)
    verify_lock(tmp_path)
    with pytest.raises(RuntimeError, match='retained'):
        lock_checkpoints(tmp_path)
    assert v10.SCHEDULES['B_Adapter']['batch'] == 128
