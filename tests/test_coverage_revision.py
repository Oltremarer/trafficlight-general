import copy

import numpy as np
import pytest
import torch

from cityflow_tsc.counterfactual.writer import atomic_json, sha256
from cityflow_tsc.effect_model import coverage_revision as v11, connection_revision as v10, local_metrics
from cityflow_tsc.effect_model.formal_model import FormalEffectModel, multiscale_loss
from cityflow_tsc.effect_model.local_revision import composed_contrast_loss, local_loss
from cityflow_tsc.evaluate_coverage_revision import LOCKED_FILES, lock_checkpoints
from cityflow_tsc.evaluate_local_revision import verify_lock
from tests.test_local_revision import static_inputs


def population():
    tasks = []
    for cohort in range(3):
        for flow in range(10):
            for policy in ('fixed_time', 'max_pressure'):
                for time in (600, 1800, 2700):
                    fid = f'c{cohort}_f{flow:02d}'
                    tasks.append(dict(root_id=f'{fid}_{policy}_{time}', flow_id=fid, cohort_id=str(cohort),
                                      policy=policy, time_s=time, split='train'))
    old = [t for t in tasks if t['flow_id'].endswith(('00', '01'))]
    return tasks, old


def test_selection_train_only_balanced_immutable_and_order_independent():
    tasks, old = population()
    before = copy.deepcopy(tasks)
    selected = v11.select_additional_tasks(tasks, old)
    assert tasks == before and len(selected) == 108
    assert len({t['flow_id'] for t in selected}) == 18
    assert not {t['root_id'] for t in old} & {t['root_id'] for t in selected}
    for c in ('0', '1', '2'):
        assert len([t for t in selected if t['cohort_id'] == c]) == 36
    extras = [dict(tasks[0], split='validation', flow_id='validation', root_id='validation')]
    assert v11.select_additional_tasks(list(reversed(tasks)) + extras, old) == selected
    assert all(t['collected_branch_count'] == 2049 for t in selected)
    with pytest.raises(ValueError):
        v11.select_additional_tasks([t for t in tasks if t['root_id'] != selected[0]['root_id']], old)


def test_margin_zero_at_truth_and_penalizes_reference_error_with_correct_gradient():
    groups = torch.arange(64).reshape(16, 4)
    y = torch.ones(64, 1, 1) * 2
    y[0] = -10
    p = torch.zeros_like(y, requires_grad=True)
    loss = v11.action_margin_loss(p, y, groups, 2)
    # node0 maximum gap12, other15 nodes gap2, then node mean and scale.
    assert float(loss.detach()) == pytest.approx((12 + 15 * 2) / 16 / 2)
    loss.backward()
    assert p.grad[0] > 0  # Descent lowers the genuinely best action prediction.
    assert v11.action_margin_loss(y, y, groups, 2) == 0
    assert v11.action_margin_loss(torch.zeros_like(y), torch.zeros_like(y), groups, 2) == 0
    permutation = torch.randperm(64)
    reverse = torch.argsort(permutation)
    torch.testing.assert_close(v11.action_margin_loss(p[permutation], y[permutation], reverse[groups], 2), loss)


def test_loss_is_exact_old_A_plus_margin_and_B_unchanged():
    torch.manual_seed(1)
    p = torch.randn(64, 272, 48)
    y, gate = torch.randn_like(p), torch.randn_like(p)
    scales = {'s5': torch.ones(272, 48) * 2, 'sp': torch.ones(272, 4) * 3, 'sj': torch.ones(4) * 5}
    incidence = torch.cat((torch.zeros(1, 64), torch.eye(64)))
    groups = torch.arange(64).reshape(16, 4)
    expected = (multiscale_loss(p, y, scales, 'A_ref', gate)
        + .1 * composed_contrast_loss(p * 2, y, incidence, 7)
        + .05 * v11.action_margin_loss(p * 2, y, groups, 5))
    torch.testing.assert_close(v11.training_loss('A_ActionMargin', p, y, scales, gate, incidence, 7, None, groups), expected)
    p, y, gate = p[:32, :, :4], y[:32, :, :4], gate[:32, :, :4]
    expected = local_loss(p, y, scales['sp'], scales['sj'], gate)
    torch.testing.assert_close(v11.training_loss('B_Coverage', p, y, scales, gate, None, 7, None), expected, rtol=0, atol=0)


def test_fresh_constructors_match_previous_weights_and_rng(tmp_path):
    static = static_inputs()
    np.savez(tmp_path / 'static.npz', **static)
    torch.save({'state_dict': FormalEffectModel(static).state_dict()}, tmp_path / 'source.pt')
    atomic_json(tmp_path / 'protocol.json', {'source_A': {'42': {'path': str(tmp_path / 'source.pt'), 'sha256': sha256(tmp_path / 'source.pt')}}})
    for family, old_family, old_factory in (('A_ActionMargin', 'A_D', local_metrics.make_model),
                                           ('B_Coverage', 'B_Adapter', v10.make_model)):
        torch.manual_seed(42)
        old = old_factory(tmp_path, old_family, 42, 'cpu')
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        new = v11.make_model(tmp_path, family, 42, 'cpu')
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)


def test_single_metric_uses_reference_and_node_action_mapping():
    nodes = np.repeat(np.arange(16), 4)
    actions = np.tile(np.arange(1, 5), 16)
    y = np.zeros((64, 272, 4))
    y[0, 0, -1] = -10
    root = {'single_nodes': nodes[:, None], 'single_actions': actions[:, None], 'single4': y}
    metric = v11.single_choice_metrics(root, np.zeros_like(y))
    assert metric['single_action_selected'] == [0] * 16
    assert metric['single_action_regret'] == 10 / 16
    assert v11.single_choice_metrics(root, y)['single_action_regret'] == 0


def test_six_complete_jobs_required_before_lock(tmp_path):
    for name in LOCKED_FILES:
        atomic_json(tmp_path / name, {'files': {}})
    for family in v11.FAMILIES:
        for seed in (42, 43, 44):
            path = tmp_path / family / f'seed_{seed}'
            path.mkdir(parents=True)
            torch.save({'seed': seed}, path / 'best.pt')
            atomic_json(path / 'complete.json', {'stage': 'complete', 'family': family, 'seed': seed,
                'updates': v11.SCHEDULES[family]['updates'], 'test_used': False,
                'checkpoint_sha256': sha256(path / 'best.pt')})
    lock_checkpoints(tmp_path)
    verify_lock(tmp_path, v11.FAMILIES, LOCKED_FILES)
    assert v11.SCHEDULES['A_ActionMargin']['updates'] == 66000
    assert v11.SCHEDULES['B_Coverage']['updates'] == 30000
