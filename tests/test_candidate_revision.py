import numpy as np
import pytest
import torch

from cityflow_tsc.counterfactual.writer import atomic_json, sha256
from cityflow_tsc.effect_model import candidate_revision as v13, local_metrics
from cityflow_tsc.effect_model.formal_model import multiscale_loss
from cityflow_tsc.effect_model.local_revision import composed_contrast_loss
from cityflow_tsc.evaluate_balanced_revision import LOCKED_FILES, lock_checkpoints
from cityflow_tsc.evaluate_local_revision import verify_lock
from tests.test_local_revision import static_inputs


def test_candidate_margin_matches_cost_augmented_formula_and_regret_bound():
    torch.manual_seed(9)
    y = torch.randn(64, 2, 3)
    p = torch.randn_like(y, requires_grad=True)
    incidence = torch.cat((torch.zeros(1, 64), torch.randint(0, 2, (64, 64)).float()))
    pred, truth = incidence @ p.sum((1, 2)), incidence @ y.sum((1, 2))
    best = truth.argmin()
    expected = (truth - truth[best] + pred[best] - pred).max() / 7
    actual = v13.candidate_margin_loss(p, y, incidence, 7)
    torch.testing.assert_close(actual, expected)
    assert actual >= (truth[pred.argmin()] - truth[best]) / 7 - 1e-6
    actual.backward()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize('zero', (False, True))
def test_perfect_prediction_has_zero_loss_and_zero_auxiliary_gradient(zero):
    torch.manual_seed(5)
    y = torch.zeros(64, 1, 1) if zero else torch.randn(64, 1, 1)
    p = y.clone().requires_grad_(True)
    incidence = torch.cat((torch.zeros(1, 64), torch.eye(64)))
    loss = v13.candidate_margin_loss(p, y, incidence, 7)
    assert loss == 0
    loss.backward()
    assert torch.count_nonzero(p.grad) == 0


def test_joint_candidate_not_individual_action_and_query_permutation():
    y = torch.zeros(64, 1, 1)
    y[0], y[4], y[1] = -4, -5, -6
    p = torch.zeros_like(y, requires_grad=True)
    incidence = torch.zeros(65, 64)
    incidence[1, [0, 4]], incidence[2, 1] = 1, 1
    loss = v13.candidate_margin_loss(p, y, incidence, 1)
    assert loss == 9  # Joint candidate {-4,-5} beats the individual -6 query.
    loss.backward()
    assert p.grad[0] > 0 and p.grad[4] > 0 and p.grad[1] == 0
    permutation = torch.randperm(64)
    torch.testing.assert_close(v13.candidate_margin_loss(p[permutation], y[permutation], incidence[:, permutation], 1), loss)
    with pytest.raises(ValueError):
        v13.candidate_margin_loss(p[:63], y[:63], incidence, 1)


def test_training_loss_keeps_old_base_and_output_gradient_bound():
    torch.manual_seed(1)
    p = torch.randn(64, 272, 48, requires_grad=True)
    y, gate = torch.randn_like(p), torch.randn_like(p)
    scales = {'s5': torch.ones(272, 48)*2, 'sp': torch.ones(272, 4)*3, 'sj': torch.ones(4)*5}
    incidence = torch.cat((torch.zeros(1, 64), torch.eye(64)))
    base = multiscale_loss(p, y, scales, 'A_ref', gate) + .1*composed_contrast_loss(p*2, y, incidence, 7)
    aux = v13.candidate_margin_loss(p*2, y, incidence, 7)
    loss, info = v13.training_loss(v13.FAMILIES[0], p, y, scales, gate, incidence, 7, None)
    torch.testing.assert_close(loss, base + info['margin_weight']*aux)
    assert info['weighted_gradient_ratio'] <= .100001
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_fresh_initialization_and_budget_match_v12(tmp_path):
    np.savez(tmp_path/'static.npz', **static_inputs())
    atomic_json(tmp_path/'protocol.json', {'source_A': {'42': {}}})
    torch.manual_seed(42)
    old = local_metrics.make_model(tmp_path, 'A_D', 42, 'cpu')
    state = torch.get_rng_state()
    torch.manual_seed(42)
    new = v13.make_model(tmp_path, v13.FAMILIES[0], 42, 'cpu')
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    for name, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
    assert v13.SCHEDULES[v13.FAMILIES[0]] == {'updates': 66000, 'batch': 64, 'warmup': 1100, 'validate': 1100, 'log': 220}


def test_additive_diagnostic_does_not_substitute_joint_truth():
    single, joint = np.zeros((64, 272, 4)), np.zeros((65, 272, 4))
    single[0, 0, -1], single[1, 0, -1] = -10, -5
    joint[1, 0, -1], joint[2, 0, -1] = 20, -3
    incidence = np.r_[np.zeros((1, 64)), np.eye(64)]
    root = {'single4': single, 'joint4': joint, 's_incidence': incidence}
    metric = v13.additive_choice_metrics(root, 2)
    assert metric == {'additive_regret': 5., 'additive_optimal_choice': False, 'true_single_oracle_joint_regret': 23.}


def test_v13_lock_requires_exact_three_new_checkpoints(tmp_path):
    for name in LOCKED_FILES:
        atomic_json(tmp_path/name, {})
    for seed in (42, 43, 44):
        family = v13.FAMILIES[0]
        path = tmp_path/family/f'seed_{seed}'
        path.mkdir(parents=True)
        torch.save({'seed': seed}, path/'best.pt')
        atomic_json(path/'complete.json', {'stage': 'complete', 'family': family, 'seed': seed,
            'updates': 66000, 'test_used': False, 'checkpoint_sha256': sha256(path/'best.pt')})
    lock_checkpoints(tmp_path, v13)
    verify_lock(tmp_path, v13.FAMILIES, LOCKED_FILES)
