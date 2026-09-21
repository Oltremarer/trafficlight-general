import json

import numpy as np
import pytest
import torch

from cityflow_tsc.effect_model.formal_model import FormalEffectModel
from cityflow_tsc.effect_model.local_data import convert_file, fit_contrast_scale, load_roots, prepare_labels, prepare_one
from cityflow_tsc.effect_model.local_metrics import field_record, predict_pair4, predict_single4, summarize
from cityflow_tsc.effect_model.local_revision import (SCHEDULES, LocalSingleModel, LocalPairModel,
    compose, composed_contrast_loss, cumulative4, local_loss, pair_batches, pair_groups, pair_indices, root_batches)
from cityflow_tsc.effect_model.revision import sequence_bank
from cityflow_tsc.evaluate_local_revision import lock_checkpoints, verify_lock, LOCKED_FILES
from cityflow_tsc.run_local_revision import dependency_ready, resource_allows
from cityflow_tsc.train_local_revision import select_checkpoint


def static_inputs():
    geometry = np.zeros((120, 5), np.float32)
    geometry[:24, -1] = 1
    return {'adjacency': np.zeros((8, 272, 272), np.float32),
        'relative': np.zeros((16, 272, 11), np.float32),
        'permission': np.ones((16, 5, 272), np.float32), 'rank_geometry': geometry}


def encoded_inputs():
    torch.set_num_threads(1)
    changed = torch.ones(1, 16, 5, dtype=torch.bool)
    changed[:, :, 0] = False
    return {'objects': torch.randn(1, 272, 64), 'global': torch.randn(1, 64),
        'actions': torch.randn(1, 16, 5, 64), 'plan_changed': changed}


def test_cumulative_horizons_preserve_cancellation_and_total():
    x = np.zeros((2, 272, 48), dtype=np.float32)
    x[0, 0, [0, 5, 6, 17, 18, 35, 36, 47]] = [10, -2, -3, 4, -5, 6, -7, 8]
    x[0, 1] = -x[0, 0]
    y = cumulative4(x)
    np.testing.assert_array_equal(y[0, 0], [8, 9, 10, 11])
    assert y[0, 1, -1] == -11
    np.testing.assert_array_equal(y[:, :, -1].sum(1), x.sum((1, 2), dtype=np.float64))
    np.testing.assert_array_equal(cumulative4(torch.from_numpy(x)).numpy(), y)
    with pytest.raises(ValueError):
        cumulative4(x[..., :4])


def test_local_loss_natural_mean_uses_physical_scales_and_nonzero_targets():
    p = torch.zeros(1, 272, 4, requires_grad=True)
    target = torch.zeros_like(p)
    target[0, 0, -1], target[0, 1, -1] = 2, -2
    scale = torch.full((272, 4), 2.)
    logits = torch.zeros_like(p, requires_grad=True)
    loss = local_loss(p, target, scale, torch.ones(4), logits)
    assert float(loss.detach()) == pytest.approx(2 / (272 * 4) + .01 * np.log(2))
    loss.backward()
    assert p.grad[0, 0, -1] < 0 < p.grad[0, 1, -1]
    assert logits.grad[0, 0, -1] < 0 < logits.grad[0, 2, -1]


def test_single_output_zero_reference_signed_and_full_action_schedule():
    encoded = encoded_inputs()
    model = LocalSingleModel(static_inputs()).eval()
    with torch.no_grad():
        model.single_head[-1].bias.fill_(-2)
    root = torch.tensor([0, 0])
    nodes, plans, base = torch.tensor([0, 0]), torch.tensor([0, 1]), torch.zeros(1, 16, dtype=torch.long)
    result = model.single(encoded, root, nodes, plans, base)
    assert result.shape == (2, 272, 4)
    assert torch.count_nonzero(result[0]) == 0
    torch.testing.assert_close(result[1], torch.full((272, 4), -1.))
    bank, requests = sequence_bank(np.zeros(16, dtype=np.int64), np.zeros((16, 5, 12)))
    actual = model.encode(torch.zeros(1, 30, 272, 20), torch.tensor(bank[None]))
    assert requests[0, 0, 0] == requests[0, 1, 0] == 0
    assert actual['plan_changed'][0, 0, 1] and not actual['plan_changed'][0, 0, 0]


def test_pair_symmetry_reference_zero_and_only_new_heads_train():
    encoded = encoded_inputs()
    source = FormalEffectModel(static_inputs())
    model = LocalPairModel(static_inputs(), source.state_dict())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all(n.startswith(('pair_head.', 'pair_gate.')) for n in names)
    model.train()
    assert not model.gru.training and not model.single_gate.training and not model.action_encoder.training
    assert model.pair_head.training and model.pair_gate.training
    with torch.no_grad():
        model.pair_head[-1].weight.normal_()
    model.eval()
    nodes, actions, roots = torch.tensor([[0, 1], [0, 2]]), torch.tensor([[1, 4], [0, 2]]), torch.tensor([0, 0])
    base = torch.zeros(1, 16, dtype=torch.long)
    forward, gate = model.pair(encoded, roots, nodes, actions, base, return_gate=True)
    reverse = model.pair(encoded, roots, nodes.flip(1), actions.flip(1), base)
    torch.testing.assert_close(forward, reverse)
    assert forward.shape == (2, 272, 4) and torch.count_nonzero(forward[1]) == 0
    (forward.square().mean() + .01 * gate.square().mean()).backward()
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith(('pair_head.', 'pair_gate.')))
    for name, value in source.state_dict().items():
        if not name.startswith('pair_head.'):
            torch.testing.assert_close(model.state_dict()[name], value)


def test_composed_contrast_is_all_2080_true_single_score_gaps():
    torch.manual_seed(42)
    p = torch.randn(64, 2, 3, requires_grad=True)
    y = torch.randn_like(p)
    incidence = torch.randint(0, 2, (65, 64)).float()
    incidence[0] = 0
    ps, ys = incidence @ p.sum((1, 2)), incidence @ y.sum((1, 2))
    i, j = torch.triu_indices(65, 65, 1)
    assert len(i) == 2080
    expected = torch.nn.functional.smooth_l1_loss((ps[i] - ps[j]) / 10, (ys[i] - ys[j]) / 10)
    actual = composed_contrast_loss(p, y, incidence, 10)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert p.grad.abs().sum() > 0
    # Perfect true singles have zero contrast loss, independent of any joint outcome.
    assert composed_contrast_loss(y, y, incidence, 10) == 0
    with pytest.raises(ValueError):
        composed_contrast_loss(p[:-1], y[:-1], incidence, 10)


def test_matched_root_batches_and_equal_A_query_budgets():
    first, again = list(root_batches(3, 42, 7)), list(root_batches(3, 42, 7))
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(first, again))
    for _, _, indices in first:
        assert len(set(indices // 64)) == 1
        np.testing.assert_array_equal(indices % 64, np.arange(64))
    assert len(set(np.concatenate([i for _, _, i in first[:3]]))) == 192
    assert {SCHEDULES[f]['batch'] * SCHEDULES[f]['updates'] for f in ('A_C', 'A_D', 'A_L4')} == {4224000}


def pair_root():
    return {'pair_query_pair_index': np.repeat(np.arange(120), 16),
            'pair_actions': np.tile(np.asarray([(a, b) for a in range(1, 5) for b in range(1, 5)]), (120, 1))}


def test_pair_groups_complete_and_adjacent_budget_selected_before_decode():
    root = pair_root()
    groups = pair_groups([root, root])
    assert groups.shape == (240, 16)
    seen = []
    for _, _, ix in pair_batches(groups, 42, 30):
        assert ix.shape == (128,)
        assert np.all(ix.reshape(8, 16) % 16 == np.arange(16))
        seen.extend(ix)
    assert len(set(seen)) == 3840
    geometry = static_inputs()['rank_geometry']
    assert len(pair_indices(root, geometry, 0)) == 0
    np.testing.assert_array_equal(pair_indices(root, geometry, 24), np.arange(384))
    assert len(pair_indices(root, geometry, 120)) == 1920
    with pytest.raises(ValueError):
        pair_indices(root, geometry, 32)


def test_sparse_composition_preserves_signed_factors():
    rng = np.random.default_rng(9)
    incidence = rng.integers(-1, 2, (65, 12))
    effects = rng.normal(size=(12, 272, 4))
    actual = compose(incidence, effects)
    expected = (incidence @ effects.reshape(12, -1)).reshape(65, 272, 4)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual.dtype == np.float64


def test_inference_never_reads_targets_and_counts_only_selected_pair_queries():
    encoded_inputs()
    static = static_inputs()
    a = LocalSingleModel(static).eval()
    b = LocalPairModel(static, FormalEffectModel(static).state_dict()).eval()
    bank, _ = sequence_bank(np.zeros(16, dtype=np.int64), np.zeros((16, 5, 12)))
    root = {**pair_root(), 'normalized_history': np.zeros((30, 272, 20), np.float32),
        'base_phase': np.zeros(16, np.int64), 'action_bank': bank,
        'single_nodes': np.repeat(np.arange(16), 4)[:, None], 'single_actions': np.tile(np.arange(1, 5), 16)[:, None],
        'pair_nodes': np.tile(np.array([[0, 1]]), (1920, 1))}
    # No single4, pair4, joint4 or future state is present.
    scales = {'sp': torch.ones(272, 4)}
    assert predict_single4(a, root, scales, 'cpu', True).shape == (64, 272, 4)
    calls = []
    hook = b.pair_head.register_forward_hook(lambda module, args, output: calls.append(len(output)))
    indices, fields = predict_pair4(b, root, scales, 'cpu', 24)
    hook.remove()
    assert fields.shape == (384, 272, 4) and len(indices) == 384
    assert calls == [128] * 6  # Two symmetric decodes, three chunks, never all 1920 queries.
    assert predict_pair4(b, {}, scales, 'cpu', 0)[1].shape == (0, 272, 4)


def test_private_cumulative_cache_does_not_modify_source(tmp_path):
    source = tmp_path / 'source.npy'
    x = np.zeros((3, 272, 48), np.float32)
    x[0, 0, 0], x[0, 0, 47] = 10, -12
    np.save(source, x)
    before = source.read_bytes()
    target = tmp_path / 'new/single4.npy'
    result = convert_file(source, target)
    assert source.read_bytes() == before and not target.is_symlink()
    assert result['shape'] == [3, 272, 4]
    np.testing.assert_array_equal(np.load(target), cumulative4(x))


def test_train_only_contrast_scale_and_pre_read_diagnostic_lock(tmp_path):
    with pytest.raises(ValueError, match='training roots only'):
        fit_contrast_scale([{'split': 'validation'}], tmp_path / 'nonexistent')
    (tmp_path / 'data').mkdir()
    singles = np.zeros((64, 3, 4))
    singles[:, 0, -1] = np.arange(64)
    incidence = np.vstack([np.zeros((1, 64)), np.eye(64)])
    np.savez(tmp_path / 'data/r.npz', single_coarse=singles, s_incidence=incidence)
    values = incidence @ singles[:, :, -1].sum(1)
    i, j = np.triu_indices(65, 1)
    gaps = abs(values[i] - values[j])
    assert fit_contrast_scale([{'root_id': 'r', 'split': 'train'}], tmp_path) == np.percentile(gaps[gaps > 0], 95)
    with pytest.raises(ValueError, match='locked'):
        load_roots(tmp_path, 'A_L4', ('test',))
    with pytest.raises(ValueError, match='locked'):
        prepare_labels(tmp_path, test=True)


def test_B_checkpoint_selection_includes_zero_and_secondary_pair_total(tmp_path):
    protocol = {'source_A': {'42': {'sha256': 'source'}}}
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    model = torch.nn.Linear(1, 1)
    metrics = {'regret': 2., 'pair_total240_mae': 3., 'joint4_mae': 1.}
    best = select_checkpoint(tmp_path, tmp_path, model, metrics, 'B_L4', 42, 0, 0, (float('inf'),) * 3)
    assert best == (2, 3, 0)
    assert select_checkpoint(tmp_path, tmp_path, model, metrics, 'B_L4', 42, 1000, 1, best) == best
    saved = torch.load(tmp_path / 'best.pt', weights_only=False)
    assert saved['updates'] == 0 and saved['source_A_sha256'] == 'source' and not saved['test_used']


def test_resource_limits_dependencies_and_startup_reservations(tmp_path):
    assert dependency_ready(tmp_path, 'A_C') and dependency_ready(tmp_path, 'A_D')
    assert not dependency_ready(tmp_path, 'B_L4') and not dependency_ready(tmp_path, 'A_L4')
    (tmp_path / 'labels_B_L4.ready.json').write_text('{}')
    assert dependency_ready(tmp_path, 'B_L4') and not dependency_ready(tmp_path, 'A_L4')
    free = {'host_available_gib': 27, 'gpu_free_gib': 31}
    assert resource_allows(free, 5, 5) and resource_allows(free, 7, 0)
    assert not resource_allows(free, 8, 0)
    assert not resource_allows({'host_available_gib': 8, 'gpu_free_gib': 30}, 2, 0)
    assert not resource_allows({'host_available_gib': 27, 'gpu_free_gib': 20}, 5, 5)


def test_lock_requires_all_twelve_fixed_budget_completions(tmp_path):
    from cityflow_tsc.counterfactual.writer import sha256
    with pytest.raises(FileNotFoundError):
        lock_checkpoints(tmp_path)
    for name in LOCKED_FILES:
        (tmp_path / name).write_text('{}')
    for family, schedule in SCHEDULES.items():
        for seed in (42, 43, 44):
            directory = tmp_path / family / f'seed_{seed}'
            directory.mkdir(parents=True)
            (directory / 'best.pt').write_bytes(b'checkpoint placeholder')
            done = {'stage': 'complete', 'family': family, 'seed': seed, 'updates': schedule['updates'],
                    'test_used': False, 'checkpoint_sha256': sha256(directory / 'best.pt')}
            (directory / 'complete.json').write_text(json.dumps(done))
    lock_checkpoints(tmp_path)
    verify_lock(tmp_path)
    (tmp_path / 'A_D/seed_43/best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        verify_lock(tmp_path)
