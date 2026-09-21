import copy
from collections import Counter

import numpy as np
import pytest
import torch

from cityflow_tsc.train_coarse_effects import aggregate_effects, target_scales, CoarseEffectModel
from cityflow_tsc.collect_interaction_v6 import select_tasks, reconstruct, slice_metrics, link_file
from cityflow_tsc.effect_model.revision import sequence_bank


def test_signed_disjoint_aggregation_and_horizon_boundaries():
    data = np.zeros((2, 272, 48), dtype=np.int64)
    data[0, 0, 0] = 7
    data[0, 1, 1] = -2
    data[0, 239, 6] = 3
    data[0, 240, 17] = -4
    data[0, 256, 47] = 9
    result = aggregate_effects(data)
    assert result.shape == (2, 3, 4)
    np.testing.assert_array_equal(result[0], [[5, 8, 8, 8], [0, -4, -4, -4], [0, 0, 0, 9]])
    np.testing.assert_array_equal(result[:, :, -1].sum(1), data.sum((1, 2)))
    assert not result[1].any()


def test_scales_train_only_and_zero_fallback():
    data = np.zeros((64, 3, 4))
    data[:, 0, 0] = -10
    scales = target_scales([{'split': 'train', 'single_coarse': data}])
    assert scales[0, 0] == 10 and scales[2, 3] == 1
    with pytest.raises(ValueError):
        target_scales([{'split': 'validation', 'single_coarse': data}])


@pytest.mark.parametrize('pooling', ['early', 'late'])
def test_model_reference_anchor_hold_semantics_and_backward(pooling):
    torch.set_num_threads(2)
    static = {'adjacency': np.zeros((8, 272, 272), np.float32),
              'relative': np.zeros((16, 272, 11), np.float32),
              'permission': np.ones((16, 5, 272), np.float32),
              'rank_geometry': np.zeros((120, 5), np.float32)}
    model = CoarseEffectModel(static, pooling=pooling).eval()
    model.coarse_head[-1].bias.data.fill_(-2)
    bank, _ = sequence_bank(np.zeros(16, np.int64), np.ones((16, 5, 12), np.float32))
    encoded = model.encode(torch.zeros(1, 30, 272, 20), torch.tensor(bank[None]))
    prediction = model.coarse(encoded, torch.tensor([0, 0]), torch.tensor([0, 0]),
                             torch.tensor([0, 1]), torch.zeros(1, 16, dtype=torch.long))
    assert prediction.shape == (2, 3, 4)
    assert not prediction[0].any()
    assert torch.all(prediction[1] == -2)  # Holding current phase differs from future reference.
    prediction.sum().backward()
    assert model.coarse_head[-1].weight.grad is not None
    assert all(not p.requires_grad for p in model.single_head.parameters())


def population():
    tasks = []
    for split in ('train', 'validation'):
        for cohort in range(3):
            for flow in range(4):
                for policy in ('fixed_time', 'max_pressure'):
                    for t in (600, 1800, 2700):
                        tasks.append({'split': split, 'cohort_id': f'{split}_{cohort}', 'flow_id': f'{flow:03}',
                                      'policy': policy, 'time_s': t, 'backlog': flow * 100})
    return {'tasks': tasks}


def test_population_fixed_independent_of_outcomes_and_input_order():
    source = population()
    before = copy.deepcopy(source)
    picked = select_tasks(source)
    assert source == before
    assert Counter(t['split'] for t in picked) == {'train': 36, 'validation': 18}
    assert len(picked) * 1920 == 103680
    source['tasks'].reverse()
    for t in source['tasks']:
        t['backlog'] = -999
    again = select_tasks(source)
    key = lambda t: (t['split'], t['cohort_id'], t['flow_id'], t['time_s'], t['policy'])
    assert [key(t) for t in picked] == [key(t) for t in again]
    assert all(t['collected_branch_count'] == 2049 for t in picked)


def test_sparse_pair_reconstruction_and_menu_mapping():
    single = np.array([[[2.]], [[3.]]])
    pair = np.array([[[-8.]], [[20.]]])
    arrays = {'s_incidence': np.array([[0, 0], [1, 1], [1, 0]]),
              'p_incidence': np.array([[0, 0], [1, 0], [0, 1]]),
              'pair_query_pair_index': np.array([0, 1])}
    full = reconstruct(single, pair, arrays)
    sparse = reconstruct(single, pair, arrays, [0])
    np.testing.assert_array_equal(full.ravel(), [0, -3, 22])
    np.testing.assert_array_equal(sparse.ravel(), [0, -3, 2])
    assert slice_metrics(full, full, np.array([0, 2]))['selected'] == 0
    truth = np.array([[[0.]], [[3.]], [[-2.]]])
    assert slice_metrics(truth, truth, np.array([0, 2]))['selected'] == 2


def test_link_reuse_keeps_parent_unchanged(tmp_path):
    source = tmp_path / 'old' / 'part_0000.npz'
    source.parent.mkdir()
    source.write_bytes(b'old')
    target = tmp_path / 'new' / source.name
    link_file(source, target)
    (target.parent / 'part_0033.npz').write_bytes(b'new')
    assert source.read_bytes() == b'old'
    assert not (source.parent / 'part_0033.npz').exists()
