import numpy as np
import pytest
import torch

from cityflow_tsc.train_coarse_effects import CoarseEffectModel
from cityflow_tsc.train_coarse_latepool import copy_frozen, link_frozen, prepare_cache


def static_inputs():
    relative = np.zeros((16, 272, 11), np.float32)
    relative[0, 0, 0], relative[0, 1, 0] = 1, -1
    return {'adjacency': np.zeros((8, 272, 272), np.float32), 'relative': relative,
        'permission': np.ones((16, 5, 272), np.float32),
        'rank_geometry': np.zeros((120, 5), np.float32)}


def test_late_pool_preserves_state_position_association_without_extra_parameters():
    torch.set_num_threads(1)
    early = CoarseEffectModel(static_inputs()).eval()
    late = CoarseEffectModel(static_inputs(), pooling='late').eval()
    with torch.no_grad():
        for parameter in early.coarse_head.parameters():
            parameter.zero_()
        early.coarse_head[0].weight[0, 0] = 1  # Recipient state.
        early.coarse_head[0].weight[0, 256] = 1  # Its relative geometry.
        early.coarse_head[3].weight[0, 0] = 1
        early.coarse_head[6].weight[0, 0] = 1
        late.load_state_dict(early.state_dict())
    assert sum(p.numel() for p in early.parameters()) == sum(p.numel() for p in late.parameters())
    encoded = {'objects': torch.zeros(1, 272, 64), 'global': torch.zeros(1, 64),
        'actions': torch.zeros(1, 16, 5, 64), 'plan_changed': torch.ones(1, 16, 5, dtype=torch.bool)}
    encoded['objects'][0, 0, 0] = 2
    moved = {**encoded, 'objects': encoded['objects'].clone()}
    moved['objects'][:, :2] = moved['objects'][:, :2].flip(1)
    args = (torch.tensor([0]), torch.tensor([0]), torch.tensor([1]), torch.zeros(1, 16, dtype=torch.long))
    with torch.no_grad():
        torch.testing.assert_close(early.coarse(encoded, *args), early.coarse(moved, *args))
        before, after = late.coarse(encoded, *args), late.coarse(moved, *args)
        assert abs(float(before[0, 0, 0] - after[0, 0, 0])) > .001
        # Merely renaming both recipient state and geometry must not change output.
        late.relative[:, :2] = late.relative[:, :2].flip(1)
        torch.testing.assert_close(before, late.coarse(moved, *args))


def test_frozen_cache_reuse_does_not_write_parent(tmp_path):
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    parent.mkdir()
    source = parent / 'input.npz'
    source.write_bytes(b'immutable data')
    digest = link_frozen(source, child / 'data.npz')
    assert digest == link_frozen(source, child / 'data.npz')
    assert (child / 'data.npz').is_symlink()
    assert copy_frozen(source, child / 'private.npz') == digest
    (child / 'private.npz').write_bytes(b'child changed')
    assert source.read_bytes() == b'immutable data'
    assert list(parent.iterdir()) == [source]
    with pytest.raises(ValueError):
        copy_frozen(source, child / 'private.npz')
    with pytest.raises(ValueError):
        link_frozen(source, child / 'private.npz')


def test_diagnostic_cache_requires_new_checkpoint_lock_before_source_read(tmp_path):
    with pytest.raises(ValueError, match='locked before diagnostic'):
        prepare_cache(tmp_path / 'new', tmp_path / 'missing_source', test=True)
    assert not (tmp_path / 'new').exists()


def test_invalid_pooling_fails():
    with pytest.raises(ValueError, match='early or late'):
        CoarseEffectModel(static_inputs(), pooling='invalid')
