import numpy as np
import pytest
import torch

from cityflow_tsc.train_single_revision import action_groups, action_contrast_loss, contrast_scale


def root(split="train"):
    labels = np.zeros((64, 2, 1), dtype=np.float32)
    labels[:, 0, 0] = np.tile([0, 10, 20, 30], 16)
    return {"split": split, "single_nodes": np.repeat(np.arange(16), 4)[:, None],
            "single_actions": np.tile(np.arange(1, 5), 16)[:, None], "single": labels}


def test_group_lookup_survives_query_reordering():
    r = root()
    order = np.random.default_rng(42).permutation(64)
    for key in ("single_nodes", "single_actions", "single"):
        r[key] = r[key][order]
    groups = action_groups([r, r])
    assert groups.shape == (32, 4)
    for group in groups:
        assert r["single_actions"][group % 64].ravel().tolist() == [1, 2, 3, 4]
        assert len(set(r["single_nodes"][group % 64].ravel())) == 1


def test_train_only_contrast_scale():
    assert contrast_scale([root()]) == 30.
    with pytest.raises(ValueError, match="training roots only"):
        contrast_scale([root("validation")])
    r = root()
    r["single"][:] = 0
    assert contrast_scale([r]) == 1.


def test_contrast_ignores_common_total_offset_but_penalizes_order_error():
    target = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)
    pred = (target + 99).clone().requires_grad_()
    assert action_contrast_loss(pred, target, torch.ones(1, 1), 1.).item() == 0.
    bad = -target.clone().requires_grad_()
    loss = action_contrast_loss(bad, target, torch.ones(1, 1), 1.)
    assert loss.item() > 0
    loss.backward()
    with pytest.raises(ValueError, match="four-action"):
        action_contrast_loss(pred[:3], target[:3], torch.ones(1, 1), 1.)


def test_grouped_full_model_forward_backward():
    from tests.test_formal_training import static, scales
    from cityflow_tsc.effect_model.formal_model import create_model, multiscale_loss
    from cityflow_tsc.effect_model.revision import sequence_bank
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = create_model(static())
    for parameter in model.pair_head.parameters():
        parameter.requires_grad_(False)
    bank, _ = sequence_bank(np.zeros(16, dtype=np.int64), np.ones((16, 5, 12), np.float32))
    encoded = model.encode(torch.zeros(2, 30, 272, 20), torch.tensor(np.stack([bank, bank])))
    root_ids = torch.repeat_interleave(torch.arange(2), 16)
    nodes = torch.arange(8).repeat_interleave(4)
    plans = torch.arange(1, 5).repeat(8)
    prediction, logits = model.single(encoded, root_ids, nodes, plans,
                                      torch.zeros(2, 16, dtype=torch.long), return_gate=True)
    target = torch.randn(32, 272, 48)
    loss = multiscale_loss(prediction, target, scales(), "A_ref", logits)
    loss = loss + .1 * action_contrast_loss(prediction, target, scales()["s5"], 100.)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.single_head.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in model.pair_head.parameters())
