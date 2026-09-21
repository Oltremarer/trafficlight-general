import numpy as np
import pytest
import torch

from cityflow_tsc.counterfactual.writer import atomic_json
from cityflow_tsc.effect_model import balanced_revision as v12, coverage_revision as v11, local_metrics
from cityflow_tsc.effect_model.formal_model import multiscale_loss
from cityflow_tsc.effect_model.local_revision import composed_contrast_loss
from tests.test_local_revision import static_inputs


def test_balanced_gradient_bound_and_no_second_order_weight():
    p=torch.tensor([1.,2.,3.],requires_grad=True)
    base=(p**2).mean()
    auxiliary=1000*p.sum()
    g0=torch.autograd.grad(base,p,retain_graph=True)[0]
    g1=torch.autograd.grad(auxiliary,p,retain_graph=True)[0]
    loss,info=v12.balance_losses(base,auxiliary,p)
    actual=torch.autograd.grad(loss,p)[0]
    torch.testing.assert_close(actual,g0+info['margin_weight']*g1)
    assert info['margin_weight']<.05
    assert info['weighted_gradient_ratio']==pytest.approx(.1)
    assert (actual-g0).norm()<=g0.norm()*.100001


def test_zero_aux_and_zero_base_are_finite():
    for base_zero in (True,False):
        p=torch.tensor([1.],requires_grad=True)
        base=p.sum()*0 if base_zero else p.square().sum()
        loss,info=v12.balance_losses(base,p.sum()*0,p)
        assert torch.isfinite(loss)
        assert info['weighted_gradient_ratio']==0
        loss.backward()
        assert torch.isfinite(p.grad).all()


def test_bound_uses_normalized_field_and_keeps_loss_targets():
    torch.manual_seed(1)
    p=torch.randn(64,272,48,requires_grad=True)
    y,gate=torch.randn_like(p),torch.randn_like(p)
    scales={'s5':torch.ones(272,48)*2,'sp':torch.ones(272,4)*3,'sj':torch.ones(4)*5}
    incidence=torch.cat((torch.zeros(1,64),torch.eye(64)))
    groups=torch.arange(64).reshape(16,4)
    base=multiscale_loss(p,y,scales,'A_ref',gate)+.1*composed_contrast_loss(p*2,y,incidence,7)
    aux=v11.action_margin_loss(p*2,y,groups,5)
    loss,info=v12.training_loss('A_MarginBalanced',p,y,scales,gate,incidence,7,None,groups)
    torch.testing.assert_close(loss,base+info['margin_weight']*aux)
    assert 0<=info['margin_weight']<=.050001 and info['weighted_gradient_ratio']<=.100001
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_identical_fresh_A_initialization_and_budget(tmp_path):
    np.savez(tmp_path/'static.npz',**static_inputs())
    atomic_json(tmp_path/'protocol.json',{'source_A':{'42':{}}})
    torch.manual_seed(42)
    old=local_metrics.make_model(tmp_path,'A_D',42,'cpu')
    state=torch.get_rng_state()
    torch.manual_seed(42)
    new=v12.make_model(tmp_path,'A_MarginBalanced',42,'cpu')
    torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
    for name,value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name],value,rtol=0,atol=0)
    assert v12.SCHEDULES['A_MarginBalanced']=={'updates':66000,'batch':64,'warmup':1100,'validate':1100,'log':220}
