"""v12: retain action margins but bound their normalized-output gradient norm."""
from pathlib import Path

import torch

from ..collect_formal_effects import _freeze_json
from ..counterfactual.writer import sha256
from ..train_coarse_effects import read
from ..train_coarse_latepool import copy_frozen
from . import coverage_revision as v11, decision_revision as v9, local_metrics
from .formal_model import multiscale_loss
from .local_revision import SCHEDULES as OLD_SCHEDULES, composed_contrast_loss

FAMILIES = ('A_MarginBalanced',)
SCHEDULES = {FAMILIES[0]: dict(OLD_SCHEDULES['A_D'])}
GRADIENT_FRACTION = .1
MAX_WEIGHT = .05


def balance_losses(base, auxiliary, prediction):
    # This bounds output-space gradients only, NOT parameter-space gradients or regret.
    main_grad = torch.autograd.grad(base, prediction, retain_graph=True)[0]
    aux_grad = torch.autograd.grad(auxiliary, prediction, retain_graph=True)[0]
    main_norm, aux_norm = main_grad.norm().detach(), aux_grad.norm().detach()
    weight = torch.minimum(main_norm.new_tensor(MAX_WEIGHT), GRADIENT_FRACTION * main_norm / (aux_norm + 1e-12))
    loss = base + weight * auxiliary
    return loss, {'margin_weight': float(weight), 'base_loss': float(base.detach()),
        'raw_margin_loss': float(auxiliary.detach()),
        'weighted_gradient_ratio': float(weight * aux_norm / main_norm.clamp_min(1e-12))}


def training_loss(family, prediction, labels, scales, logits, incidence, contrast_scale, pair_scale, groups=None):
    if family != FAMILIES[0]:
        raise ValueError('Unknown v12 family')
    physical = prediction * scales['s5']
    base = multiscale_loss(prediction, labels, scales, 'A_ref', logits)
    base = base + .1 * composed_contrast_loss(physical, labels, incidence, contrast_scale)
    auxiliary = v11.action_margin_loss(physical, labels, groups, scales['sj'][-1])
    return balance_losses(base, auxiliary, prediction)


def initialize_run(run, source_v11):
    from ..evaluate_connection_revision import verify_lock
    run, source_v11 = Path(run).resolve(), Path(source_v11).resolve()
    if run.is_relative_to(source_v11) or source_v11.is_relative_to(run):
        raise ValueError('Independent directory required')
    source_protocol = read(source_v11 / 'protocol.json')
    for seed in (42, 43, 44):
        path = source_v11 / 'A_ActionMargin' / f'seed_{seed}'
        done = read(path / 'complete.json')
        if done['stage'] != 'complete' or done['checkpoint_sha256'] != sha256(path / 'best.pt'):
            raise ValueError('Completed v11 A evidence required; B may remain running')
    reuse = Path(source_protocol['reuse_v10'])
    verify_lock(reuse)
    prior = read(reuse / 'protocol.json')
    for name in v9.COPIED_FILES:
        copy_frozen(reuse / name, run / name)
    protocol = {k: prior[k] for k in ('single_source','pair_source','coarse_source','source_A','source_fixed_A_D',
        'reuse_v8','reuse_v9','A_population','horizons_s','history_s','intervention_s','candidate_count',
        'optimizer','precision','normalization','pair_query_batch')}
    protocol.update(schema='output-gradient-balanced-margin-v12', source_v11=str(source_v11),
        source_v11_protocol_sha256=sha256(source_v11 / 'protocol.json'), reuse_v10=str(reuse),
        seeds=[42,43,44], schedules=SCHEDULES, max_training_workers=3, collection_jobs=0, B_jobs=0,
        gradient_fraction=GRADIENT_FRACTION, max_margin_weight=MAX_WEIGHT,
        A_loss='L_old_A_D + stopgrad(min(.05,.1*norm(dL_old/dp)/(norm(dL_margin/dp)+1e-12)))*L_margin',
        gradient_scope='p is the normalized64x272x48 output field; no parameter-space or decision guarantee',
        A_selection='same78 validation roots; candidate regret, joint4 MAE, earlier update',
        initial_weights='fresh seed-matched v8 A-D constructor; no fine tuning',
        normalization='all scales frozen from v10; no refitting or threshold search',
        evaluation='validation78 only; independent of v11 B; old A-D comparison; record local5-action regret',
        test_scope='No diagnostic/test roots opened in v12; later held-out evaluation requires separate scope')
    _freeze_json(run / 'protocol.json', protocol)
    v9.reuse_labels(run)


load_roots = v9.load_roots
validate_model = v9.validate_model


def make_model(run, family, seed, device, selected=False):
    run = Path(run)
    if family == 'A_D_v8':
        return v9.make_model(run, family, seed, device, selected=True)
    if family != FAMILIES[0]:
        raise ValueError('Unknown v12 family')
    model = local_metrics.make_model(run, 'A_D', seed, device)
    if selected:
        saved = torch.load(run / family / f'seed_{seed}/best.pt', map_location='cpu', weights_only=False)
        if (saved['family'] != family or saved['seed'] != seed or saved['test_used'] or
                saved['protocol_sha256'] != sha256(run / 'protocol.json')):
            raise ValueError('Selected v12 checkpoint changed')
        model.load_state_dict(saved['state_dict'])
    return model
