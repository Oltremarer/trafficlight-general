"""v13: bounded decision margins over the deployed candidate set, using true singles only."""
from pathlib import Path

import numpy as np
import torch

from ..collect_formal_effects import _freeze_json
from ..counterfactual.writer import sha256
from ..train_coarse_effects import read
from ..train_coarse_latepool import copy_frozen
from . import balanced_revision as v12, decision_revision as v9, local_metrics
from .formal_model import multiscale_loss, tie_argmin
from .local_revision import SCHEDULES as OLD_SCHEDULES, composed_contrast_loss

FAMILIES = ('A_CandidateBalanced',)
SCHEDULES = {FAMILIES[0]: dict(OLD_SCHEDULES['A_D'])}
SCHEMA = 'candidate-gradient-balanced-margin-v13'


def candidate_margin_loss(physical, truth, incidence, scale):
    """Cost-augmented hinge; no true-joint targets and no gradient at zero violation."""
    if physical.shape != truth.shape or len(physical) != 64 or incidence.shape != (65, 64):
        raise ValueError('Expected 64 single-effect queries and the frozen 65-candidate incidence')
    p, y = incidence @ physical.sum((1, 2)), incidence @ truth.sum((1, 2))
    best = y.detach().argmin()  # Candidate order breaks truth ties; reference is first.
    # Algebraically y-y_best+p_best-p, but exact p==y gives exactly zero.
    error = (p - y) / scale
    violation = (error[best] - error).max()
    return torch.where(violation > 0, violation, violation * 0)


def training_loss(family, prediction, labels, scales, logits, incidence, contrast_scale, pair_scale):
    if family != FAMILIES[0]:
        raise ValueError('Unknown v13 family')
    physical = prediction * scales['s5']
    base = multiscale_loss(prediction, labels, scales, 'A_ref', logits)
    base = base + .1 * composed_contrast_loss(physical, labels, incidence, contrast_scale)
    auxiliary = candidate_margin_loss(physical, labels, incidence, contrast_scale)
    return v12.balance_losses(base, auxiliary, prediction)


def initialize_run(run, source_v12):
    from ..evaluate_balanced_revision import LOCKED_FILES
    from ..evaluate_local_revision import verify_lock
    run, source_v12 = Path(run).resolve(), Path(source_v12).resolve()
    if run.is_relative_to(source_v12) or source_v12.is_relative_to(run):
        raise ValueError('Independent directory required')
    if (read(source_v12 / 'execution.json')['stage'] != 'complete' or
            read(source_v12 / 'evaluation/summary.json')['stage'] != 'complete'):
        raise ValueError('Completed v12 training and evaluation required')
    verify_lock(source_v12, v12.FAMILIES, LOCKED_FILES)
    prior = read(source_v12 / 'protocol.json')
    for name in v9.COPIED_FILES:
        copy_frozen(source_v12 / name, run / name)
    protocol = dict(prior)
    protocol.update(schema=SCHEMA, source_v12=str(source_v12),
        source_v12_protocol_sha256=sha256(source_v12 / 'protocol.json'),
        source_balanced_A={str(seed): {'path': str(source_v12 / 'A_MarginBalanced' / f'seed_{seed}/best.pt'),
            'sha256': sha256(source_v12 / 'A_MarginBalanced' / f'seed_{seed}/best.pt')} for seed in (42, 43, 44)},
        schedules=SCHEDULES,
        A_loss='unchanged old A-D base + output-gradient-balanced candidate margin; replace v12 local-action margin',
        auxiliary_target='y=incidence@true_single_total240; never true_joint or pair targets',
        auxiliary_formula='max_c[(y_c-y_best)+(p_best-p_c)]/frozen_contrast_scale; zero subgradient at zero violation',
        candidate_scope='same frozen 65 candidates per root, including exact-zero reference; no candidate search',
        evaluation='validation78 only; fixed old A-D and v12 A_MarginBalanced vs new A; additive regret diagnostic',
        test_scope='No diagnostic/test roots opened in v13; not an independent confirmation')
    _freeze_json(run / 'protocol.json', protocol)
    v9.reuse_labels(run)


load_roots = v9.load_roots
validate_model = v9.validate_model


def make_model(run, family, seed, device, selected=False):
    run = Path(run)
    if family == 'A_D_v8':
        return v9.make_model(run, family, seed, device, selected=True)
    if family == 'A_MarginBalanced':
        protocol = read(run / 'protocol.json')
        source = protocol['source_balanced_A'][str(seed)]
        if sha256(Path(source['path'])) != source['sha256']:
            raise ValueError('Frozen v12 comparison checkpoint changed')
        return v12.make_model(Path(protocol['source_v12']), family, seed, device, selected=True)
    if family != FAMILIES[0]:
        raise ValueError('Unknown v13 family')
    model = local_metrics.make_model(run, 'A_D', seed, device)
    if selected:
        saved = torch.load(run / family / f'seed_{seed}/best.pt', map_location='cpu', weights_only=False)
        if (saved['family'] != family or saved['seed'] != seed or saved['test_used'] or
                saved['protocol_sha256'] != sha256(run / 'protocol.json')):
            raise ValueError('Selected v13 checkpoint changed')
        model.load_state_dict(saved['state_dict'])
    return model


def additive_choice_metrics(root, selected):
    single_total = np.asarray(root['single4'], dtype=np.float64)[:, :, -1].sum(1)
    additive = np.asarray(root['s_incidence'], dtype=np.float64) @ single_total
    oracle = tie_argmin(additive)
    joint = np.asarray(root['joint4'], dtype=np.float64)[:, :, -1].sum(1)
    return {'additive_regret': float(additive[selected] - additive[oracle]),
            'additive_optimal_choice': bool(abs(additive[selected] - additive[oracle]) <= 1e-6),
            'true_single_oracle_joint_regret': float(joint[oracle] - joint.min())}
