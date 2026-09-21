"""v10: frozen A, direction-aware connecting-lane context, conservative B use."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..collect_formal_effects import _freeze_json
from ..counterfactual.writer import sha256
from ..train_coarse_effects import read
from ..train_coarse_latepool import copy_frozen, link_frozen
from ..train_temporal_effects import SEEDS
from . import decision_revision as v9
from .formal_data import _read_npz
from .formal_model import tie_argmin
from .local_revision import LocalPairModel, SCHEDULES as OLD_SCHEDULES, local_loss

FAMILIES = ('B_Adapter',)
SCHEDULES = {'B_Adapter': dict(OLD_SCHEDULES['B_L4'])}


def connection_pool(relative):
    """i -> j lanes: outgoing from i AND incoming to j; no label-derived mask."""
    rel = np.asarray(relative)
    if rel.shape != (16, 272, 11):
        raise ValueError('Expected fixed 4x4 static geometry')
    outgoing, incoming = rel[:, :240, 5] > .5, rel[:, :240, 4] > .5
    mask = outgoing[:, None, :] & incoming[None, :, :]
    counts = mask.sum(-1, keepdims=True)
    weights = mask.astype(np.float32) / np.maximum(counts, 1)
    return weights.astype(np.float32), (counts > 0).astype(np.float32)


class ConnectionPairModel(LocalPairModel):
    """Train only the B heads and a residual pair-context adapter.

    The frozen encoding cache holds label-free features, NOT adapter outputs,
    so training the adapter remains differentiable when the encoder is cached.
    """

    def __init__(self, static, frozen_single_state):
        super().__init__(static, frozen_single_state)
        pool, present = connection_pool(static['relative'])
        self.register_buffer('connection_weights', torch.from_numpy(pool))
        self.register_buffer('connection_present', torch.from_numpy(present))
        # Two endpoint state/action vectors (2*128), two directed lane contexts
        # (2*(64 encoded + 20 latest + 20 historical mean + 1 presence)) = 466.
        self.connection_adapter = nn.Sequential(nn.Linear(466, 64), nn.SiLU(), nn.Linear(64, 64))
        nn.init.zeros_(self.connection_adapter[-1].weight)
        nn.init.zeros_(self.connection_adapter[-1].bias)

    def encode(self, history, actions):
        encoded = super().encode(history, actions)
        lane = torch.cat((encoded['objects'][:, :240], history[:, -1, :240],
                          history[:, :, :240].mean(1)), -1)
        pooled = torch.matmul(self.connection_weights.reshape(256, 240), lane).reshape(-1, 16, 16, 104)
        encoded['connection_context'] = torch.cat((pooled,
            self.connection_present[None].expand(len(history), -1, -1, -1)), -1)
        return encoded

    def pair(self, encoded, roots, nodes, targets, base_phase, return_gate=False):
        state, glob, first, r1, c1 = self.query_parts(encoded, roots, nodes[:, 0], targets[:, 0], base_phase)
        _, _, second, r2, c2 = self.query_parts(encoded, roots, nodes[:, 1], targets[:, 1], base_phase)
        forward_context = encoded['connection_context'][roots, nodes[:, 0], nodes[:, 1]]
        reverse_context = encoded['connection_context'][roots, nodes[:, 1], nodes[:, 0]]
        forward_delta = self.connection_adapter(torch.cat((first[:, 0], second[:, 0],
                                                            forward_context, reverse_context), -1))
        reverse_delta = self.connection_adapter(torch.cat((second[:, 0], first[:, 0],
                                                            reverse_context, forward_context), -1))
        forward = torch.cat((state + forward_delta[:, None], glob, first, second, r1, r2), -1)
        reverse = torch.cat((state + reverse_delta[:, None], glob, second, first, r2, r1), -1)
        value = (self.pair_head(forward) + self.pair_head(reverse)) * .5
        logits = (self.pair_gate(forward) + self.pair_gate(reverse)) * .5
        prediction = value * logits.sigmoid() * (c1 & c2)[:, None, None]
        return (prediction, logits) if return_gate else prediction


def consensus_choice(single_scores, joint_scores, tolerance=1e-6):
    """Three fixed seeds, FP64 equal means; unanimity is a heuristic, not a CI."""
    single, joint = np.asarray(single_scores, dtype=np.float64), np.asarray(joint_scores, dtype=np.float64)
    if single.shape != joint.shape or single.ndim != 2 or single.shape[0] != 3:
        raise ValueError('Expected three seed-matched candidate score vectors')
    if not np.isfinite(single).all() or not np.isfinite(joint).all():
        raise FloatingPointError('Nonfinite ensemble scores')
    baseline, proposal = tie_argmin(single.mean(0)), tie_argmin(joint.mean(0))
    gains = joint[:, baseline] - joint[:, proposal]
    accepted = proposal != baseline and bool(np.all(gains > tolerance))
    return {'selected': proposal if accepted else baseline, 'baseline_selected': baseline,
        'proposed_selected': proposal, 'seed_predicted_gains': gains.tolist(),
        'proposed_switch': proposal != baseline, 'accepted_switch': accepted}


def initialize_run(run, source_v9):
    from ..evaluate_decision_revision import verify_lock
    run, source_v9 = Path(run).resolve(), Path(source_v9).resolve()
    if run.is_relative_to(source_v9) or source_v9.is_relative_to(run):
        raise ValueError('Source and new run must be disjoint')
    if read(source_v9 / 'execution.json')['stage'] != 'complete':
        raise ValueError('v9 source must be complete')
    verify_lock(source_v9)
    source, lock = read(source_v9 / 'protocol.json'), read(source_v9 / 'checkpoints_locked.json')
    for name in v9.COPIED_FILES:
        copy_frozen(source_v9 / name, run / name)
    fixed_b = {}
    for seed in SEEDS:
        name = f'B_Match/seed_{seed}/best.pt'
        fixed_b[str(seed)] = {'path': str(source_v9 / name), 'sha256': lock['checkpoints'][name]}
    keys = ('single_source', 'pair_source', 'coarse_source', 'source_A', 'source_fixed_A_D',
        'reuse_v8', 'A_population', 'B_population', 'horizons_s', 'history_s', 'intervention_s',
        'candidate_count', 'optimizer', 'precision', 'normalization', 'pair_query_batch')
    protocol = {key: source[key] for key in keys}
    protocol.update(schema='connection-decision-v10', reuse_v9=str(source_v9),
        reuse_v9_protocol_sha256=sha256(source_v9 / 'protocol.json'), source_B_Match=fixed_b,
        seeds=list(SEEDS), schedules=SCHEDULES, max_training_workers=3, cpu_threads_per_training_worker=1,
        B_encoder='frozen seed-matched v5 A_ref; no updates to any A or frozen encoder parameter',
        B_adapter='466->64->64 zero-output-initialized residual context; shared forward/reverse adapter; no extra dropout',
        connection_context='directional connecting-lane mean of frozen 64D encoding, latest and 150s mean normalized 20D history, plus presence; absent links exactly zero',
        B_target='unchanged true counterfactual pair effect at 272 positions and 30/90/180/240 seconds',
        B_loss='unchanged natural position L1 + .1 total240 Huber + .01 nonzero gate BCE',
        B_selection='same18 validation roots, fixed v8 A-D + all120 B regret; pair total240 MAE; earlier update; zero eligible',
        initial_weights='same seeded B_L4 constructor and frozen v5 encoder, plus fresh zero-residual adapter; no B-Match fine-tuning',
        inference_pair_budgets=[24, 120], ensemble_pair_budget=24,
        ensemble_methods=['A_Mean3', 'A_Mean3+B_Match_Mean3_24', 'A_Mean3+B_Match_Consensus3_24'],
        ensemble_rule='equal physical-field means; propose argmin mean(A+B); replace argmin mean(A) only if all three matched A+B scores improve by >1e-6 vehicle-seconds',
        ensemble_calibration='none; no threshold or weight fitted on any labels; unanimity is not a calibrated guarantee',
        evaluation='new B paired only with fixed seed-matched v8 A-D; old-model ensembles independent of training; split selection18/remaining60',
        test_scope='previously inspected 12-root diagnostic, not blind/new demand; only after three new checkpoints lock',
        decomposition_scope='only18 validation roots with true pair labels; all120 factors; descriptive substitution, never supervision',
        parallelism='three independent B training processes plus old-ensemble validation; no new collection')
    _freeze_json(run / 'protocol.json', protocol)
    v9.reuse_labels(run)
    ready = read(source_v9 / 'fixed_A_D.ready.json')
    for name, digest in ready['files'].items():
        if link_frozen(source_v9 / name, run / name) != digest:
            raise ValueError('Fixed A selection prediction changed')
    copy_frozen(source_v9 / 'fixed_A_D.ready.json', run / 'fixed_A_D.ready.json')


load_roots = v9.load_roots
attach_fixed_predictions = v9.attach_fixed_predictions
validate_model = v9.validate_model


def training_loss(family, prediction, labels, scales, logits, incidence=None, contrast_scale=None, pair_scale=None):
    if family != 'B_Adapter':
        raise ValueError('Unknown v10 training family')
    return local_loss(prediction, labels, scales['sp'], scales['sj'], logits)


def make_model(run, family, seed, device, selected=False):
    run = Path(run)
    protocol = read(run / 'protocol.json')
    if family == 'A_D_v8':
        return v9.make_model(run, family, seed, device, selected=True)
    if family == 'B_Match':
        source = protocol['source_B_Match'][str(seed)]
        if sha256(Path(source['path'])) != source['sha256']:
            raise ValueError('Frozen B-Match changed')
        return v9.make_model(Path(protocol['reuse_v9']), family, seed, device, selected=True)
    if family != 'B_Adapter':
        raise ValueError('Unknown v10 model')
    source = protocol['source_A'][str(seed)]
    if sha256(Path(source['path'])) != source['sha256']:
        raise ValueError('Frozen source encoder changed')
    saved = torch.load(source['path'], map_location='cpu', weights_only=False)
    model = ConnectionPairModel(_read_npz(run / 'static.npz'), saved['state_dict'])
    if selected:
        saved = torch.load(run / family / f'seed_{seed}/best.pt', map_location='cpu', weights_only=False)
        if (saved['family'] != family or saved['seed'] != seed or saved['test_used'] or
                saved['protocol_sha256'] != sha256(run / 'protocol.json') or
                saved['source_A_sha256'] != source['sha256']):
            raise ValueError('Selected v10 checkpoint identity changed')
        model.load_state_dict(saved['state_dict'])
    return model.to(device)
