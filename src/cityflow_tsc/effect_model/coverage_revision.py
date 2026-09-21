"""v11: one fixed action-margin loss and expanded training-only pair coverage."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from ..collect_formal_effects import _freeze_json
from ..counterfactual.writer import atomic_json, sha256
from ..train_coarse_effects import read
from ..train_coarse_latepool import copy_frozen, link_frozen
from ..train_single_revision import action_groups
from ..train_temporal_effects import SEEDS
from . import connection_revision as v10, decision_revision as v9, local_metrics
from .formal_model import multiscale_loss
from .local_data import META_KEYS
from .local_revision import SCHEDULES as OLD_SCHEDULES, composed_contrast_loss, local_loss

FAMILIES = ('A_ActionMargin', 'B_Coverage')
SCHEDULES = {FAMILIES[0]: dict(OLD_SCHEDULES['A_D']), FAMILIES[1]: dict(OLD_SCHEDULES['B_L4'])}
SELECTION_SEED = 20260920


def select_additional_tasks(tasks, existing):
    """Outcome-blind selection, one RNG over sorted cohorts and flow IDs."""
    old_flows = {t['flow_id'] for t in existing if t['split'] == 'train'}
    training = [t for t in tasks if t['split'] == 'train']
    cohorts = sorted({t['cohort_id'] for t in training})
    if len(cohorts) != 3 or len(old_flows) != 6:
        raise ValueError('Expected three training cohorts and six existing pair flows')
    rng, selected = np.random.default_rng(SELECTION_SEED), []
    expected = {(p, t) for p in ('fixed_time', 'max_pressure') for t in (600, 1800, 2700)}
    for cohort in cohorts:
        members = [t for t in training if t['cohort_id'] == cohort and t['flow_id'] not in old_flows]
        flows = sorted({t['flow_id'] for t in members})
        for flow in sorted(rng.choice(flows, 6, replace=False).tolist()):
            roots = [t for t in members if t['flow_id'] == flow]
            if len(roots) != 6 or {(t['policy'], t['time_s']) for t in roots} != expected:
                raise ValueError('Each selected flow needs both policies at three times')
            selected.extend(copy.deepcopy(roots))
    selected.sort(key=lambda t: (t['time_s'], t['cohort_id'], t['flow_id'], t['policy']))
    for task in selected:
        task['collected_branch_count'] = 2049
    if len(selected) != 108 or len({t['root_id'] for t in selected}) != 108:
        raise ValueError('Expected 108 distinct new roots')
    return selected


def action_margin_loss(physical, truth, groups, scale):
    if physical.shape != truth.shape or len(physical) != 64 or groups.shape != (16, 4):
        raise ValueError('Expected a complete root and its node/action index groups')
    p, y = physical.sum((1, 2))[groups], truth.sum((1, 2))[groups]
    p = torch.cat((p.new_zeros(16, 1), p), 1)
    y = torch.cat((y.new_zeros(16, 1), y), 1)
    best = y.detach().argmin(1, keepdim=True)  # Reference first, then plan IDs 1..4.
    violation = y - y.gather(1, best) + p.gather(1, best) - p
    return violation.max(1).values.mean() / scale


def training_loss(family, prediction, labels, scales, logits, incidence, contrast_scale, pair_scale, groups=None):
    if family == 'B_Coverage':
        return local_loss(prediction, labels, scales['sp'], scales['sj'], logits)
    if family != 'A_ActionMargin':
        raise ValueError('Unknown v11 family')
    physical = prediction * scales['s5']
    return (multiscale_loss(prediction, labels, scales, 'A_ref', logits)
        + .1 * composed_contrast_loss(physical, labels, incidence, contrast_scale)
        + .05 * action_margin_loss(physical, labels, groups, scales['sj'][-1]))


def initialize_run(run, source_v10):
    from ..evaluate_connection_revision import verify_lock
    run, source_v10 = Path(run).resolve(), Path(source_v10).resolve()
    if run.is_relative_to(source_v10) or source_v10.is_relative_to(run):
        raise ValueError('Disjoint new run required')
    if read(source_v10 / 'execution.json')['stage'] != 'complete':
        raise ValueError('Completed v10 required')
    verify_lock(source_v10)
    prior = read(source_v10 / 'protocol.json')
    collection_source = Path(read(Path(prior['pair_source']) / 'protocol.json')['source_run'])
    parent = read(collection_source / 'selection.json')
    old_b = read(source_v10 / 'b_roots.json')
    additional = select_additional_tasks(parent['tasks'], old_b)
    for name in v9.COPIED_FILES:
        if name != 'b_roots.json':
            copy_frozen(source_v10 / name, run / name)
    b_records = old_b + [{k: t[k] for k in META_KEYS} for t in additional]
    a_records = read(run / 'a_roots.json')
    if (len(b_records) != 162 or len({r['root_id'] for r in b_records}) != 162 or
            sum(r['split'] == 'train' for r in b_records) != 144 or
            any(r not in a_records for r in b_records)):
        raise ValueError('Expanded pairs must match the frozen A root identities and splits')
    _freeze_json(run / 'b_roots.json', b_records)
    _freeze_json(run / 'collection/selection.json', {'tasks': additional, 'split_roots': {'train': 108},
        'selection_seed': SELECTION_SEED, 'source_selection_sha256': sha256(collection_source / 'selection.json'),
        'rule': 'sorted cohorts and flow IDs; select six new train flows per cohort without replacement; no labels'})
    keys = ('single_source', 'coarse_source', 'source_A', 'source_fixed_A_D', 'reuse_v8', 'reuse_v9',
            'A_population', 'horizons_s', 'history_s', 'intervention_s', 'candidate_count', 'optimizer',
            'precision', 'pair_query_batch')
    protocol = {k: prior[k] for k in keys}
    protocol.update(schema='action-margin-pair-coverage-v11', reuse_v10=str(source_v10),
        reuse_v10_protocol_sha256=sha256(source_v10 / 'protocol.json'),
        old_pair_source=prior['pair_source'], collection_source=str(collection_source),
        pair_source=str(run / 'collection'), B_population={'train': 144, 'validation': 18},
        seeds=list(SEEDS), schedules=SCHEDULES, collection_workers=4, max_training_workers=3,
        A_loss='unchanged v8 A-D + .05 max_a[(y_a-y_best)+(p_best-p_a)]/single_sj[-1], mean over16 nodes; reference0 first',
        A_selection='same78 roots; candidate regret, joint4 MAE, earlier update',
        B_loss=prior['B_loss'], B_target=prior['B_target'], B_encoder=prior['B_encoder'],
        B_adapter=prior['B_adapter'], B_selection=prior['B_selection'],
        normalization='all input, single, pair and candidate contrast scales frozen from v10; no refitting',
        collection={'new_flows': 18, 'new_roots': 108, 'new_pair_branches': 207360,
                    'reused_initial_branches': 13932, 'total_train_flows': 24, 'consistency_replay': False},
        initial_weights='fresh matched seeded A-D and v10 B-Adapter constructors; no fine tuning',
        inference_pair_budgets=[24, 120],
        evaluation='new A alone vs old A; new B paired only with fixed v8 A-D vs v10 B; no new A/B cross selection',
        test_scope='previously inspected12 diagnostic roots, only after all six checkpoints lock; not blind')
    _freeze_json(run / 'protocol.json', protocol)
    old_ids = {r['root_id'] for r in old_b}
    manifests = {}
    committed_cache = read(source_v10 / 'training_cache.json')
    for record in read(run / 'a_roots.json'):
        if record['split'] == 'test':
            continue
        rid = record['root_id']
        names = ['single4'] + (['joint4'] if record['split'] != 'train' else [])
        names += ['pair4'] if rid in old_ids else []
        committed = committed_cache[rid]
        manifests[rid] = {}
        for name in names:
            digest = link_frozen(source_v10 / 'labels' / rid / (name + '.npy'), run / 'labels' / rid / (name + '.npy'))
            if digest != committed[name]:
                raise ValueError('Frozen label changed')
            manifests[rid][name] = digest
    _freeze_json(run / 'training_cache.json', manifests)
    ready = read(source_v10 / 'fixed_A_D.ready.json')
    for name, digest in ready['files'].items():
        if link_frozen(source_v10 / name, run / name) != digest:
            raise ValueError('Fixed A prediction changed')
    copy_frozen(source_v10 / 'fixed_A_D.ready.json', run / 'fixed_A_D.ready.json')


def load_roots(run, family, splits=('train', 'validation')):
    if family == 'B_Coverage' and 'train' in splits:
        if read(Path(run) / 'collection/complete.json')['stage'] != 'complete':
            raise ValueError('Pair coverage not ready')
    return v9.load_roots(run, family, splits)


attach_fixed_predictions = v9.attach_fixed_predictions
validate_model = v9.validate_model


def make_model(run, family, seed, device, selected=False):
    run = Path(run)
    if family == 'A_D_v8':
        return v9.make_model(run, family, seed, device, selected=True)
    if family == 'A_ActionMargin':
        model = local_metrics.make_model(run, 'A_D', seed, device)
    elif family == 'B_Coverage':
        model = v10.make_model(run, 'B_Adapter', seed, device)
    else:
        raise ValueError('Unknown v11 family')
    if selected:
        saved = torch.load(run / family / f'seed_{seed}/best.pt', map_location='cpu', weights_only=False)
        if (saved['family'] != family or saved['seed'] != seed or saved['test_used'] or
                saved['protocol_sha256'] != sha256(run / 'protocol.json')):
            raise ValueError('Selected checkpoint identity changed')
        if family == 'B_Coverage' and saved['source_A_sha256'] != read(run / 'protocol.json')['source_A'][str(seed)]['sha256']:
            raise ValueError('Frozen encoder source changed')
        model.load_state_dict(saved['state_dict'])
    return model


def single_choice_metrics(root, single4):
    groups = action_groups([root])
    p = np.asarray(single4, dtype=np.float64)[:, :, -1].sum(1)[groups]
    y = np.asarray(root['single4'], dtype=np.float64)[:, :, -1].sum(1)[groups]
    p, y = np.c_[np.zeros(16), p], np.c_[np.zeros(16), y]
    selected = p.argmin(1)
    regret = y[np.arange(16), selected] - y.min(1)
    return {'single_action_regret': float(regret.mean()), 'single_action_regret_per_node': regret.tolist(),
            'single_action_selected': selected.tolist(), 'single_action_optimal_count': int((regret <= 1e-6).sum())}
