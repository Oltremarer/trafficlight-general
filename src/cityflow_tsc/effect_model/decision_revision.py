"""v9: dense A contrast/local auxiliaries and deployment-matched pair selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..collect_formal_effects import _freeze_json
from ..counterfactual.writer import atomic_json, sha256
from ..train_coarse_effects import read
from ..train_coarse_latepool import copy_frozen, link_frozen
from ..train_formal_effects import initialize, now
from ..train_temporal_effects import SEEDS
from .formal_data import _read_npz, _write_numpy
from .formal_model import multiscale_loss, tensor_scales
from . import local_data, local_metrics
from .local_revision import (SCHEDULES as V8_SCHEDULES, HORIZON_INDICES,
    composed_contrast_loss, local_loss, pair_groups)

FAMILIES = ('A_D_Top8', 'A_D_Local', 'B_Match', 'B_Contrast')
SCHEDULES = {f: dict(V8_SCHEDULES['B_L4' if f.startswith('B_') else 'A_D']) for f in FAMILIES}
COPIED_FILES = ('static.npz', 'normalization.npz', 'pair_normalization.npz',
                'contrast_scale.json', 'a_roots.json', 'b_roots.json')


def top8_contrast_loss(prediction, target, incidence, scale):
    if prediction.shape != target.shape or len(prediction) != 64 or incidence.shape != (65, 64):
        raise ValueError('Expected one complete 64-query root and 65 candidates')
    p, y = incidence @ prediction.sum((1, 2)), incidence @ target.sum((1, 2))
    i, j = torch.triu_indices(65, 65, 1, device=p.device)
    top = torch.zeros(65, dtype=torch.bool, device=p.device)
    # Frozen candidate order breaks exact truth ties; targets never include joint labels.
    top[torch.argsort(y.detach(), stable=True)[:8]] = True
    mask = top[i] | top[j]
    gaps = F.smooth_l1_loss((p[i] - p[j]) / scale, (y[i] - y[j]) / scale,
                            reduction='none', beta=1.)
    return .05 * gaps.mean() + .05 * gaps[mask].mean()


def dense_local_loss(prediction_physical, target_physical, scale):
    if prediction_physical.shape != target_physical.shape or prediction_physical.shape[-2:] != (272, 48):
        raise ValueError('Expected dense signed 272 x 48 effects')
    error = prediction_physical.cumsum(-1) - target_physical.cumsum(-1)
    return (error[..., list(HORIZON_INDICES)] / scale).abs().mean()


def pair_action_contrast_loss(prediction_physical, target_physical, scale):
    if prediction_physical.shape != target_physical.shape or prediction_physical.shape != (128, 272, 4):
        raise ValueError('Expected eight complete groups of sixteen pair queries')
    p = prediction_physical[:, :, -1].sum(1).reshape(8, 16)
    y = target_physical[:, :, -1].sum(1).reshape(8, 16)
    p = torch.cat((p, p.new_zeros(8, 1)), 1)
    y = torch.cat((y, y.new_zeros(8, 1)), 1)
    i, j = torch.triu_indices(17, 17, 1, device=p.device)
    return F.smooth_l1_loss((p[:, i] - p[:, j]) / scale,
                            (y[:, i] - y[:, j]) / scale, beta=1.)


def training_loss(family, prediction, labels, scales, logits, incidence, contrast_scale, pair_scale):
    if family.startswith('A_'):
        loss = multiscale_loss(prediction, labels, scales, 'A_ref', logits)
        physical = prediction * scales['s5']
        if family == 'A_D_Top8':
            auxiliary = top8_contrast_loss(physical, labels, incidence, contrast_scale)
        elif family == 'A_D_Local':
            auxiliary = .1 * composed_contrast_loss(physical, labels, incidence, contrast_scale)
            auxiliary = auxiliary + .1 * dense_local_loss(physical, labels, scales['sp'])
        else:
            raise ValueError('Unapproved dense family')
        return loss + auxiliary
    loss = local_loss(prediction, labels, scales['sp'], scales['sj'], logits)
    if family == 'B_Contrast':
        loss = loss + .1 * pair_action_contrast_loss(prediction * scales['sp'], labels, pair_scale)
    elif family != 'B_Match':
        raise ValueError('Unapproved pair family')
    return loss


def reuse_labels(run, test=False):
    run = Path(run)
    if test and not (run / 'checkpoints_locked.json').exists():
        raise ValueError('Diagnostic labels remain locked')
    source = Path(read(run / 'protocol.json')['reuse_v8'])
    records = [r for r in read(run / 'a_roots.json') if (r['split'] == 'test') == test]
    pairs = {r['root_id'] for r in read(run / 'b_roots.json')}
    manifests = {}
    for record in records:
        rid = record['root_id']
        original = source / 'labels' / rid
        complete = read(original / 'complete.json')
        if complete['stage'] != 'complete' or any(complete[k] != record[k] for k in local_data.META_KEYS):
            raise ValueError('Source label identity mismatch')
        names = ['single4'] + (['joint4'] if record['split'] != 'train' else [])
        names += ['pair4'] if rid in pairs else []
        manifests[rid] = {}
        for name in names:
            digest = link_frozen(original / (name + '.npy'), run / 'labels' / rid / (name + '.npy'))
            if digest != complete['files'][name]['target_sha256']:
                raise ValueError('Previously committed label cache changed')
            manifests[rid][name] = digest
    _freeze_json(run / ('diagnostic_cache.json' if test else 'training_cache.json'), manifests)


def initialize_run(run, reuse):
    run, reuse = Path(run).resolve(), Path(reuse).resolve()
    if run.is_relative_to(reuse) or reuse.is_relative_to(run):
        raise ValueError('New run and source must be disjoint')
    if read(reuse / 'execution.json')['stage'] != 'complete':
        raise ValueError('v8 source is not complete')
    lock = read(reuse / 'checkpoints_locked.json')
    for name in ('protocol.json', *COPIED_FILES):
        if sha256(reuse / name) != lock['files'][name]:
            raise ValueError('Frozen v8 metadata changed: ' + name)
    for name in COPIED_FILES:
        copy_frozen(reuse / name, run / name)
    source = read(reuse / 'protocol.json')
    fixed = {}
    for seed in SEEDS:
        name = f'A_D/seed_{seed}/best.pt'
        digest = sha256(reuse / name)
        if digest != lock['checkpoints'][name]:
            raise ValueError('Frozen v8 A-D changed')
        fixed[str(seed)] = {'path': str(reuse / name), 'sha256': digest}
    protocol = {k: source[k] for k in ('single_source', 'pair_source', 'coarse_source', 'source_A',
        'A_population', 'B_population', 'horizons_s', 'history_s', 'intervention_s', 'candidate_count',
        'optimizer', 'precision', 'normalization', 'test_scope', 'inference_pair_budgets', 'pair_query_batch')}
    protocol.update(schema='matched-decision-v9', reuse_v8=str(reuse),
        reuse_protocol_sha256=sha256(reuse / 'protocol.json'), source_fixed_A_D=fixed,
        seeds=list(SEEDS), schedules=SCHEDULES, max_training_workers=8, cpu_threads_per_training_worker=1,
        A_D_Top8='v8 dense A base + .05 all2080 Huber + .05 top8-endpoint484 Huber; true-single sums only',
        A_D_Local='unchanged v8 A-D + .1 natural mean normalized signed cumulative position L1 at 30/90/180/240s',
        B_Match='unchanged v8 B_L4 training; selection fixed seed-matched v8 A-D + all120 predicted pairs',
        B_Contrast='B_Match + .1 normalized Huber of136 contrasts among16 pair actions + exact zero reference',
        B_target='true pair effects; never true joint minus learned single residual',
        B_encoder='frozen seed-matched v5 A_ref; identical initialization and grouped sampler to v8 B_L4',
        B_selection='same18 validation roots; fixed v8 A-D+predictedB regret, pair_total240_mae, earlier update; zero eligible',
        A_selection='same78 validation roots; regret, joint4_mae, earlier update',
        final_comparisons='main B claims use fixed v8 A-D; new A crosses are predeclared secondary; separate B selection18 and remaining60',
        frozen_prediction_preparation='separate process before workers; no B RNG consumption or gradient',
        initial_weights='same seeded fresh constructors as v8; no A-D checkpoint fine-tuning')
    _freeze_json(run / 'protocol.json', protocol)
    reuse_labels(run)


def fit_pair_contrast_scale(roots):
    if not roots or any(r['split'] != 'train' for r in roots):
        raise ValueError('Pair contrast scale requires training roots only')
    i, j = np.triu_indices(17, 1)
    values = []
    for root in roots:
        groups = pair_groups([root])
        totals = np.asarray(root['pair4'], dtype=np.float64)[:, :, -1].sum(1)[groups]
        totals = np.concatenate((totals, np.zeros((120, 1))), axis=1)
        gaps = abs(totals[:, i] - totals[:, j])
        values.append(gaps[gaps > 0])
    nonzero = np.concatenate(values)
    return max(1., float(np.percentile(nonzero, 95))) if len(nonzero) else 1.


def prepare_frozen_predictions(run):
    """Invoked in its own process; training workers have their own seeded RNG."""
    run = Path(run)
    roots = local_data.load_roots(run, 'B_L4')
    train = [r for r in roots if r['split'] == 'train']
    validation = [r for r in roots if r['split'] == 'validation']
    if (len(train), len(validation)) != (36, 18):
        raise ValueError('Pair population changed')
    _freeze_json(run / 'pair_contrast_scale.json', {'scale': fit_pair_contrast_scale(train),
        'fit_split': 'train', 'roots': 36, 'pairs_per_root': 120, 'contrasts_per_pair': 136,
        'percentile': 95, 'exclude_exact_zero': True, 'floor': 1, 'weight': .1})
    initialize(42)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device('cuda:0')
    scales = tensor_scales(_read_npz(run / 'normalization.npz'), 'single', device)
    source = Path(read(run / 'protocol.json')['reuse_v8'])
    digests = {}
    for seed in SEEDS:
        model = make_model(run, 'A_D_v8', seed, device, selected=True).eval()
        for root in validation:
            prediction = local_metrics.predict_single4(model, root, scales, device)
            target = run / 'fixed_A_D' / f'seed_{seed}' / (root['root_id'] + '.npy')
            _write_numpy(target, prediction)
            digests[str(target.relative_to(run))] = sha256(target)
        del model
    atomic_json(run / 'fixed_A_D.ready.json', {'stage': 'complete', 'roots': 18, 'seeds': list(SEEDS),
        'files': digests, 'source_run': str(source), 'finished_at': now()})


def load_roots(run, family, splits=('train', 'validation')):
    run = Path(run)
    pair = family.startswith('B_')
    roots = local_data.load_roots(run, 'B_L4' if pair else 'A_L4', splits)
    if not pair:
        source = Path(read(run / 'protocol.json')['single_source'])
        for root in roots:
            if root['split'] == 'train':
                root['single'] = np.load(source / 'derived/roots' / root['root_id'] / 'single.npy', mmap_mode='r')
    return roots


def attach_fixed_predictions(run, roots, seed):
    ready = read(Path(run) / 'fixed_A_D.ready.json')
    if ready['stage'] != 'complete' or ready['roots'] != 18:
        raise ValueError('Frozen validation predictions missing')
    for root in roots:
        name = f"fixed_A_D/seed_{seed}/{root['root_id']}.npy"
        if root['split'] != 'validation' or sha256(Path(run) / name) != ready['files'][name]:
            raise ValueError('Frozen prediction identity changed')
        root['fixed_A_D4'] = np.load(Path(run) / name, mmap_mode='r')


def make_model(run, family, seed, device, selected=False):
    run = Path(run)
    protocol = read(run / 'protocol.json')
    if family == 'A_D_v8':
        source = protocol['source_fixed_A_D'][str(seed)]
        if sha256(Path(source['path'])) != source['sha256']:
            raise ValueError('Frozen A-D changed')
        return local_metrics.make_model(Path(protocol['reuse_v8']), 'A_D', seed, device, selected=True)
    if family not in FAMILIES:
        raise ValueError('Unknown v9 family')
    model = local_metrics.make_model(run, 'B_L4' if family.startswith('B_') else 'A_D', seed, device)
    if selected:
        saved = torch.load(run / family / f'seed_{seed}/best.pt', map_location='cpu', weights_only=False)
        if (saved['family'] != family or saved['seed'] != seed or saved['test_used'] or
                saved['protocol_sha256'] != sha256(run / 'protocol.json')):
            raise ValueError('Selected v9 checkpoint identity changed')
        if family.startswith('B_') and saved['source_A_sha256'] != protocol['source_A'][str(seed)]['sha256']:
            raise ValueError('Frozen B encoder source changed')
        model.load_state_dict(saved['state_dict'])
    return model


def validate_model(model, roots, single_scales, pair_scales, device, family):
    return local_metrics.validate_model(model, roots, single_scales, pair_scales, device,
        'B_L4' if family.startswith('B_') else 'A_D',
        single_baseline_key='fixed_A_D4' if family.startswith('B_') else 'single4')
