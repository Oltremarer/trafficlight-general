"""Read-only source reuse and private cumulative-label caches for v8."""
from __future__ import annotations

import concurrent.futures
import multiprocessing
from pathlib import Path

import numpy as np

from ..collect_formal_effects import _freeze_json
from ..counterfactual.writer import atomic_json, sha256
from ..train_coarse_effects import read
from ..train_coarse_latepool import copy_frozen
from ..train_formal_effects import now
from ..train_temporal_effects import ensure_file_limit, SEEDS
from .formal_data import _read_npz, _write_numpy, ScaleAccumulator
from .local_revision import cumulative4, SCHEDULES

META_KEYS = ('root_id', 'split', 'cohort_id', 'flow_id', 'policy', 'time_s')


def fit_contrast_scale(records, coarse_source):
    if not records or any(r['split'] != 'train' for r in records):
        raise ValueError('Contrast scales use training roots only')
    differences = []
    i, j = np.triu_indices(65, 1)
    for record in records:
        with np.load(Path(coarse_source) / 'data' / (record['root_id'] + '.npz')) as data:
            totals = data['single_coarse'][:, :, -1].sum(1, dtype=np.float64)
            scores = data['s_incidence'] @ totals
        values = np.abs(scores[i] - scores[j])
        differences.append(values[values > 0])
    values = np.concatenate(differences)
    return max(1., float(np.percentile(values, 95))) if len(values) else 1.


def initialize_run(run, single_source, pair_source, coarse_source):
    run = Path(run)
    single_source, pair_source, coarse_source = map(lambda p: Path(p).resolve(),
                                                  (single_source, pair_source, coarse_source))
    for source in (single_source, pair_source, coarse_source):
        if run.resolve().is_relative_to(source) or source.is_relative_to(run.resolve()):
            raise ValueError('Run and source directories must be disjoint')
    selection = read(single_source / 'selection.json')
    if selection['split_roots'] != {'train': 660, 'validation': 78, 'test': 12}:
        raise ValueError('Expected v5 multi population')
    if read(single_source / 'training_temporal/summary.json')['stage'] != 'complete':
        raise ValueError('Single source training is incomplete')
    if read(pair_source / 'summary.json')['stage'] != 'complete':
        raise ValueError('Pair collection is incomplete')
    coarse_protocol = read(coarse_source / 'protocol.json')
    if (Path(coarse_protocol['source_run']).resolve() != single_source or
            coarse_protocol['source_input_normalization_sha256'] != sha256(single_source / 'derived/normalization.npz')):
        raise ValueError('Compact input cache does not match single source')
    a_records = [{k: t[k] for k in META_KEYS} for t in selection['tasks']]
    b_records = [{k: t[k] for k in META_KEYS} for t in read(pair_source / 'selection.json')['tasks']]
    if {s: sum(r['split'] == s for r in b_records) for s in ('train', 'validation')} != {'train': 36, 'validation': 18} or len(b_records) != 54:
        raise ValueError('Expected 36 train and 18 validation pair roots')
    if any(r not in a_records for r in b_records):
        raise ValueError('Pair roots must match the single-root identities and splits')
    if [r for r in a_records if r['split'] != 'test'] != read(coarse_source / 'trainval_roots.json'):
        raise ValueError('Compact input cache population or ordering changed')
    run.mkdir(parents=True, exist_ok=True)
    _freeze_json(run / 'a_roots.json', a_records)
    _freeze_json(run / 'b_roots.json', b_records)
    copy_frozen(single_source / 'derived/static.npz', run / 'static.npz')
    copy_frozen(single_source / 'derived/normalization.npz', run / 'normalization.npz')
    source_lock = read(single_source / 'checkpoints_locked.json')
    checkpoints = {}
    for seed in SEEDS:
        name = f'A_ref/seed_{seed}/best.pt'
        path = single_source / name
        digest = sha256(path)
        if digest != source_lock['checkpoints'][name]:
            raise ValueError('Frozen source A checkpoint changed')
        checkpoints[str(seed)] = {'path': str(path), 'sha256': digest}
    protocol = {'schema': 'local-decision-v8', 'single_source': str(single_source),
        'pair_source': str(pair_source), 'coarse_source': str(coarse_source),
        'source_A': checkpoints, 'seeds': list(SEEDS), 'schedules': SCHEDULES,
        'A_population': {'train': 660, 'validation': 78, 'diagnostic': 12},
        'B_population': {'train': 36, 'validation': 18}, 'horizons_s': [30, 90, 180, 240],
        'history_s': 150, 'intervention_s': 90, 'candidate_count': 65,
        'A_C_loss': 'unchanged A_ref: L5 + 0.1 total240 Huber + 0.01 gate BCE',
        'A_D_loss': 'A_C + 0.1 Huber on all 2080 same-root composed candidate gaps; TRUE SINGLES ONLY',
        'local_loss': 'natural mean position L1 + 0.1 total240 Huber + 0.01 nonzero BCE',
        'local_targets': '272 positions x 4 signed cumulative horizons; no position pooling',
        'B_target': 'true pair effect from integer inclusion-exclusion; never residual of learned A',
        'B_encoder': 'frozen seed-matched v5 A_ref; only new symmetric pair head and gate train',
        'B_selection': 'validation True-S + predicted all120 regret; pair total240 MAE; earlier update; zero baseline eligible',
        'A_selection': 'validation regret; common 272x4 joint MAE; earlier update',
        'inference_pair_budgets': [0, 24, 120], 'pair_query_batch': 128,
        'optimizer': {'name': 'AdamW', 'lr_peak': 3e-4, 'lr_final': 3e-5, 'weight_decay': 1e-4, 'clip': 1.},
        'cpu_threads_per_training_worker': 1, 'label_workers': 4, 'max_training_workers': 8,
        'precision': 'FP32; TF32 and AMP disabled; FP64 effect composition and decision metrics',
        'test_scope': 'previously inspected diagnostic; only after all twelve new checkpoints lock',
        'normalization': 'single scales copied from the exact 660-root source; pair scales only from 36 train roots',
        'source_hashes': {'single_selection': sha256(single_source / 'selection.json'),
            'pair_selection': sha256(pair_source / 'selection.json'), 'pair_summary': sha256(pair_source / 'summary.json'),
            'coarse_protocol': sha256(coarse_source / 'protocol.json'),
            'coarse_trainval_manifest': sha256(coarse_source / 'trainval_roots.json'),
            'static': sha256(run / 'static.npz'), 'normalization': sha256(run / 'normalization.npz')}}
    _freeze_json(run / 'protocol.json', protocol)
    accumulator = ScaleAccumulator()
    for record in b_records:
        if record['split'] == 'train':
            accumulator.merge(read(pair_source / 'derived/roots' / record['root_id'] / 'pair_scale_counts.json'))
    _write_numpy(run / 'pair_normalization.npz', accumulator.arrays('pair'), compressed=True)
    scale = fit_contrast_scale([r for r in a_records if r['split'] == 'train'], coarse_source)
    _freeze_json(run / 'contrast_scale.json', {'scale': scale, 'fit_split': 'train', 'roots': 660,
        'target': 'signed true-single sums, reference-first 65 candidates, 2080 upper-triangle gaps', 'weight': .1})


def convert_file(source, target):
    """Bound working memory for pair arrays; never materialize 48-window FP64 roots."""
    source, target = Path(source), Path(target)
    values = np.load(source, mmap_mode='r', allow_pickle=False)
    output = np.empty((len(values), 272, 4), dtype=np.float32)
    for offset in range(0, len(values), 64):
        output[offset:offset + 64] = cumulative4(values[offset:offset + 64])
    _write_numpy(target, output)
    return {'source_sha256': sha256(source), 'target_sha256': sha256(target), 'shape': list(output.shape)}


def prepare_one(run, record, include_pair):
    run = Path(run)
    protocol = read(run / 'protocol.json')
    directory = run / 'labels' / record['root_id']
    if record['split'] == 'test' and not (run / 'checkpoints_locked.json').exists():
        raise ValueError('Diagnostic labels are locked')
    if (directory / 'complete.json').exists():
        raise RuntimeError('Existing label cache is retained; explicit recovery required')
    directory.mkdir(parents=True, exist_ok=True)
    single = Path(protocol['single_source']) / 'derived/roots' / record['root_id']
    files = {'single4': convert_file(single / 'single.npy', directory / 'single4.npy')}
    if record['split'] != 'train':
        files['joint4'] = convert_file(single / 'joint.npy', directory / 'joint4.npy')
    if include_pair:
        pair = Path(protocol['pair_source']) / 'derived/roots' / record['root_id']
        files['pair4'] = convert_file(pair / 'pair.npy', directory / 'pair4.npy')
    result = {**record, 'stage': 'complete', 'pair': include_pair, 'files': files}
    atomic_json(directory / 'complete.json', result)
    return result


def prepare_labels(run, test=False):
    run = Path(run)
    if test and not (run / 'checkpoints_locked.json').exists():
        raise ValueError('Diagnostic labels require all twelve locked checkpoints')
    records = [r for r in read(run / 'a_roots.json') if (r['split'] == 'test') == test]
    pair_ids = {r['root_id'] for r in read(run / 'b_roots.json')} if not test else set()
    # Each root is written exactly once, even when shared by A and B.
    records.sort(key=lambda r: r['root_id'] not in pair_ids)
    done, pair_done = 0, 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(prepare_one, str(run), r, r['root_id'] in pair_ids) for r in records]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            done += 1
            pair_done += int(result['pair'])
            atomic_json(run / ('prepare_test_progress.json' if test else 'prepare_progress.json'),
                {'stage': 'preparing', 'roots': done, 'total': len(records), 'pair_roots': pair_done, 'updated_at': now()})
            if pair_done == 54 and not (run / 'labels_B_L4.ready.json').exists():
                atomic_json(run / 'labels_B_L4.ready.json', {'stage': 'complete', 'roots': 54, 'finished_at': now()})
    atomic_json(run / ('labels_test.ready.json' if test else 'labels_A_L4.ready.json'),
                {'stage': 'complete', 'roots': done, 'finished_at': now()})


def load_roots(run, family, splits=('train', 'validation')):
    run = Path(run)
    if 'test' in splits and not (run / 'checkpoints_locked.json').exists():
        raise ValueError('Diagnostic labels remain locked')
    protocol = read(run / 'protocol.json')
    records = read(run / ('b_roots.json' if family == 'B_L4' else 'a_roots.json'))
    records = [r for r in records if r['split'] in splits]
    ensure_file_limit(len(records) * 2)
    result = []
    for record in records:
        rid = record['root_id']
        # Open only explicitly selected splits; no test cache is opened during training.
        with np.load(Path(protocol['coarse_source']) / 'data' / (rid + '.npz')) as data:
            root = {**record, **{k: data[k] for k in ('normalized_history', 'base_phase', 'action_bank',
                                                      'single_nodes', 'single_actions', 's_incidence')}}
        source = Path(protocol['single_source']) / 'derived/roots' / rid
        label = run / 'labels' / rid
        if family in ('A_L4', 'B_L4'):
            root['single4'] = np.load(label / 'single4.npy', mmap_mode='r')
            if record['split'] != 'train':
                root['joint4'] = np.load(label / 'joint4.npy', mmap_mode='r')
        else:
            root['single'] = np.load(source / 'single.npy', mmap_mode='r')
            if record['split'] != 'train':
                root['single4'] = cumulative4(root['single'])
                root['joint4'] = cumulative4(np.load(source / 'joint.npy', mmap_mode='r'))
        if family == 'B_L4' or record['split'] != 'train':
            with np.load(source / 'inputs.npz') as inputs:
                for key in ('pair_nodes', 'pair_actions', 'pair_query_pair_index', 'p_incidence', 'joint_sizes'):
                    root[key] = inputs[key]
        if family == 'B_L4':
            root['pair4'] = np.load(label / 'pair4.npy', mmap_mode='r')
        result.append(root)
    return result
