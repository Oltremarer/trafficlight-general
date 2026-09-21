"""Balanced multi-time pair collection and oracle structural analysis, no training."""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import copy
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil

import numpy as np

from . import collect_formal_effects as formal
from .collect_counterfactual import stamp
from .collect_temporal_effects import verify_reused
from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_data import _derive_pairs, _read_npz
from .effect_model.formal_model import decision_metrics, group_mean


def read(path):
    return json.loads(Path(path).read_text())


def select_tasks(selection):
    """Predeclared lexicographic flow IDs in each cohort, never outcome based."""
    chosen = []
    for split, flows_per_cohort in (('train', 2), ('validation', 1)):
        tasks = [t for t in selection['tasks'] if t['split'] == split]
        cohorts = sorted({t['cohort_id'] for t in tasks})
        if len(cohorts) != 3:
            raise ValueError('Expected three cohorts per split')
        for cohort in cohorts:
            members = [t for t in tasks if t['cohort_id'] == cohort]
            flows = sorted({t['flow_id'] for t in members})[:flows_per_cohort]
            if len(flows) != flows_per_cohort:
                raise ValueError('Insufficient flows in cohort')
            for flow in flows:
                roots = [t for t in members if t['flow_id'] == flow]
                expected = {(p, t) for p in ('fixed_time', 'max_pressure') for t in (600, 1800, 2700)}
                if len(roots) != 6 or {(t['policy'], t['time_s']) for t in roots} != expected:
                    raise ValueError('Each chosen flow requires both policies at all three times')
                chosen.extend(copy.deepcopy(roots))
    if Counter(t['split'] for t in chosen) != {'train': 36, 'validation': 18}:
        raise ValueError('Unexpected population')
    chosen.sort(key=lambda t: (t['split'] != 'validation', t['time_s'], t['cohort_id'], t['flow_id'], t['policy']))
    for t in chosen:
        t['collected_branch_count'] = 2049
    return chosen


def link_file(source, target):
    source, target = Path(source).resolve(), Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source:
            raise ValueError('Changed immutable reuse source')
    elif target.exists():
        raise ValueError('Refusing to replace an existing payload')
    else:
        target.symlink_to(source)


def prepare(run, source):
    run, source = Path(run).resolve(), Path(source).resolve()
    if run == source or run.is_relative_to(source) or source.is_relative_to(run):
        raise ValueError('Separate source and destination trees required')
    if read(source / 'complete.json').get('stage') != 'complete':
        raise ValueError('Completed v5 source required')
    protocol = read(source / 'protocol.json')
    if sha256(importlib.import_module('cityflow').__file__) != protocol['engine_binary_sha256']:
        raise ValueError('CityFlow binary differs from the paired source')
    parent = read(source / 'selection.json')
    tasks = select_tasks(parent)
    run.mkdir(parents=True, exist_ok=True)
    selection = {'schema': 'multi-time-interaction-v6', 'tasks': tasks, 'split_roots': {'train': 36, 'validation': 18},
        'source_run': str(source), 'source_selection_sha256': sha256(source / 'selection.json'),
        'selection_rule': 'first 2 training / first 1 validation flow IDs in each cohort; both policies; all 3 times'}
    formal._freeze_json(run / 'selection.json', selection)
    formal._freeze_json(run / 'protocol.json', {'schema': 'multi-time-interaction-v6',
        'source_protocol_sha256': sha256(source / 'protocol.json'),
        'engine_binary_sha256': protocol['engine_binary_sha256'],
        'history_s': 150, 'intervention_s': 90, 'horizon_s': 240,
        'source_run': str(source), 'root_count': 54, 'branch_count': 110646, 'branch_upper_bound': 110646,
        'new_branches': 103680, 'reused_initial_branches': 6966, 'train_roots': 36, 'validation_roots': 18,
        'test_roots': 0, 'training_jobs': 0, 'workers_max': 20,
        'oracle_top_k': 24, 'pair_strength': 'maximum absolute 240s global signed effect over 16 action combinations',
        'oracle_scope': 'uses true pair labels to measure sparsity potential; not a deployable selection policy'})
    if (run / 'reuse_manifest.json').exists():
        return selection
    hashes = {}
    for i, task in enumerate(tasks):
        rid = task['root_id']
        hashes.update(verify_reused(source, task))
        root_target = run / 'roots' / rid
        if not root_target.exists():
            # Root metadata is private; a collector can never rewrite the old root.
            shutil.copytree(source / 'roots' / rid, root_target)
        for path in (source / 'shards' / rid).iterdir():
            if path.is_file():
                # Only immutable existing payloads are linked. The shard directory is new.
                link_file(path, run / 'shards' / rid / path.name)
        for name in (f'{rid}.initial.complete.json', f"{task['task_id']}.roots.json"):
            path = source / 'indexes' / name
            if path.exists():
                formal._freeze_json(run / 'indexes' / name, read(path))
        for name in ('inputs.npz', 'single.npy', 'single_int64.npy', 'joint.npy', 'baseline.npy'):
            link_file(source / 'arms/multi/derived/roots' / rid / name, run / 'derived/roots' / rid / name)
        atomic_json(run / 'prepare_progress.json', {'stage': 'reusing_initial', 'prepared_roots': i + 1, 'total_roots': 54})
    formal._freeze_json(run / 'reuse_manifest.json', {'source_run': str(source), 'files': hashes,
        'source_selection_sha256': sha256(source / 'selection.json'), 'reused_branches': 6966})
    return selection


def collect(run, source, workers=20):
    import fcntl
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'collector.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = prepare(run, source)
        results, errors = [], []
        def save(stage):
            value = {'stage': stage, 'workers': workers, 'completed_roots': len(results), 'total_roots': 54,
                'new_completed_branches': len(results) * 1920, 'new_branch_budget': 103680,
                'reused_branches': 6966, 'errors': errors, 'updated_at': stamp()}
            atomic_json(run / 'status.json', value)
            return value
        save('running')
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(formal.collect_root, str(run), t, 'remaining'): t for t in selection['tasks']}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result['branch_count'] != 2049 or result['stage'] != 'complete':
                        raise ValueError('Unexpected pair collection commit')
                    results.append(result)
                except Exception as exc:
                    errors.append({'root_id': futures[future]['root_id'], 'error': repr(exc)})
                save('running')
        summary = save('failed' if errors or len(results) != 54 else 'complete')
        summary['results'] = results
        summary['unresolved_events'] = sum(r['unresolved_events'] for r in results)
        atomic_json(run / 'summary.json', summary)
        if summary['stage'] != 'complete':
            raise RuntimeError('Pair collection errors retained')


def reconstruct(single, pair, arrays, keep_pairs=None):
    """Sum only active queries in float64; avoid dense 65 x 1920 field products."""
    output = []
    for si, pi in zip(arrays['s_incidence'], arrays['p_incidence']):
        value = np.asarray(single[np.flatnonzero(si)], dtype=np.float64).sum(0)
        indices = np.flatnonzero(pi)
        if keep_pairs is not None:
            indices = indices[np.isin(arrays['pair_query_pair_index'][indices], keep_pairs)]
        if pair is not None and len(indices):
            value += np.asarray(pair[indices], dtype=np.float64).sum(0)
        output.append(value)
    return np.stack(output)


def slice_metrics(predicted, truth, indices):
    p, y = predicted[indices], truth[indices]
    scores, real = p.sum((1, 2)), y.sum((1, 2))
    # Reference stays first in every size-specific menu.
    metrics = decision_metrics(scores, real)
    metrics['selected'] = int(indices[metrics['selected']])
    denominator = np.abs(y).sum()
    metrics.update(field_mae=float(np.abs(p - y).mean()),
        relative_field_l1=float(np.abs(p - y).sum() / denominator) if denominator else None)
    return metrics


def analyze(run):
    run = Path(run)
    if read(run / 'summary.json')['stage'] != 'complete':
        raise ValueError('All pair branches required')
    selection = read(run / 'selection.json')
    rows = []
    for ordinal, task in enumerate(selection['tasks']):
        path = run / 'derived/roots' / task['root_id']
        _derive_pairs(run, task)
        arrays = _read_npz(path / 'inputs.npz')
        single = np.load(path / 'single.npy', mmap_mode='r', allow_pickle=False)
        pair = np.load(path / 'pair.npy', mmap_mode='r', allow_pickle=False)
        truth = np.asarray(np.load(path / 'joint.npy', mmap_mode='r'), dtype=np.float64)
        strength = np.load(path / 'rank_targets.npy')
        top = np.argsort(-strength, kind='stable')[:24]
        base = {k: task[k] for k in ('root_id', 'split', 'flow_id', 'cohort_id', 'policy', 'time_s')}
        root_rows = []
        for method, keep in (('True-S', []), ('True-SP', None), ('Oracle-Top24', top)):
            predicted = reconstruct(single, None if method == 'True-S' else pair, arrays, keep)
            for size in (0, 3, 4, 8, 16):
                indices = np.arange(65) if size == 0 else np.flatnonzero((arrays['joint_sizes'] == size) | (arrays['joint_sizes'] == 0))
                root_rows.append({**base, 'method': method, 'menu': 'all65' if size == 0 else f'size{size}+reference',
                    **slice_metrics(predicted, truth, indices),
                    'top24_strength_mass': float(strength[top].sum() / strength.sum()) if strength.sum() else None})
        atomic_json(run / 'analysis/roots' / f"{task['root_id']}.json", root_rows)
        rows.extend(root_rows)
        atomic_json(run / 'analysis/progress.json', {'roots_analyzed': ordinal + 1, 'total': 54, 'updated_at': stamp()})
        del single, pair, truth
    summaries = []
    for split in ('train', 'validation'):
        for t in ('all', 600, 1800, 2700):
            for method in ('True-S', 'True-SP', 'Oracle-Top24'):
                for menu in ('all65', 'size3+reference', 'size4+reference', 'size8+reference', 'size16+reference'):
                    subset = [r for r in rows if r['split'] == split and r['method'] == method and r['menu'] == menu
                              and (t == 'all' or r['time_s'] == t)]
                    metrics = {key: group_mean(subset, key) for key in ('regret', 'field_mae', 'optimal_choice', 'worse_than_reference')}
                    relative = [r for r in subset if r['relative_field_l1'] is not None]
                    metrics['relative_field_l1'] = group_mean(relative, 'relative_field_l1') if relative else None
                    summaries.append({'split': split, 'time_s': t, 'method': method, 'menu': menu, 'roots': len(subset), **metrics})
    atomic_json(run / 'analysis/summary.json', {'stage': 'complete', 'summaries': summaries,
        'records': rows, 'finished_at': stamp(), 'learned_pair_model': False,
        'scope': 'true-factor structural analysis; Oracle-Top24 uses true pair labels and is not deployable'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--stage', choices=('collect', 'analyze'), required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not Path('/mnt/pan').is_mount() or not run.is_relative_to(Path('/mnt/pan')):
        raise ValueError('Mounted /mnt/pan required')
    if not 1 <= args.workers <= 20:
        parser.error('workers must be 1..20')
    if args.stage == 'collect':
        if args.source_run is None:
            parser.error('--source-run required for collection')
        collect(run, args.source_run, args.workers)
    else:
        analyze(run)


if __name__ == '__main__':
    main()
