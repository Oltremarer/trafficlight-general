"""Isolated v7 A revision: delay aggregation, retaining the v6 data/loss/budget."""
from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .collect_formal_effects import _freeze_json
from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_data import _read_npz
from .train_coarse_effects import CoarseEffectModel, evaluate_model, load_roots, read, train_job
from .train_formal_effects import initialize, now
from .train_temporal_effects import MAX_UPDATES, SEEDS


def copy_frozen(source, target):
    source, target = Path(source), Path(target)
    digest = sha256(source)
    if target.exists():
        if target.is_symlink() or sha256(target) != digest:
            raise ValueError(f'Retained private metadata differs: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return digest


def link_frozen(source, target):
    source, target = Path(source).resolve(strict=True), Path(target)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source:
            raise ValueError(f'Retained data link differs: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    return sha256(source)


def prepare_cache(run, reuse, test=False):
    run, reuse = Path(run), Path(reuse).resolve()
    if test and not (run / 'checkpoints_locked.json').is_file():
        raise ValueError('All new checkpoints must be locked before diagnostic labels')
    source_protocol = read(reuse / 'protocol.json')
    if source_protocol['schema'] != 'coarse-action-consequences-v6':
        raise ValueError('Expected the completed v6 early-pooling cache')
    if read(reuse / 'training/summary.json')['stage'] != 'complete':
        raise ValueError('Source training must be complete')
    source_lock = read(reuse / 'checkpoints_locked.json')
    for name, key in (('protocol.json', 'protocol_sha256'), ('target_scales.npy', 'scales_sha256'),
                      ('static.npz', 'static_sha256')):
        if sha256(reuse / name) != source_lock[key]:
            raise ValueError(f'Source training metadata changed: {name}')
    run.mkdir(parents=True, exist_ok=True)
    protocol = {**source_protocol, 'schema': 'coarse-latepool-v7', 'pooling': 'late',
        'decoder': 'same 272-128-64-4 MLP per recipient, then mean by location group',
        'learned_output': '12 group consequences; recipient contributions have no physical supervision',
        'reuse_coarse_run': str(reuse), 'reuse_protocol_sha256': sha256(reuse / 'protocol.json'),
        'reuse_trainval_manifest_sha256': sha256(reuse / 'trainval_roots.json'),
        'cpu_threads_per_worker': 1, 'workers': 3,
        'architecture_change': 'nonlinear decoding before instead of after location aggregation',
        'parameters_loss_scales_and_update_budget_unchanged': True,
        'resource_policy': 'new processes only: nice 15, idle IO; never signal or modify B'}
    _freeze_json(run / 'protocol.json', protocol)
    for name in ('static.npz', 'target_scales.npy'):
        copy_frozen(reuse / name, run / name)
    manifest = 'test_roots.json' if test else 'trainval_roots.json'
    records = read(reuse / manifest)
    counts = {s: sum(r['split'] == s for r in records) for s in {r['split'] for r in records}}
    expected = {'test': 12} if test else {'train': 660, 'validation': 78}
    if counts != expected or len({r['root_id'] for r in records}) != len(records):
        raise ValueError('Frozen root population mismatch')
    hashes = {}
    for i, record in enumerate(records):
        name = record['root_id'] + '.npz'
        hashes[name] = link_frozen(reuse / 'data' / name, run / 'data' / name)
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            atomic_json(run / 'prepare_progress.json', {'stage': 'test' if test else 'trainval',
                'prepared': i + 1, 'total': len(records)})
    _freeze_json(run / manifest, records)
    _freeze_json(run / ('test_reuse_hashes.json' if test else 'trainval_reuse_hashes.json'), hashes)
    atomic_json(run / ('test_ready.json' if test else 'trainval_ready.json'),
                {'stage': 'complete', 'roots': len(records), 'finished_at': now()})


def lock_checkpoints(run):
    atomic_json(run / 'checkpoints_locked.json', {'checkpoints': {
        f'coarse/seed_{s}/best.pt': sha256(run / 'coarse' / f'seed_{s}' / 'best.pt') for s in SEEDS},
        'scales_sha256': sha256(run / 'target_scales.npy'), 'protocol_sha256': sha256(run / 'protocol.json'),
        'static_sha256': sha256(run / 'static.npz'),
        'trainval_reuse_sha256': sha256(run / 'trainval_reuse_hashes.json'),
        'locked_at': now(), 'test_used': False})


def evaluate(run, reuse):
    lock = read(run / 'checkpoints_locked.json')
    for name, digest in lock['checkpoints'].items():
        if sha256(run / name) != digest:
            raise ValueError('Locked checkpoint changed')
    for name, key in (('target_scales.npy', 'scales_sha256'), ('protocol.json', 'protocol_sha256'),
                      ('static.npz', 'static_sha256'), ('trainval_reuse_hashes.json', 'trainval_reuse_sha256')):
        if sha256(run / name) != lock[key]:
            raise ValueError('Locked training metadata changed')
    prepare_cache(run, reuse, test=True)
    roots = load_roots(run, test=True)
    initialize(42)
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    scales = torch.as_tensor(np.load(run / 'target_scales.npy').astype(np.float32), device=device)
    results, validation_results = {}, {}
    for seed in SEEDS:
        model = CoarseEffectModel(_read_npz(run / 'static.npz'), pooling='late').to(device)
        saved = torch.load(run / 'coarse' / f'seed_{seed}' / 'best.pt', map_location=device, weights_only=False)
        if saved['pooling'] != 'late':
            raise ValueError('Checkpoint model order mismatch')
        model.load_state_dict(saved['state_dict'])
        results[str(seed)] = evaluate_model(model, roots, scales, device)
        validation_results[str(seed)] = read(run / 'coarse' / f'seed_{seed}' / 'best_validation.json')
        del model, saved
    prior = read(reuse / 'evaluation/summary.json')
    baselines = {'coarse12_early': prior, 'dense_A_ref': prior['baseline_A_ref']}
    comparison = {}
    for name, result in {**baselines, 'coarse12_late': {'results': results, 'validation_results': validation_results}}.items():
        comparison[name] = {}
        for split in ('validation_results', 'results'):
            values = [result[split][f'A_ref/seed_{s}' if name == 'dense_A_ref' else str(s)] for s in SEEDS]
            metrics = [v.get('metrics', v) for v in values]
            comparison[name][split] = {'regret_seeds': [m['regret'] for m in metrics],
                'regret_mean': float(np.mean([m['regret'] for m in metrics])),
                'worse_than_reference_count': sum(r['worse_than_reference'] for m in metrics for r in m['records']),
                'root_seed_count': sum(len(m['records']) for m in metrics)}
    atomic_json(run / 'evaluation/summary.json', {'stage': 'complete', 'results': results,
        'validation_results': validation_results, 'comparison': comparison,
        'test_scope': 'previously inspected diagnostic, not blind test',
        'aggregate_MAE_is_not_field_MAE': True, 'finished_at': now()})


def supervise(run, reuse):
    import fcntl
    run, reuse = Path(run).resolve(), Path(reuse).resolve()
    if (not Path('/mnt/pan').is_mount() or not run.is_relative_to(Path('/mnt/pan'))
            or run == Path('/mnt/pan') or run.is_relative_to(reuse) or reuse.is_relative_to(run)):
        raise ValueError('Separate output tree on mounted /mnt/pan required')
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'supervisor.lock').open('a') as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'execution.json').exists():
            raise RuntimeError('Duplicate launch refused; existing work is preserved')
        state = {'stage': 'preparing', 'started_at': now(), 'pid': os.getpid(),
            'run_dir': str(run), 'reuse_coarse_run': str(reuse), 'seeds': list(SEEDS),
            'updates_per_seed': MAX_UPDATES, 'cpu_threads_per_worker': 1}
        atomic_json(run / 'execution.json', state)
        try:
            prepare_cache(run, reuse)
            state.update(stage='training', updated_at=now())
            atomic_json(run / 'execution.json', state)
            results, errors = [], []
            with concurrent.futures.ProcessPoolExecutor(max_workers=3,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {pool.submit(train_job, str(run), seed): seed for seed in SEEDS}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        errors.append({'seed': futures[future], 'error': repr(exc)})
                    atomic_json(run / 'training/queue.json', {'completed': len(results), 'total': 3, 'errors': errors})
            atomic_json(run / 'training/summary.json',
                {'stage': 'failed' if errors else 'complete', 'results': results, 'errors': errors})
            if errors:
                raise RuntimeError('Seed training failed; artifacts retained')
            lock_checkpoints(run)
            state.update(stage='evaluating', updated_at=now())
            atomic_json(run / 'execution.json', state)
            evaluate(run, reuse)
            state.update(stage='complete', finished_at=now())
            atomic_json(run / 'execution.json', state)
            atomic_json(run / 'complete.json', state)
        except BaseException as exc:
            state.update(stage='failed', error=repr(exc), updated_at=now())
            atomic_json(run / 'execution.json', state)
            atomic_json(run / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--reuse-coarse-run', type=Path, required=True)
    args = parser.parse_args()
    supervise(args.run_dir, args.reuse_coarse_run)


if __name__ == '__main__':
    main()
