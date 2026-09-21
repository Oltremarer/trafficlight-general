"""Collect only the 108 approved new pair roots, with four independent workers."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import multiprocessing
from pathlib import Path
import shutil

from . import collect_formal_effects as formal
from .collect_interaction_v6 import link_file
from .collect_temporal_effects import verify_reused
from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_data import _derive_pairs
from .effect_model.local_data import convert_file
from .train_coarse_effects import read
from .train_formal_effects import now


def collect_one(run, task):
    run = Path(run)
    collection, rid = run / 'collection', task['root_id']
    source = Path(read(run / 'protocol.json')['collection_source'])
    hashes = verify_reused(source, task)  # Commit integrity, no simulator consistency replay.
    target = collection / 'roots' / rid
    if not target.exists():
        shutil.copytree(source / 'roots' / rid, target)
    for path in (source / 'shards' / rid).iterdir():
        if path.is_file():
            link_file(path, collection / 'shards' / rid / path.name)
    for name in (f'{rid}.initial.complete.json', f"{task['task_id']}.roots.json"):
        path = source / 'indexes' / name
        if path.exists():
            formal._freeze_json(collection / 'indexes' / name, read(path))
    for name in ('inputs.npz', 'single.npy', 'single_int64.npy', 'joint.npy', 'baseline.npy'):
        link_file(source / 'arms/multi/derived/roots' / rid / name, collection / 'derived/roots' / rid / name)
    atomic_json(collection / 'reuse' / f'{rid}.json', hashes)
    result = formal.collect_root(str(collection), task, 'remaining')
    if result['stage'] != 'complete' or result['branch_count'] != 2049:
        raise ValueError('Incomplete pair root')
    _derive_pairs(collection, task)
    label = run / 'labels' / rid / 'pair4.npy'
    converted = convert_file(collection / 'derived/roots' / rid / 'pair.npy', label)
    atomic_json(collection / 'converted' / f'{rid}.json', converted)
    return {**result, 'pair4': converted}


def collect(run):
    import fcntl
    run = Path(run).resolve()
    if not Path('/mnt/pan').is_mount() or not run.is_relative_to(Path('/mnt/pan')):
        raise ValueError('Mounted /mnt/pan required')
    collection = run / 'collection'
    for name in ('logs', 'locks', 'tmp', 'cache'):
        (collection / name).mkdir(parents=True, exist_ok=True)
    with (collection / 'collector.lock').open('a') as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = Path(read(run / 'protocol.json')['collection_source'])
        source_protocol = read(source / 'protocol.json')
        if sha256(importlib.import_module('cityflow').__file__) != source_protocol['engine_binary_sha256']:
            raise ValueError('CityFlow binary differs from source counterfactuals')
        tasks = read(collection / 'selection.json')['tasks']
        if len(tasks) != 108 or any(t['split'] != 'train' for t in tasks):
            raise ValueError('Only approved training roots allowed')
        formal._freeze_json(collection / 'protocol.json', {'source_run': str(source),
            'source_protocol_sha256': sha256(source / 'protocol.json'), 'engine_binary_sha256': source_protocol['engine_binary_sha256'],
            'new_pair_branches': 207360, 'reused_initial_branches': 13932, 'workers': 4, 'consistency_replay': False})
        results, errors = [], []

        def save(stage):
            status = {'stage': stage, 'workers': 4, 'completed_roots': len(results), 'total_roots': 108,
                'new_completed_branches': len(results) * 1920, 'new_branch_budget': 207360,
                'errors': errors, 'updated_at': now()}
            atomic_json(collection / 'status.json', status)
            return status

        save('running')
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(collect_one, str(run), t): t for t in tasks}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({'root_id': futures[future]['root_id'], 'error': repr(exc)})
                save('running')
        status = save('complete' if len(results) == 108 and not errors else 'failed')
        atomic_json(collection / 'summary.json', {**status, 'results': results})
        if status['stage'] != 'complete':
            raise RuntimeError('Pair collection failures retained; no B training')
        atomic_json(collection / 'complete.json', {**status,
            'label_hashes': {r['root_id']: r['pair4']['target_sha256'] for r in results},
            'unresolved_events': sum(r['unresolved_events'] for r in results), 'finished_at': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    collect(parser.parse_args().run_dir)
