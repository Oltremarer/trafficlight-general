"""Persistent single-effect revision scheduler; no repeated agent monitoring."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .collect_counterfactual import stamp
from .counterfactual.writer import atomic_json, sha256
from .run_formal_effects import command_environment


def read(path):
    return json.loads(Path(path).read_text())


def supervise(run, dataset, source_run, workers=20, gpu_jobs=6):
    import fcntl
    from .collect_single_revision import prepare

    run = Path(run).resolve()
    if not Path('/mnt/pan').is_mount() or not run.is_relative_to(Path('/mnt/pan')):
        raise ValueError('Mounted /mnt/pan required; no system-disk output fallback')
    run.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'tmp', 'cache'):
        (run / name).mkdir(exist_ok=True)
    with (run / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'launcher.json').exists():
            raise RuntimeError('Existing launch retained; refusing duplicate dispatch')
        selection = prepare(run, Path(dataset), Path(source_run))
        hashes = {str(p.relative_to(Path(__file__).parent)): sha256(p)
                  for p in sorted(Path(__file__).parent.rglob('*.py'))}
        atomic_json(run / 'source_hashes.json', hashes)
        state = {'stage': 'running', 'created_at': stamp(), 'supervisor_pid': os.getpid(),
                 'run_dir': str(run), 'workers': workers, 'gpu_jobs': gpu_jobs,
                 'root_count': len(selection['tasks']), 'collection_branches': len(selection['tasks']) * 129,
                 'stages': {'collection': 'pending', 'data': 'pending', 'training': 'pending',
                            'evaluation': 'pending'},
                 'test_scope': 'previously inspected demand; diagnostic, not a fresh blind test'}
        children, launches = {}, {}

        def save():
            state['updated_at'] = stamp()
            atomic_json(run / 'execution.json', state)

        def launch(name, module, extras, threads):
            command = [sys.executable, '-u', '-m', module, '--run-dir', str(run), *extras]
            logfile = run / 'logs' / (name + '.log')
            with logfile.open('ab') as stream:
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        env=command_environment(run, threads))
            children[name] = proc
            launches[name] = {'pid': proc.pid, 'command': command, 'log': str(logfile), 'at': stamp()}
            state['stages'][name] = 'running'
            atomic_json(run / 'launcher.json', {'supervisor_pid': os.getpid(), 'jobs': launches})
            save()

        save()
        try:
            launch('collection', 'cityflow_tsc.collect_single_revision',
                   ['--dataset', str(dataset), '--source-run', str(source_run),
                    '--stage', 'collect', '--workers', str(workers)], 1)
            while True:
                for name, proc in children.items():
                    result = proc.poll()
                    if result is None or state['stages'][name] != 'running':
                        continue
                    if result:
                        state['stages'][name] = 'failed'
                        raise RuntimeError(f'{name} exited {result}; see {launches[name]["log"]}')
                    markers = {'collection': 'summary.json', 'training': 'checkpoints_locked.json',
                               'evaluation': 'evaluation/summary.json'}
                    marker = run / markers[name]
                    if not marker.is_file():
                        raise RuntimeError(f'{name} exited without completion evidence: {marker}')
                    if name != 'training' and read(marker).get('stage') not in ('complete', 'completed'):
                        raise RuntimeError(f'{name} summary is not complete')
                    state['stages'][name] = 'complete'
                    save()
                stages = state['stages']
                if (run / 'trainval_ready.json').exists() and stages['data'] == 'pending':
                    from .effect_model.formal_data import prepare_dataset
                    stages['data'] = 'running'
                    save()
                    prepare_dataset(run, include_pairs=False, splits=('train', 'validation'))
                    stages['data'] = 'complete'
                    launch('training', 'cityflow_tsc.train_single_revision',
                           ['--stage', 'train', '--workers', str(gpu_jobs)], 2)
                if stages['training'] == 'complete' and stages['collection'] == 'complete' and stages['evaluation'] == 'pending':
                    from .effect_model.formal_data import prepare_dataset
                    prepare_dataset(run, include_pairs=False, splits=('test',))
                    launch('evaluation', 'cityflow_tsc.train_single_revision', ['--stage', 'evaluate'], 2)
                if stages['evaluation'] == 'complete':
                    state.update(stage='complete', finished_at=stamp())
                    save()
                    atomic_json(run / 'complete.json', state)
                    return
                time.sleep(20)
        except BaseException as exc:
            state.update(stage='attention_required', error=repr(exc))
            save()
            atomic_json(run / 'failure.json', {'at': stamp(), 'error': repr(exc),
                        'traceback': traceback.format_exc(),
                        'active_children_retained': {k: p.pid for k, p in children.items() if p.poll() is None}})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--gpu-jobs', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24 or not 1 <= args.gpu_jobs <= 6:
        parser.error('workers must be 1..24 and GPU jobs 1..6')
    supervise(args.run_dir, args.dataset, args.source_run, args.workers, args.gpu_jobs)


if __name__ == '__main__':
    main()
