"""Persistent scheduling for the matched early versus multi-time experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .collect_counterfactual import stamp
from .counterfactual.writer import atomic_json, sha256
from .run_formal_effects import command_environment
from .run_single_revision import read


def supervise(run, source_run, workers=20, resume=False):
    import fcntl
    from .collect_temporal_effects import prepare
    from .effect_model.formal_data import prepare_dataset

    run = Path(run).resolve()
    if not Path('/mnt/pan').is_mount() or not run.is_relative_to(Path('/mnt/pan')):
        raise ValueError('Mounted /mnt/pan required; no output fallback')
    run.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'tmp', 'cache'):
        (run / name).mkdir(exist_ok=True)
    with (run / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'launcher.json').exists() and not resume:
            raise RuntimeError('Existing dispatch retained; refusing duplicate launch')
        state = {'stage': 'preparing', 'created_at': stamp(), 'supervisor_pid': os.getpid(),
                 'run_dir': str(run), 'workers': workers, 'training_jobs_per_arm': 3,
                 'reused_branches': 33282, 'new_branches': 63468, 'total_branches': 96750,
                 'updates_per_job': 132000, 'seeds': [42, 43, 44],
                 'stages': {name: 'pending' for name in ('collection', 'early_data',
                     'early_training', 'multi_data', 'multi_training', 'early_evaluation', 'multi_evaluation')},
                 'test_scope': 'Previously inspected diagnostic; not a fresh blind test'}
        children, launches = {}, {}
        if resume:
            state = read(run / 'execution.json')
            if state['stage'] != 'attention_required' or state['stages']['collection'] != 'complete':
                raise RuntimeError('Recovery requires a stopped run with completed collection')
            state.update(stage='resuming', supervisor_pid=os.getpid(), resumed_at=stamp())
            state.pop('error', None)
            for arm in ('early', 'multi'):
                if state['stages'][f'{arm}_training'] == 'failed':
                    state['stages'][f'{arm}_training'] = 'pending'
            launches = read(run / 'launcher.json')['jobs']

        def save():
            state['updated_at'] = stamp()
            atomic_json(run / 'execution.json', state)

        def launch(name, module, target, extras, threads):
            command = [sys.executable, '-u', '-m', module, '--run-dir', str(target), *extras]
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
            if not resume:
                prepare(run, Path(source_run))
            atomic_json(run / ('recovery_source_hashes.json' if resume else 'source_hashes.json'), {
                str(p.relative_to(Path(__file__).parent)): sha256(p)
                for p in sorted(Path(__file__).parent.rglob('*.py'))})
            state['stage'] = 'running'
            if not resume:
                launch('collection', 'cityflow_tsc.collect_temporal_effects', run,
                       ['--source-run', str(source_run), '--stage', 'collect', '--workers', str(workers)], 1)
            while True:
                for name, proc in children.items():
                    code = proc.poll()
                    if code is None or state['stages'][name] != 'running':
                        continue
                    if code:
                        state['stages'][name] = 'failed'
                        raise RuntimeError(f'{name} exited {code}; see {launches[name]["log"]}')
                    if name == 'collection':
                        marker = run / 'summary.json'
                    else:
                        arm, kind = name.split('_', 1)
                        marker = run / 'arms' / arm / ('checkpoints_locked.json' if kind == 'training'
                                                       else 'evaluation/summary.json')
                    if not marker.is_file():
                        raise RuntimeError(f'{name} missing completion evidence: {marker}')
                    if not name.endswith('_training') and read(marker).get('stage') not in ('complete', 'completed'):
                        raise RuntimeError(f'{name} summary is not complete')
                    state['stages'][name] = 'complete'
                    save()
                stages = state['stages']
                for arm in ('early', 'multi'):
                    target = run / 'arms' / arm
                    if (run / f'{arm}_ready.json').is_file() and stages[f'{arm}_data'] == 'pending':
                        stages[f'{arm}_data'] = 'running'
                        save()
                        prepare_dataset(target, include_pairs=False, splits=('train', 'validation'))
                        stages[f'{arm}_data'] = 'complete'
                    if stages[f'{arm}_data'] == 'complete' and stages[f'{arm}_training'] == 'pending':
                        launch(f'{arm}_training', 'cityflow_tsc.train_temporal_effects', target,
                               ['--stage', 'train', '--workers', '3', '--arm', arm], 2)
                # Neither arm may read diagnostic labels until both models are locked.
                if stages['collection'] == 'complete' and all(
                        stages[f'{arm}_training'] == 'complete' for arm in ('early', 'multi')):
                    for arm in ('early', 'multi'):
                        if stages[f'{arm}_evaluation'] == 'pending':
                            target = run / 'arms' / arm
                            prepare_dataset(target, include_pairs=False, splits=('test',))
                            launch(f'{arm}_evaluation', 'cityflow_tsc.train_temporal_effects', target,
                                   ['--stage', 'evaluate', '--arm', arm], 2)
                if all(stages[f'{arm}_evaluation'] == 'complete' for arm in ('early', 'multi')):
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
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 20:
        parser.error('workers must be 1..20, leaving CPU capacity for overlapping training')
    supervise(args.run_dir, args.source_run, args.workers, args.resume)


if __name__ == '__main__':
    main()
