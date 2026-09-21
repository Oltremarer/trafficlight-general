"""Persistent v11 queue: three A jobs alongside four CPU collectors, then B."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .counterfactual.writer import atomic_json, sha256
from .effect_model import coverage_revision as v11
from .evaluate_coverage_revision import lock_checkpoints
from .run_formal_effects import command_environment, read
from .run_local_revision import resource_snapshot, resource_allows
from .train_formal_effects import now
from .train_temporal_effects import SEEDS


def supervise(run, source_v10):
    import fcntl
    run = Path(run).resolve()
    if (not Path('/mnt/pan').is_mount() or run == Path('/mnt/pan') or not run.is_relative_to(Path('/mnt/pan'))):
        raise ValueError('Mounted /mnt/pan required; no system-disk fallback')
    run.mkdir(parents=True, exist_ok=True)
    if not os.access(run, os.W_OK) or shutil.disk_usage(run).free < 100_000_000_000:
        raise OSError('Writable run and sufficient collection storage required')
    with (run / 'supervisor.lock').open('a') as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'execution.json').exists():
            raise RuntimeError('Duplicate launch refused; retained jobs require explicit recovery')
        for name in ('logs', 'tmp', 'cache'):
            (run / name).mkdir(exist_ok=True)
        children = {}
        state = {'stage': 'preparing', 'pid': os.getpid(), 'run_dir': str(run), 'started_at': now(),
                 'max_training_workers': 3, 'collection_workers': 4, 'jobs': {}, 'errors': []}

        def save():
            state['updated_at'] = now()
            atomic_json(run / 'execution.json', state)

        def launch(name, module, extras=()):
            logfile = run / 'logs' / (name.replace('/', '_') + '.log')
            command = ['nice', '-n', '15', 'ionice', '-c', '3', sys.executable, '-u', '-m', module,
                       '--run-dir', str(run), *extras]
            with logfile.open('x') as stream:
                process = subprocess.Popen(command, env=command_environment(run, 1), stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT)
            children[name] = process
            return {'stage': 'running', 'pid': process.pid, 'command': command, 'log': str(logfile), 'started_at': now()}

        save()
        try:
            v11.initialize_run(run, source_v10)
            source = Path(__file__).parent
            atomic_json(run / 'source_hashes.json', {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*.py'))})
            state['collection'] = launch('collection', 'cityflow_tsc.collect_coverage_revision')
            for family in v11.FAMILIES:
                for seed in SEEDS:
                    state['jobs'][f'{family}/seed_{seed}'] = {'stage': 'pending', 'family': family, 'seed': seed}
            state['stage'] = 'training_and_collecting'
            save()
            while True:
                collector = children['collection']
                if state['collection']['stage'] == 'running' and collector.poll() is not None:
                    done = run / 'collection/complete.json'
                    success = collector.returncode == 0 and done.exists() and read(done)['stage'] == 'complete'
                    state['collection'].update(stage='complete' if success else 'failed', exit_code=collector.returncode, finished_at=now())
                    if not success:
                        state['errors'].append('Collection failed; A continues, B remains blocked')
                for name, job in state['jobs'].items():
                    if job['stage'] == 'running' and children[name].poll() is not None:
                        code = children[name].returncode
                        done = run / name / 'complete.json'
                        success = code == 0 and done.exists() and read(done)['stage'] == 'complete'
                        job.update(stage='complete' if success else 'failed', exit_code=code, finished_at=now())
                        if success:
                            job['result'] = read(done)
                        else:
                            state['errors'].append(name + ' failed; independent jobs continue')
                for name, job in state['jobs'].items():
                    if job['stage'] != 'pending':
                        continue
                    if job['family'] == 'B_Coverage' and state['collection']['stage'] != 'complete':
                        if state['collection']['stage'] == 'failed':
                            job.update(stage='blocked', reason='Required pair data incomplete')
                        continue
                    active = [(n, j) for n, j in state['jobs'].items() if j['stage'] == 'running']
                    loading = sum(not (run / n / 'status.json').exists() or read(run / n / 'status.json').get('stage') == 'loading'
                                  for n, _ in active)
                    resources = resource_snapshot()
                    state['resources'] = resources
                    if len(active) >= 3 or not resource_allows(resources, len(active), loading):
                        job['waiting_reason'] = 'three-job limit or reserved memory headroom'
                        continue
                    job.update(launch(name, 'cityflow_tsc.train_local_revision',
                        ['--stage', 'train', '--family', job['family'], '--seed', str(job['seed'])]))
                    job.pop('waiting_reason', None)
                    save()
                state['counts'] = {s: sum(j['stage'] == s for j in state['jobs'].values())
                                   for s in ('pending', 'running', 'complete', 'failed', 'blocked')}
                save()
                if not state['counts']['pending'] and not state['counts']['running'] and state['collection']['stage'] != 'running':
                    break
                time.sleep(15)
            if state['errors'] or state['counts']['complete'] != 6:
                raise RuntimeError('Required work incomplete; diagnostic remains unopened')
            lock_checkpoints(run)
            state['stage'] = 'evaluating'
            state['evaluation'] = launch('evaluation', 'cityflow_tsc.evaluate_coverage_revision')
            save()
            code = children['evaluation'].wait()
            summary = run / 'evaluation/summary.json'
            if code != 0 or not summary.exists() or read(summary)['stage'] != 'complete':
                raise RuntimeError('Final evaluation failed; locked checkpoints retained')
            state['evaluation'].update(stage='complete', exit_code=code, finished_at=now())
            state.update(stage='complete', finished_at=now())
            save()
        except Exception as exc:
            state.update(stage='failed', error=repr(exc), traceback=traceback.format_exc(), failed_at=now(),
                         still_running={n: p.pid for n, p in children.items() if p.poll() is None})
            save()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-v10', type=Path, required=True)
    args = parser.parse_args()
    supervise(args.run_dir, args.source_v10)
