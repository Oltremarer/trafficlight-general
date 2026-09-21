"""Three independent v10 training workers plus one old-model evaluation worker."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .counterfactual.writer import atomic_json, sha256
from .effect_model import connection_revision as v10
from .evaluate_connection_revision import lock_checkpoints
from .run_formal_effects import command_environment, read
from .run_local_revision import resource_snapshot, resource_allows
from .train_formal_effects import now
from .train_temporal_effects import SEEDS


def supervise(run, source_v9):
    import fcntl
    run = Path(run).resolve()
    if (not Path('/mnt/pan').is_mount() or run == Path('/mnt/pan') or
            not run.is_relative_to(Path('/mnt/pan'))):
        raise ValueError('Mounted /mnt/pan required; no system-disk fallback')
    run.mkdir(parents=True, exist_ok=True)
    if not os.access(run, os.W_OK):
        raise PermissionError('Run directory is not writable')
    with (run / 'supervisor.lock').open('a') as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'execution.json').exists():
            raise RuntimeError('Existing run retained; duplicate launch refused')
        for name in ('logs', 'tmp', 'cache'):
            (run / name).mkdir(exist_ok=True)
        state = {'stage': 'preparing', 'pid': os.getpid(), 'run_dir': str(run), 'started_at': now(),
            'max_training_workers': 3, 'max_evaluation_workers': 1, 'jobs': {}, 'errors': []}
        children = {}

        def save():
            state['updated_at'] = now()
            atomic_json(run / 'execution.json', state)

        def launch(name, module, args):
            logfile = run / 'logs' / (name.replace('/', '_') + '.log')
            command = ['nice', '-n', '15', 'ionice', '-c', '3', sys.executable, '-u', '-m', module,
                       '--run-dir', str(run), *args]
            with logfile.open('x') as output:
                process = subprocess.Popen(command, env=command_environment(run, 1), stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT)
            children[name] = process
            return {'stage': 'running', 'pid': process.pid, 'command': command,
                'log': str(logfile), 'started_at': now()}

        save()
        try:
            v10.initialize_run(run, source_v9)
            source = Path(__file__).parent
            atomic_json(run / 'source_hashes.json', {str(p.relative_to(source)): sha256(p)
                for p in sorted(source.rglob('*.py'))})
            for seed in SEEDS:
                state['jobs'][f'B_Adapter/seed_{seed}'] = {'stage': 'pending', 'family': 'B_Adapter', 'seed': seed}
            state['ensemble_validation'] = launch('ensemble_validation', 'cityflow_tsc.evaluate_connection_revision',
                ['--stage', 'ensemble-validation'])
            state['stage'] = 'training_and_ensemble_validation'
            save()
            while True:
                for name, job in state['jobs'].items():
                    if job['stage'] == 'running' and children[name].poll() is not None:
                        code = children[name].returncode
                        success = code == 0 and (run / name / 'complete.json').exists()
                        job.update(stage='complete' if success else 'failed', exit_code=code, finished_at=now())
                        if success:
                            job['result'] = read(run / name / 'complete.json')
                        else:
                            state['errors'].append(name + ' failed; other workers remain running')
                evaluation = state['ensemble_validation']
                if evaluation['stage'] == 'running' and children['ensemble_validation'].poll() is not None:
                    code = children['ensemble_validation'].returncode
                    summary = run / 'ensemble_validation/summary.json'
                    success = code == 0 and summary.exists() and read(summary)['stage'] == 'complete'
                    evaluation.update(stage='complete' if success else 'failed', exit_code=code, finished_at=now())
                    if not success:
                        state['errors'].append('Old-model ensemble validation failed; training continues')
                for name, job in state['jobs'].items():
                    if job['stage'] != 'pending':
                        continue
                    active = [(n, j) for n, j in state['jobs'].items() if j['stage'] == 'running']
                    loading = sum(not (run / n / 'status.json').exists() or
                        read(run / n / 'status.json').get('stage') == 'loading' for n, _ in active)
                    resources = resource_snapshot()
                    state['resources'] = resources
                    if not resource_allows(resources, len(active) + int(evaluation['stage'] == 'running'), loading):
                        job['waiting_reason'] = 'reserved host/GPU memory headroom'
                        continue
                    job.update(launch(name, 'cityflow_tsc.train_local_revision',
                        ['--stage', 'train', '--family', 'B_Adapter', '--seed', str(job['seed'])]))
                    job.pop('waiting_reason', None)
                    save()
                state['counts'] = {s: sum(j['stage'] == s for j in state['jobs'].values())
                                   for s in ('pending', 'running', 'complete', 'failed')}
                save()
                if not state['counts']['pending'] and not state['counts']['running'] and evaluation['stage'] != 'running':
                    break
                time.sleep(15)
            if state['errors'] or state['counts']['complete'] != 3:
                raise RuntimeError('Training/evaluation incomplete; diagnostic remains unopened')
            lock_checkpoints(run)
            state['stage'] = 'evaluating'
            state['evaluation'] = launch('evaluation', 'cityflow_tsc.evaluate_connection_revision', ['--stage', 'final'])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--source-v9', required=True, type=Path)
    args = parser.parse_args()
    supervise(args.run_dir, args.source_v9)


if __name__ == '__main__':
    main()
