"""Durable twelve-job dependency queue; independent failures preserve other jobs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .counterfactual.writer import atomic_json, sha256
from .effect_model.local_data import initialize_run
from .effect_model.local_revision import SCHEDULES
from .effect_model import decision_revision as v9
from .evaluate_local_revision import lock_checkpoints
from .run_formal_effects import command_environment, read
from .train_formal_effects import now
from .train_temporal_effects import SEEDS


def resource_snapshot():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    host_gib = int(fields['MemAvailable'].split()[0]) / 1024 ** 2
    output = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    return {'host_available_gib': host_gib, 'gpu_free_gib': float(output.strip()) / 1024}


def resource_allows(resources, active_count, loading_count):
    # Loading jobs have not allocated their steady-state tensors yet.
    return (active_count < 8 and resources['host_available_gib'] - 2.5 * loading_count >= 8.5
            and resources['gpu_free_gib'] - 3. * loading_count >= 9.)


def dependency_ready(run, family):
    return family in ('A_C', 'A_D') or (run / f'labels_{family}.ready.json').exists()


def supervise(run, single_source=None, pair_source=None, coarse_source=None, reuse_v8=None):
    import fcntl
    run = Path(run).resolve()
    if not Path('/mnt/pan').is_mount() or not run.is_relative_to(Path('/mnt/pan')) or run == Path('/mnt/pan'):
        raise ValueError('Mounted /mnt/pan required; no system-disk fallback')
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'supervisor.lock').open('a') as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'execution.json').exists():
            raise RuntimeError('Existing run retained; duplicate launch refused')
        for name in ('logs', 'tmp', 'cache'):
            (run / name).mkdir(exist_ok=True)
        revised = reuse_v8 is not None
        state = {'stage': 'preparing', 'pid': os.getpid(), 'run_dir': str(run), 'started_at': now(),
            'max_training_workers': 8, 'label_workers': 0 if revised else 4, 'jobs': {}, 'errors': []}
        children = {}

        def save():
            state['updated_at'] = now()
            atomic_json(run / 'execution.json', state)

        def launch(name, module, args):
            logfile = run / 'logs' / (name.replace('/', '_') + '.log')
            command = ['nice', '-n', '15', 'ionice', '-c', '3', sys.executable, '-u', '-m', module,
                       '--run-dir', str(run), *args]
            with logfile.open('x') as stream:
                process = subprocess.Popen(command, env=command_environment(run, 1), stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT)
            children[name] = process
            return {'stage': 'running', 'pid': process.pid, 'command': command, 'log': str(logfile), 'started_at': now()}

        save()
        try:
            if revised:
                v9.initialize_run(run, reuse_v8)
            else:
                initialize_run(run, single_source, pair_source, coarse_source)
            source = Path(__file__).parent
            atomic_json(run / 'source_hashes.json', {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*.py'))})
            state['preparation'] = launch('labels', 'cityflow_tsc.train_local_revision',
                ['--stage', 'prepare-decision' if revised else 'prepare-labels'])
            # Six matched A jobs enter first; the two B families follow for v9.
            order = training_order(revised)
            for family, seed in order:
                state['jobs'][f'{family}/seed_{seed}'] = {'stage': 'pending', 'family': family, 'seed': seed}
            state['stage'] = 'training_and_preparing'
            save()
            while True:
                prep = children['labels']
                if prep.poll() is not None and state['preparation']['stage'] == 'running':
                    ready_file = 'fixed_A_D.ready.json' if revised else 'labels_A_L4.ready.json'
                    success = prep.returncode == 0 and (run / ready_file).exists()
                    state['preparation'].update(stage='complete' if success else 'failed', exit_code=prep.returncode, finished_at=now())
                    if not success:
                        state['errors'].append('Label preparation failed; see labels.log')
                for name, job in state['jobs'].items():
                    if job['stage'] == 'running' and children[name].poll() is not None:
                        code = children[name].returncode
                        success = code == 0 and (run / name / 'complete.json').exists()
                        job.update(stage='complete' if success else 'failed', exit_code=code, finished_at=now())
                        if success:
                            job['result'] = read(run / name / 'complete.json')
                        else:
                            state['errors'].append(name + ' failed; its artifacts are retained')
                for name, job in state['jobs'].items():
                    if job['stage'] != 'pending':
                        continue
                    ready = (state['preparation']['stage'] == 'complete' if revised else dependency_ready(run, job['family']))
                    if not ready:
                        if state['preparation']['stage'] == 'failed':
                            job.update(stage='blocked', reason='required label cache unavailable')
                        continue
                    active = [(n, j) for n, j in state['jobs'].items() if j['stage'] == 'running']
                    loading = sum(not (run / n / 'status.json').exists() or
                                  read(run / n / 'status.json').get('stage') == 'loading' for n, _ in active)
                    resources = resource_snapshot()
                    state['resources'] = resources
                    if not resource_allows(resources, len(active), loading):
                        job['waiting_reason'] = 'concurrency or reserved memory headroom'
                        continue
                    job.update(launch(name, 'cityflow_tsc.train_local_revision',
                        ['--stage', 'train', '--family', job['family'], '--seed', str(job['seed'])]))
                    job.pop('waiting_reason', None)
                    save()
                state['counts'] = {s: sum(j['stage'] == s for j in state['jobs'].values())
                                   for s in ('pending', 'running', 'complete', 'failed', 'blocked')}
                if state['preparation']['stage'] == 'complete':
                    state['stage'] = 'training'
                save()
                if not state['counts']['pending'] and not state['counts']['running'] and state['preparation']['stage'] != 'running':
                    break
                time.sleep(15)
            if state['errors'] or state['counts']['complete'] != 12:
                raise RuntimeError('Not all twelve jobs completed; no diagnostic labels opened')
            if revised:
                from .evaluate_decision_revision import lock_checkpoints as lock_decision
                lock_decision(run)
            else:
                lock_checkpoints(run)
            state['stage'] = 'evaluating'
            state['evaluation'] = launch('evaluation',
                'cityflow_tsc.evaluate_decision_revision' if revised else 'cityflow_tsc.evaluate_local_revision', [])
            save()
            code = children['evaluation'].wait()
            summary = run / 'evaluation/summary.json'
            if code != 0 or not summary.exists() or read(summary)['stage'] != 'complete':
                raise RuntimeError('Final evaluation failed; checkpoints remain locked')
            state['evaluation'].update(stage='complete', exit_code=code, finished_at=now())
            state.update(stage='complete', finished_at=now())
            save()
        except Exception as exc:
            state.update(stage='failed', error=repr(exc), traceback=traceback.format_exc(), failed_at=now(),
                         still_running={n: p.pid for n, p in children.items() if p.poll() is None})
            save()
            raise


def training_order(revised=False):
    first = ('A_D_Top8', 'A_D_Local') if revised else ('A_C', 'A_D')
    second = ('B_Match', 'B_Contrast') if revised else ('B_L4', 'A_L4')
    return [(f, s) for s in SEEDS for f in first] + [(f, s) for s in SEEDS for f in second]


def main():
    parser = argparse.ArgumentParser()
    for name in ('run-dir', 'single-source', 'pair-source', 'coarse-source', 'reuse-v8'):
        parser.add_argument('--' + name, type=Path, required=name == 'run-dir')
    args = parser.parse_args()
    if args.reuse_v8 is None and any(v is None for v in (args.single_source, args.pair_source, args.coarse_source)):
        parser.error('Provide --reuse-v8 or all three source paths')
    if args.reuse_v8 is not None and any(v is not None for v in (args.single_source, args.pair_source, args.coarse_source)):
        parser.error('--reuse-v8 cannot be combined with legacy source paths')
    supervise(args.run_dir, args.single_source, args.pair_source, args.coarse_source, args.reuse_v8)


if __name__ == '__main__':
    main()
