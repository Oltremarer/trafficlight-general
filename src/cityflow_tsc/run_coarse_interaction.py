"""Run the approved coarse-model and paired-interaction branches concurrently."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .collect_counterfactual import stamp
from .counterfactual.writer import atomic_json, sha256
from .run_formal_effects import command_environment, read


def supervise(run, source):
    import fcntl
    run, source = Path(run).resolve(), Path(source).resolve()
    if not Path('/mnt/pan').is_mount() or not run.is_relative_to(Path('/mnt/pan')) or run == source:
        raise ValueError('A separate output tree on mounted /mnt/pan is required')
    run.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'tmp', 'cache', 'coarse', 'interaction'):
        (run / name).mkdir(exist_ok=True)
    with (run / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'launcher.json').exists():
            raise RuntimeError('Duplicate dispatch refused; retained run needs explicit recovery')
        atomic_json(run / 'source_hashes.json', {
            str(p.relative_to(Path(__file__).parent)): sha256(p) for p in sorted(Path(__file__).parent.rglob('*.py'))})
        stages = {key: 'pending' for key in ('coarse_training', 'coarse_evaluation', 'pair_collection', 'pair_analysis')}
        state = {'stage': 'running', 'created_at': stamp(), 'supervisor_pid': os.getpid(),
            'run_dir': str(run), 'source_run': str(source), 'stages': stages, 'errors': [],
            'training_seeds': [42, 43, 44], 'updates_per_seed': 132000, 'collector_workers': 20,
            'pair_roots': 54, 'new_pair_branch_budget': 103680}
        children, launches = {}, {}
        markers = {'coarse_training': run / 'coarse/checkpoints_locked.json',
            'coarse_evaluation': run / 'coarse/evaluation/summary.json',
            'pair_collection': run / 'interaction/summary.json',
            'pair_analysis': run / 'interaction/analysis/summary.json'}

        def save():
            state['updated_at'] = stamp()
            atomic_json(run / 'execution.json', state)

        def launch(name, module, target, extras, threads):
            command = [sys.executable, '-u', '-m', module, '--run-dir', str(target), *extras]
            logfile = run / 'logs' / (name + '.log')
            with logfile.open('ab') as stream:
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True, env=command_environment(run, threads))
            children[name] = proc
            stages[name] = 'running'
            launches[name] = {'pid': proc.pid, 'command': command, 'log': str(logfile), 'started_at': stamp()}
            atomic_json(run / 'launcher.json', {'supervisor_pid': os.getpid(), 'jobs': launches})
            save()

        save()
        launch('coarse_training', 'cityflow_tsc.train_coarse_effects', run / 'coarse',
               ['--source-run', str(source / 'arms/multi'), '--stage', 'train'], 2)
        launch('pair_collection', 'cityflow_tsc.collect_interaction_v6', run / 'interaction',
               ['--source-run', str(source), '--stage', 'collect', '--workers', '20'], 1)
        while True:
            for name, process in children.items():
                code = process.poll()
                if code is None or stages[name] != 'running':
                    continue
                marker = markers[name]
                okay = code == 0 and marker.is_file()
                if okay and name != 'coarse_training':
                    okay = read(marker).get('stage') == 'complete'
                stages[name] = 'complete' if okay else 'failed'
                if not okay:
                    state['errors'].append({'stage': name, 'exit_code': code, 'log': launches[name]['log']})
                    state['stage'] = 'attention_required'
                    atomic_json(run / 'failure.json', {'errors': state['errors'], 'independent_jobs_continue': True})
                save()
            for first, second in (('coarse_training', 'coarse_evaluation'), ('pair_collection', 'pair_analysis')):
                if stages[first] == 'failed' and stages[second] == 'pending':
                    stages[second] = 'skipped_due_to_failed_dependency'
                    save()
            if stages['coarse_training'] == 'complete' and stages['coarse_evaluation'] == 'pending':
                launch('coarse_evaluation', 'cityflow_tsc.train_coarse_effects', run / 'coarse',
                       ['--source-run', str(source / 'arms/multi'), '--stage', 'evaluate'], 2)
            if stages['pair_collection'] == 'complete' and stages['pair_analysis'] == 'pending':
                launch('pair_analysis', 'cityflow_tsc.collect_interaction_v6', run / 'interaction', ['--stage', 'analyze'], 1)
            if all(s not in ('pending', 'running') for s in stages.values()):
                state.update(stage='attention_required' if state['errors'] else 'complete', finished_at=stamp())
                save()
                if not state['errors']:
                    atomic_json(run / 'complete.json', state)
                return int(bool(state['errors']))
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    args = parser.parse_args()
    return supervise(args.run_dir, args.source_run)


if __name__ == '__main__':
    raise SystemExit(main())
