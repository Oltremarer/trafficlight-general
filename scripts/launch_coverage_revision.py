"""Launch the approved v11/v12/v13/v14 supervisor once with a durable dispatch record."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from cityflow_tsc.counterfactual.writer import atomic_json
from cityflow_tsc.run_formal_effects import command_environment
from cityflow_tsc.train_formal_effects import now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--source-v10', type=Path)
    sources.add_argument('--source-v11', type=Path)
    sources.add_argument('--source-v12', type=Path)
    sources.add_argument('--source-v13', type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not Path('/mnt/pan').is_mount() or run == Path('/mnt/pan') or not run.is_relative_to(Path('/mnt/pan')):
        raise ValueError('Mounted /mnt/pan required')
    run.mkdir(parents=True, exist_ok=True)
    # Exclusive dispatch guard prevents accidental repeated SSH launches.
    fd = os.open(run / 'dispatch.guard', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    if (run / 'execution.json').exists() or (run / 'dispatch.json').exists():
        raise RuntimeError('Existing run retained; refusing duplicate dispatch')
    for name in ('logs', 'tmp', 'cache'):
        (run / name).mkdir(exist_ok=True)
    module = 'cityflow_tsc.run_coverage_revision' if args.source_v10 else 'cityflow_tsc.run_balanced_revision'
    if args.source_v13:
        source_flag, source = '--source-v13', args.source_v13
    elif args.source_v12:
        source_flag, source = '--source-v12', args.source_v12
    elif args.source_v11:
        source_flag, source = '--source-v11', args.source_v11
    else:
        source_flag, source = '--source-v10', args.source_v10
    command = [sys.executable, '-u', '-m', module, '--run-dir', str(run), source_flag, str(source.resolve())]
    with (run / 'logs/supervisor.log').open('x') as output:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            env=command_environment(run, 1), start_new_session=True)
    record = {'stage': 'launched_not_completed', 'pid': process.pid, 'run_dir': str(run),
              'command': command, 'started_at': now()}
    atomic_json(run / 'dispatch.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
