"""Per-experiment W&B logging with an independent, append-only local history."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def add_tracking_arguments(parser) -> None:
    parser.add_argument('--wandb-mode', choices=('offline', 'online', 'disabled'),
                        default=os.environ.get('WANDB_MODE', 'offline'),
                        help='Default: offline W&B files in the experiment output; disabled keeps JSON logs only')
    parser.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', 'rl-trafficlight'))
    parser.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--wandb-group')
    parser.add_argument('--wandb-name')


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def _source_version() -> dict:
    source = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(source.rglob('*.py')):
        digest.update(str(path.relative_to(source)).encode() + b'\0')
        digest.update(path.read_bytes())
    result = {'source_sha256': digest.hexdigest(), 'git_commit': None, 'git_dirty': None}
    try:
        result['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=source, stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip()
        result['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=source, stderr=subprocess.DEVNULL, timeout=3,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return result


class ExperimentTracker:
    """SDK errors never discard local metrics or change the learning procedure.

    A resumed training invocation creates a new run segment. Its config records
    the source checkpoint, and its round axis continues at completed_episodes.
    """

    def __init__(self, args, output: Path, config: dict, *, baseline: str, job_type: str):
        self.directory = output.resolve() / 'tracking'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.history_path = self.directory / 'history.jsonl'
        self.run = None
        self.remote_logging = False
        self.finished = False
        mode = getattr(args, 'wandb_mode', 'offline')
        if mode not in ('offline', 'online', 'disabled'):
            raise ValueError(f'unsupported W&B mode: {mode}')
        flow = args.flow.expanduser().resolve()
        self.config = {
            **config, 'baseline': baseline, 'job_type': job_type,
            'flow_path': str(flow), 'roadnet_path': str(args.roadnet.expanduser().resolve()),
            'invocation_seed': args.seed, 'source_version': _source_version(),
            'resume_checkpoint': str(args.resume.resolve()) if getattr(args, 'resume', None) else None,
            'requested_episodes': getattr(args, 'episodes', None),
        }
        self.status = {
            'id': uuid.uuid4().hex[:12], 'requested_mode': mode, 'effective_mode': 'local',
            'name': getattr(args, 'wandb_name', None) or f'{baseline}__{flow.stem}__seed{args.seed}__{job_type}',
            'group': getattr(args, 'wandb_group', None) or f'{baseline}__{flow.stem}',
            'project': getattr(args, 'wandb_project', 'rl-trafficlight'),
            'entity': getattr(args, 'wandb_entity', None),
            'url': None, 'errors': [], 'state': 'running',
        }
        _write_json(self.directory / 'config.json', self.config)
        self._save_status()
        if mode == 'disabled':
            return
        try:
            wandb = importlib.import_module('wandb')
        except Exception as exc:
            self._error('import', exc)
            print("Install W&B with: pip install -e '.[tracking]'", file=sys.stderr)
            return
        # All run files stay beside the experiment, including offline runs on /mnt/pan.
        for effective_mode in (('online', 'offline') if mode == 'online' else ('offline',)):
            try:
                self.run = wandb.init(
                    project=self.status['project'], entity=self.status['entity'],
                    name=self.status['name'], group=self.status['group'],
                    id=self.status['id'], job_type=job_type, config=self.config,
                    mode=effective_mode, dir=str(self.directory), reinit='create_new',
                    settings=wandb.Settings(init_timeout=15, login_timeout=5, console='off',
                                            disable_git=True, disable_code=True),
                )
                self.status['effective_mode'] = effective_mode
                self.status['url'] = self.run.url if effective_mode == 'online' else None
                self.remote_logging = True
                break
            except Exception as exc:
                self._error(f'init_{effective_mode}', exc)
        if self.run is not None:
            try:
                self.run.define_metric('round')
                for prefix in ('train', 'env', 'eval'):
                    self.run.define_metric(f'{prefix}/*', step_metric='round')
                self.run.define_metric('evaluation_index')
                self.run.define_metric('final_eval/*', step_metric='evaluation_index')
            except Exception as exc:
                self.remote_logging = False
                self._error('define_metric', exc)
        self._save_status()

    def _save_status(self) -> None:
        _write_json(self.directory / 'status.json', self.status)

    def _error(self, stage: str, exc: Exception) -> None:
        # SDK exception text can contain credentials/URLs; retain only its type.
        self.status['errors'].append({'stage': stage, 'type': type(exc).__name__})
        self._save_status()
        print(f'W&B {stage} failed ({type(exc).__name__}); local tracking remains at {self.directory}',
              file=sys.stderr)

    def log(self, values: dict) -> None:
        with self.history_path.open('a') as stream:
            stream.write(json.dumps(values, sort_keys=True, allow_nan=False) + '\n')
        if self.run is not None and self.remote_logging:
            try:
                # W&B's internal step is monotonic; round is the explicit custom x axis.
                self.run.log(values)
            except Exception as exc:
                self.remote_logging = False
                self._error('log', exc)

    def summarize(self, values: dict) -> None:
        _write_json(self.directory / 'summary.json', values)
        if self.run is not None and self.remote_logging:
            try:
                self.run.summary.update(values)
            except Exception as exc:
                self.remote_logging = False
                self._error('summary', exc)

    def finish(self, exit_code: int) -> None:
        if self.finished:
            return
        self.finished = True
        self.status['state'] = 'finished' if exit_code == 0 else 'failed'
        self.status['exit_code'] = exit_code
        self._save_status()
        if self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as exc:
                self._error('finish', exc)


def training_record(learner, metrics: dict, seed: int) -> dict:
    values = {'round': learner.completed_episodes, 'train/environment_steps': learner.environment_steps,
              'train/gradient_steps': learner.gradient_steps, 'train/environment_seed': seed}
    for key, value in metrics.items():
        name = 'train/' + key[len('training_'):] if key.startswith('training_') else 'env/' + key
        values[name] = value
    epsilon = getattr(learner, 'epsilon', None)
    if epsilon is not None:
        values['train/epsilon'] = float(epsilon() if callable(epsilon) else epsilon)
    if hasattr(learner, 'replay'):
        values['train/replay_size'] = len(learner.replay)
    return values


def final_training_summary(summary: dict) -> dict:
    values = {'completed_episodes': summary['completed_episodes'],
              'environment_steps': summary['environment_steps'],
              'gradient_steps': summary['gradient_steps'],
              'checkpoint': summary['checkpoint'],
              'final_eval_replication_unit': summary['evaluation']['replication_unit']}
    for key, statistics in summary['evaluation']['aggregate'].items():
        for statistic, value in statistics.items():
            values[f'final_eval_{statistic}/{key}'] = value
    curves = summary['learning_curve']['runs'][-10:]
    rounds = [run['completed_episodes'] for run in curves]
    values['last10_eval/count'] = len(curves)
    values['last10_eval/rounds'] = rounds
    values['last10_eval/is_final_ten_rounds'] = (
        rounds == list(range(summary['completed_episodes'] - 9, summary['completed_episodes'] + 1))
        and len(rounds) == 10
    )
    values['last10_eval/replication_unit'] = 'evaluated checkpoints of one trained model, not independent training seeds'
    if curves:
        for key in sorted(set.intersection(*(set(run['metrics']) for run in curves))):
            values[f'last10_eval/mean/{key}'] = sum(run['metrics'][key] for run in curves) / len(curves)
    return values
