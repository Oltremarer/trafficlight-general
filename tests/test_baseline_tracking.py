from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cityflow_tsc.experiment_tracking import ExperimentTracker, final_training_summary


class RecordingRun:
    def __init__(self):
        self.rows = []
        self.axes = {}
        self.summary = {}
        self.url = 'https://wandb.invalid/test/run'
        self.exit_code = None

    def define_metric(self, name, **kwargs):
        self.axes[name] = kwargs

    def log(self, values):
        self.rows.append(dict(values))

    def finish(self, exit_code):
        self.exit_code = exit_code


def tracking_args(**overrides):
    return SimpleNamespace(flow=Path(__file__), roadnet=Path(__file__), seed=7,
                           **{'wandb_mode': 'offline', **overrides})


@pytest.fixture
def sdk(monkeypatch):
    calls, runs = [], []

    def init(**kwargs):
        calls.append(kwargs)
        run = RecordingRun()
        runs.append(run)
        return run

    fake = SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs, calls=calls, runs=runs)
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    return fake


def read_history(directory):
    return [json.loads(line) for line in (directory / 'tracking/history.jsonl').read_text().splitlines()]


def test_tracking_axes_local_mirror_and_run_isolation(tmp_path, sdk):
    trackers = [ExperimentTracker(tracking_args(), tmp_path / name, {'gamma': .8},
                                  baseline='colight', job_type='train') for name in ('one', 'two')]
    for tracker in trackers:
        tracker.log({'round': 5, 'env/average_travel_time_s': 300.})
        tracker.log({'round': 5, 'eval/average_travel_time_s': 280.})
        tracker.summarize({'last10_eval/count': 1})
        tracker.finish(0)
    assert sdk.calls[0]['id'] != sdk.calls[1]['id']
    assert sdk.calls[0]['group'] == sdk.calls[1]['group']
    for name, call, run in zip(('one', 'two'), sdk.calls, sdk.runs):
        assert call['dir'] == str(tmp_path / name / 'tracking')
        assert run.axes['eval/*']['step_metric'] == 'round'
        assert run.axes['final_eval/*']['step_metric'] == 'evaluation_index'
        assert run.rows == read_history(tmp_path / name)
        assert run.summary['last10_eval/count'] == 1
        assert run.exit_code == 0
        assert len(call['config']['source_version']['source_sha256']) == 64


def test_sdk_log_failure_keeps_every_local_record_and_failure_state(tmp_path, sdk, capsys):
    tracker = ExperimentTracker(tracking_args(), tmp_path, {}, baseline='mplight', job_type='train')

    def broken(values):
        raise RuntimeError('secret_token_must_not_appear')

    sdk.runs[0].log = broken
    tracker.log({'round': 1, 'train/loss': .5})
    tracker.log({'round': 2, 'train/loss': .4})
    tracker.summarize({'completed_episodes': 2})
    tracker.finish(1)
    assert [row['round'] for row in read_history(tmp_path)] == [1, 2]
    status = json.loads((tmp_path / 'tracking/status.json').read_text())
    assert status['state'] == 'failed'
    assert status['errors'] == [{'stage': 'log', 'type': 'RuntimeError'}]
    assert sdk.runs[0].exit_code == 1
    assert 'secret_token' not in capsys.readouterr().err
    assert 'secret_token' not in (tmp_path / 'tracking/status.json').read_text()


def test_online_init_failure_falls_back_to_offline(tmp_path, sdk):
    init = sdk.init
    attempted = []

    def online_unavailable(**kwargs):
        attempted.append(kwargs['mode'])
        if kwargs['mode'] == 'online':
            raise ConnectionError('network unavailable')
        return init(**kwargs)

    sdk.init = online_unavailable
    tracker = ExperimentTracker(tracking_args(wandb_mode='online'), tmp_path, {},
                                baseline='mappo', job_type='train')
    tracker.log({'round': 1})
    tracker.finish(0)
    assert attempted == ['online', 'offline']
    assert tracker.status['effective_mode'] == 'offline'
    assert tracker.status['url'] is None
    assert sdk.runs[0].rows == read_history(tmp_path)


def test_missing_sdk_still_records_results(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'wandb', None)
    tracker = ExperimentTracker(tracking_args(), tmp_path, {}, baseline='mp', job_type='rule_evaluate')
    tracker.log({'round': 0, 'eval/average_queue_vehicles': 4.})
    tracker.finish(0)
    assert len(read_history(tmp_path)) == 1
    assert tracker.status['effective_mode'] == 'local'
    assert tracker.status['errors'][0]['stage'] == 'import'


def test_rule_controller_logs_one_evaluation_point(tmp_path, monkeypatch, sdk):
    from cityflow_tsc import cli
    from .fakes import DeterministicBackend

    monkeypatch.setattr(cli, 'CityFlowBackend', DeterministicBackend)
    fixtures = Path(__file__).parent / 'fixtures'
    args = cli.build_parser().parse_args([
        '--roadnet', str(fixtures / 'roadnet_two_intersections.json'),
        '--flow', str(fixtures / 'flow_empty.json'), '--output', str(tmp_path / 'rule'),
        '--policy', 'fixed-time', '--duration', '10', '--decision-interval', '5',
        '--yellow-time', '2', '--green-phases', '1,2', '--wandb-mode', 'offline',
    ])
    result = cli.run_from_args(args)
    assert len(sdk.runs) == 1
    assert sdk.runs[0].rows == [{'round': 0, 'eval/seed': 0,
                                **{f'eval/{k}': v for k, v in result.metrics.items()}}]
    assert sdk.runs[0].exit_code == 0


@pytest.mark.parametrize('rounds,complete', [([1, 2, 3], False), (list(range(1, 11)), True),
                                           (list(range(10, 101, 10)), False)])
def test_tail_summary_does_not_confuse_sparse_evaluations_with_ten_rounds(rounds, complete):
    result = final_training_summary({
        'completed_episodes': rounds[-1], 'environment_steps': 100, 'gradient_steps': 10,
        'checkpoint': 'test.pt',
        'evaluation': {'replication_unit': 'evaluation seeds of one trained model',
                       'aggregate': {'att': {'mean': 300., 'std': 2.}}},
        'learning_curve': {'runs': [{'completed_episodes': n, 'metrics': {'att': float(n)}} for n in rounds]},
    })
    assert result['last10_eval/is_final_ten_rounds'] is complete
    assert result['last10_eval/rounds'] == rounds
    assert result['last10_eval/count'] == len(rounds)
    assert result['final_eval_mean/att'] == 300.
    assert result['last10_eval/mean/att'] == sum(rounds) / len(rounds)


def test_real_wandb_offline_creates_syncable_run(tmp_path, monkeypatch):
    pytest.importorskip('wandb')
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    tracker = ExperimentTracker(tracking_args(), tmp_path, {}, baseline='colight', job_type='train')
    try:
        assert tracker.status['effective_mode'] == 'offline'
        tracker.log({'round': 1, 'env/average_travel_time_s': 300.})
        tracker.log({'round': 2, 'env/average_travel_time_s': 280.})
        tracker.summarize({'completed_episodes': 2})
    finally:
        tracker.finish(0)
    assert not tracker.status['errors']
    assert list((tmp_path / 'tracking/wandb').glob('offline-run-*/run-*.wandb'))
    assert len(read_history(tmp_path)) == 2
