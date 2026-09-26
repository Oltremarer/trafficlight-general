from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from cityflow_tsc.lifecycle_metrics import LifecycleLedger as Ledger


def test_finished_active_and_waiting_are_disjoint(tmp_path):
    ledger = Ledger({'a','b','c'},3)
    ledger.observe(0,[],[],0)
    ledger.observe(1,['a','b','c'],['a'],1)
    ledger.observe(2,['b','c'],['b'],1)
    ledger.observe(2,['b','c'],['b'],1)
    ledger.observe(3,['b','c'],['b'],1)
    result = ledger.write_evidence(tmp_path/'lifecycle.manifest.json')
    assert result['finished_vehicles'] == 1
    assert result['active_unfinished_vehicles'] == 1
    assert result['not_entered_vehicles'] == 1
    assert result['completion_rate'] == pytest.approx(1/3)
    assert result['entered_vehicles'] == 2
    assert json.loads((tmp_path/'lifecycle.manifest.json').read_text())['finished_vehicle_ids'] == ['a']


def test_unobserved_generation_is_not_mislabeled_finished():
    ledger=Ledger({'a'},1)
    ledger.observe(0,[],[],0)
    ledger.observe(1,[],[],0)
    with pytest.raises(ValueError,match='generation not fully observed'):
        ledger.summary()


def test_disappearing_waiting_vehicle_is_not_completion():
    ledger=Ledger({'a'},2)
    ledger.observe(0,[],[],0)
    ledger.observe(1,['a'],[],0)
    with pytest.raises(ValueError,match='before entering'):
        ledger.observe(2,[],[],0)


def test_reappearing_vehicle_is_rejected():
    ledger=Ledger({'a'},3)
    ledger.observe(0,['a'],['a'],1)
    ledger.observe(1,[],[],0)
    with pytest.raises(ValueError,match='reappeared'):
        ledger.observe(2,['a'],['a'],1)


def test_missing_tick_and_inconsistent_count():
    ledger=Ledger({'a'},3)
    ledger.observe(0,[],[],0)
    with pytest.raises(ValueError,match='every simulator second'):
        ledger.observe(2,['a'],['a'],1)
    with pytest.raises(ValueError,match='active count'):
        ledger.observe(1,['a'],['a'],2)


def test_supported_flow_contract(tmp_path):
    flow=tmp_path/'flow.json'
    scenario=SimpleNamespace(flow_path=flow,duration_s=10)
    flow.write_text(json.dumps([{'startTime':0,'endTime':0,'interval':1,'route':['road']}]))
    assert Ledger.from_scenario(scenario,1).expected == {'flow_0_0'}
    flow.write_text(json.dumps([{'startTime':0,'endTime':5,'interval':1,'route':['road']}]))
    with pytest.raises(ValueError,match='one-shot'):
        Ledger.from_scenario(scenario,1)


@pytest.mark.parametrize('mode', ['rule', 'rl'])
def test_main_backend_and_episode_runner_collect_full_lifecycle(tmp_path, monkeypatch, mode):
    import sys
    from cityflow_tsc.config import ControlConfig, ScenarioConfig
    from cityflow_tsc.runtime import build_environment
    from cityflow_tsc.topology import load_network_spec
    from cityflow_tsc.runner import EpisodeRunner
    from cityflow_tsc.policies import FixedTimePolicy

    class Engine:
        def __init__(self, *args, **kwargs): self.time = 0
        def get_current_time(self): return self.time
        def next_step(self): self.time += 1
        def set_tl_phase(self, *args): pass
        def get_vehicles(self, include_waiting=False):
            if not self.time: return []
            if include_waiting: return ['flow_0_0', 'flow_1_0', 'flow_2_0'] if self.time == 1 else ['flow_1_0', 'flow_2_0']
            return ['flow_0_0'] if self.time == 1 else ['flow_1_0']
        def get_vehicle_count(self): return len(self.get_vehicles())
        def get_vehicle_speed(self): return {v: 0.0 for v in self.get_vehicles()}
        def get_vehicle_distance(self): return {v: 0.0 for v in self.get_vehicles()}
        def get_lane_vehicles(self): return {}
        def get_lane_vehicle_count(self): return {}
        def get_lane_waiting_vehicle_count(self): return {}
        def get_average_travel_time(self): return 7.5

    monkeypatch.setitem(sys.modules, 'cityflow', SimpleNamespace(Engine=Engine))
    flow = tmp_path / 'flow.json'
    flow.write_text(json.dumps([{'startTime': 0, 'endTime': 0, 'interval': 1, 'route': ['road']}] * 3))
    control = ControlConfig(decision_interval_s=3, yellow_time_s=0, green_phase_ids=(1, 2))
    scenario = ScenarioConfig(Path(__file__).parent / 'fixtures/roadnet_two_intersections.json',
                              flow, tmp_path / 'episode', duration_s=3)
    network = load_network_spec(scenario.roadnet_path, control)
    if mode == 'rule':
        result = EpisodeRunner().run(build_environment(control, network), FixedTimePolicy(), scenario)
        metric_file = 'metrics.json'
    else:
        pytest.importorskip('torch')
        from cityflow_tsc.baselines.registry import create_learner, make_train_config
        from cityflow_tsc.baselines.training import TrainingRunner
        env = build_environment(control, network, profile='shared-dqn')
        initial = env.reset(scenario)
        config = make_train_config('shared-dqn', hidden_dim=8, batch_size=1, warmup_transitions=1)
        learner = create_learner('shared-dqn', network, initial[0], config)
        result = TrainingRunner().run_episode(env, learner, scenario, initial=initial)
        metric_file = 'training.metrics.json'
    assert result.metrics['lifecycle_metrics_available'] == 1
    assert result.metrics['finished_vehicles'] == 1
    assert result.metrics['active_unfinished_vehicles'] == 1
    assert result.metrics['not_entered_vehicles'] == 1
    assert result.metrics['completion_rate'] == pytest.approx(1/3)
    assert result.metrics['average_travel_time_s'] == 7.5
    saved = json.loads((scenario.output_dir / metric_file).read_text())
    assert saved['metrics'] == result.metrics
    evidence = json.loads((scenario.output_dir / 'lifecycle.manifest.json').read_text())
    assert evidence['waiting_to_enter_vehicle_ids'] == ['flow_2_0']


def test_unsupported_flow_is_explicit_without_fabricated_completion_rate(tmp_path):
    from cityflow_tsc.metrics import MetricCollector
    from cityflow_tsc.types import NetworkSnapshot
    flow = tmp_path / 'flow.json'
    flow.write_text(json.dumps([{'startTime': 0, 'endTime': 5, 'interval': 1, 'route': ['road']}]))
    scenario = SimpleNamespace(flow_path=flow, duration_s=10, output_dir=tmp_path / 'out')
    snapshot = NetworkSnapshot(0, {}, {}, {}, vehicle_pool_ids=(), active_vehicle_ids=(), active_vehicle_count=0)
    metrics = MetricCollector()
    metrics.configure(scenario, 1, snapshot)
    metrics.observe(snapshot, 0)
    result = metrics.summary(0)
    assert result['lifecycle_metrics_available'] == 0
    assert 'completion_rate' not in result
    evidence = json.loads((scenario.output_dir / 'lifecycle.manifest.json').read_text())
    assert not evidence['available'] and 'one-shot' in evidence['reason']
