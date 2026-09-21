from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cityflow_tsc.baselines.contracts import TrainConfig
from cityflow_tsc.baselines.profiles import get_profile, list_profiles
from cityflow_tsc.baselines.registry import create_learner, make_train_config
from cityflow_tsc.baselines.training import TrainingRunner
from cityflow_tsc.baselines.trajectory import BaselineTrajectoryReader, BaselineTrajectoryWriter
from cityflow_tsc.config import ControlConfig, ScenarioConfig
from cityflow_tsc.runner import EpisodeRunner
from cityflow_tsc.runtime import build_environment
from cityflow_tsc.topology import load_network_spec
from cityflow_tsc.trajectory import TrajectoryReader
from cityflow_tsc.types import NetworkSnapshot
from cityflow_tsc import train_baseline


FIXTURES = Path(__file__).parent / "fixtures"


class TrafficBackend:
    """Deterministic lane records for interface tests; not a traffic simulator."""

    def __init__(self, control, network):
        self.control, self.network = control, network
        self.time = 0.
        self.phases = {}
        self.closed = False

    def reset(self, scenario):
        self.time = 0.
        self.phases = {}
        self.closed = False
        return self.snapshot()

    def set_phase(self, intersection_id, engine_phase_id):
        self.phases[intersection_id] = engine_phase_id

    def next_step(self):
        self.time += self.control.simulator_step_s

    def current_time(self):
        return self.time

    def average_travel_time(self):
        return 12.

    def snapshot(self):
        counts, queues, vehicles, speeds, distances = {}, {}, {}, {}, {}
        for inter in self.network.intersections:
            phase = self.phases.get(inter.intersection_id, 1)
            for index, lane in enumerate(inter.incoming_lanes + inter.outgoing_lanes):
                queue = 1 + ((index + int(self.time) + phase) % 3)
                counts[lane], queues[lane] = queue + 2, queue
                ids = tuple(f"{lane}:vehicle:{k}" for k in range(queue + 2))
                vehicles[lane] = ids
                for k, name in enumerate(ids):
                    speeds[name], distances[name] = (0. if k < queue else 8.), 450.
        return NetworkSnapshot(self.time, counts, queues, speeds, vehicles, distances)

    def close(self):
        self.closed = True


def make_stack(tmp_path, baseline):
    control = ControlConfig(decision_interval_s=2, yellow_time_s=1,
                            green_phase_ids=(3, 1, 4, 2))
    scenario = ScenarioConfig(FIXTURES / "roadnet_baseline_four_arms.json",
                              FIXTURES / "flow_empty.json", tmp_path / "train", duration_s=8, seed=7)
    network = load_network_spec(scenario.roadnet_path, control)
    backend = TrafficBackend(control, network)
    env = build_environment(control, network, profile=baseline, backend=backend)
    config = TrainConfig(hidden_dim=8, batch_size=2, warmup_transitions=2, replay_capacity=16,
                         updates_per_round=2, target_update_steps=1, ppo_epochs=2,
                         epsilon_start=.2, epsilon_end=.1)
    return env, scenario, config


@pytest.mark.parametrize("name", [p.baseline_id for p in list_profiles()])
def test_every_profile_trains_saves_reloads_and_evaluates(tmp_path, name):
    torch.set_num_threads(1)
    env, scenario, config = make_stack(tmp_path, name)
    initial = env.reset(scenario)
    learner = create_learner(name, env.network, initial[0], config, seed=7)
    result = TrainingRunner().run_episode(env, learner, scenario, initial=initial)
    assert learner.gradient_steps > 0
    assert result.metrics['training_updates'] == learner.gradient_steps
    assert env.backend.closed
    loaded = BaselineTrajectoryReader().load(Path(result.manifest_path))
    arrays = loaded['arrays']
    assert arrays['actions'].shape == (4, 1)
    assert arrays['observation_baseline_features'].shape[0] == 4
    np.testing.assert_array_equal(arrays['observation_baseline_source_action_to_local'][0],
                                  initial[0].baseline_view.source_action_to_local)
    assert arrays['elapsed_s'].tolist() == [2., 2., 2., 2.]
    assert arrays['behavior_present'].all()
    assert arrays['reward_component_incoming_queue'].shape == (4, 1)
    component = {'queue': 'incoming_queue', 'queue_mean': 'queue_mean_road',
                 'absolute_queue_pressure': 'absolute_queue_pressure'}[learner.profile.reward_kind]
    np.testing.assert_allclose(arrays['rewards'],
        arrays[f'reward_component_{component}'] * learner.profile.reward_factor)
    if name in {'ippo', 'mappo'}:
        assert 'behavior_log_prob' in arrays and 'behavior_hidden_state' in arrays
        assert len(set(arrays['behavior_policy_version'])) == 1
    with pytest.raises(ValueError, match='unsupported trajectory schema'):
        TrajectoryReader().load(Path(result.manifest_path))
    checkpoint = tmp_path / 'model.pt'
    learner.save(checkpoint)
    restored = create_learner(get_profile(name), env.network, initial[0], config, seed=91)
    restored.load(checkpoint)
    learner.policy.reset(77, env.network)
    restored.policy.reset(77, env.network)
    assert np.array_equal(learner.policy.act(initial[0], True).actions,
                          restored.policy.act(initial[0], True).actions)
    updates = restored.gradient_steps
    eval_scenario = ScenarioConfig(scenario.roadnet_path, scenario.flow_path,
                                   tmp_path / 'evaluation', duration_s=6, seed=77)
    eval_env = build_environment(env.control, env.network, profile=name,
                                 backend=TrafficBackend(env.control, env.network))
    evaluation = EpisodeRunner().run(eval_env, restored.policy, eval_scenario,
                                    writer_factory=BaselineTrajectoryWriter)
    assert restored.gradient_steps == updates
    assert BaselineTrajectoryReader().load(Path(evaluation.manifest_path))['manifest']['steps'] == 3


@pytest.mark.parametrize('baseline', ['a-colight', 'ippo', 'mappo', 'maddpg'])
def test_cli_training_evaluation_and_resume(tmp_path, monkeypatch, baseline):
    def fake_build(control, network, waiting_speed_threshold=.1, *, profile=None):
        return build_environment(control, network, waiting_speed_threshold, profile=profile,
                                 backend=TrafficBackend(control, network))
    monkeypatch.setattr(train_baseline, 'build_environment', fake_build)
    parser = train_baseline.build_parser()
    common = ['--roadnet', str(FIXTURES / 'roadnet_baseline_four_arms.json'),
              '--flow', str(FIXTURES / 'flow_empty.json'), '--duration', '8']
    train_options = ['--baseline', baseline, '--episodes', '1', '--eval-seeds', '101',
                     '--decision-interval', '2', '--yellow-time', '1', '--batch-size', '2',
                     '--replay-capacity', '16', '--warmup-transitions', '2', '--hidden-dim', '8',
                     '--updates-per-round', '2', '--ppo-epochs', '1']
    result = train_baseline.train(parser.parse_args(['train', *common, *train_options,
                                                   '--output', str(tmp_path / 'first')]))
    assert result['gradient_steps'] > 0
    checkpoint = result['checkpoint']
    evaluated = train_baseline.evaluate(parser.parse_args([
        'evaluate', *common, '--checkpoint', checkpoint, '--output', str(tmp_path / 'eval')]))
    assert evaluated['metrics']['metric_ticks'] == 8
    resumed = train_baseline.train(parser.parse_args([
        'train', *common, *train_options, '--resume', checkpoint, '--output', str(tmp_path / 'resume')]))
    assert resumed['completed_episodes'] == 2
    assert resumed['gradient_steps'] > result['gradient_steps']
    assert resumed['training'][0]['episode'] == 1
    protocol = json.loads(Path(checkpoint).with_suffix('.protocol.json').read_text())
    assert protocol['profile']['implementation'] == get_profile(baseline).implementation


@pytest.mark.parametrize('baseline', ['colight', 'frap', 'ippo', 'mappo', 'maddpg'])
def test_cli_and_python_defaults_and_explicit_overrides_agree(tmp_path, baseline):
    env, scenario, _ = make_stack(tmp_path, baseline)
    try:
        initial, _ = env.reset(scenario)
        args = train_baseline.build_parser().parse_args([
            'train', '--baseline', baseline, '--roadnet', str(scenario.roadnet_path),
            '--flow', str(scenario.flow_path), '--output', str(tmp_path / 'cli'),
        ])
        learner = create_learner(baseline, env.network, initial)
        assert learner.config == train_baseline._training_config(args, get_profile(baseline))
        overrides = dict(gamma=.7, reward_scale=.3, learning_rate=.002, hidden_dim=8)
        config = make_train_config(baseline, **overrides)
        for key, value in overrides.items():
            setattr(args, key, value)
        assert config == train_baseline._training_config(args, get_profile(baseline))
        assert create_learner(baseline, env.network, initial, config).config is config
    finally:
        env.close()


def test_registry_rejects_a_different_observation_profile(tmp_path):
    env, scenario, config = make_stack(tmp_path, 'e-colight')
    observation, _ = env.reset(scenario)
    with pytest.raises(ValueError, match='profile does not match'):
        create_learner('colight', env.network, observation, config)


def test_evaluation_and_training_close_on_initialization_failure(tmp_path, monkeypatch):
    env, scenario, config = make_stack(tmp_path, 'idqn')
    observation, _ = env.reset(scenario)
    learner = create_learner('idqn', env.network, observation, config)
    def broken_reset(*args):
        raise RuntimeError('reset failed')
    monkeypatch.setattr(env, 'reset', broken_reset)
    with pytest.raises(RuntimeError, match='reset failed'):
        EpisodeRunner().run(env, learner.policy, scenario)
    assert env.backend.closed
    env.backend.closed = False
    monkeypatch.setattr(learner.policy, 'reset', broken_reset)
    with pytest.raises(RuntimeError, match='reset failed'):
        TrainingRunner().run_episode(env, learner, scenario)
    assert env.backend.closed


def test_failed_checkpoint_sidecar_preserves_previous_version(tmp_path, monkeypatch):
    class Learner:
        completed_episodes = 1
        def save(self, path):
            path.write_bytes(str(self.completed_episodes).encode())
    learner = Learner()
    previous = tmp_path / 'episode_0000.pt'
    train_baseline._save(learner, previous, {'profile': {'baseline_id': 'test'}})
    previous_pointer = (tmp_path / 'latest.json').read_bytes()
    write = train_baseline.write_json
    def fail_next_sidecar(path, value):
        if path.name == 'episode_0001.protocol.json':
            raise OSError('disk failure')
        write(path, value)
    monkeypatch.setattr(train_baseline, 'write_json', fail_next_sidecar)
    learner.completed_episodes = 2
    with pytest.raises(OSError, match='disk failure'):
        train_baseline._save(learner, tmp_path / 'episode_0001.pt', {'profile': {}})
    assert previous.read_bytes() == b'1'
    assert train_baseline._read_protocol(previous)['completed_episodes'] == 1
    assert (tmp_path / 'latest.json').read_bytes() == previous_pointer
