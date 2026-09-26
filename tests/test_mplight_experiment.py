from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from cityflow_tsc.baselines.mplight import MPLightConfig, MPLightLearner, MPLightNetwork, PROFILE
from cityflow_tsc.baselines.q_learning import _state_arrays, _tensor_batch
from cityflow_tsc.baselines.training import TrainingRunner
from cityflow_tsc import mplight_experiment as experiment
from cityflow_tsc.colight_results import CoLightMetrics
from cityflow_tsc.runtime import build_environment
from .test_baseline_integration import TrafficBackend, make_stack, FIXTURES


def test_mplight_round_update_checkpoint_and_physical_phase_permutation(tmp_path):
    torch.set_num_threads(1)
    env, scenario, _ = make_stack(tmp_path, PROFILE)
    initial = env.reset(scenario)
    config = MPLightConfig(batch_size=2, replay_capacity=16, sample_size=4, fit_epochs=2)
    learner = MPLightLearner(env.network, initial[0], config, seed=2)
    assert isinstance(learner.online, MPLightNetwork)
    assert learner.online.demand_conv.out_features == 20
    assert learner.optimizer.param_groups[0]['eps'] == 1e-8
    before = {k: v.clone() for k,v in learner.online.state_dict().items()}
    TrainingRunner().run_episode(env, learner, scenario, initial=initial)
    assert learner.last_fit['updates'] == 2
    assert any(not torch.equal(v, learner.online.state_dict()[k]) for k,v in before.items())
    state = _tensor_batch([_state_arrays(initial[0])], torch.device('cpu'))
    q = learner.online(state)
    permutation = torch.tensor([2, 0, 3, 1])
    changed = dict(state)
    changed['phase_lane_mask'] = state['phase_lane_mask'][..., permutation, :]
    changed['action_mask'] = state['action_mask'][..., permutation]
    torch.testing.assert_close(learner.online(changed), q[..., permutation])
    checkpoint = tmp_path/'mplight.pt'
    learner.save(checkpoint)
    restored = MPLightLearner(env.network, initial[0], config, seed=99)
    restored.load(checkpoint)
    for i, agent in enumerate((learner, restored)):
        next_env = build_environment(env.control, env.network, profile=PROFILE,
                                    backend=TrafficBackend(env.control, env.network))
        TrainingRunner().run_episode(next_env, agent, replace(scenario, output_dir=tmp_path/f'next{i}'))
    assert learner.last_fit == restored.last_fit
    for k,v in learner.online.state_dict().items():
        torch.testing.assert_close(v, restored.online.state_dict()[k], rtol=0, atol=0)


def test_two_stage_replay_retains_source_node_major_tail_cap():
    # Four nodes with three timesteps: cap six leaves only the last two nodes.
    learner = MPLightLearner.__new__(MPLightLearner)
    learner.config = SimpleNamespace(sample_size=3, replay_capacity=6)
    learner.network = SimpleNamespace(num_intersections=4)
    learner.rng = np.random.default_rng(42)
    learner.replay = [dict(state={'features':np.arange(4)[:,None]},
                          next_state={'features':np.arange(4)[:,None]+10},
                          actions=np.arange(4), reward=np.arange(4)+t, timestep=t)
                      for t in range(3)]
    items = learner._sample_fit_items()
    assert len(items) == 3
    assert all(item['actions'].shape == (1,) for item in items)
    assert all(int(item['actions'][0]) in {2,3} for item in items)
    assert len({(item['timestep'],int(item['actions'][0])) for item in items}) == 3


def test_cli_training_and_evaluation_keep_mplight_identity(tmp_path, monkeypatch):
    def environment(control, network, lane_change):
        env = build_environment(control, network, profile=PROFILE,
                                backend=TrafficBackend(control, network))
        env.metrics = CoLightMetrics(network, lane_change=lane_change)
        return env
    monkeypatch.setattr(experiment, 'make_environment', environment)
    parser = experiment.build_parser()
    roadnet, flow = FIXTURES/'roadnet_baseline_four_arms.json', FIXTURES/'flow_empty.json'
    output = tmp_path/'train'
    args = parser.parse_args(['train','--roadnet',str(roadnet),'--flow',str(flow),
                              '--output',str(output),'--rounds','2','--duration','8',
                              '--decision-interval','2','--yellow-time','1',
                              '--fit-epochs','2','--batch-size','2','--sample-size','4'])
    result = experiment.train(args)
    assert result['protocol']['baseline'] == 'MPLight'
    assert result['protocol']['config']['hidden_dim'] == 20
    assert 'attention_heads' not in result['protocol']['config']
    assert 'MPLight' in (output/'paper_comparison.md').read_text()
    evaluation = experiment.evaluate(parser.parse_args([
        'evaluate','--checkpoint',str(output/'seed_0/latest.pt'), '--roadnet',str(roadnet),
        '--flow',str(flow),'--output',str(tmp_path/'eval'),'--duration','8']))
    assert evaluation['training_protocol']['baseline'] == 'MPLight'
    assert experiment.references('Jinan1')[0]['paper_att_s'] == 307.82
    with pytest.raises(ValueError, match='MPLight'):
        experiment.report(tmp_path/'wrong', {'baseline':'CoLight'}, {})
