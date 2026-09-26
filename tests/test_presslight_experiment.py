from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from cityflow_tsc.baselines.presslight import PressLightConfig, PressLightLearner, PressLightNetwork, PROFILE
from cityflow_tsc.baselines.q_learning import _state_arrays, _tensor_batch
from cityflow_tsc.baselines.training import TrainingRunner
from cityflow_tsc import presslight_experiment as experiment
from cityflow_tsc.colight_results import CoLightMetrics
from cityflow_tsc.runtime import build_environment
from .test_baseline_integration import TrafficBackend, make_stack, FIXTURES


def test_cli_training_and_evaluation_keep_presslight_identity(tmp_path, monkeypatch):
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
    assert result['protocol']['baseline'] == 'PressLight'
    assert result['protocol']['config']['hidden_dim'] == 20
    assert 'attention_heads' not in result['protocol']['config']
    assert 'PressLight' in (output/'paper_comparison.md').read_text()
    evaluation = experiment.evaluate(parser.parse_args([
        'evaluate','--checkpoint',str(output/'seed_0/latest.pt'), '--roadnet',str(roadnet),
        '--flow',str(flow),'--output',str(tmp_path/'eval'),'--duration','8']))
    assert evaluation['training_protocol']['baseline'] == 'PressLight'
    assert experiment.references('Jinan1')[0]['paper_att_s'] == 291.57
    with pytest.raises(ValueError, match='PressLight'):
        experiment.report(tmp_path/'wrong', {'baseline':'CoLight'}, {})


def test_phase_branch_selector_and_physical_output_mapping():
    from cityflow_tsc.baselines.presslight import PressLightNetwork
    net = PressLightNetwork(PressLightConfig())
    with torch.no_grad():
        for i, branch in enumerate(net.branches):
            branch[-1].weight.zero_()
            branch[-1].bias.copy_(torch.arange(4) + i * 10)
    phase = net.phase_codes[2].reshape(1,1,8)
    state = {'features':torch.zeros(1,1,20), 'phase_encoding':phase,
             'source_action_to_local':torch.tensor([[[2,0,3,1,-1,-1,-1,-1]]])}
    torch.testing.assert_close(net(state), torch.tensor([[[21.,23.,20.,22.]]]))
    # An all-zero yellow phase matches none of the source selectors.
    torch.testing.assert_close(net({**state,'phase_encoding':torch.zeros_like(phase)}), torch.zeros(1,1,4))
