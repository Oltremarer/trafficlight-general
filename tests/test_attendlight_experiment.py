from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from cityflow_tsc.baselines.attendlight import AttendLightConfig, AttendLightLearner, AttendLightNetwork, AttendLightObservationBuilder, PROFILE
from cityflow_tsc.baselines.q_learning import _state_arrays, _tensor_batch
from cityflow_tsc.baselines.training import TrainingRunner
from cityflow_tsc import attendlight_experiment as experiment
from cityflow_tsc.colight_results import CoLightMetrics
from cityflow_tsc.runtime import build_environment
from .test_baseline_integration import TrafficBackend, make_stack, FIXTURES


def test_cli_training_and_evaluation_keep_attendlight_identity(tmp_path, monkeypatch):
    def environment(control, network, lane_change):
        env = build_environment(control, network, profile=PROFILE,
                                backend=TrafficBackend(control, network))
        env.observation_builder = AttendLightObservationBuilder(network)
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
    assert result['protocol']['baseline'] == 'AttendLight'
    assert result['protocol']['config']['hidden_dim'] == 20
    assert 'attention_heads' not in result['protocol']['config']
    assert 'AttendLight' in (output/'paper_comparison.md').read_text()
    evaluation = experiment.evaluate(parser.parse_args([
        'evaluate','--checkpoint',str(output/'seed_0/latest.pt'), '--roadnet',str(roadnet),
        '--flow',str(flow),'--output',str(tmp_path/'eval'),'--duration','8']))
    assert evaluation['training_protocol']['baseline'] == 'AttendLight'
    assert experiment.references('Jinan1')[0]['paper_att_s'] == 291.29
    with pytest.raises(ValueError, match='AttendLight'):
        experiment.report(tmp_path/'wrong', {'baseline':'CoLight'}, {})




def test_attend_segments_boundaries_outgoing_order_and_shadow():
    from .test_baseline_observations import canonical_network, snapshot, observe, CANONICAL
    network=canonical_network()
    snap=snapshot(network)
    vehicles={lane:() for lane in snap.lane_waiting_count}
    vehicles['in_W_2']=('near','edge100','edge200','edge300','stopped','near_shadow')
    vehicles['out_E_0']=('exit',)
    snap=replace(snap,lane_vehicles=vehicles,
        vehicle_distances=dict(near=401,edge100=400,edge200=300,edge300=200,stopped=490,exit=450),
        vehicle_speeds=dict(near=1,edge100=1,edge200=1,edge300=1,stopped=.1,exit=1))
    builder=AttendLightObservationBuilder(network,CANONICAL)
    view=observe(builder,snap)
    tokens=view.features.reshape(1,24,4)
    np.testing.assert_array_equal(tokens[0,0],[1,1,1,snap.lane_waiting_count['in_W_2']])
    assert tokens[0,12,0]==1  # exiting along the W incoming direction is eastbound
    assert view.schema_id==f'baseline-{PROFILE.schema_hash}'
    np.testing.assert_array_equal(observe(builder,snap,action=2).features,view.features)


def test_attend_attention_and_physical_action_mapping():
    from cityflow_tsc.baselines.attendlight import SourceAttention
    net=AttendLightNetwork()
    features=torch.randn(2,3,96)
    state={'features':features,'source_action_to_local':torch.tensor([0,1,2,3,-1,-1,-1,-1]).expand(2,3,8)}
    original=net(state)
    mapping=torch.tensor([2,0,3,1,-1,-1,-1,-1]).expand(2,3,8)
    changed=net({**state,'source_action_to_local':mapping})
    torch.testing.assert_close(changed.gather(-1,mapping[...,:4]),original)
    original.square().sum().backward()
    assert net.lane_attention.q.grad.abs().sum()>0
    assert net.phase_attention.q.grad.abs().sum()>0
    # A single key has attention weight one regardless of query/key scores.
    a=SourceAttention();q=torch.randn(2,1,32);v=torch.randn(2,1,32)
    values=torch.einsum('btd,dhk->bthk',v,a.v)+a.v_bias
    expected=torch.einsum('bthk,hkd->btd',values,a.output)+a.output_bias
    torch.testing.assert_close(a(q,v),expected)
