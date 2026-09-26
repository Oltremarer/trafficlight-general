import pytest
import numpy as np

torch = pytest.importorskip('torch')

from cityflow_tsc import ecolight_experiment as experiment
from cityflow_tsc.baselines.ecolight import PROFILE
from cityflow_tsc.baselines.observations import BaselineObservationBuilder
from cityflow_tsc.colight_results import CoLightMetrics
from cityflow_tsc.runtime import build_environment
from .test_baseline_integration import TrafficBackend, FIXTURES
from .test_baseline_observations import canonical_network, snapshot, observe, CANONICAL


def test_ecolight_uses_efficient_queue_pressure_not_counts_or_general_pressure():
    network = canonical_network()
    snap = snapshot(network)
    efficient = observe(BaselineObservationBuilder(network, PROFILE, CANONICAL), snap)
    general = observe(BaselineObservationBuilder(network, 'presslight', CANONICAL), snap)
    counts = observe(BaselineObservationBuilder(network, 'colight', CANONICAL), snap)
    incoming = counts.lane_features[..., 0] / 10
    expected = incoming - (incoming - general.lane_features[..., 0]) / 3
    np.testing.assert_allclose(efficient.lane_features[..., 0], expected, rtol=1e-6)
    assert not np.allclose(efficient.features, counts.features)
    assert PROFILE.reward_kind == 'queue'


def test_ecolight_train_evaluate_compare_and_checkpoint_identity(tmp_path, monkeypatch):
    def environment(control, network, lane_change):
        env = build_environment(control, network, profile=PROFILE,
                                backend=TrafficBackend(control, network))
        env.metrics = CoLightMetrics(network, lane_change=lane_change)
        return env
    monkeypatch.setattr(experiment, 'make_environment', environment)
    parser = experiment.build_parser()
    roadnet, flow = FIXTURES/'roadnet_baseline_four_arms.json', FIXTURES/'flow_empty.json'
    output = tmp_path/'train'
    result = experiment.train(parser.parse_args([
        'train','--roadnet',str(roadnet),'--flow',str(flow),'--output',str(output),
        '--rounds','2','--duration','8','--decision-interval','2','--yellow-time','1',
        '--fit-epochs','2','--batch-size','2','--sample-size','4']))
    assert result['protocol']['baseline'] == 'E-CoLight'
    assert result['protocol']['profile']['feature_kind'] == 'efficient_queue_pressure'
    assert all(r['fit']['updates'] > 0 for r in result['runs'][0]['rounds'])
    checkpoint = output/'seed_0/latest.pt'
    payload = torch.load(checkpoint, weights_only=False)
    assert payload['profile']['baseline_id'] == 'e-colight'
    ev = experiment.evaluate(parser.parse_args([
        'evaluate','--checkpoint',str(checkpoint),'--roadnet',str(roadnet),'--flow',str(flow),
        '--output',str(tmp_path/'eval'),'--duration','8']))
    assert ev['training_protocol']['baseline'] == 'E-CoLight'
    combined = experiment.compare(parser.parse_args([
        'compare','--runs',str(output),'--output',str(tmp_path/'comparison')]))
    assert combined == result['aggregate']
    assert experiment.references('Jinan1')[0]['paper_att_s'] == 277.11
