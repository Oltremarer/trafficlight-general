from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cityflow_tsc.baselines.colight import CoLightConfig, CoLightLearner, CoLightNetwork, PROFILE
from cityflow_tsc.baselines.q_learning import _state_arrays, _tensor_batch
from cityflow_tsc.baselines.training import TrainingRunner
from cityflow_tsc.colight_results import (CoLightMetrics, aggregate_seed_runs, identify_dataset,
                                         references, FLOW_HASHES, ROADNET_HASHES)
from cityflow_tsc import colight_experiment as experiment
from cityflow_tsc.runtime import build_environment
from cityflow_tsc.simulator import CityFlowBackend
from cityflow_tsc.types import NetworkSnapshot
from .test_baseline_integration import TrafficBackend, make_stack, FIXTURES


def small_config(**kw):
    return CoLightConfig(**dict(hidden_dim=8, batch_size=2, replay_capacity=32,
                               sample_size=4, fit_epochs=2, patience=2,
                               attention_heads=2, attention_head_dim=4, **kw))


def test_round_fit_targets_resume_and_frozen_evaluation(tmp_path):
    torch.set_num_threads(1)
    env, scenario, _ = make_stack(tmp_path, PROFILE)
    initial = env.reset(scenario)
    config = small_config(target_lag_rounds=2)
    learner = CoLightLearner(env.network, initial[0], config, seed=7)
    assert learner.online.attention.head_dim == 4
    original = {k: v.clone() for k, v in learner.online.state_dict().items()}
    initial_target = {k: v.clone() for k, v in learner.target.state_dict().items()}
    result = TrainingRunner().run_episode(env, learner, scenario, initial=initial)
    assert result.metrics["training_updates"] == 2
    assert learner.last_fit["sample_size"] == 4
    assert learner.last_fit["target_round"] == -1
    assert learner.epsilon == pytest.approx(.8 * .95)
    assert any(not torch.equal(v, learner.online.state_dict()[k]) for k, v in original.items())
    assert all(torch.equal(v, learner.target.state_dict()[k]) for k, v in initial_target.items())
    checkpoint = tmp_path / "model.pt"
    learner.save(checkpoint)
    restored = CoLightLearner(env.network, initial[0], config, seed=19)
    restored.load(checkpoint)
    assert all(torch.equal(v, restored.target_history[0][k])
               for k, v in learner.target_history[0].items())
    # Complete several rounds through both instances; target history survives pruning/load.
    for r in range(1, 4):
        for label, agent in (("original", learner), ("restored", restored)):
            next_scenario = replace(scenario, output_dir=tmp_path / label / str(r), seed=8 + r)
            next_env = build_environment(env.control, env.network, profile=PROFILE,
                                         backend=TrafficBackend(env.control, env.network))
            TrainingRunner().run_episode(next_env, agent, next_scenario)
        assert learner.last_fit == restored.last_fit
        assert learner.last_fit["target_round"] == max(r - 2, 0)
        assert all(torch.equal(v, restored.online.state_dict()[k])
                   for k, v in learner.online.state_dict().items())
    before_rng = json.dumps(learner.policy.rng.bit_generator.state)
    before_updates = learner.gradient_steps
    before_replay = learner.environment_steps
    evaluation_env = build_environment(env.control, env.network, profile=PROFILE,
                                       backend=TrafficBackend(env.control, env.network))
    experiment.evaluate_episode(env.control, env.network, learner,
                                replace(scenario, output_dir=tmp_path / "eval", seed=999),
                                True, env=evaluation_env)
    assert learner.gradient_steps == before_updates
    assert learner.environment_steps == before_replay
    assert json.dumps(learner.policy.rng.bit_generator.state) == before_rng


def test_four_phase_q_values_follow_physical_phase_permutation(tmp_path):
    env, scenario, _ = make_stack(tmp_path, PROFILE)
    observation, _ = env.reset(scenario)
    state = _tensor_batch([_state_arrays(observation)], torch.device("cpu"))
    torch.manual_seed(4)
    model = CoLightNetwork(small_config())
    q = model(state)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(state)
    permuted["source_action_to_local"] = state["source_action_to_local"].clone()
    mapping = permuted["source_action_to_local"][..., :4]
    permuted["source_action_to_local"][..., :4] = permutation[mapping]
    torch.testing.assert_close(model(permuted)[..., permutation], q)
    assert q.shape == (1, 1, 4)


def test_source_att_counts_incoming_lanes_and_keeps_engine_att_separate():
    network = SimpleNamespace(intersections=[SimpleNamespace(incoming_lanes=("in",))])
    collector = CoLightMetrics(network)
    def observe(time, vehicles):
        snap = NetworkSnapshot(time, {}, {}, {}, {"in": tuple(vehicles), "out": ("ignored",)}, {})
        collector.observe(snap, 0 if time == 0 else 1)
    observe(0, [])
    observe(1, ["a"])
    # The source empty-previous branch sets the first vehicle's leave time to arrival.
    assert collector.summary(17)["llmtscs_att_s"] == 0
    observe(3, ["a", "b", "b_shadow"])
    observe(4, ["b", "b_shadow"])
    observe(5, ["b", "b_shadow"])
    result = collector.summary(17)
    assert result["llmtscs_att_s"] == pytest.approx((3 + 2) / 2)
    assert result["llmtscs_att_vehicle_count"] == 2
    assert result["average_travel_time_s"] == 17


@pytest.mark.parametrize("lane_change", [False, True])
def test_cityflow_config_preserves_requested_lane_change(tmp_path, monkeypatch, lane_change):
    env, scenario, _ = make_stack(tmp_path, PROFILE)
    calls = []
    def engine(config_path, thread_num):
        calls.append(json.loads(Path(config_path).read_text()))
        return object()
    monkeypatch.setattr("cityflow_tsc.simulator.importlib.import_module",
                        lambda _: SimpleNamespace(Engine=engine))
    monkeypatch.setattr(CityFlowBackend, "snapshot", lambda _: None)
    backend = CityFlowBackend(env.control, lane_change=lane_change)
    backend.reset(scenario)
    assert calls[0]["laneChange"] is lane_change
    backend.close()


def test_aggregation_uses_training_seeds_not_pooled_rounds():
    def run(seed, values):
        return {"status": "completed", "seed": seed,
                "rounds": [{"evaluation": {"att": v}} for v in values]}
    runs = [run(0, [999, 10, 20]), run(1, [999, 30, 40])]
    result = aggregate_seed_runs(runs, 2)
    assert result["metrics"]["att"]["mean"] == 25
    assert result["metrics"]["att"]["std"] == pytest.approx(np.std([15, 35], ddof=1))
    assert aggregate_seed_runs(runs[:1], 2)["metrics"]["att"]["std"] is None
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_seed_runs([runs[0], runs[0]], 2)


def test_lane_change_shadow_count_does_not_inflate_physical_completion(tmp_path):
    flow = tmp_path / 'flow.json'
    flow.write_text(json.dumps([
        {'startTime': 0, 'endTime': 0, 'interval': 1, 'route': ['road']}
    ] * 2))
    scenario = SimpleNamespace(flow_path=flow, duration_s=3, output_dir=tmp_path)
    network = SimpleNamespace(intersections=[])
    initial = NetworkSnapshot(0, {}, {}, {}, vehicle_pool_ids=(),
                              active_vehicle_ids=(), active_vehicle_count=0)
    metrics = CoLightMetrics(network, lane_change=True)
    metrics.configure(scenario, 1, initial)
    metrics.observe(initial, 0)
    with_shadow = replace(initial, time_s=1, vehicle_pool_ids=('flow_0_0', 'flow_1_0'),
                          active_vehicle_ids=('flow_0_0',), active_vehicle_count=2)
    metrics.observe(with_shadow, 1)
    metrics.observe(replace(with_shadow, time_s=2, vehicle_pool_ids=('flow_1_0',),
                            active_vehicle_ids=('flow_1_0',)), 1)
    metrics.observe(replace(with_shadow, time_s=3, vehicle_pool_ids=('flow_1_0',),
                            active_vehicle_ids=('flow_1_0',), active_vehicle_count=1), 1)
    result = metrics.summary(17)
    assert result['finished_vehicles'] == 1
    assert result['active_unfinished_vehicles'] == 1
    assert result['not_entered_vehicles'] == 0
    assert result['completion_rate'] == .5
    assert result['average_travel_time_s'] == 17
    assert with_shadow.active_vehicle_count == 2  # raw engine snapshot is intact
    evidence = json.loads((tmp_path / 'lifecycle.manifest.json').read_text())
    assert 'excluding lane-change shadows' in evidence['active_count_source']
    strict = CoLightMetrics(network, lane_change=False)
    strict.configure(scenario, 1, initial)
    strict.observe(initial, 0)
    with pytest.raises(ValueError, match='active count'):
        strict.observe(with_shadow, 1)


def test_reference_mapping_does_not_confuse_jinan_flow_names():
    assert identify_dataset(ROADNET_HASHES["Jinan"], FLOW_HASHES["Jinan1"]) == "Jinan1"
    assert identify_dataset("wrong-roadnet", FLOW_HASHES["Jinan1"]) == "custom"
    assert not references("custom")
    j1 = references("Jinan1")
    assert j1[0]["paper_att_s"] == 279.60
    assert not any(r["paper"] == "FutureLight" for r in j1)
    assert next(r for r in references("Jinan2") if r["paper"] == "FutureLight")["paper_att_s"] == 271.2
    assert "ZERO-SHOT" in next(r for r in j1 if r["table"] == "2" and r["paper"] == "Traffic-R1")["setting"]


def test_cli_train_evaluate_compare_complete_with_test_backend(tmp_path, monkeypatch):
    def fake_environment(control, network, lane_change):
        env = build_environment(control, network, profile=PROFILE,
                                backend=TrafficBackend(control, network))
        env.metrics = CoLightMetrics(network)
        return env
    monkeypatch.setattr(experiment, "make_environment", fake_environment)
    roadnet = FIXTURES / "roadnet_baseline_four_arms.json"
    flow = FIXTURES / "flow_empty.json"
    output = tmp_path / "experiment"
    parser = experiment.build_parser()
    args = parser.parse_args([
        "train", "--roadnet", str(roadnet), "--flow", str(flow), "--output", str(output),
        "--seeds", "0,1", "--rounds", "2", "--duration", "8", "--decision-interval", "2",
        "--yellow-time", "1", "--fit-epochs", "2", "--batch-size", "2", "--sample-size", "4",
    ])
    result = experiment.train(args)
    assert result["aggregate"]["n_training_seeds"] == 2
    assert result["protocol"]["dataset"] == "custom"
    assert (output / "paper_comparison.csv").is_file()
    assert result["runs"][0]["rounds"][1]["epsilon"] == pytest.approx(.76)
    checkpoint = output / "seed_0" / "latest.pt"
    assert checkpoint.is_file()
    ev = experiment.evaluate(parser.parse_args([
        "evaluate", "--checkpoint", str(checkpoint), "--roadnet", str(roadnet), "--flow", str(flow),
        "--output", str(tmp_path / "frozen"), "--duration", "8",
    ]))
    assert "llmtscs_att_s" in ev["metrics"]
    combined = experiment.compare(parser.parse_args([
        "compare", "--runs", str(output), "--output", str(tmp_path / "compared"),
    ]))
    assert combined == result["aggregate"]
    with pytest.raises(FileExistsError):
        experiment.train(args)
