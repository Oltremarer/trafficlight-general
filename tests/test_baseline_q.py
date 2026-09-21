from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cityflow_tsc.baselines.contracts import BaselineView, JointTransition, TrainConfig
from cityflow_tsc.baselines.codecs import SOURCE_PHASES
from cityflow_tsc.baselines.observations import BaselineObservationBuilder
from cityflow_tsc.baselines.registry import create_learner
from cityflow_tsc.baselines.profiles import get_profile
from cityflow_tsc.baselines.q_learning import QLearner, _state_arrays, _tensor_batch
from cityflow_tsc.baselines.q_networks import (
    CoLightQNetwork, IndependentQNetworks, PhaseCompetitionQNetwork,
    PressLightQNetwork,
)
from cityflow_tsc.types import IntersectionSpec, MovementSpec, NetworkObservation, NetworkSpec, PolicyOutput
from cityflow_tsc.types import NetworkSnapshot
from cityflow_tsc.config import ControlConfig
from cityflow_tsc.topology import load_network_spec


def make_case(name="mplight", n=2):
    profile = get_profile(name)
    canonical = profile.layout == "canonical12"
    lanes, actions = (12, 4) if canonical else (4, 2)
    phase = np.zeros((actions, lanes), dtype=bool)
    if canonical:
        for action, members in enumerate(((1, 4), (7, 10), (0, 3), (6, 9))):
            phase[action, list(members)] = True
        phase[:, [2, 5, 8, 11]] = True
    else:
        phase[0, :2] = True
        phase[1, 2:] = True
    intersections = []
    for i in range(n):
        lane_ids = tuple(f"in_{i}_{j}" for j in range(lanes))
        movements = tuple(MovementSpec(j, "go_straight", f"road_{i}_{j}", f"out_{i}_{j}",
                                       (lane_ids[j],), (f"out_{i}_{j}_0",)) for j in range(lanes))
        intersections.append(IntersectionSpec(i, f"node_{i}", (float(i), 0.), lane_ids, (), movements,
                                               tuple(range(1, actions + 1)), phase.copy()))
    index = np.stack([np.array([i, (i + 1) % n]) for i in range(n)])
    neighbor_mask = np.ones_like(index, dtype=bool)
    network = NetworkSpec(tuple(intersections), index, neighbor_mask, lanes, actions)
    channels = 2 if profile.feature_kind == "advanced" else 1
    lane_features = np.arange(1, n * lanes * channels + 1, dtype=np.float32).reshape(n, lanes, channels) / 10
    current = np.zeros(n, dtype=np.int64)
    phase_encoding = np.tile(phase[0, [0, 1, 3, 4, 6, 7, 9, 10]], (n, 1)).astype(np.float32) if canonical else np.tile([1., 0.], (n, 1)).astype(np.float32)
    features = np.concatenate([phase_encoding] + [lane_features[..., c] for c in range(channels)], axis=-1)
    view = BaselineView(features, lane_features, np.ones((n, lanes), dtype=bool),
                        np.tile(phase, (n, 1, 1)), phase_encoding, index, neighbor_mask,
                        "test-" + name, network.intersection_ids,
                        tuple(f"feature_{i}" for i in range(features.shape[-1])),
                        np.tile([0, 1, 2, 3, -1, -1, -1, -1], (n, 1)) if canonical else None)
    observation = NetworkObservation(lane_features.copy(), np.ones((n, lanes), dtype=bool),
                                     np.ones_like(lane_features, dtype=bool), current,
                                     np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.float32),
                                     index, neighbor_mask, network.action_mask(), 0., ("test",), view)
    return profile, network, observation


def config(**kwargs):
    return TrainConfig(**{"hidden_dim": 8, "batch_size": 2, "replay_capacity": 8,
                          "warmup_transitions": 0, "updates_per_round": 2,
                          "target_update_steps": 2, **kwargs})


def transition(observation, *, reward=2., terminated=False, truncated=False):
    return JointTransition(observation, PolicyOutput(np.zeros(len(observation.current_phase), dtype=np.int64)),
                           np.full(len(observation.current_phase), reward, dtype=np.float32),
                           replace(observation, time_s=observation.time_s + 30),
                           terminated, truncated, 30., "episode", "scenario")


Q_NAMES = ("presslight", "e-presslight", "mplight", "e-mplight", "a-mplight",
           "colight", "e-colight", "a-colight", "idqn", "shared-dqn", "frap",
           "libsignal-mplight", "libsignal-colight")


@pytest.mark.parametrize("name", Q_NAMES)
def test_each_q_profile_has_actual_td_update_and_target_sync(name):
    profile, network, observation = make_case(name)
    learner = QLearner(profile, network, observation, config(), seed=7)
    before = {k: v.clone() for k, v in learner.online.state_dict().items()}
    for step in range(3):
        learner.observe(transition(observation, reward=step + 1))
        learner.update_if_due("step")
    final_metrics = learner.end_episode()
    assert learner.gradient_steps == 2
    assert learner.completed_episodes == 1
    assert learner.policy.policy_version == 2
    assert any(not torch.equal(before[k], v) for k, v in learner.online.state_dict().items())
    assert all(torch.equal(v, learner.target.state_dict()[k]) for k, v in learner.online.state_dict().items())
    if profile.update_schedule == "round":
        assert final_metrics["updates"] == 2
        assert learner.update_if_due("round") == {}
    assert learner.end_episode() == {}
    output = learner.policy.act(observation, deterministic=True)
    assert output.actions.shape == (network.num_intersections,)
    assert observation.action_mask[np.arange(network.num_intersections), output.actions].all()


def test_presslight_routes_current_phase_to_a_separate_q_branch():
    profile, network, observation = make_case("presslight")
    model = PressLightQNetwork(observation.baseline_view.features.shape[-1], 8, 4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for i, branch in enumerate(model.branches):
            branch[-1].bias.fill_(i + 1.)
    state = _tensor_batch([_state_arrays(observation)], torch.device("cpu"))
    state["current_phase"][0, 1] = 3
    actual = model(state)
    torch.testing.assert_close(actual[0, 0], torch.ones(4))
    torch.testing.assert_close(actual[0, 1], torch.full((4,), 4.))


def test_mplight_competition_is_phase_equivariant_and_ignores_right_turn_slots():
    _, _, observation = make_case("mplight")
    torch.manual_seed(4)
    model = PhaseCompetitionQNetwork(1, 8, canonical=True)
    state = _tensor_batch([_state_arrays(observation)], torch.device("cpu"))
    expected = model(state)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(state)
    permuted["phase_lane_mask"] = state["phase_lane_mask"][:, :, permutation]
    permuted["action_mask"] = state["action_mask"][:, :, permutation]
    torch.testing.assert_close(model(permuted), expected[:, :, permutation])
    state["lane_features"][..., [2, 5, 8, 11], :] = 100000.
    torch.testing.assert_close(model(state), expected)


def test_advanced_mplight_running_branch_receives_learning_gradient():
    _, _, observation = make_case("a-mplight")
    torch.manual_seed(1)
    model = PhaseCompetitionQNetwork(2, 16, canonical=True, advanced=True)
    state = _tensor_batch([_state_arrays(observation)], torch.device("cpu"))
    model(state).square().mean().backward()
    assert model.value_embeddings[1].weight.grad.abs().sum() > 0
    assert model.relation_embedding.weight.grad.abs().sum() > 0


def test_colight_uses_only_connected_neighbors_and_handles_isolated_node():
    _, _, observation = make_case("colight", n=3)
    state = _tensor_batch([_state_arrays(observation)], torch.device("cpu"))
    state["neighbor_index"] = torch.tensor([[[0, 1], [1, 0], [2, -1]]])
    state["neighbor_mask"] = torch.tensor([[[True, True], [True, True], [False, False]]])
    torch.manual_seed(10)
    model = CoLightQNetwork(state["features"].shape[-1], 16, 4)
    before = model(state)
    changed = {k: v.clone() for k, v in state.items()}
    changed["features"][:, 1] += 20
    after = model(changed)
    assert not torch.allclose(after[:, 0], before[:, 0])
    torch.testing.assert_close(after[:, 2], before[:, 2])
    assert torch.isfinite(after).all()


def test_independent_dqn_has_disjoint_parameters():
    profile, network, observation = make_case("idqn")
    learner = QLearner(profile, network, observation, config())
    assert isinstance(learner.online, IndependentQNetworks)
    state = _tensor_batch([_state_arrays(observation)], learner.device)
    before = learner.online(state).detach()
    with torch.no_grad():
        learner.online.models[0].layers[-1].bias.add_(3)
    after = learner.online(state)
    torch.testing.assert_close(after[:, 1], before[:, 1])
    torch.testing.assert_close(after[:, 0], before[:, 0] + 3)


@pytest.mark.parametrize("terminated,truncated,bootstrap,target", [
    (False, False, True, 4.5), (True, False, True, .5),
    (False, True, True, 4.5), (False, True, False, .5),
])
def test_td_masks_illegal_actions_scales_reward_once_and_respects_end_flags(terminated, truncated, bootstrap, target):
    profile, network, observation = make_case("shared-dqn")
    mask = observation.action_mask.copy()
    mask[:, 1] = False
    observation = replace(observation, action_mask=mask)
    learner = QLearner(profile, network, observation,
                       config(batch_size=1, bootstrap_truncated=bootstrap, epsilon_start=1., epsilon_end=1.))
    with torch.no_grad():
        for parameter in learner.target.parameters():
            parameter.zero_()
        learner.target.layers[-1].bias.copy_(torch.tensor([5., 10000.]))
        learner.online.layers[-1].bias.copy_(torch.tensor([0., 10000.]))
    for deterministic in (True, False):
        for _ in range(5):
            assert (learner.policy.act(observation, deterministic).actions == 0).all()
    learner.observe(transition(observation, reward=10., terminated=terminated, truncated=truncated))
    metrics = learner.update_if_due("step")
    assert metrics["mean_target"] == pytest.approx(target)
    assert learner.update_if_due("step") == {}


def test_joint_replay_is_a_snapshot_and_cannot_alias_environment_arrays():
    profile, network, observation = make_case("colight")
    learner = QLearner(profile, network, observation, config())
    learner.observe(transition(observation))
    recorded = learner.replay[0]["state"]["features"].copy()
    observation.baseline_view.features[:] = 999.
    np.testing.assert_array_equal(learner.replay[0]["state"]["features"], recorded)
    assert recorded.shape[0] == network.num_intersections


def test_checkpoint_restores_learning_and_rng_and_rejects_schema_or_config(tmp_path):
    profile, network, observation = make_case("mplight")
    cfg = config(epsilon_start=1., epsilon_end=1.)
    original = QLearner(profile, network, observation, cfg, seed=12)
    for reward in (1., 2., 3.):
        original.observe(transition(observation, reward=reward))
    with pytest.raises(ValueError, match="end_episode"):
        original.save(tmp_path / "mid.pt")
    original.end_episode()
    path = tmp_path / "checkpoint.pt"
    original.save(path)
    restored = QLearner(profile, network, observation, cfg, seed=999)
    restored.load(path)
    assert restored.environment_steps == original.environment_steps
    assert restored.gradient_steps == original.gradient_steps
    assert restored.completed_episodes == original.completed_episodes
    for _ in range(3):
        np.testing.assert_array_equal(original.policy.act(observation, False).actions,
                                      restored.policy.act(observation, False).actions)
    for learner in (original, restored):
        learner.observe(transition(observation, reward=5.))
        learner.end_episode()
    for key, value in original.online.state_dict().items():
        torch.testing.assert_close(value, restored.online.state_dict()[key], rtol=0, atol=0)
    bad_config = QLearner(profile, network, observation, replace(cfg, gamma=.3))
    with pytest.raises(ValueError, match="config"):
        bad_config.load(path)
    old = observation.baseline_view
    changed_view = replace(old, neighbor_index=old.neighbor_index[:, ::-1].copy())
    changed = QLearner(profile, network, replace(observation, baseline_view=changed_view), cfg)
    with pytest.raises(ValueError, match="schema"):
        changed.load(path)


def test_invalid_schema_and_action_are_rejected_before_replay():
    profile, network, observation = make_case("mplight")
    learner = QLearner(profile, network, observation, config())
    view = replace(observation.baseline_view, feature_names=tuple(reversed(observation.baseline_view.feature_names)))
    with pytest.raises(ValueError, match="schema"):
        learner.policy.act(replace(observation, baseline_view=view), True)
    item = transition(observation)
    with pytest.raises(ValueError, match="invalid action"):
        learner.observe(replace(item, output=PolicyOutput(np.full(network.num_intersections, 100))))
    assert learner.environment_steps == 0


def heterogeneous_phase_case(tmp_path, name):
    original = json.loads((Path(__file__).parent / "fixtures" / "roadnet_baseline_four_arms.json").read_text())
    identifiers = {item["id"] for key in ("intersections", "roads") for item in original[key]}
    raw = {"intersections": [], "roads": []}
    for prefix in ("a_", "b_"):
        def rename(value):
            if isinstance(value, dict):
                return {k: rename(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rename(v) for v in value]
            return prefix + value if isinstance(value, str) and value in identifiers else value
        node = rename(original)
        if prefix == "b_":
            phases = node["intersections"][0]["trafficLight"]["lightphases"]
            phases[2], phases[3] = phases[3], phases[2]
        for key in raw:
            raw[key].extend(node[key])
    path = tmp_path / "different_phase_orders.json"
    path.write_text(json.dumps(raw))
    network = load_network_spec(path, ControlConfig(green_phase_ids=(1, 2, 3, 4)))
    builder = BaselineObservationBuilder(network, name, path)
    counts = {lane: 5 for inter in network.intersections for lane in inter.incoming_lanes + inter.outgoing_lanes}
    snapshot = NetworkSnapshot(0., counts, counts, {}, {lane: () for lane in counts}, {})
    observation = builder.build(snapshot, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int8), np.zeros(2))
    return network, builder, snapshot, observation


@pytest.mark.parametrize("name", ["presslight", "e-presslight", "colight", "e-colight", "a-colight"])
def test_fixed_heads_map_physical_phases_for_execution_replay_and_td(tmp_path, name):
    network, builder, snap, observation = heterogeneous_phase_case(tmp_path, name)
    mapping = observation.baseline_view.source_action_to_local
    np.testing.assert_array_equal(mapping[:, :4], [[0, 1, 2, 3], [0, 2, 1, 3]])
    np.testing.assert_array_equal(observation.baseline_view.features[0], observation.baseline_view.features[1])
    learner = create_learner(name, network, observation,
                             config(batch_size=1, epsilon_start=0., epsilon_end=0.))
    model = learner.online.model
    with torch.no_grad():
        for parameter in learner.online.parameters():
            parameter.zero_()
        heads = [branch[-1] for branch in model.branches] if name.endswith("presslight") else [model.action_layer]
        for head in heads:
            # Unavailable source phase 7 must not win execution or TD bootstrap.
            head.bias.copy_(torch.tensor([0., 10., 0., 0., 0., 0., 0., 10000.]))
        learner.target.load_state_dict(learner.online.state_dict())
    output = learner.policy.act(observation, True)
    np.testing.assert_array_equal(output.actions, [1, 2])
    actual_sources = [np.flatnonzero(mapping[i] == action)[0] for i, action in enumerate(output.actions)]
    assert actual_sources == [1, 1]
    assert [network.engine_phase(i, action) for i, action in enumerate(output.actions)] == [2, 3]
    following = replace(observation, time_s=30.)
    learner.observe(JointTransition(observation, output, np.zeros(2, np.float32), following,
                                    False, True, 30., "episode", "scenario"))
    np.testing.assert_array_equal(learner.replay[0]["actions"], [1, 2])
    learner.config = replace(learner.config, updates_per_round=1, target_update_steps=100)
    result = learner.end_episode()
    assert result["mean_target"] == pytest.approx(8.)
    assert result["loss"] == pytest.approx(4.)  # both selected physical Qs were 10
    trained_heads = [model.branches[0][-1]] if name.endswith("presslight") else heads
    for head in trained_heads:
        assert head.bias[1] < 10.
        assert head.bias[2] == 0. and head.bias[7] == 10000.
    # Current physical phase 1 has different local indices, but uses the same PressLight branch.
    if name.endswith("presslight"):
        current = np.array([1, 2])
        same_phase = builder.build(snap, current, np.zeros(2, dtype=np.int8), np.zeros(2))
        with torch.no_grad():
            for i, branch in enumerate(model.branches):
                branch[-1].bias.fill_(float(i))
        state = _tensor_batch([_state_arrays(same_phase)], learner.device)
        torch.testing.assert_close(learner.online(state), torch.ones((1, 2, 4)))
    # Exploration keeps local legality even though the learned heads have eight source slots.
    learner.config = replace(learner.config, epsilon_start=1., epsilon_end=1.)
    for _ in range(20):
        actions = learner.policy.act(observation, False).actions
        assert observation.action_mask[np.arange(2), actions].all()
    path = tmp_path / "mapped.pt"
    learner.save(path)
    restored = create_learner(name, network, observation, learner.config, seed=99)
    restored.load(path)
    np.testing.assert_array_equal(learner.policy.act(observation, True).actions,
                                  restored.policy.act(observation, True).actions)


def test_canonical_mapping_cannot_disagree_with_physical_phase_mask():
    profile, network, observation = make_case("colight")
    view = observation.baseline_view
    wrong = view.source_action_to_local.copy()
    wrong[:, [1, 2]] = wrong[:, [2, 1]]
    with pytest.raises(ValueError, match="physical phase meanings"):
        QLearner(profile, network, replace(observation, baseline_view=replace(view, source_action_to_local=wrong)), config())


@pytest.mark.parametrize("name", ["frap", "libsignal-mplight"])
def test_libsignal_q_and_gradients_ignore_permissive_right_turn_demand(name):
    path = Path(__file__).parent / "fixtures" / "roadnet_baseline_four_arms.json"
    network = load_network_spec(path, ControlConfig())
    builder = BaselineObservationBuilder(network, name, path)
    counts = {lane: i + 1 for i, lane in enumerate(network.intersections[0].incoming_lanes)}
    observation = builder.build(NetworkSnapshot(0., counts, counts, {}),
                                 np.array([0]), np.array([0]), np.array([0.]))
    learner = create_learner(name, network, observation, config(), seed=4)
    state = _tensor_batch([_state_arrays(observation)], learner.device)
    right = [i for i, lane in enumerate(builder.codec.lane_ids[0]) if lane.endswith("_1")]
    state["lane_features"].requires_grad_(True)
    expected = learner.online(state)
    expected.sum().backward()
    gradient = state["lane_features"].grad
    assert torch.count_nonzero(gradient[..., right, :]) == 0
    assert torch.count_nonzero(gradient) > 0
    changed = {key: value.detach().clone() for key, value in state.items()}
    changed["lane_features"][..., right, :] = 100000.
    torch.testing.assert_close(learner.online(changed), expected)


def test_q_checkpoints_before_phase_semantics_fix_are_rejected(tmp_path):
    profile, network, observation = make_case("colight")
    learner = QLearner(profile, network, observation, config())
    path = tmp_path / "old.pt"
    learner.save(path)
    payload = torch.load(path, weights_only=False)
    payload["checkpoint_version"] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint_version mismatch"):
        learner.load(path)


def test_libsignal_relation_embedding_clips_negative_values_before_convolution():
    _, _, observation = make_case("libsignal-mplight")
    model = PhaseCompetitionQNetwork(1, 8, canonical=False, phase_pairs=True)
    state = _tensor_batch([_state_arrays(observation)], torch.device("cpu"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(.1)
        model.relation_conv.bias.fill_(1.)
        model.relation_embedding.weight.fill_(-2.)
        negative = model(state)
        model.relation_embedding.weight.zero_()
        zero = model(state)
        model.relation_embedding.weight.fill_(2.)
        positive = model(state)
    torch.testing.assert_close(negative, zero)
    assert torch.all(positive > zero)
