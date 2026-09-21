from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cityflow_tsc.baselines.contracts import BaselineView, JointTransition, TrainConfig
from cityflow_tsc.baselines.maddpg import MADDPGLearner
from cityflow_tsc.baselines.ppo import PPOLearner, estimate_returns
from cityflow_tsc.baselines.profiles import get_profile
from cityflow_tsc.types import IntersectionSpec, NetworkObservation, NetworkSpec


def network_fixture():
    intersections = tuple(
        IntersectionSpec(i, name, (float(i), 0.0), (f"{name}_in",), (f"{name}_out",), (), phases,
                         np.array([[1], [1], [1]], dtype=np.bool_))
        for i, (name, phases) in enumerate((("a", (1, 2, -1)), ("b", (1, -1, 3))))
    )
    return NetworkSpec(intersections, np.array([[1, -1, -1, -1], [0, -1, -1, -1]]),
                       np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.bool_), 1, 3)


def observation(network, t=0.0):
    actor_features = np.array([[1 + t * .3, 3 + t * .4, .2], [2 - t * .1, .5 + t * .2, 1]], dtype=np.float32)
    view = BaselineView(
        features=actor_features, lane_features=np.ones((2, 2, 2), dtype=np.float32),
        lane_mask=np.ones((2, 2), dtype=np.bool_),
        phase_lane_mask=np.array([[[1, 0], [0, 1], [0, 0]], [[1, 0], [0, 0], [0, 1]]], dtype=np.bool_),
        phase_encoding=np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32),
        neighbor_index=network.neighbor_index.copy(), neighbor_mask=network.neighbor_mask.copy(),
        schema_id="test-lane-view-v1", node_ids=network.intersection_ids, feature_names=("x", "queue", "phase"))
    return NetworkObservation(
        features=np.zeros((2, 1, 6), dtype=np.float32), movement_mask=np.ones((2, 1), dtype=np.bool_),
        valid_mask=np.ones((2, 1, 6), dtype=np.bool_), current_phase=np.zeros(2, dtype=np.int64),
        signal_stage=np.zeros(2, dtype=np.int8), phase_elapsed_s=np.full(2, t, dtype=np.float32),
        neighbor_index=network.neighbor_index.copy(), neighbor_mask=network.neighbor_mask.copy(),
        action_mask=network.action_mask(), time_s=float(t), feature_names=tuple(f"f{i}" for i in range(6)), baseline_view=view)


def small_config(**kwargs):
    values = dict(hidden_dim=16, batch_size=2, warmup_transitions=2, replay_capacity=16,
                  ppo_epochs=2, learning_rate=3e-3, reward_scale=1.0, epsilon_start=.5,
                  epsilon_end=.1, tau=.2, n_steps=2)
    values.update(kwargs)
    return TrainConfig(**values)


def collect_episode(learner, network, length=4, episode_id="e0"):
    outputs = []
    for t in range(length):
        obs, next_obs = observation(network, t), observation(network, t + 1)
        output = learner.policy.act(obs, deterministic=False)
        transition = JointTransition(obs, output, np.array([-1.0 - t, -.2 - .2 * t], dtype=np.float32),
                                     next_obs, False, t == length - 1, 1.0, episode_id)
        learner.observe(transition)
        outputs.append(output)
    return outputs


def copied_parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def changed(before, module):
    return any(not torch.equal(a, b) for a, b in zip(before, module.parameters()))


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_ppo_fresh_recurrent_rollout_updates_actor_and_critic(algorithm):
    network = network_fixture()
    learner = PPOLearner(get_profile(algorithm), network, observation(network), small_config(), seed=7)
    learner.policy.reset(7, network)
    actor_before, critic_before = copied_parameters(learner.actor), copied_parameters(learner.critic)
    outputs = collect_episode(learner, network)
    assert np.count_nonzero(outputs[0].behavior.hidden_state) == 0
    assert np.count_nonzero(outputs[1].behavior.hidden_state) > 0
    for output in outputs:
        assert network.action_mask()[np.arange(2), output.actions].all()
        assert np.isfinite(output.behavior.log_prob).all()
    np.testing.assert_allclose(learner.rollout[0]["reward"], [-.6, -.6])
    with torch.no_grad():
        logs, _ = learner._sequence()
    np.testing.assert_allclose(logs.cpu().numpy(), np.stack([o.behavior.log_prob for o in outputs]), atol=2e-6)
    assert learner.update_if_due("step") == {}
    assert not changed(actor_before, learner.actor)
    metrics = learner.end_episode()
    assert metrics["updates"] == 2 and learner.gradient_steps == 2
    assert changed(actor_before, learner.actor) and changed(critic_before, learner.critic)
    assert not learner.rollout and learner.policy_version == 1 and learner.completed_episodes == 1
    with pytest.raises(ValueError, match="current policy version"):
        learner.observe(JointTransition(observation(network), outputs[0], np.zeros(2), observation(network, 1), True, False, 1))


@pytest.mark.parametrize("estimator", ["nstep", "gae"])
@pytest.mark.parametrize("terminal,bootstrap,expected", [(False, True, [9.5, 17]), (False, False, [2, 2]), (True, True, [2, 2])])
def test_ppo_return_targets_distinguish_terminal_and_truncation(estimator, terminal, bootstrap, expected):
    config = small_config(gamma=.5, return_estimator=estimator, gae_lambda=1.0, bootstrap_truncated=bootstrap)
    targets, _ = estimate_returns(np.array([[1], [2]], dtype=np.float32), np.array([[10], [20]], dtype=np.float32),
                                 np.array([30], dtype=np.float32), np.array([False, terminal]),
                                 np.array([False, not terminal]), config)
    np.testing.assert_allclose(targets[:, 0], expected)


def test_ippo_critic_is_local_mappo_critic_uses_other_nodes():
    network = network_fixture()
    features = torch.ones(2, 3)
    perturbed = features.clone()
    perturbed[1] += 10
    for algorithm in ("ippo", "mappo"):
        learner = PPOLearner(get_profile(algorithm), network, observation(network), small_config(), seed=9)
        with torch.no_grad():
            for parameter in learner.critic.parameters():
                parameter.fill_(.1)
            before, after = learner.critic(features), learner.critic(perturbed)
        if algorithm == "ippo":
            assert before[0] == after[0]
        else:
            assert before[0] != after[0]


def test_ppo_rejects_corrupted_behavior_and_partial_checkpoint(tmp_path):
    network = network_fixture()
    learner = PPOLearner(get_profile("ippo"), network, observation(network), small_config())
    obs, next_obs = observation(network), observation(network, 1)
    output = learner.policy.act(obs, False)
    learner.observe(JointTransition(obs, output, np.zeros(2), next_obs, False, False, 1, "e"))
    with pytest.raises(ValueError, match="episode boundary"):
        learner.save(tmp_path / "partial.pt")
    with pytest.raises(ValueError, match="complete episode"):
        learner.end_episode()
    final_output = learner.policy.act(next_obs, False)
    corrupted = replace(final_output, behavior=replace(final_output.behavior, log_prob=final_output.behavior.log_prob + .5))
    learner.observe(JointTransition(next_obs, corrupted, np.ones(2), observation(network, 2), True, False, 1, "e"))
    with pytest.raises(ValueError, match="behavior probabilities"):
        learner.end_episode()


def test_ppo_checkpoint_preserves_sampling_and_checks_semantics(tmp_path):
    network = network_fixture()
    learner = PPOLearner(get_profile("mappo"), network, observation(network), small_config(), seed=5)
    collect_episode(learner, network)
    learner.end_episode()
    path = tmp_path / "ppo.pt"
    learner.save(path)
    restored = PPOLearner(get_profile("mappo"), network, observation(network), small_config(), seed=999)
    restored.load(path)
    for _ in range(3):
        expected, actual = learner.policy.act(observation(network, 4), False), restored.policy.act(observation(network, 4), False)
        np.testing.assert_array_equal(expected.actions, actual.actions)
        np.testing.assert_array_equal(expected.behavior.log_prob, actual.behavior.log_prob)
    learner.policy.reset(12, network)
    restored.policy.reset(12, network)
    collect_episode(learner, network, episode_id="second")
    collect_episode(restored, network, episode_id="second")
    assert restored.end_episode() == pytest.approx(learner.end_episode())
    for a, b in zip(learner.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(a, b)
    alternate = observation(network)
    alternate = replace(alternate, baseline_view=replace(alternate.baseline_view,
                        phase_lane_mask=np.flip(alternate.baseline_view.phase_lane_mask, axis=-1).copy()))
    wrong = PPOLearner(get_profile("mappo"), network, alternate, small_config())
    with pytest.raises(ValueError, match="contract"):
        wrong.load(path)


def test_maddpg_joint_updates_masked_targets_and_resume(tmp_path):
    network = network_fixture()
    learner = MADDPGLearner(get_profile("maddpg"), network, observation(network), small_config(), seed=13)
    actor_before, critic_before = copied_parameters(learner.actors), copied_parameters(learner.critics)
    target_before = copied_parameters(learner.target_actors)
    outputs = collect_episode(learner, network)
    for output in outputs:
        vectors = output.behavior.action_vector
        assert (vectors[~network.action_mask()] == 0).all()
        np.testing.assert_array_equal(vectors.argmax(-1), output.actions)
    metrics = learner.update_if_due("step")
    assert metrics["updates"] == 1
    assert changed(actor_before, learner.actors) and changed(critic_before, learner.critics)
    for previous, source, target in zip(target_before, learner.actors.parameters(), learner.target_actors.parameters()):
        torch.testing.assert_close(target, previous.lerp(source.detach(), learner.config.tau))
    with torch.no_grad():
        for i, actor in enumerate(learner.target_actors):
            actor.network[-1].bias[~torch.as_tensor(network.action_mask()[i])] = 1e6
        target_vectors = learner.actor_vectors(torch.tensor(observation(network).baseline_view.features),
                                              torch.tensor(network.action_mask()), target=True)
    assert (target_vectors[~torch.tensor(network.action_mask())] == 0).all()
    learner.end_episode()
    path = tmp_path / "maddpg.pt"
    learner.save(path)
    restored = MADDPGLearner(get_profile("maddpg"), network, observation(network), small_config(), seed=999)
    restored.load(path)
    assert len(restored.replay) == len(learner.replay) == 4
    for _ in range(3):
        expected, actual = learner.policy.act(observation(network), False), restored.policy.act(observation(network), False)
        np.testing.assert_array_equal(expected.actions, actual.actions)
        np.testing.assert_array_equal(expected.behavior.action_vector, actual.behavior.action_vector)
    assert restored._learn_once() == pytest.approx(learner._learn_once())
    for a, b in zip(learner.actors.parameters(), restored.actors.parameters()):
        assert torch.equal(a, b)


def test_maddpg_requires_matching_behavior_vector_and_boundary(tmp_path):
    network = network_fixture()
    learner = MADDPGLearner(get_profile("maddpg"), network, observation(network), small_config())
    obs, next_obs = observation(network), observation(network, 1)
    output = learner.policy.act(obs, False)
    missing = replace(output, behavior=None)
    with pytest.raises(ValueError, match="recorded"):
        learner.observe(JointTransition(obs, missing, np.zeros(2), next_obs, False, False, 1))
    learner.observe(JointTransition(obs, output, np.zeros(2), next_obs, False, False, 1))
    with pytest.raises(ValueError, match="episode boundary"):
        learner.save(tmp_path / "partial.pt")
