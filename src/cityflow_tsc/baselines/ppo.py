"""Recurrent PPO adapters with fresh episodes, not a cMALC environment reproduction.

The actor shares parameters across intersections. IPPO uses a local V critic;
MAPPO uses a joint-state V critic. Lane views/rewards belong to the local profile.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..dqn import network_schema_sha256
from ..types import NetworkObservation, NetworkSpec, PolicyOutput
from .contracts import BehaviorInfo, JointTransition, TrainConfig, require_view
from .profiles import BaselineProfile


def observation_contract(network: NetworkSpec, observation: NetworkObservation) -> dict[str, Any]:
    view = require_view(observation)
    return {
        "network": network_schema_sha256(network),
        "schema_id": view.schema_id,
        "node_ids": list(view.node_ids),
        "feature_names": list(view.feature_names),
        "feature_shape": list(view.features.shape),
        "lane_shape": list(view.lane_features.shape),
        "lane_mask": view.lane_mask.tolist(),
        "phase_lane_mask": view.phase_lane_mask.tolist(),
        "neighbor_index": view.neighbor_index.tolist(),
        "neighbor_mask": view.neighbor_mask.tolist(),
        "action_mask": observation.action_mask.tolist(),
    }


def contract_hash(contract: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def check_observation(network: NetworkSpec, observation: NetworkObservation,
                      expected: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    view = require_view(observation)
    if tuple(view.node_ids) != network.intersection_ids:
        raise ValueError("baseline node order does not match the environment")
    if observation_contract(network, observation) != expected:
        raise ValueError("baseline observation/action/topology schema changed")
    features = np.asarray(view.features, dtype=np.float32)
    mask = np.asarray(observation.action_mask, dtype=np.bool_)
    if features.ndim != 2 or features.shape[0] != network.num_intersections:
        raise ValueError("baseline features must have shape [intersections, features]")
    if not np.all(np.isfinite(features)):
        raise ValueError("baseline features must be finite")
    if mask.shape != (network.num_intersections, network.max_actions) or not mask.any(axis=1).all():
        raise ValueError("every intersection needs a valid action")
    return features.copy(), mask.copy()


def validate_transition_actions(transition: JointTransition, mask: np.ndarray) -> np.ndarray:
    actions = np.asarray(transition.actions)
    if actions.shape != (mask.shape[0],) or actions.dtype.kind not in "iu":
        raise ValueError("actions must be an integer vector in environment node order")
    if np.any(actions < 0) or np.any(actions >= mask.shape[1]):
        raise ValueError("action index is outside the action space")
    if not mask[np.arange(len(actions)), actions].all():
        raise ValueError("transition contains an illegal action")
    if not np.isfinite(transition.elapsed_s) or transition.elapsed_s <= 0:
        raise ValueError("transition elapsed time must be positive")
    return actions.astype(np.int64, copy=True)


def reward_vector(transition: JointTransition, profile: BaselineProfile,
                  config: TrainConfig, n_agents: int) -> np.ndarray:
    reward = np.asarray(transition.reward, dtype=np.float32)
    if reward.shape != (n_agents,) or not np.all(np.isfinite(reward)):
        raise ValueError("environment reward must be a finite per-intersection vector")
    if profile.global_reward:
        reward = np.full(n_agents, reward.mean(), dtype=np.float32)
    return reward.copy() * config.reward_scale


def atomic_save(value: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(target)


class RecurrentActor(nn.Module):
    def __init__(self, feature_dim: int, n_agents: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(feature_dim + n_agents, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_actions)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gru(F.relu(self.encoder(inputs)), hidden)
        return self.output(hidden), hidden


class ValueCritic(nn.Module):
    def __init__(self, feature_dim: int, n_agents: int, hidden_dim: int, centralized: bool):
        super().__init__()
        self.n_agents = n_agents
        self.centralized = centralized
        width = feature_dim * (n_agents if centralized else 1) + n_agents
        self.network = nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Supports [N,F] and [T,N,F], preserving time and node alignment.
        leading = features.shape[:-2]
        identity = torch.eye(self.n_agents, device=features.device).expand(*leading, self.n_agents, self.n_agents)
        if self.centralized:
            inputs = features.flatten(-2).unsqueeze(-2).expand(*leading, self.n_agents, -1)
        else:
            inputs = features
        return self.network(torch.cat((inputs, identity), dim=-1)).squeeze(-1)


def estimate_returns(rewards: np.ndarray, values: np.ndarray, next_value: np.ndarray,
                     terminated: np.ndarray, truncated: np.ndarray,
                     config: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return targets and advantages, preserving terminal vs time-limit bootstrap."""
    next_values = np.concatenate((values[1:], next_value[None]), axis=0)
    bootstrap = ~(terminated | (truncated & (not config.bootstrap_truncated)))
    continuation = ~(terminated | truncated)
    targets = np.zeros_like(rewards, dtype=np.float32)
    if config.return_estimator == "gae":
        advantage = np.zeros(rewards.shape[1], dtype=np.float32)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + config.gamma * float(bootstrap[t]) * next_values[t] - values[t]
            advantage = delta + config.gamma * config.gae_lambda * float(continuation[t]) * advantage
            targets[t] = advantage + values[t]
    else:
        for start in range(len(rewards)):
            discount = 1.0
            for offset in range(config.n_steps):
                t = start + offset
                if t >= len(rewards):
                    break
                targets[start] += discount * rewards[t]
                discount *= config.gamma
                if not continuation[t] or t == len(rewards) - 1 or offset == config.n_steps - 1:
                    if bootstrap[t]:
                        targets[start] += discount * next_values[t]
                    break
    return targets, targets - values


class _PPOPolicy:
    def __init__(self, learner: "PPOLearner"):
        self.learner = learner
        self.name = learner.profile.baseline_id
        self.hidden = torch.zeros(learner.n_agents, learner.config.hidden_dim, device=learner.device)

    def reset(self, seed: int, network: NetworkSpec) -> None:
        if network_schema_sha256(network) != self.learner.contract["network"]:
            raise ValueError("policy reset received an incompatible network")
        if self.learner.rollout:
            raise ValueError("finish the PPO training episode before resetting policy context")
        self.learner.action_rng.manual_seed(seed)
        self.hidden = torch.zeros_like(self.hidden)

    def act(self, observation: NetworkObservation, deterministic: bool) -> PolicyOutput:
        learner = self.learner
        features, mask = check_observation(learner.network, observation, learner.contract)
        hidden_before = self.hidden.detach().cpu().numpy().copy()
        with torch.no_grad():
            logits, next_hidden = learner.actor(learner.actor_inputs(features), self.hidden)
            logits = logits.masked_fill(~torch.as_tensor(mask, device=learner.device), -torch.inf)
            probabilities = torch.softmax(logits, dim=-1)
            if deterministic:
                actions = probabilities.argmax(dim=-1)
            else:
                actions = torch.multinomial(probabilities.cpu(), 1, generator=learner.action_rng).squeeze(-1).to(learner.device)
            log_prob = torch.log_softmax(logits, dim=-1).gather(-1, actions[:, None]).squeeze(-1)
            values = learner.critic(torch.as_tensor(features, device=learner.device))
        self.hidden = next_hidden.detach()
        return PolicyOutput(
            actions=actions.cpu().numpy().copy(),
            recurrent_state=self.hidden.cpu().numpy().copy(),
            behavior=BehaviorInfo(log_prob=log_prob.cpu().numpy().copy(), value=values.cpu().numpy().copy(),
                                  hidden_state=hidden_before, policy_version=learner.policy_version),
        )

    def metadata(self) -> dict[str, Any]:
        return {"profile": self.learner.profile.to_dict(), "policy_version": self.learner.policy_version,
                "observation_contract_hash": contract_hash(self.learner.contract),
                "training_config": self.learner.config.to_dict()}


class PPOLearner:
    """Updates only completed fresh episodes. Checkpoints require no pending rollout."""
    checkpoint_version = "cityflow-baseline-recurrent-ppo-v1"

    def __init__(self, profile: BaselineProfile, network: NetworkSpec,
                 initial_observation: NetworkObservation, config: TrainConfig,
                 seed: int = 0, device: str = "cpu"):
        if profile.algorithm not in {"ippo", "mappo"}:
            raise ValueError("PPOLearner requires an IPPO or MAPPO profile")
        self.profile, self.network, self.config = profile, network, config
        self.device = torch.device(device)
        self.contract = observation_contract(network, initial_observation)
        features, _ = check_observation(network, initial_observation, self.contract)
        self.n_agents, self.feature_dim = features.shape
        self.action_rng = torch.Generator(device="cpu").manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = RecurrentActor(self.feature_dim, self.n_agents, network.max_actions, config.hidden_dim).to(self.device)
            self.critic = ValueCritic(self.feature_dim, self.n_agents, config.hidden_dim,
                                      profile.algorithm == "mappo").to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.policy_version = 0
        self.environment_steps = 0
        self.gradient_steps = 0
        self.completed_episodes = 0
        self.rollout: list[dict[str, Any]] = []
        self.policy = _PPOPolicy(self)

    def actor_inputs(self, features: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        return torch.cat((tensor, torch.eye(self.n_agents, device=self.device)), dim=-1)

    def observe(self, transition: JointTransition) -> None:
        features, mask = check_observation(self.network, transition.observation, self.contract)
        next_features, _ = check_observation(self.network, transition.next_observation, self.contract)
        actions = validate_transition_actions(transition, mask)
        behavior = transition.behavior
        if behavior is None or behavior.policy_version != self.policy_version:
            raise ValueError("PPO requires behavior from the current policy version")
        arrays = {}
        for name, shape in (("log_prob", (self.n_agents,)), ("value", (self.n_agents,)),
                            ("hidden_state", (self.n_agents, self.config.hidden_dim))):
            value = np.asarray(getattr(behavior, name), dtype=np.float32)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"PPO requires finite stored behavior {name}")
            arrays[name] = value.copy()
        if self.rollout:
            previous = self.rollout[-1]
            if previous["terminated"] or previous["truncated"]:
                raise ValueError("PPO episode must update before collecting another episode")
            if previous["episode_id"] != transition.episode_id:
                raise ValueError("PPO rollout cannot cross episode IDs")
            if not np.array_equal(previous["next_features"], features):
                raise ValueError("PPO rollout observations are not temporally contiguous")
        self.rollout.append({"features": features, "next_features": next_features, "mask": mask,
                             "actions": actions, "reward": reward_vector(transition, self.profile, self.config, self.n_agents),
                             "terminated": bool(transition.terminated), "truncated": bool(transition.truncated),
                             "episode_id": transition.episode_id, **arrays})
        self.environment_steps += 1

    def _sequence(self) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.as_tensor(self.rollout[0]["hidden_state"], device=self.device)
        log_probs, entropies = [], []
        for item in self.rollout:
            logits, hidden = self.actor(self.actor_inputs(item["features"]), hidden)
            mask = torch.as_tensor(item["mask"], device=self.device)
            logits = logits.masked_fill(~mask, -torch.inf)
            logs = torch.log_softmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            actions = torch.as_tensor(item["actions"], device=self.device)
            log_probs.append(logs.gather(-1, actions[:, None]).squeeze(-1))
            # Avoid 0 * -inf in padded action positions.
            entropies.append(-(probabilities * logs.masked_fill(~mask, 0.0)).sum(-1))
        return torch.stack(log_probs), torch.stack(entropies)

    def update_if_due(self, event: str = "step") -> dict[str, float]:
        if event not in {"step", "round", "rollout"}:
            raise ValueError("unknown training event")
        if event != "rollout" or not self.rollout:
            return {}
        if not (self.rollout[-1]["terminated"] or self.rollout[-1]["truncated"]):
            raise ValueError("recurrent PPO only updates at complete episode boundaries")
        return self._update_episode()

    def end_episode(self) -> dict[str, float]:
        return self.update_if_due("rollout")

    def _update_episode(self) -> dict[str, float]:
        old_logs = torch.as_tensor(np.stack([x["log_prob"] for x in self.rollout]), device=self.device)
        with torch.no_grad():
            checked_logs, _ = self._sequence()
            if not torch.allclose(checked_logs, old_logs, atol=2e-5, rtol=2e-5):
                raise ValueError("stored behavior probabilities do not match the fresh recurrent rollout")
            next_value = self.critic(torch.as_tensor(self.rollout[-1]["next_features"], device=self.device)).cpu().numpy()
        targets, advantages = estimate_returns(
            np.stack([x["reward"] for x in self.rollout]), np.stack([x["value"] for x in self.rollout]), next_value,
            np.array([x["terminated"] for x in self.rollout]), np.array([x["truncated"] for x in self.rollout]), self.config)
        advantages_t = torch.as_tensor(advantages, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / advantages_t.std(unbiased=False).clamp_min(1e-8)
        targets_t = torch.as_tensor(targets, device=self.device)
        features_t = torch.as_tensor(np.stack([x["features"] for x in self.rollout]), device=self.device)
        actor_losses, critic_losses, entropy_values = [], [], []
        for _ in range(self.config.ppo_epochs):
            logs, entropy = self._sequence()
            ratio = torch.exp(logs - old_logs)
            surrogate = torch.minimum(ratio * advantages_t,
                                      ratio.clamp(1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantages_t)
            actor_loss = -surrogate.mean() - self.config.entropy_coef * entropy.mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
            self.actor_optimizer.step()
            critic_loss = self.config.value_coef * F.mse_loss(self.critic(features_t), targets_t)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
            self.critic_optimizer.step()
            self.gradient_steps += 1
            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            entropy_values.append(float(entropy.mean().detach().cpu()))
        count = len(self.rollout)
        self.rollout.clear()
        self.policy_version += 1
        self.completed_episodes += 1
        return {"actor_loss": float(np.mean(actor_losses)), "critic_loss": float(np.mean(critic_losses)),
                "entropy": float(np.mean(entropy_values)), "updates": float(self.config.ppo_epochs),
                "rollout_steps": float(count), "policy_version": float(self.policy_version)}

    def save(self, path: str | Path) -> None:
        if self.rollout:
            raise ValueError("PPO checkpoint requires an episode boundary with no pending rollout")
        atomic_save({"version": self.checkpoint_version, "profile": self.profile.to_dict(),
                     "config": self.config.to_dict(), "contract": self.contract,
                     "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                     "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
                     "action_rng": self.action_rng.get_state(), "policy_hidden": self.policy.hidden.detach().cpu(),
                     "policy_version": self.policy_version, "environment_steps": self.environment_steps,
                     "gradient_steps": self.gradient_steps, "completed_episodes": self.completed_episodes}, path)

    def load(self, path: str | Path) -> None:
        if self.rollout:
            raise ValueError("cannot load PPO over a pending rollout")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for key, expected in (("version", self.checkpoint_version), ("profile", self.profile.to_dict()),
                              ("config", self.config.to_dict()), ("contract", self.contract)):
            if checkpoint.get(key) != expected:
                raise ValueError(f"PPO checkpoint {key} does not match")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.action_rng.set_state(checkpoint["action_rng"].cpu())
        self.policy.hidden = checkpoint["policy_hidden"].to(self.device)
        for key in ("policy_version", "environment_steps", "gradient_steps", "completed_episodes"):
            setattr(self, key, int(checkpoint[key]))
