"""Discrete-phase MADDPG with simplex actors and joint centralized critics.

This is a local adaptation: phase execution takes argmax of the actor vector;
training differentiates through that vector. Epsilon exploration uses a legal
one-hot vector, recorded with the action rather than recomputed later.
"""
from __future__ import annotations

import copy
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..dqn import network_schema_sha256
from ..types import NetworkObservation, NetworkSpec, PolicyOutput
from .contracts import BehaviorInfo, JointTransition, TrainConfig
from .profiles import BaselineProfile
from .ppo import (atomic_save, check_observation, contract_hash, observation_contract,
                  reward_vector, validate_transition_actions)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class _MADDPGPolicy:
    def __init__(self, learner: "MADDPGLearner"):
        self.learner = learner
        self.name = learner.profile.baseline_id

    def reset(self, seed: int, network: NetworkSpec) -> None:
        if network_schema_sha256(network) != self.learner.contract["network"]:
            raise ValueError("policy reset received an incompatible network")
        if self.learner.episode_open:
            raise ValueError("finish the training episode before resetting MADDPG context")
        self.learner.action_rng = np.random.default_rng(seed)

    def act(self, observation: NetworkObservation, deterministic: bool) -> PolicyOutput:
        learner = self.learner
        features, mask = check_observation(learner.network, observation, learner.contract)
        with torch.no_grad():
            vectors = learner.actor_vectors(torch.as_tensor(features, device=learner.device),
                                             torch.as_tensor(mask, device=learner.device)).cpu().numpy().copy()
        if not deterministic:
            for index in range(learner.n_agents):
                if learner.action_rng.random() < learner.epsilon():
                    action = learner.action_rng.choice(np.flatnonzero(mask[index]))
                    vectors[index] = 0.0
                    vectors[index, action] = 1.0
        actions = vectors.argmax(axis=-1).astype(np.int64)
        return PolicyOutput(actions=actions, behavior=BehaviorInfo(
            action_vector=vectors, policy_version=learner.gradient_steps))

    def metadata(self) -> dict[str, Any]:
        return {"profile": self.learner.profile.to_dict(), "training_config": self.learner.config.to_dict(),
                "observation_contract_hash": contract_hash(self.learner.contract),
                "action_relaxation": "masked simplex actor, argmax execution; legal one-hot epsilon exploration"}


class MADDPGLearner:
    """Independent local actors and critics over synchronized whole-network replay."""
    checkpoint_version = "cityflow-baseline-discrete-maddpg-v1"

    def __init__(self, profile: BaselineProfile, network: NetworkSpec,
                 initial_observation: NetworkObservation, config: TrainConfig,
                 seed: int = 0, device: str = "cpu"):
        if profile.algorithm != "maddpg":
            raise ValueError("MADDPGLearner requires a MADDPG profile")
        self.profile, self.network, self.config = profile, network, config
        self.device = torch.device(device)
        self.contract = observation_contract(network, initial_observation)
        features, _ = check_observation(network, initial_observation, self.contract)
        self.n_agents, self.feature_dim = features.shape
        self.n_actions = network.max_actions
        self.action_rng = np.random.default_rng(seed)
        self.replay_rng = np.random.default_rng(seed + 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actors = nn.ModuleList([_MLP(self.feature_dim, config.hidden_dim, self.n_actions)
                                         for _ in range(self.n_agents)]).to(self.device)
            joint_width = self.n_agents * (self.feature_dim + self.n_actions)
            self.critics = nn.ModuleList([_MLP(joint_width, config.hidden_dim, 1)
                                          for _ in range(self.n_agents)]).to(self.device)
        self.target_actors = copy.deepcopy(self.actors)
        self.target_critics = copy.deepcopy(self.critics)
        self.target_actors.requires_grad_(False)
        self.target_critics.requires_grad_(False)
        self.actor_optimizers = [torch.optim.Adam(model.parameters(), lr=config.learning_rate) for model in self.actors]
        self.critic_optimizers = [torch.optim.Adam(model.parameters(), lr=config.learning_rate) for model in self.critics]
        self.replay: deque[dict[str, Any]] = deque(maxlen=config.replay_capacity)
        self.environment_steps = 0
        self.gradient_steps = 0
        self.completed_episodes = 0
        self.episode_open = False
        self.policy = _MADDPGPolicy(self)

    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def actor_vectors(self, features: torch.Tensor, mask: torch.Tensor, target: bool = False) -> torch.Tensor:
        actors = self.target_actors if target else self.actors
        return torch.stack([torch.softmax(actor(features[..., i, :]).masked_fill(~mask[..., i, :], -torch.inf), dim=-1)
                            for i, actor in enumerate(actors)], dim=-2)

    def observe(self, transition: JointTransition) -> None:
        features, mask = check_observation(self.network, transition.observation, self.contract)
        next_features, next_mask = check_observation(self.network, transition.next_observation, self.contract)
        actions = validate_transition_actions(transition, mask)
        if transition.behavior is None or transition.behavior.action_vector is None:
            raise ValueError("MADDPG requires the actor vector recorded when the action was chosen")
        vectors = np.asarray(transition.behavior.action_vector, dtype=np.float32)
        if (vectors.shape != mask.shape or not np.all(np.isfinite(vectors)) or np.any(vectors < 0)
                or not np.allclose(vectors.sum(-1), 1.0, atol=1e-5) or np.any(vectors[~mask] != 0)
                or not np.array_equal(vectors.argmax(-1), actions)):
            raise ValueError("MADDPG behavior vector must be a legal simplex matching the executed argmax action")
        bootstrap = not transition.terminated and (not transition.truncated or self.config.bootstrap_truncated)
        self.replay.append({"features": features, "next_features": next_features, "mask": mask,
                            "next_mask": next_mask, "actions": actions, "action_vector": vectors.copy(),
                            "reward": reward_vector(transition, self.profile, self.config, self.n_agents),
                            "bootstrap": float(bootstrap)})
        self.environment_steps += 1
        self.episode_open = not (transition.terminated or transition.truncated)

    def _learn_once(self) -> dict[str, float]:
        indices = self.replay_rng.choice(len(self.replay), self.config.batch_size, replace=False)
        rows = [self.replay[int(i)] for i in indices]
        def tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(np.stack([item[name] for item in rows]), device=self.device)
        features, next_features = tensor("features"), tensor("next_features")
        masks, next_masks = tensor("mask"), tensor("next_mask")
        vectors, rewards = tensor("action_vector"), tensor("reward")
        bootstrap = tensor("bootstrap").float()
        states = features.flatten(1)
        critic_input = torch.cat((states, vectors.flatten(1)), dim=-1)
        with torch.no_grad():
            next_vectors = self.actor_vectors(next_features, next_masks, target=True)
            target_input = torch.cat((next_features.flatten(1), next_vectors.flatten(1)), dim=-1)
            targets = [rewards[:, i] + self.config.gamma * bootstrap * critic(target_input).squeeze(-1)
                       for i, critic in enumerate(self.target_critics)]
        critic_losses, actor_losses = [], []
        for index, (critic, optimizer) in enumerate(zip(self.critics, self.critic_optimizers)):
            loss = F.mse_loss(critic(critic_input).squeeze(-1), targets[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), self.config.gradient_clip)
            optimizer.step()
            critic_losses.append(float(loss.detach().cpu()))
        # Other agents use their recorded behavior vectors; only the selected actor receives policy gradients.
        for index, (actor, optimizer, critic) in enumerate(zip(self.actors, self.actor_optimizers, self.critics)):
            predicted = torch.softmax(actor(features[:, index]).masked_fill(~masks[:, index], -torch.inf), dim=-1)
            joint_vectors = torch.stack([predicted if i == index else vectors[:, i].detach()
                                         for i in range(self.n_agents)], dim=1)
            critic.requires_grad_(False)
            try:
                loss = -critic(torch.cat((states, joint_vectors.flatten(1)), dim=-1)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), self.config.gradient_clip)
                optimizer.step()
                actor_losses.append(float(loss.detach().cpu()))
            finally:
                critic.requires_grad_(True)
        with torch.no_grad():
            for source, target in ((self.actors, self.target_actors), (self.critics, self.target_critics)):
                for parameter, target_parameter in zip(source.parameters(), target.parameters()):
                    target_parameter.lerp_(parameter, self.config.tau)
        self.gradient_steps += 1
        return {"actor_loss": float(np.mean(actor_losses)), "critic_loss": float(np.mean(critic_losses)),
                "updates": 1.0, "replay_size": float(len(self.replay)), "epsilon": self.epsilon()}

    def update_if_due(self, event: str = "step") -> dict[str, float]:
        if event not in {"step", "round", "rollout"}:
            raise ValueError("unknown training event")
        if event != self.profile.update_schedule or len(self.replay) < max(self.config.batch_size, self.config.warmup_transitions):
            return {}
        count = self.config.updates_per_round if event == "round" else 1
        updates = [self._learn_once() for _ in range(count)]
        result = {key: float(np.mean([item[key] for item in updates])) for key in updates[0]}
        result["updates"] = float(count)
        return result

    def end_episode(self) -> dict[str, float]:
        if self.episode_open:
            raise ValueError("MADDPG end_episode requires a terminal or truncated transition")
        self.completed_episodes += 1
        return self.update_if_due("round")

    def save(self, path: str | Path) -> None:
        if self.episode_open:
            raise ValueError("MADDPG checkpoint requires an episode boundary")
        atomic_save({"version": self.checkpoint_version, "profile": self.profile.to_dict(),
                     "config": self.config.to_dict(), "contract": self.contract,
                     "actors": self.actors.state_dict(), "critics": self.critics.state_dict(),
                     "target_actors": self.target_actors.state_dict(), "target_critics": self.target_critics.state_dict(),
                     "actor_optimizers": [optimizer.state_dict() for optimizer in self.actor_optimizers],
                     "critic_optimizers": [optimizer.state_dict() for optimizer in self.critic_optimizers],
                     "replay": list(self.replay), "action_rng": self.action_rng.bit_generator.state,
                     "replay_rng": self.replay_rng.bit_generator.state,
                     "environment_steps": self.environment_steps, "gradient_steps": self.gradient_steps,
                     "completed_episodes": self.completed_episodes}, path)

    def load(self, path: str | Path) -> None:
        if self.episode_open:
            raise ValueError("cannot load MADDPG during a training episode")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for key, expected in (("version", self.checkpoint_version), ("profile", self.profile.to_dict()),
                              ("config", self.config.to_dict()), ("contract", self.contract)):
            if checkpoint.get(key) != expected:
                raise ValueError(f"MADDPG checkpoint {key} does not match")
        for name in ("actors", "critics", "target_actors", "target_critics"):
            getattr(self, name).load_state_dict(checkpoint[name])
        for name in ("actor_optimizers", "critic_optimizers"):
            for optimizer, state in zip(getattr(self, name), checkpoint[name]):
                optimizer.load_state_dict(state)
        self.replay = deque(checkpoint["replay"], maxlen=self.config.replay_capacity)
        self.action_rng.bit_generator.state = checkpoint["action_rng"]
        self.replay_rng.bit_generator.state = checkpoint["replay_rng"]
        for key in ("environment_steps", "gradient_steps", "completed_episodes"):
            setattr(self, key, int(checkpoint[key]))
        self.episode_open = False
