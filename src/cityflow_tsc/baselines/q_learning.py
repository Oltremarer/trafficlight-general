"""Joint replay and actual TD learning for the explicit local PyTorch profiles."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..dqn import network_schema_sha256
from ..types import NetworkObservation, NetworkSpec, PolicyOutput
from .contracts import BehaviorInfo, JointTransition, TrainConfig, require_view
from .codecs import PHASE_LANE_INDICES, SOURCE_PHASES
from .profiles import BaselineProfile, get_profile
from .q_networks import build_q_network


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    h = hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def observation_schema(observation: NetworkObservation) -> dict[str, Any]:
    view = require_view(observation)
    return {
        "schema_id": view.schema_id,
        "node_ids": list(view.node_ids),
        "feature_names": list(view.feature_names),
        "shapes": {name: list(np.asarray(getattr(view, name)).shape) for name in
                   ("features", "lane_features", "lane_mask", "phase_lane_mask", "phase_encoding",
                    "neighbor_index", "neighbor_mask")},
        "static": {name: _array_hash(np.asarray(getattr(view, name))) for name in
                   ("lane_mask", "phase_lane_mask", "neighbor_index", "neighbor_mask")},
        "action_mask": _array_hash(np.asarray(observation.action_mask)),
        "source_action_to_local": (None if view.source_action_to_local is None
                                   else _array_hash(np.asarray(view.source_action_to_local))),
    }


def _state_arrays(observation: NetworkObservation) -> dict[str, np.ndarray]:
    view = require_view(observation)
    result = {name: np.asarray(getattr(view, name)).copy() for name in
              ("features", "lane_features", "lane_mask", "phase_lane_mask", "phase_encoding",
               "neighbor_index", "neighbor_mask")}
    result["current_phase"] = np.asarray(observation.current_phase, dtype=np.int64).copy()
    result["action_mask"] = np.asarray(observation.action_mask, dtype=np.bool_).copy()
    if view.source_action_to_local is not None:
        result["source_action_to_local"] = np.asarray(view.source_action_to_local, dtype=np.int64).copy()
    return result


def _tensor_batch(states: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key in states[0]:
        dtype = torch.long if key in {"current_phase", "neighbor_index", "source_action_to_local"} else (
            torch.bool if key in {"lane_mask", "phase_lane_mask", "neighbor_mask", "action_mask"}
            else torch.float32)
        result[key] = torch.as_tensor(np.stack([item[key] for item in states]), dtype=dtype, device=device)
    return result


class QPolicy:
    def __init__(self, learner: "QLearner", seed: int) -> None:
        self.learner = learner
        self.name = learner.profile.baseline_id
        self.policy_version = 0
        self.rng = np.random.default_rng(seed)
        self.checkpoint_sha256: str | None = None

    def reset(self, seed: int, network: NetworkSpec) -> None:
        if network_schema_sha256(network) != self.learner.network_hash:
            raise ValueError("policy reset received a different network schema")
        self.rng = np.random.default_rng(seed)

    def act(self, observation: NetworkObservation, deterministic: bool) -> PolicyOutput:
        self.learner._validate_observation(observation)
        batch = _tensor_batch([_state_arrays(observation)], self.learner.device)
        self.learner.online.eval()
        with torch.no_grad():
            q = self.learner.online(batch)[0]
            q = q.masked_fill(~batch["action_mask"][0], -torch.inf)
            actions = q.argmax(dim=-1).cpu().numpy().astype(np.int64)
        epsilon = 0.0 if deterministic else self.learner.epsilon
        if epsilon:
            explore = self.rng.random(len(actions)) < epsilon
            for i in np.flatnonzero(explore):
                actions[i] = self.rng.choice(np.flatnonzero(observation.action_mask[i]))
        return PolicyOutput(actions=actions,
                            diagnostics={"epsilon": epsilon, "implementation": self.learner.profile.implementation},
                            behavior=BehaviorInfo(policy_version=self.policy_version))

    def metadata(self) -> dict[str, Any]:
        return {"profile": self.learner.profile.to_dict(),
                "policy_version": self.policy_version,
                "checkpoint_sha256": self.checkpoint_sha256,
                "observation_schema_sha256": self.learner.schema_hash,
                "network_schema_sha256": self.learner.network_hash}


class QLearner:
    """One replay item is one network decision, including ALL graph nodes.

    Source state/reward profiles are retained. Training is deliberately a bounded
    mini-batch PyTorch port, not the upstream Keras epoch/early-stopping protocol.
    """
    CHECKPOINT_VERSION = 2

    def __init__(self, profile: str | BaselineProfile, network: NetworkSpec,
                 initial_observation: NetworkObservation, config: TrainConfig,
                 seed: int = 0, device: str = "cpu") -> None:
        self.profile = get_profile(profile)
        self.network = network
        self.config = config
        self.device = torch.device(device)
        self.network_hash = network_schema_sha256(network)
        self.schema = observation_schema(initial_observation)
        self.schema_hash = hashlib.sha256(json.dumps(self.schema, sort_keys=True).encode()).hexdigest()
        self._validate_observation(initial_observation)
        view = require_view(initial_observation)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = build_q_network(self.profile, view.features.shape[-1],
                                          view.lane_features.shape[-1], config.hidden_dim,
                                          network.max_actions, network.num_intersections).to(self.device)
        self.target = copy.deepcopy(self.online).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.rng = np.random.default_rng(seed)
        self.replay: list[dict[str, Any]] = []
        self.replay_cursor = 0
        self.environment_steps = 0
        self.gradient_steps = 0
        self.completed_episodes = 0
        self._last_step_processed = 0
        self._last_round_processed = 0
        self._episode_open = False
        self.policy = QPolicy(self, seed)

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def _validate_observation(self, observation: NetworkObservation) -> None:
        view = require_view(observation)
        if observation_schema(observation) != self.schema:
            raise ValueError("observation feature, topology, phase or graph schema changed")
        n, a = self.network.num_intersections, self.network.max_actions
        if tuple(view.node_ids) != self.network.intersection_ids:
            raise ValueError("baseline node order differs from environment node order")
        if view.features.ndim != 2 or view.features.shape[0] != n:
            raise ValueError("baseline features must have shape [N,F]")
        if view.lane_features.ndim != 3 or view.lane_features.shape[0] != n:
            raise ValueError("lane features must have shape [N,L,C]")
        lanes = view.lane_features.shape[1]
        if view.lane_mask.shape != (n, lanes) or view.phase_lane_mask.shape != (n, a, lanes):
            raise ValueError("lane and phase-lane mask dimensions do not match")
        if self.profile.layout == "canonical12" and (lanes != 12 or view.phase_encoding.shape != (n, 8)):
            raise ValueError("canonical12 requires twelve lanes and phase8 encoding")
        if view.phase_encoding.ndim != 2 or view.phase_encoding.shape[0] != n:
            raise ValueError("phase encoding must have shape [N,P]")
        if np.any((view.phase_encoding != 0) & (view.phase_encoding != 1)):
            raise ValueError("phase encoding must be binary")
        if observation.action_mask.shape != (n, a) or not observation.action_mask.any(axis=-1).all():
            raise ValueError("each node requires a valid action")
        if self.profile.layout == "canonical12":
            mapping = view.source_action_to_local
            if mapping is None or mapping.shape != (n, len(SOURCE_PHASES)) or mapping.dtype.kind not in "iu":
                raise ValueError("canonical12 requires integer source_action_to_local [N,8]")
            for i in range(n):
                available = mapping[i] >= 0
                local = mapping[i, available]
                if (np.any(mapping[i] < -1)
                        or not np.array_equal(np.sort(local), np.flatnonzero(observation.action_mask[i]))):
                    raise ValueError("source_action_to_local must map each legal local action exactly once")
                physical = view.phase_lane_mask[i, local][:, PHASE_LANE_INDICES]
                if not np.array_equal(physical, SOURCE_PHASES[available]):
                    raise ValueError("source_action_to_local disagrees with physical phase meanings")
        if self.profile.layout == "generic" and self.profile.algorithm in {"frap", "mplight"}:
            pairs = view.phase_lane_mask[observation.action_mask]
            if np.any(pairs.sum(axis=-1) != 2):
                raise ValueError("LibSignal phase_pairs require exactly two demand lanes per action")
        current = np.asarray(observation.current_phase)
        if current.shape != (n,) or not np.issubdtype(current.dtype, np.integer):
            raise ValueError("current phase must be one integer per node")
        if np.any(current < 0) or np.any(current >= a) or not observation.action_mask[np.arange(n), current].all():
            raise ValueError("current phase is not a valid local action")
        if view.neighbor_index.ndim != 2 or view.neighbor_index.shape[0] != n or view.neighbor_mask.shape != view.neighbor_index.shape:
            raise ValueError("neighbor index/mask dimensions do not match")
        neighbors = view.neighbor_index[view.neighbor_mask]
        if np.any(neighbors < 0) or np.any(neighbors >= n):
            raise ValueError("valid neighbor index is outside the node range")
        for array in (view.features, view.lane_features, view.phase_encoding):
            if not np.isfinite(array).all():
                raise ValueError("baseline observations must be finite")

    def observe(self, transition: JointTransition) -> None:
        self._validate_observation(transition.observation)
        self._validate_observation(transition.next_observation)
        actions = np.asarray(transition.actions)
        n, a = self.network.num_intersections, self.network.max_actions
        if actions.shape != (n,) or not np.issubdtype(actions.dtype, np.integer) or np.issubdtype(actions.dtype, np.bool_):
            raise ValueError("actions must contain one integer per node")
        if np.any(actions < 0) or np.any(actions >= a) or not transition.observation.action_mask[np.arange(n), actions].all():
            raise ValueError("transition contains an invalid action")
        reward = np.asarray(transition.reward, dtype=np.float32)
        if reward.shape != (n,) or not np.isfinite(reward).all():
            raise ValueError("rewards must be a finite [N] vector")
        if not np.isfinite(transition.elapsed_s) or transition.elapsed_s <= 0:
            raise ValueError("transition elapsed time must be positive")
        record = {"state": _state_arrays(transition.observation),
                  "next_state": _state_arrays(transition.next_observation),
                  "actions": actions.astype(np.int64, copy=True), "reward": reward.copy(),
                  "terminated": bool(transition.terminated), "truncated": bool(transition.truncated),
                  "episode_id": transition.episode_id, "scenario_id": transition.scenario_id,
                  "elapsed_s": float(transition.elapsed_s)}
        if len(self.replay) < self.config.replay_capacity:
            self.replay.append(record)
        else:
            self.replay[self.replay_cursor] = record
        self.replay_cursor = (self.replay_cursor + 1) % self.config.replay_capacity
        self.environment_steps += 1
        self._episode_open = True

    def _learn_once(self) -> dict[str, float]:
        indices = self.rng.choice(len(self.replay), self.config.batch_size, replace=False)
        items = [self.replay[int(i)] for i in indices]
        states = _tensor_batch([item["state"] for item in items], self.device)
        next_states = _tensor_batch([item["next_state"] for item in items], self.device)
        actions = torch.as_tensor(np.stack([item["actions"] for item in items]), dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(np.stack([item["reward"] for item in items]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item["terminated"] or (item["truncated"] and not self.config.bootstrap_truncated)
                                 for item in items], dtype=torch.float32, device=self.device)
        self.online.train()
        predicted = self.online(states).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target(next_states).masked_fill(~next_states["action_mask"], -torch.inf)
            # The environment applies reward_factor; training scale appears once.
            targets = rewards * self.config.reward_scale + self.config.gamma * (1 - dones[:, None]) * next_q.max(dim=-1).values
        loss = F.mse_loss(predicted, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Q-learning loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite Q-learning gradient")
        self.optimizer.step()
        self.gradient_steps += 1
        self.policy.policy_version = self.gradient_steps
        if self.gradient_steps % self.config.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.detach().cpu()), "mean_q": float(predicted.detach().mean().cpu()),
                "mean_target": float(targets.mean().cpu()), "gradient_norm": float(grad_norm.cpu())}

    def update_if_due(self, event: str = "step") -> dict[str, Any]:
        if event not in {"step", "round", "rollout"}:
            raise ValueError("unknown learner update event")
        if event != self.profile.update_schedule:
            return {}
        if event == "step":
            if self.environment_steps <= self._last_step_processed:
                return {}
            self._last_step_processed = self.environment_steps
        else:
            if self.completed_episodes <= self._last_round_processed:
                return {}
            self._last_round_processed = self.completed_episodes
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_transitions):
            return {}
        updates = 1 if event == "step" else self.config.updates_per_round
        metrics = [self._learn_once() for _ in range(updates)]
        return {**{key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
                "updates": updates, "gradient_steps": self.gradient_steps,
                "replay_network_transitions": len(self.replay)}

    def end_episode(self) -> dict[str, Any]:
        if not self._episode_open:
            return {}
        self._episode_open = False
        self.completed_episodes += 1
        return self.update_if_due("round")

    def save(self, path: str | Path) -> None:
        if self._episode_open:
            raise ValueError("save Q learner after end_episode for a boundary-resumable checkpoint")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"checkpoint_version": self.CHECKPOINT_VERSION,
                      "profile": self.profile.to_dict(), "config": self.config.to_dict(),
                      "network_hash": self.network_hash, "schema": self.schema,
                      "online": self.online.state_dict(), "target": self.target.state_dict(),
                      "optimizer": self.optimizer.state_dict(), "replay": self.replay,
                      "replay_cursor": self.replay_cursor, "rng": self.rng.bit_generator.state,
                      "policy_rng": self.policy.rng.bit_generator.state,
                      "environment_steps": self.environment_steps, "gradient_steps": self.gradient_steps,
                      "completed_episodes": self.completed_episodes,
                      "last_step_processed": self._last_step_processed,
                      "last_round_processed": self._last_round_processed,
                      "policy_version": self.policy.policy_version}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        self.policy.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for key, expected in (("checkpoint_version", self.CHECKPOINT_VERSION),
                              ("profile", self.profile.to_dict()), ("config", self.config.to_dict()),
                              ("network_hash", self.network_hash), ("schema", self.schema)):
            if checkpoint.get(key) != expected:
                raise ValueError(f"checkpoint {key} mismatch")
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.replay = checkpoint["replay"]
        self.replay_cursor = checkpoint["replay_cursor"]
        self.rng.bit_generator.state = checkpoint["rng"]
        self.policy.rng.bit_generator.state = checkpoint["policy_rng"]
        self.environment_steps = checkpoint["environment_steps"]
        self.gradient_steps = checkpoint["gradient_steps"]
        self.completed_episodes = checkpoint["completed_episodes"]
        self._last_step_processed = checkpoint["last_step_processed"]
        self._last_round_processed = checkpoint["last_round_processed"]
        self.policy.policy_version = checkpoint["policy_version"]
        self._episode_open = False
        self.policy.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
