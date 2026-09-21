from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import ScenarioConfig
from .dqn import DQNConfig, QNetwork, ReplayBuffer, SharedDQNPolicy
from .environment import TrafficEnv
from .trajectory import TrajectoryWriter
from .types import EpisodeResult


class DQNTrainer:
    """Online shared-parameter DQN training over whole-network environment steps."""

    def __init__(self, policy: SharedDQNPolicy, config: DQNConfig, seed: int) -> None:
        self.policy = policy
        self.config = config
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.target = QNetwork(
            policy.vectorizer.output_dim,
            config.hidden_dim,
            policy.network.max_actions,
        ).to(policy.device)
        self.target.load_state_dict(policy.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            policy.online.parameters(), lr=config.learning_rate
        )
        self.replay = ReplayBuffer(
            config.replay_capacity,
            policy.vectorizer.output_dim,
            policy.network.max_actions,
        )
        self.environment_steps = 0
        self.gradient_steps = 0
        self.completed_episodes = 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _learn_once(self) -> float:
        batch = self.replay.sample(self.config.batch_size, self.rng)
        device = self.policy.device
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(
            batch["next_states"], dtype=torch.float32, device=device
        )
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)
        next_mask = torch.as_tensor(
            batch["next_action_masks"], dtype=torch.bool, device=device
        )

        q_values = self.policy.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).masked_fill(~next_mask, -torch.inf)
            if torch.any(torch.all(~next_mask, dim=1)):
                raise ValueError("replay contains a next state with no valid action")
            target = (
                rewards * self.config.reward_scale
                + self.config.gamma * (1.0 - dones) * next_q.max(dim=1).values
            )
        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.online.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_steps == 0:
            self.target.load_state_dict(self.policy.online.state_dict())
        return float(loss.detach().cpu())

    def run_episode(self, env: TrafficEnv, scenario: ScenarioConfig) -> EpisodeResult:
        self.policy.reset(scenario.seed, env.network)
        observation, _ = env.reset(scenario)
        writer = TrajectoryWriter(
            scenario=scenario,
            control=env.control,
            network=env.network,
            policy_name=self.policy.name,
            reward_name=env.reward_calculator.name,
            policy_metadata={
                "mode": "online_training",
                "starting_environment_steps": self.environment_steps,
                "starting_gradient_steps": self.gradient_steps,
                "starting_checkpoint_sha256": self.policy.checkpoint_sha256,
                "observation_schema_id": self.policy.observation_schema_id,
                "roadnet_sha256": self.policy.roadnet_sha256,
                "network_schema_sha256": self.policy.network_schema_sha256,
            },
        )
        losses = []
        steps = 0
        try:
            while True:
                output = self.policy.select_actions(observation, self.epsilon())
                next_observation, rewards, terminated, truncated, _ = env.step(
                    output.actions
                )
                writer.append(
                    observation,
                    output.actions,
                    rewards,
                    next_observation,
                    terminated,
                    truncated,
                )
                states = self.policy.vectorizer.transform(observation)
                next_states = self.policy.vectorizer.transform(next_observation)
                done_batch = np.full(
                    env.network.num_intersections,
                    terminated or truncated,
                    dtype=np.bool_,
                )
                self.replay.append_batch(
                    states=states,
                    actions=output.actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=done_batch,
                    next_action_masks=next_observation.action_mask,
                )
                self.environment_steps += 1
                if (
                    self.replay.size >= max(
                        self.config.batch_size, self.config.warmup_transitions
                    )
                    and self.environment_steps % self.config.train_every_steps == 0
                ):
                    losses.append(self._learn_once())
                observation = next_observation
                steps += 1
                if terminated or truncated:
                    break

            metrics = env.metrics_summary()
            metrics.update(
                {
                    "training_mean_loss": float(np.mean(losses)) if losses else 0.0,
                    "training_updates": float(len(losses)),
                    "replay_size": float(self.replay.size),
                    "epsilon_final": float(self.epsilon()),
                }
            )
            paths = writer.write(scenario.output_dir)
            result = EpisodeResult(
                metrics=metrics,
                steps=steps,
                trajectory_path=paths["trajectory_path"],
                manifest_path=paths["manifest_path"],
            )
            result_file = scenario.output_dir / "training.metrics.json"
            temporary = scenario.output_dir / ".training.metrics.json.tmp"
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"steps": steps, "metrics": metrics},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            os.replace(temporary, result_file)
            self.completed_episodes += 1
            return result
        finally:
            env.close()

    def save_checkpoint(self, path: Path) -> None:
        self.policy.save(
            path,
            extra={
                "environment_steps": self.environment_steps,
                "gradient_steps": self.gradient_steps,
                "completed_episodes": self.completed_episodes,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "replay_state_dict": self.replay.state_dict(),
                "numpy_rng_state": copy.deepcopy(self.rng.bit_generator.state),
                "torch_rng_state": torch.get_rng_state(),
            },
        )

    def load_checkpoint(self, path: Path) -> None:
        extra = self.policy.load(path)
        self.environment_steps = int(extra.get("environment_steps", 0))
        self.gradient_steps = int(extra.get("gradient_steps", 0))
        self.completed_episodes = int(extra.get("completed_episodes", 0))
        if "optimizer_state_dict" in extra:
            self.optimizer.load_state_dict(extra["optimizer_state_dict"])
        if "target_state_dict" in extra:
            self.target.load_state_dict(extra["target_state_dict"])
        else:
            self.target.load_state_dict(self.policy.online.state_dict())
        if "replay_state_dict" in extra:
            self.replay.load_state_dict(extra["replay_state_dict"])
        if "numpy_rng_state" in extra:
            self.rng.bit_generator.state = extra["numpy_rng_state"]
        if "torch_rng_state" in extra:
            torch.set_rng_state(extra["torch_rng_state"].cpu())
