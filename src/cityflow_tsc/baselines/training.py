from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from ..types import EpisodeResult
from .contracts import JointTransition
from .trajectory import BaselineTrajectoryWriter, write_json


class TrainingRunner:
    """One environment loop for Q, graph Q, recurrent PPO and joint actor-critic."""

    def run_episode(self, env, learner, scenario, *, initial=None) -> EpisodeResult:
        policy = learner.policy
        start_updates = learner.gradient_steps
        losses: list[dict[str, Any]] = []
        steps = 0
        try:
            policy.reset(scenario.seed, env.network)
            observation, _ = env.reset(scenario) if initial is None else initial
            writer = BaselineTrajectoryWriter(
                scenario, env.control, env.network, policy.name, env.reward_calculator.name,
                policy_metadata={
                    "mode": "training", "profile": learner.profile.to_dict(),
                    "train_config": learner.config.to_dict(),
                },
            )
            while True:
                output = policy.act(observation, deterministic=False)
                following, rewards, terminated, truncated, info = env.step(output.actions)
                transition = JointTransition(
                    observation, output, np.asarray(rewards, dtype=np.float32).copy(), following,
                    terminated, truncated, following.time_s - observation.time_s,
                    episode_id=scenario.output_dir.name, scenario_id=str(scenario.flow_path), info=info,
                )
                writer.append(observation, output.actions, rewards, following, terminated, truncated,
                              output=output, info=info)
                callback = getattr(policy, "observe_context", None)
                if callable(callback):
                    callback(transition)
                learner.observe(transition)
                statistics = learner.update_if_due(event="step")
                if statistics:
                    losses.append(statistics)
                observation = following
                steps += 1
                if terminated or truncated:
                    break
            statistics = learner.end_episode()
            if statistics:
                losses.append(statistics)
            metrics = env.metrics_summary()
            for key in sorted({key for record in losses for key in record}):
                if key == "updates":
                    continue
                numbers = [float(record[key]) for record in losses
                           if key in record and isinstance(record[key], (int, float, np.number))]
                if numbers:
                    metrics[f"training_{key}"] = float(np.mean(numbers))
            metrics["training_updates"] = float(learner.gradient_steps - start_updates)
            paths = writer.write(scenario.output_dir)
            result = EpisodeResult(metrics, steps, paths["trajectory_path"], paths["manifest_path"])
            write_json(scenario.output_dir / "training.metrics.json", {
                "steps": steps, "metrics": metrics, "profile": learner.profile.to_dict(),
            })
            return result
        finally:
            env.close()
