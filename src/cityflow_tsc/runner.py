from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import ScenarioConfig
from .environment import TrafficEnv
from .trajectory import TrajectoryWriter
from .types import EpisodeResult, Policy


class EpisodeRunner:
    """Execute one complete policy/environment episode and retain its evidence."""

    def run(
        self,
        env: TrafficEnv,
        policy: Policy,
        scenario: ScenarioConfig,
        deterministic: bool = True,
        policy_metadata: Optional[Mapping[str, Any]] = None,
    ) -> EpisodeResult:
        policy.reset(scenario.seed, env.network)
        observation, _ = env.reset(scenario)
        writer = TrajectoryWriter(
            scenario=scenario,
            control=env.control,
            network=env.network,
            policy_name=policy.name,
            reward_name=env.reward_calculator.name,
            policy_metadata=policy_metadata,
        )
        steps = 0
        try:
            while True:
                output = policy.act(observation, deterministic=deterministic)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(output.actions)
                writer.append(
                    observation=observation,
                    actions=output.actions,
                    rewards=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                )
                observation = next_observation
                steps += 1
                if terminated or truncated:
                    break

            metrics = env.metrics_summary()
            paths = writer.write(scenario.output_dir)
            result = EpisodeResult(
                metrics=metrics,
                steps=steps,
                trajectory_path=paths["trajectory_path"],
                manifest_path=paths["manifest_path"],
            )
            result_path = scenario.output_dir / "metrics.json"
            temporary_path = scenario.output_dir / ".metrics.json.tmp"
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "policy": policy.name,
                        "steps": steps,
                        "metrics": metrics,
                        "trajectory_path": result.trajectory_path,
                        "manifest_path": result.manifest_path,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            os.replace(temporary_path, result_path)
            return result
        finally:
            env.close()
