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
        *,
        writer_factory=None,
        initial=None,
    ) -> EpisodeResult:
        steps = 0
        try:
            policy.reset(scenario.seed, env.network)
            observation, _ = env.reset(scenario) if initial is None else initial
            writer_class = writer_factory or TrajectoryWriter
            writer = writer_class(
                scenario=scenario,
                control=env.control,
                network=env.network,
                policy_name=policy.name,
                reward_name=env.reward_calculator.name,
                policy_metadata=policy_metadata,
            )
            while True:
                output = policy.act(observation, deterministic=deterministic)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(output.actions)
                extra = {}
                if writer_factory is not None:
                    extra = {"output": output, "info": info}
                writer.append(
                    observation=observation,
                    actions=output.actions,
                    rewards=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                    **extra,
                )
                callback = getattr(policy, "observe_context", None)
                if callable(callback):
                    from .baselines.contracts import JointTransition
                    callback(JointTransition(
                        observation, output, reward, next_observation, terminated, truncated,
                        next_observation.time_s - observation.time_s,
                        episode_id=scenario.output_dir.name, scenario_id=str(scenario.flow_path),
                        info=info,
                    ))
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
