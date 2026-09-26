from __future__ import annotations

from numbers import Integral
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import ControlConfig, ScenarioConfig
from .metrics import MetricCollector
from .simulator import SimulatorBackendProtocol
from .types import (
    NetworkObservation,
    NetworkSnapshot,
    NetworkSpec,
    ObservationBuilder,
    RewardCalculator,
)


class TrafficEnv:
    """Algorithm-independent traffic-control environment over a simulator backend."""

    def __init__(
        self,
        network: NetworkSpec,
        control: ControlConfig,
        backend: SimulatorBackendProtocol,
        observation_builder: ObservationBuilder,
        reward_calculator: RewardCalculator,
        metrics: Optional[MetricCollector] = None,
        *,
        reward_accumulator: Optional[Any] = None,
    ) -> None:
        self.network = network
        self.control = control
        self.backend = backend
        self.observation_builder = observation_builder
        self.reward_calculator = reward_calculator
        self.metrics = metrics or MetricCollector()
        self.reward_accumulator = reward_accumulator
        self.scenario: Optional[ScenarioConfig] = None
        self.current_phase = np.zeros(network.num_intersections, dtype=np.int64)
        self.signal_stage = np.zeros(network.num_intersections, dtype=np.int8)
        self.phase_elapsed_s = np.zeros(network.num_intersections, dtype=np.float32)
        self._last_snapshot: Optional[NetworkSnapshot] = None

    def reset(
        self, scenario: ScenarioConfig
    ) -> Tuple[NetworkObservation, Dict[str, Any]]:
        duration_ticks = scenario.duration_s / self.control.simulator_step_s
        if abs(duration_ticks - round(duration_ticks)) > 1e-9:
            raise ValueError(
                "scenario duration_s must be divisible by simulator_step_s; "
                "CityFlow cannot execute a partial simulator tick"
            )
        self.scenario = scenario
        configure_builder = getattr(self.observation_builder, "configure", None)
        if callable(configure_builder):
            configure_builder(scenario)
        if self.reward_accumulator is not None:
            self.reward_accumulator.reset()
        snapshot = self.backend.reset(scenario)
        reset_builder = getattr(self.observation_builder, "reset", None)
        if callable(reset_builder):
            reset_builder()
        action_mask = self.network.action_mask()
        for index, valid_mask in enumerate(action_mask):
            valid = np.flatnonzero(valid_mask)
            if not len(valid):
                raise ValueError(f"intersection {index} has no valid phase")
            self.current_phase[index] = int(valid[0])
            self.backend.set_phase(
                self.network.intersections[index].intersection_id,
                self.network.engine_phase(index, int(valid[0])),
            )
        self.phase_elapsed_s.fill(0.0)
        self.signal_stage.fill(0)
        self.metrics.reset()
        configure_metrics = getattr(self.metrics, "configure", None)
        if callable(configure_metrics):
            configure_metrics(scenario, self.control.simulator_step_s, snapshot)
        self.metrics.observe(snapshot, elapsed_s=0.0)
        self._last_snapshot = snapshot
        observation = self.observation_builder.build(
            snapshot, self.current_phase, self.signal_stage, self.phase_elapsed_s
        )
        return observation, {
            "simulation_time_s": snapshot.time_s,
            "reward_name": self.reward_calculator.name,
        }

    def _validate_actions(self, actions: np.ndarray) -> np.ndarray:
        raw_values = np.asarray(actions, dtype=object)
        if raw_values.shape != (self.network.num_intersections,):
            raise ValueError(
                f"actions must have shape ({self.network.num_intersections},), "
                f"received {raw_values.shape}"
            )
        mask = self.network.action_mask()
        values = np.empty(self.network.num_intersections, dtype=np.int64)
        for index, raw_action in enumerate(raw_values):
            if isinstance(raw_action, (bool, np.bool_)) or not isinstance(
                raw_action, Integral
            ):
                raise ValueError(
                    "actions must contain discrete integer values without "
                    "implicit casting"
                )
            action = int(raw_action)
            if action < 0 or action >= self.network.max_actions or not mask[index, action]:
                raise ValueError(
                    f"invalid action {action} for "
                    f"{self.network.intersections[index].intersection_id}"
                )
            values[index] = action
        return values

    def _remaining_ticks(self) -> int:
        if self.scenario is None:
            raise RuntimeError("environment has not been reset")
        remaining_s = max(
            0.0, self.scenario.duration_s - float(self.backend.current_time())
        )
        interval_s = min(float(self.control.decision_interval_s), remaining_s)
        ticks_float = interval_s / self.control.simulator_step_s
        ticks = int(round(ticks_float))
        if abs(ticks_float - ticks) > 1e-9:
            raise RuntimeError(
                "remaining episode time is not aligned to simulator_step_s"
            )
        if remaining_s > 1e-9 and ticks == 0:
            raise RuntimeError(
                "environment cannot advance a positive remaining duration"
            )
        return ticks

    def _advance(self, ticks: int) -> Tuple[int, NetworkSnapshot]:
        if ticks < 0:
            raise ValueError("ticks cannot be negative")
        snapshot = self.backend.snapshot()
        advanced = 0
        for _ in range(ticks):
            if self.scenario is not None and self.backend.current_time() >= self.scenario.duration_s:
                break
            previous_time = float(self.backend.current_time())
            before_snapshot = snapshot
            self.backend.next_step()
            current_time = float(self.backend.current_time())
            if current_time <= previous_time:
                raise RuntimeError(
                    "simulator backend did not advance time after next_step()"
                )
            snapshot = self.backend.snapshot()
            if self.reward_accumulator is not None:
                self.reward_accumulator.observe_tick(
                    before_snapshot, snapshot, current_time - previous_time
                )
            self.metrics.observe(snapshot, elapsed_s=self.control.simulator_step_s)
            advanced += 1
        return advanced, snapshot

    def _set_stage_phase(self, changed: np.ndarray, engine_phase_id: int) -> None:
        for index in np.flatnonzero(changed):
            self.backend.set_phase(
                self.network.intersections[int(index)].intersection_id,
                int(engine_phase_id),
            )

    def step(
        self, actions: np.ndarray
    ) -> Tuple[NetworkObservation, np.ndarray, bool, bool, Dict[str, Any]]:
        if self.scenario is None:
            raise RuntimeError("environment has not been reset")
        action_values = self._validate_actions(actions)
        if self.reward_accumulator is not None:
            self.reward_accumulator.begin()
        total_ticks = self._remaining_ticks()
        changed = action_values != self.current_phase
        ticks_used = 0
        yellow_ticks = 0
        all_red_ticks = 0
        green_ticks = 0
        target_applied = False
        snapshot = self.backend.snapshot()

        if np.any(changed) and total_ticks > 0:
            requested_yellow = int(
                round(self.control.yellow_time_s / self.control.simulator_step_s)
            )
            if requested_yellow:
                self.signal_stage[changed] = 1
                self._set_stage_phase(changed, self.control.yellow_phase_id)
                yellow_ticks, snapshot = self._advance(
                    min(requested_yellow, total_ticks - ticks_used)
                )
                ticks_used += yellow_ticks

            requested_all_red = int(
                round(self.control.all_red_time_s / self.control.simulator_step_s)
            )
            if requested_all_red and ticks_used < total_ticks:
                assert self.control.all_red_phase_id is not None
                self.signal_stage[changed] = 2
                self._set_stage_phase(changed, self.control.all_red_phase_id)
                all_red_ticks, snapshot = self._advance(
                    min(requested_all_red, total_ticks - ticks_used)
                )
                ticks_used += all_red_ticks

            if ticks_used < total_ticks:
                for index in np.flatnonzero(changed):
                    self.backend.set_phase(
                        self.network.intersections[int(index)].intersection_id,
                        self.network.engine_phase(int(index), int(action_values[index])),
                    )
                target_applied = True
                self.signal_stage[changed] = 0
                green_ticks, snapshot = self._advance(total_ticks - ticks_used)
                ticks_used += green_ticks
        else:
            green_ticks, snapshot = self._advance(total_ticks)
            ticks_used += green_ticks

        elapsed_s = ticks_used * self.control.simulator_step_s
        unchanged = ~changed
        self.phase_elapsed_s[unchanged] += elapsed_s
        if target_applied:
            self.current_phase[changed] = action_values[changed]
            self.phase_elapsed_s[changed] = (
                green_ticks * self.control.simulator_step_s
            )
        else:
            self.phase_elapsed_s[changed] = np.where(
                self.signal_stage[changed] == 1,
                yellow_ticks * self.control.simulator_step_s,
                all_red_ticks * self.control.simulator_step_s,
            )

        self._last_snapshot = snapshot
        observation = self.observation_builder.build(
            snapshot, self.current_phase, self.signal_stage, self.phase_elapsed_s
        )
        reward = (
            self.reward_accumulator.finish(snapshot)
            if self.reward_accumulator is not None
            else self.reward_calculator.compute(snapshot)
        )
        terminated = False
        truncated = self.backend.current_time() >= self.scenario.duration_s
        info = {
            "simulation_time_s": float(self.backend.current_time()),
            "changed_intersections": int(np.count_nonzero(changed)),
            "yellow_time_s": yellow_ticks * self.control.simulator_step_s,
            "all_red_time_s": all_red_ticks * self.control.simulator_step_s,
            "green_time_s": green_ticks * self.control.simulator_step_s,
            "target_applied": target_applied or not np.any(changed),
        }
        if self.reward_accumulator is not None:
            info["baseline"] = {
                "elapsed_s": elapsed_s,
                "requested_actions": action_values.copy(),
                "executed_actions": self.current_phase.copy(),
                "signal_stage": self.signal_stage.copy(),
                "local_reward": np.asarray(reward, dtype=np.float32).copy(),
                "reward_components": {
                    name: values.copy()
                    for name, values in self.reward_accumulator.last_components.items()
                },
            }
        return observation, reward, terminated, truncated, info

    def metrics_summary(self) -> Dict[str, float]:
        return self.metrics.summary(self.backend.average_travel_time())

    def close(self) -> None:
        self.backend.close()
