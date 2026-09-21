from __future__ import annotations

from typing import List, Tuple

from cityflow_tsc.config import ControlConfig, ScenarioConfig
from cityflow_tsc.types import NetworkSnapshot


class DeterministicBackend:
    """Small deterministic traffic process that exercises the real environment contract."""

    def __init__(self, control: ControlConfig) -> None:
        self.control = control
        self.time_s = 0.0
        self.phases = {"intersection_1_1": 1, "intersection_2_1": 1}
        self.phase_events: List[Tuple[float, str, int]] = []
        self.closed = False

    def reset(self, scenario: ScenarioConfig) -> NetworkSnapshot:
        self.time_s = 0.0
        self.phases = {"intersection_1_1": 1, "intersection_2_1": 1}
        self.phase_events = []
        self.closed = False
        return self.snapshot()

    def set_phase(self, intersection_id: str, engine_phase_id: int) -> None:
        self.phases[intersection_id] = engine_phase_id
        self.phase_events.append((self.time_s, intersection_id, engine_phase_id))

    def next_step(self) -> None:
        self.time_s += self.control.simulator_step_s

    def current_time(self) -> float:
        return self.time_s

    def average_travel_time(self) -> float:
        return 7.5

    def snapshot(self) -> NetworkSnapshot:
        tick = int(round(self.time_s / self.control.simulator_step_s))
        vehicle_ids = {}
        if tick < 3:
            vehicle_ids["vehicle_0"] = 0.0 if tick < 2 else 2.0
        if 1 <= tick < 7:
            vehicle_ids["vehicle_1"] = 0.0 if tick < 4 else 3.0
        return NetworkSnapshot(
            time_s=self.time_s,
            lane_vehicle_count={
                "road_w_a_0": max(0, 4 - tick),
                "road_a_b_0": max(0, 2 - tick // 2),
                "road_b_a_0": tick % 3,
                "road_b_e_0": 1,
                "road_e_b_0": 1 if tick >= 2 else 0,
                "road_a_w_0": 0,
            },
            lane_waiting_count={
                "road_w_a_0": max(0, 3 - tick),
                "road_a_b_0": 1 if tick < 2 else 0,
                "road_b_a_0": tick % 2,
                "road_b_e_0": 0,
                "road_e_b_0": 1 if tick == 2 else 0,
                "road_a_w_0": 0,
            },
            vehicle_speeds=vehicle_ids,
        )

    def close(self) -> None:
        self.closed = True
