from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ControlConfig:
    """Traffic-light timing and action-space configuration."""

    decision_interval_s: int = 30
    simulator_step_s: float = 1.0
    yellow_time_s: int = 5
    all_red_time_s: int = 0
    yellow_phase_id: int = 0
    all_red_phase_id: Optional[int] = None
    green_phase_ids: Tuple[int, ...] = (1, 2, 3, 4)

    def __post_init__(self) -> None:
        if self.decision_interval_s <= 0:
            raise ValueError("decision_interval_s must be positive")
        if self.simulator_step_s <= 0:
            raise ValueError("simulator_step_s must be positive")
        if self.yellow_time_s < 0 or self.all_red_time_s < 0:
            raise ValueError("transition times cannot be negative")
        if self.yellow_phase_id < 0:
            raise ValueError("yellow_phase_id cannot be negative")
        if self.all_red_phase_id is not None and self.all_red_phase_id < 0:
            raise ValueError("all_red_phase_id cannot be negative")
        if self.yellow_time_s + self.all_red_time_s >= self.decision_interval_s:
            raise ValueError(
                "yellow plus all-red time must leave positive green time in a decision interval"
            )
        if self.all_red_time_s and self.all_red_phase_id is None:
            raise ValueError("all_red_phase_id is required when all_red_time_s is positive")
        if not self.green_phase_ids:
            raise ValueError("at least one green phase is required")
        if len(set(self.green_phase_ids)) != len(self.green_phase_ids):
            raise ValueError("green_phase_ids must be unique")
        for seconds, name in (
            (self.decision_interval_s, "decision_interval_s"),
            (self.yellow_time_s, "yellow_time_s"),
            (self.all_red_time_s, "all_red_time_s"),
        ):
            ticks = seconds / self.simulator_step_s
            if abs(ticks - round(ticks)) > 1e-9:
                raise ValueError(f"{name} must be divisible by simulator_step_s")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["green_phase_ids"] = list(self.green_phase_ids)
        return data


@dataclass(frozen=True)
class ScenarioConfig:
    """Files and runtime settings for one CityFlow episode."""

    roadnet_path: Path
    flow_path: Path
    output_dir: Path
    duration_s: int = 3600
    seed: int = 0
    thread_num: int = 1
    save_replay: bool = False

    def __post_init__(self) -> None:
        roadnet = Path(self.roadnet_path).expanduser().resolve()
        flow = Path(self.flow_path).expanduser().resolve()
        output = Path(self.output_dir).expanduser().resolve()
        object.__setattr__(self, "roadnet_path", roadnet)
        object.__setattr__(self, "flow_path", flow)
        object.__setattr__(self, "output_dir", output)
        if not roadnet.is_file():
            raise FileNotFoundError(f"roadnet file does not exist: {roadnet}")
        if not flow.is_file():
            raise FileNotFoundError(f"flow file does not exist: {flow}")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.thread_num <= 0:
            raise ValueError("thread_num must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roadnet_path": str(self.roadnet_path),
            "flow_path": str(self.flow_path),
            "output_dir": str(self.output_dir),
            "duration_s": self.duration_s,
            "seed": self.seed,
            "thread_num": self.thread_num,
            "save_replay": self.save_replay,
        }
