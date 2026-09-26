from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Optional, Protocol

from .config import ControlConfig, ScenarioConfig
from .types import NetworkSnapshot


class CityFlowBackend:
    """Narrow owner of the CityFlow Engine and its raw simulator API."""

    def __init__(self, control: ControlConfig, *, lane_change: bool = False) -> None:
        self.control = control
        self.lane_change = lane_change
        self._engine: Optional[Any] = None
        self._config_path: Optional[Path] = None

    @property
    def config_path(self) -> Path:
        if self._config_path is None:
            raise RuntimeError("backend has not been reset")
        return self._config_path

    def reset(self, scenario: ScenarioConfig) -> NetworkSnapshot:
        scenario.output_dir.mkdir(parents=True, exist_ok=True)
        common_dir = Path(
            os.path.commonpath([scenario.roadnet_path, scenario.flow_path])
        )
        cityflow_config = {
            "interval": self.control.simulator_step_s,
            "seed": scenario.seed,
            "dir": str(common_dir) + os.sep,
            "roadnetFile": os.path.relpath(scenario.roadnet_path, common_dir),
            "flowFile": os.path.relpath(scenario.flow_path, common_dir),
            "rlTrafficLight": True,
            "laneChange": self.lane_change,
            "saveReplay": scenario.save_replay,
            "roadnetLogFile": str(scenario.output_dir / "roadnetLogFile.json"),
            "replayLogFile": str(scenario.output_dir / "replayLogFile.txt"),
        }
        self._config_path = scenario.output_dir / "cityflow.config.json"
        with self._config_path.open("w", encoding="utf-8") as handle:
            json.dump(cityflow_config, handle, indent=2, sort_keys=True)

        try:
            cityflow = importlib.import_module("cityflow")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CityFlow is not installed. Run the simulator path on a supported "
                "Linux environment with CityFlow 0.1 available."
            ) from exc
        self._engine = cityflow.Engine(
            str(self._config_path), thread_num=scenario.thread_num
        )
        return self.snapshot()

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("CityFlow backend has not been reset")
        return self._engine

    def set_phase(self, intersection_id: str, engine_phase_id: int) -> None:
        self._require_engine().set_tl_phase(intersection_id, int(engine_phase_id))

    def next_step(self) -> None:
        self._require_engine().next_step()

    def current_time(self) -> float:
        return float(self._require_engine().get_current_time())

    def average_travel_time(self) -> float:
        return float(self._require_engine().get_average_travel_time())

    @property
    def engine(self) -> Any:
        """Raw readout/archive access for the physical counterfactual recorder."""
        return self._require_engine()

    def capture_archive(self) -> Any:
        return self._require_engine().snapshot()

    def restore_archive(self, archive: Any) -> None:
        self._require_engine().load(archive)

    def load_archive_file(self, path: Path) -> None:
        self._require_engine().load_from_file(str(path))

    def snapshot(self) -> NetworkSnapshot:
        engine = self._require_engine()
        speeds = engine.get_vehicle_speed()
        lane_vehicles = engine.get_lane_vehicles()
        distances = engine.get_vehicle_distance()
        return NetworkSnapshot(
            time_s=float(engine.get_current_time()),
            lane_vehicle_count={
                key: int(value) for key, value in engine.get_lane_vehicle_count().items()
            },
            lane_waiting_count={
                key: int(value)
                for key, value in engine.get_lane_waiting_vehicle_count().items()
            },
            vehicle_speeds={key: float(value) for key, value in speeds.items()},
            lane_vehicles={
                key: tuple(str(vehicle_id) for vehicle_id in value)
                for key, value in lane_vehicles.items()
            },
            vehicle_distances={
                key: float(value) for key, value in distances.items()
            },
            vehicle_pool_ids=tuple(engine.get_vehicles(True)),
            active_vehicle_ids=tuple(engine.get_vehicles(False)),
            active_vehicle_count=int(engine.get_vehicle_count()),
        )

    def close(self) -> None:
        self._engine = None


class SimulatorBackendProtocol(Protocol):
    """Structural interface used by TrafficEnv and deterministic test backends."""

    def reset(self, scenario: ScenarioConfig) -> NetworkSnapshot:
        ...

    def set_phase(self, intersection_id: str, engine_phase_id: int) -> None:
        ...

    def next_step(self) -> None:
        ...

    def current_time(self) -> float:
        ...

    def average_travel_time(self) -> float:
        ...

    def snapshot(self) -> NetworkSnapshot:
        ...

    def close(self) -> None:
        ...
