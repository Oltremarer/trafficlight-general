from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cityflow_tsc.config import ControlConfig, ScenarioConfig
from cityflow_tsc.environment import TrafficEnv
from cityflow_tsc.metrics import MetricCollector
from cityflow_tsc.observations import QueuePressureObservationBuilder
from cityflow_tsc.policies import FixedTimePolicy, MaxPressurePolicy
from cityflow_tsc.rewards import QueueReward
from cityflow_tsc.runner import EpisodeRunner
from cityflow_tsc.topology import load_network_spec
from cityflow_tsc.trajectory import TrajectoryReader

from .fakes import DeterministicBackend


FIXTURES = Path(__file__).parent / "fixtures"


def make_stack(tmp_path: Path, duration_s: int = 10):
    control = ControlConfig(
        decision_interval_s=5,
        simulator_step_s=1.0,
        yellow_time_s=2,
        green_phase_ids=(1, 2),
    )
    scenario = ScenarioConfig(
        roadnet_path=FIXTURES / "roadnet_two_intersections.json",
        flow_path=FIXTURES / "flow_empty.json",
        output_dir=tmp_path / "run",
        duration_s=duration_s,
        seed=17,
    )
    network = load_network_spec(scenario.roadnet_path, control)
    backend = DeterministicBackend(control)
    env = TrafficEnv(
        network=network,
        control=control,
        backend=backend,
        observation_builder=QueuePressureObservationBuilder(network),
        reward_calculator=QueueReward(network),
        metrics=MetricCollector(),
    )
    return control, scenario, network, backend, env


def test_topology_is_stable_and_masked(tmp_path: Path) -> None:
    _, _, network, _, _ = make_stack(tmp_path)
    assert network.intersection_ids == ("intersection_1_1", "intersection_2_1")
    assert network.neighbor_index.tolist() == [[-1, -1, 1, -1], [-1, -1, -1, 0]]
    assert network.neighbor_mask.tolist() == [
        [False, False, True, False],
        [False, False, False, True],
    ]
    assert network.action_mask().tolist() == [[True, True], [True, True]]
    assert network.padded_phase_movement_mask().shape == (2, 2, 2)


def test_environment_executes_yellow_and_green_data_flow(tmp_path: Path) -> None:
    _, scenario, _, backend, env = make_stack(tmp_path)
    observation, _ = env.reset(scenario)
    assert observation.features.shape == (2, 2, 6)

    policy = FixedTimePolicy()
    policy.reset(scenario.seed, env.network)
    first = policy.act(observation, deterministic=True)
    observation, reward, _, truncated, info = env.step(first.actions)
    assert not truncated
    assert info["yellow_time_s"] == 0
    assert observation.time_s == 5
    assert reward.shape == (2,)

    second = policy.act(observation, deterministic=True)
    observation, _, _, truncated, info = env.step(second.actions)
    assert truncated
    assert observation.time_s == 10
    assert info == {
        "simulation_time_s": 10.0,
        "changed_intersections": 2,
        "yellow_time_s": 2.0,
        "all_red_time_s": 0.0,
        "green_time_s": 3.0,
        "target_applied": True,
    }
    yellow_events = [event for event in backend.phase_events if event[2] == 0]
    phase_two_events = [event for event in backend.phase_events if event[2] == 2]
    assert yellow_events[-2:] == [
        (5.0, "intersection_1_1", 0),
        (5.0, "intersection_2_1", 0),
    ]
    assert phase_two_events[-2:] == [
        (7.0, "intersection_1_1", 2),
        (7.0, "intersection_2_1", 2),
    ]


def test_complete_episode_round_trips_trajectory(tmp_path: Path) -> None:
    _, scenario, network, backend, env = make_stack(tmp_path)
    result = EpisodeRunner().run(
        env=env,
        policy=MaxPressurePolicy(),
        scenario=scenario,
        deterministic=True,
    )
    assert result.steps == 2
    assert backend.closed
    assert result.metrics["average_travel_time_s"] == 7.5
    assert result.metrics["metric_ticks"] == 10
    assert result.metrics["observed_vehicles"] == 2
    assert result.metrics["throughput_vehicles"] == 2

    loaded = TrajectoryReader().load(Path(result.manifest_path))
    manifest = loaded["manifest"]
    arrays = loaded["arrays"]
    assert manifest["steps"] == 2
    assert manifest["num_intersections"] == network.num_intersections
    assert manifest["policy"] == "max_pressure"
    assert arrays["observation_features"].shape == (2, 2, 2, 6)
    assert arrays["observation_valid_mask"].dtype == np.bool_
    assert arrays["actions"].shape == (2, 2)
    assert arrays["rewards"].shape == (2, 2)
    assert arrays["truncated"].tolist() == [False, True]
    assert Path(scenario.output_dir / "metrics.json").is_file()


def test_invalid_action_fails_before_simulator_advance(tmp_path: Path) -> None:
    _, scenario, _, backend, env = make_stack(tmp_path)
    env.reset(scenario)
    with pytest.raises(ValueError, match="invalid action"):
        env.step(np.asarray([9, 0], dtype=np.int64))
    assert backend.current_time() == 0


@pytest.mark.parametrize(
    "actions",
    (
        np.asarray([1.9, 0.2]),
        np.asarray([True, False]),
        [True, 0],
        np.asarray(["1", "0"]),
        np.asarray([np.nan, 0.0]),
    ),
)
def test_non_integer_actions_fail_before_simulator_advance(
    tmp_path: Path, actions: np.ndarray
) -> None:
    _, scenario, _, backend, env = make_stack(tmp_path)
    env.reset(scenario)
    with pytest.raises(ValueError, match="discrete integer"):
        env.step(actions)
    assert backend.current_time() == 0


def test_control_interval_must_align_with_simulator_step() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ControlConfig(
            decision_interval_s=5,
            simulator_step_s=2.0,
            yellow_time_s=1,
        )


def test_episode_duration_must_align_with_simulator_step(tmp_path: Path) -> None:
    control = ControlConfig(
        decision_interval_s=4,
        simulator_step_s=2.0,
        yellow_time_s=0,
        green_phase_ids=(1, 2),
    )
    scenario = ScenarioConfig(
        roadnet_path=FIXTURES / "roadnet_two_intersections.json",
        flow_path=FIXTURES / "flow_empty.json",
        output_dir=tmp_path / "unaligned",
        duration_s=5,
    )
    network = load_network_spec(scenario.roadnet_path, control)
    backend = DeterministicBackend(control)
    env = TrafficEnv(
        network,
        control,
        backend,
        QueuePressureObservationBuilder(network),
        QueueReward(network),
        MetricCollector(),
    )
    with pytest.raises(ValueError, match="duration_s must be divisible"):
        env.reset(scenario)
    assert backend.current_time() == 0


def test_backend_must_advance_time(tmp_path: Path) -> None:
    _, scenario, _, backend, env = make_stack(tmp_path)
    env.reset(scenario)
    backend.next_step = lambda: None
    with pytest.raises(RuntimeError, match="did not advance time"):
        env.step(np.asarray([0, 0], dtype=np.int64))


def test_partial_final_interval_preserves_yellow_stage_semantics(tmp_path: Path) -> None:
    _, scenario, _, backend, env = make_stack(tmp_path, duration_s=6)
    observation, _ = env.reset(scenario)
    policy = FixedTimePolicy()
    policy.reset(scenario.seed, env.network)
    observation, _, _, _, _ = env.step(
        policy.act(observation, deterministic=True).actions
    )
    observation, _, _, truncated, info = env.step(
        policy.act(observation, deterministic=True).actions
    )
    assert truncated
    assert info["yellow_time_s"] == 1.0
    assert info["green_time_s"] == 0.0
    assert not info["target_applied"]
    assert observation.current_phase.tolist() == [0, 0]
    assert observation.signal_stage.tolist() == [1, 1]
    assert observation.phase_elapsed_s.tolist() == [1.0, 1.0]
    assert backend.current_time() == 6


def test_transition_time_must_leave_green_time() -> None:
    with pytest.raises(ValueError, match="positive green time"):
        ControlConfig(
            decision_interval_s=5,
            yellow_time_s=3,
            all_red_time_s=2,
            all_red_phase_id=9,
        )


@pytest.mark.parametrize(
    "control",
    (
        ControlConfig(
            decision_interval_s=5,
            yellow_time_s=1,
            yellow_phase_id=99,
            green_phase_ids=(1, 2),
        ),
        ControlConfig(
            decision_interval_s=5,
            yellow_time_s=0,
            all_red_time_s=1,
            all_red_phase_id=99,
            green_phase_ids=(1, 2),
        ),
    ),
)
def test_transition_phase_must_exist_in_roadnet(control: ControlConfig) -> None:
    with pytest.raises(ValueError, match="phase 99 does not exist"):
        load_network_spec(FIXTURES / "roadnet_two_intersections.json", control)
