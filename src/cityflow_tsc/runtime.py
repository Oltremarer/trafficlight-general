from __future__ import annotations

from .config import ControlConfig
from .environment import TrafficEnv
from .metrics import MetricCollector
from .observations import (
    QueuePressureObservationBuilder,
    RichTrafficObservationBuilder,
)
from .rewards import QueueReward
from .simulator import CityFlowBackend
from .topology import load_network_spec
from .types import NetworkSpec


def build_environment(
    control: ControlConfig,
    network: NetworkSpec,
    waiting_speed_threshold: float = 0.1,
) -> TrafficEnv:
    return TrafficEnv(
        network=network,
        control=control,
        backend=CityFlowBackend(control),
        observation_builder=QueuePressureObservationBuilder(network),
        reward_calculator=QueueReward(network),
        metrics=MetricCollector(waiting_speed_threshold),
    )


def build_rich_environment(
    control: ControlConfig,
    network: NetworkSpec,
    waiting_speed_threshold: float = 0.1,
) -> TrafficEnv:
    return TrafficEnv(
        network=network,
        control=control,
        backend=CityFlowBackend(control),
        observation_builder=RichTrafficObservationBuilder(
            network, waiting_speed_threshold
        ),
        reward_calculator=QueueReward(network),
        metrics=MetricCollector(waiting_speed_threshold),
    )
