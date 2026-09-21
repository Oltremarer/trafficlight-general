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
    *,
    profile=None,
    backend=None,
) -> TrafficEnv:
    if profile is not None:
        from .baselines.observations import BaselineObservationBuilder
        from .baselines.profiles import get_profile
        from .baselines.rewards import BaselineReward, RewardAccumulator

        profile = get_profile(profile)
        reward = BaselineReward(network, profile)
        return TrafficEnv(
            network=network,
            control=control,
            backend=backend if backend is not None else CityFlowBackend(control),
            observation_builder=BaselineObservationBuilder(network, profile),
            reward_calculator=reward,
            metrics=MetricCollector(waiting_speed_threshold),
            reward_accumulator=RewardAccumulator(reward),
        )
    return TrafficEnv(
        network=network,
        control=control,
        backend=backend if backend is not None else CityFlowBackend(control),
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
