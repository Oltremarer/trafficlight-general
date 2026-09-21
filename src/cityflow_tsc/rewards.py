from __future__ import annotations

import numpy as np

from .types import NetworkSnapshot, NetworkSpec


class QueueReward:
    """Per-intersection negative incoming-lane queue."""

    name = "negative_incoming_queue"

    def __init__(self, network: NetworkSpec) -> None:
        self.network = network

    def compute(self, snapshot: NetworkSnapshot) -> np.ndarray:
        rewards = np.zeros(self.network.num_intersections, dtype=np.float32)
        for intersection in self.network.intersections:
            rewards[intersection.index] = -float(
                sum(
                    snapshot.lane_waiting_count.get(lane_id, 0)
                    for lane_id in intersection.incoming_lanes
                )
            )
        return rewards
