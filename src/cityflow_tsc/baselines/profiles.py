from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import hashlib
import json


LLM_SHA = "d5d4180f34edb843e1d1b462d5846c75d6d4533a"
LIB_SHA = "127af9f93902778e556de2eedb2b606c4c9447e6"
CMALC_SHA = "93a9d9f60f77153c25fbac739ad7615e194ad7cb"


@dataclass(frozen=True)
class BaselineProfile:
    baseline_id: str
    algorithm: str
    feature_kind: str
    layout: str = "canonical12"
    reward_kind: str = "absolute_queue_pressure"
    reward_factor: float = -0.25
    reward_window: str = "pre_mean"
    graph: str = "none"
    parameter_sharing: bool = True
    update_schedule: str = "round"
    global_reward: bool = False
    source_repo: str = "usail-hkust/LLMTSCS"
    source_commit: str = LLM_SHA
    implementation: str = "pytorch-port-v1"
    adaptation_notes: tuple[str, ...] = (
        "Native PyTorch port, not upstream runtime or a paper reproduction.",
        "Local horizon/bootstrap rules and bounded mini-batch updates are explicit adaptations.",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "adaptation_notes", tuple(self.adaptation_notes))
        if self.layout not in {"canonical12", "generic"}:
            raise ValueError("unknown lane layout")
        if self.reward_window not in {"last", "pre_mean", "post_mean", "sum", "integral"}:
            raise ValueError("unknown reward window")
        if self.graph not in {"none", "knn", "road"}:
            raise ValueError("unknown graph mode")
        if self.update_schedule not in {"step", "round", "rollout"}:
            raise ValueError("unknown update schedule")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def schema_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


PROFILES: dict[str, BaselineProfile] = {}


def _add(name: str, algorithm: str, feature: str, **kwargs: Any) -> None:
    PROFILES[name] = BaselineProfile(name, algorithm, feature, **kwargs)


_add("presslight", "presslight", "general_queue_pressure", implementation="pytorch-port-v2")
_add("e-presslight", "presslight", "efficient_queue_pressure", implementation="pytorch-port-v2")
_add("mplight", "mplight", "general_vehicle_pressure")
_add("e-mplight", "mplight", "efficient_queue_pressure")
_add("a-mplight", "advanced_mplight", "advanced")
_add("colight", "colight", "vehicle_count", reward_kind="queue", graph="knn", implementation="pytorch-port-v2")
_add("e-colight", "colight", "efficient_queue_pressure", reward_kind="queue", graph="knn", implementation="pytorch-port-v2")
_add("a-colight", "colight", "advanced", reward_kind="queue", graph="knn", implementation="pytorch-port-v2")
_lib = dict(layout="generic", reward_kind="queue_mean", reward_factor=-1.0,
            reward_window="post_mean", source_repo="DaRL-LibSignal/LibSignal",
            source_commit=LIB_SHA, update_schedule="step",
            adaptation_notes=(
                "Native PyTorch port using local Q/actor-critic updates and local training defaults.",
                "FRAP/MPLight use two dedicated L/T demand lanes per phase; permissive right turns are excluded.",
                "DQN/CoLight source x12 reward multiplier is omitted; local CoLight includes phase input.",
            ))
_add("idqn", "idqn", "vehicle_count", parameter_sharing=False, **_lib)
_add("shared-dqn", "dqn", "vehicle_count", **_lib)
_add("frap", "frap", "vehicle_count", parameter_sharing=False, implementation="pytorch-port-v2", **_lib)
_add("libsignal-mplight", "mplight", "vehicle_count", implementation="pytorch-port-v2", **_lib)
_add("libsignal-colight", "colight", "vehicle_count", graph="road", **_lib)
_add("maddpg", "maddpg", "vehicle_count", parameter_sharing=False, **_lib)
for _name in ("ippo", "mappo"):
    _add(_name, _name, "counts_queue", layout="generic", reward_kind="queue",
         reward_factor=-1.0, reward_window="post_mean", update_schedule="rollout",
         global_reward=True, source_repo="DaRL-LibSignal/cMALC-D", source_commit=CMALC_SHA,
         adaptation_notes=(
             "Local PyTorch recurrent PPO with fresh on-policy episodes and stored behavior probabilities.",
             "Uses explicit local lane count/queue observations and negative queue reward, not cMALC TSflow.",
             "IPPO/MAPPO share an actor; local/global critic respectively; n-step or explicit GAE.",
         ))


def get_profile(name: str | BaselineProfile) -> BaselineProfile:
    if isinstance(name, BaselineProfile):
        return name
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown baseline {name!r}; choose from {', '.join(PROFILES)}") from exc


def list_profiles() -> tuple[BaselineProfile, ...]:
    return tuple(PROFILES.values())
