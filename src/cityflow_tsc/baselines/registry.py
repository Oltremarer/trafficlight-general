from __future__ import annotations

from .contracts import TrainConfig, require_view
from .profiles import BaselineProfile, get_profile, list_profiles


def make_train_config(profile, **overrides):
    """Shared CLI/Python defaults; explicit overrides always take precedence."""
    profile = get_profile(profile)
    settings = {}
    if profile.algorithm in {"ippo", "mappo", "maddpg"}:
        settings.update(gamma=0.99, reward_scale=1.0, learning_rate=3e-4)
    settings.update(overrides)
    return TrainConfig(**settings)


def create_learner(profile, network, initial_observation, config=None, seed=0, device="cpu"):
    """Construct only the requested backend after an observation schema is known."""
    profile = get_profile(profile)
    view = require_view(initial_observation)
    if view.schema_id != f"baseline-{profile.schema_hash}":
        raise ValueError("learner profile does not match the environment observation profile")
    config = make_train_config(profile) if config is None else config
    if profile.algorithm in {"ippo", "mappo"}:
        from .ppo import PPOLearner
        cls = PPOLearner
    elif profile.algorithm == "maddpg":
        from .maddpg import MADDPGLearner
        cls = MADDPGLearner
    else:
        from .q_learning import QLearner
        cls = QLearner
    return cls(profile, network, initial_observation, config, seed=seed, device=device)


__all__ = ["BaselineProfile", "get_profile", "list_profiles", "make_train_config", "create_learner"]
