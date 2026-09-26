"""Four-phase CoLight: shared graph Q network and round-based fitted Q learning.

Architecture/fit defaults follow LLMTSCS's CoLight baseline, expressed in PyTorch.
This is not a port of the LLM method or a claim of identical published scores.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import TrainConfig, require_view
from .profiles import get_profile
from .q_learning import QLearner, _tensor_batch
from .q_networks import NeighborAttention


PROFILE = replace(
    get_profile("colight"), implementation="colight-round-fit-v1",
    adaptation_notes=(
        "PyTorch four-phase CoLight; whole-network replay and round-based fixed TD targets.",
        "Actual boundary next states and explicit time-limit bootstrapping; no T-1 workaround.",
        "Per-intersection exploration, persistent Adam, project seeds; not identical author RNG.",
    ),
)


@dataclass(frozen=True)
class CoLightConfig(TrainConfig):
    hidden_dim: int = 32
    batch_size: int = 20
    replay_capacity: int = 12000
    warmup_transitions: int = 0
    epsilon_end: float = 0.2
    sample_size: int = 3000
    fit_epochs: int = 100
    validation_fraction: float = 0.3
    patience: int = 10
    epsilon_decay: float = 0.95
    target_lag_rounds: int = 5
    attention_heads: int = 5
    attention_head_dim: int = 32
    adam_epsilon: float = 1e-7

    def __post_init__(self):
        super().__post_init__()
        for name in ("sample_size", "fit_epochs", "patience", "target_lag_rounds",
                     "attention_heads", "attention_head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0,1)")
        if not 0 < self.epsilon_decay <= 1 or self.adam_epsilon <= 0:
            raise ValueError("invalid epsilon_decay or adam_epsilon")
        if self.sample_size < 2 or self.replay_capacity < 2:
            raise ValueError("round fit needs at least two replay samples")

    def to_dict(self):
        # Do not advertise unused generic Q/PPO knobs as active CoLight settings.
        names = ("hidden_dim", "learning_rate", "gamma", "reward_scale", "batch_size",
                 "replay_capacity", "epsilon_start", "epsilon_end", "bootstrap_truncated",
                 "sample_size", "fit_epochs", "validation_fraction", "patience", "epsilon_decay",
                 "target_lag_rounds", "attention_heads", "attention_head_dim", "adam_epsilon")
        return {name: getattr(self, name) for name in names}


class CoLightNetwork(nn.Module):
    """Feature inputs -> shared MLP -> neighbor attention -> four canonical phase Qs."""

    def __init__(self, config: CoLightConfig, input_dim=20):
        super().__init__()
        width = config.hidden_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU())
        self.attention = NeighborAttention(width, config.attention_heads,
                                          head_dim=config.attention_head_dim)
        self.action_layer = nn.Linear(width, 4)
        # Source Dense layers use random_normal; equivalent distribution, not RNG stream.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.05)
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        hidden = self.encoder(state["features"])
        hidden = self.attention(hidden, state["neighbor_index"], state["neighbor_mask"])
        canonical_q = self.action_layer(hidden)
        # A roadnet may enumerate the four physical phases in another order.
        mapping = state["source_action_to_local"][..., :4]
        return torch.zeros_like(canonical_q).scatter(-1, mapping, canonical_q)


class CoLightLearner(QLearner):
    CHECKPOINT_VERSION = 3
    EXPERIMENT_PROFILE = PROFILE
    INPUT_DIM = 20

    def __init__(self, network, initial_observation, config=None, seed=0, device="cpu"):
        config = config or CoLightConfig()
        profile = self.EXPERIMENT_PROFILE
        if require_view(initial_observation).schema_id != f"baseline-{profile.schema_hash}":
            raise ValueError("CoLight needs an environment built with its experiment PROFILE")
        super().__init__(profile, network, initial_observation, config, seed, device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = CoLightNetwork(config, input_dim=self.INPUT_DIM).to(self.device)
        self.target = copy.deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate,
                                          eps=config.adam_epsilon)
        self.target_history = {}
        self.last_fit = {}

    def _validate_observation(self, observation):
        super()._validate_observation(observation)
        view = require_view(observation)
        if (self.network.max_actions != 4 or view.features.shape[1] != self.INPUT_DIM
                or not observation.action_mask.all()
                or np.any(view.source_action_to_local[:, :4] < 0)
                or np.any(view.source_action_to_local[:, 4:] != -1)):
            raise ValueError(f"CoLight experiment requires the four L/T phase pairs and {self.INPUT_DIM} input features")

    @property
    def epsilon(self):
        return max(self.config.epsilon_start * self.config.epsilon_decay ** self.completed_episodes,
                   self.config.epsilon_end)

    def _sample_fit_items(self):
        size = min(self.config.sample_size, len(self.replay))
        indices = self.rng.choice(len(self.replay), size, replace=False)
        return [self.replay[int(i)] for i in indices]

    def _fit_round(self):
        config = self.config
        round_index = self.completed_episodes - 1
        if round_index:
            source_round = max(round_index - config.target_lag_rounds, 0)
            self.target.load_state_dict(self.target_history[source_round])
        else:
            source_round = -1  # initial network
        items = self._sample_fit_items()
        size = len(items)
        if size < 2:
            raise ValueError("CoLight round contains fewer than two network transitions")
        states = _tensor_batch([item["state"] for item in items], self.device)
        following = _tensor_batch([item["next_state"] for item in items], self.device)
        actions = torch.as_tensor(np.stack([item["actions"] for item in items]),
                                  dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(np.stack([item["reward"] for item in items]),
                                  dtype=torch.float32, device=self.device)
        done = torch.as_tensor([item["terminated"] or
                               (item["truncated"] and not config.bootstrap_truncated)
                               for item in items], dtype=torch.float32, device=self.device)
        self.online.eval()
        with torch.no_grad():
            targets = self._predict_batches(self.online, states).clone()
            future = self._predict_batches(self.target, following).max(-1).values
            td = rewards * config.reward_scale + config.gamma * (1 - done[:, None]) * future
            targets.scatter_(-1, actions[..., None], td[..., None])
        # As in fit(validation_split=0.3, shuffle=False): fixed tail validation split.
        split = max(1, min(size - 1, int(size * (1 - config.validation_fraction))))
        validation = {key: value[split:] for key, value in states.items()}
        best, wait, updates = float("inf"), 0, 0
        losses = []
        for epoch in range(config.fit_epochs):
            self.online.train()
            for start in range(0, split, config.batch_size):
                stop = min(start + config.batch_size, split)
                batch = {key: value[start:stop] for key, value in states.items()}
                loss = F.mse_loss(self.online(batch), targets[start:stop])
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite CoLight loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in self.online.parameters()):
                    raise FloatingPointError("non-finite CoLight gradient")
                self.optimizer.step()
                self.gradient_steps += 1
                updates += 1
                losses.append(float(loss.detach().cpu()))
            self.online.eval()
            with torch.no_grad():
                val_loss = float(F.mse_loss(self._predict_batches(self.online, validation), targets[split:]).cpu())
            if not np.isfinite(val_loss):
                raise FloatingPointError("non-finite CoLight validation loss")
            if val_loss < best:
                best, wait = val_loss, 0
            else:
                wait += 1
                if wait >= config.patience:
                    break
        # Keep the last epoch (no restore_best_weights), and the exact round-lag targets.
        self.target_history[round_index] = {k: v.detach().cpu().clone()
                                            for k, v in self.online.state_dict().items()}
        oldest_needed = max(round_index + 1 - config.target_lag_rounds, 0)
        self.target_history = {r: weights for r, weights in self.target_history.items()
                               if r >= oldest_needed}
        self.policy.policy_version = self.gradient_steps
        self.last_fit = {"loss": float(np.mean(losses)), "validation_loss": val_loss,
                         "fit_epochs": epoch + 1, "updates": updates, "sample_size": size,
                         "target_round": source_round, "gradient_steps": self.gradient_steps}
        return self.last_fit

    def _predict_batches(self, model, states):
        size = states["features"].shape[0]
        return torch.cat([model({key: value[start:start + self.config.batch_size]
                                 for key, value in states.items()})
                          for start in range(0, size, self.config.batch_size)], dim=0)

    def update_if_due(self, event="step"):
        if event not in {"step", "round", "rollout"}:
            raise ValueError("unknown update event")
        if event != "round" or self.completed_episodes <= self._last_round_processed:
            return {}
        result = self._fit_round()
        self._last_round_processed = self.completed_episodes
        return result

    def _checkpoint_extra(self):
        return {"colight_target_history": self.target_history, "colight_last_fit": self.last_fit}

    def _restore_checkpoint_extra(self, checkpoint):
        self.target_history = checkpoint["colight_target_history"]
        self.last_fit = checkpoint["colight_last_fit"]
