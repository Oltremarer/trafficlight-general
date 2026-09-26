"""Four-phase MPLight with source FRAP structure and two-stage replay sampling."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from .colight import CoLightConfig, CoLightLearner
from .contracts import require_view
from .profiles import get_profile
from .q_learning import QLearner, QPolicy
from .q_networks import PhaseCompetitionQNetwork


PROFILE = replace(
    get_profile('mplight'), implementation='mplight-round-fit-v1',
    adaptation_notes=(
        'PyTorch FRAP: pressure/phase embeddings 4, lane16, pair convolutions20, four phases.',
        'Source two-stage replay: shared timestep sample per node; node-major concat; tail cap; resample.',
        'Actual boundary next states and time-limit bootstrap; one exploration coin for the network; project RNG.',
        'Glorot-uniform linear kernels, zero biases, uniform[-.05,.05] embeddings; persistent Adam eps1e-8.',
    ),
)


@dataclass(frozen=True)
class MPLightConfig(CoLightConfig):
    hidden_dim: int = 20
    adam_epsilon: float = 1e-8

    def to_dict(self):
        return {k: v for k, v in super().to_dict().items() if not k.startswith('attention_')}


class MPLightNetwork(PhaseCompetitionQNetwork):
    def __init__(self, config):
        super().__init__(1, config.hidden_dim, canonical=True)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Embedding):
                nn.init.uniform_(layer.weight, -.05, .05)


class MPLightPolicy(QPolicy):
    def act(self, observation, deterministic):
        output = super().act(observation, deterministic=True)
        epsilon = 0.0 if deterministic else self.learner.epsilon
        actions = output.actions
        if epsilon and self.rng.random() < epsilon:
            actions = self.rng.integers(0, self.learner.network.max_actions, size=len(actions))
        return replace(output, actions=actions,
                       diagnostics={**output.diagnostics, 'epsilon': epsilon})


class MPLightLearner(CoLightLearner):
    """Reuse fitted-Q mechanics, with MPLight's network, reward and sample unit."""
    def __init__(self, network, initial_observation, config=None, seed=0, device='cpu'):
        config = config or MPLightConfig()
        if require_view(initial_observation).schema_id != f'baseline-{PROFILE.schema_hash}':
            raise ValueError('MPLight needs its own experiment PROFILE')
        QLearner.__init__(self, PROFILE, network, initial_observation, config, seed, device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = MPLightNetwork(config).to(self.device)
        self.target = copy.deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate,
                                          eps=config.adam_epsilon)
        self.target_history = {}
        self.last_fit = {}
        self.policy = MPLightPolicy(self, seed)

    def _validate_observation(self, observation):
        QLearner._validate_observation(self, observation)
        view = require_view(observation)
        if (self.network.max_actions != 4 or view.lane_features.shape[1:] != (12, 1)
                or not observation.action_mask.all()
                or np.any(view.source_action_to_local[:, :4] < 0)
                or np.any(view.source_action_to_local[:, 4:] != -1)):
            raise ValueError('MPLight experiment requires four L/T phase pairs and 12 pressures')

    def _sample_fit_items(self):
        # Joint storage retains the last capacity timesteps of EACH intersection.
        # The official updater uses identical sampled indices for all nodes.
        indices = self.rng.choice(len(self.replay), min(self.config.sample_size, len(self.replay)),
                                  replace=False)
        n = self.network.num_intersections
        pairs = [(int(t), node) for node in range(n) for t in indices]
        # Preserve the source agent's second cap, including its node-order bias.
        pairs = pairs[-self.config.replay_capacity:]
        selected = self.rng.choice(len(pairs), min(self.config.sample_size, len(pairs)), replace=False)
        items = []
        for i in selected:
            timestep, node = pairs[int(i)]
            original = self.replay[timestep]
            item = dict(original)
            for key in ('state', 'next_state'):
                item[key] = {k: v[node:node + 1] for k, v in original[key].items()}
            for key in ('actions', 'reward'):
                item[key] = original[key][node:node + 1]
            items.append(item)
        return items
