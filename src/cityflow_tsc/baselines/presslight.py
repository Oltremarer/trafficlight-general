"""Four-phase PressLight with source current-phase Q branches and two-stage replay."""
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
from .q_networks import PressLightQNetwork
from .mplight import MPLightLearner, MPLightPolicy


PROFILE = replace(
    get_profile('presslight'), implementation='presslight-round-fit-v1',
    adaptation_notes=(
        'PyTorch source PressLight: sigmoid shared20; four current-phase branches, ReLU20 then four Qs.',
        'Source run_presslight.py uses model key EfficientPressLight with general queue pressure and pressure reward.',
        'Source two-stage node-major replay with tail cap; one whole-network exploration coin.',
        'Actual boundary next states and time-limit bootstrap; project RNG; persistent Adam epsilon1e-7.',
    ),
)


@dataclass(frozen=True)
class PressLightConfig(CoLightConfig):
    hidden_dim: int = 20
    adam_epsilon: float = 1e-7

    def to_dict(self):
        return {k: v for k, v in super().to_dict().items() if not k.startswith('attention_')}


class PressLightNetwork(PressLightQNetwork):
    def __init__(self, config):
        super().__init__(20, config.hidden_dim, 4)
        self.register_buffer('phase_codes', torch.tensor([
            [0,1,0,1,0,0,0,0], [0,0,0,0,0,1,0,1],
            [1,0,1,0,0,0,0,0], [0,0,0,0,1,0,1,0]], dtype=torch.float32))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        encoded = torch.sigmoid(self.shared(state['features']))
        branches = torch.stack([branch(encoded) for branch in self.branches], dim=-2)
        selected = (state['phase_encoding'].unsqueeze(-2) == self.phase_codes).all(-1)
        source_q = (branches * selected.unsqueeze(-1)).sum(-2)
        mapping = state['source_action_to_local'][..., :4]
        return torch.zeros_like(source_q).scatter(-1, mapping, source_q)


class PressLightLearner(MPLightLearner):
    """Reuse fitted-Q mechanics, with PressLight's network, reward and sample unit."""
    def __init__(self, network, initial_observation, config=None, seed=0, device='cpu'):
        config = config or PressLightConfig()
        if require_view(initial_observation).schema_id != f'baseline-{PROFILE.schema_hash}':
            raise ValueError('PressLight needs its own experiment PROFILE')
        QLearner.__init__(self, PROFILE, network, initial_observation, config, seed, device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = PressLightNetwork(config).to(self.device)
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
            raise ValueError('PressLight experiment requires four L/T phase pairs and 12 pressures')
