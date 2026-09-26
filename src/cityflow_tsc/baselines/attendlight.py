"""LLMTSCS AttendLight: 24 lane tokens, lane-to-phase and phase attention."""
from dataclasses import dataclass, replace
import math
import numpy as np
import torch
from torch import nn
from .profiles import BaselineProfile
from .observations import BaselineObservationBuilder
from .colight import CoLightConfig, CoLightLearner
from .mplight import MPLightLearner
from .q_learning import QLearner
from .contracts import require_view

PROFILE = BaselineProfile(
    'attendlight', 'attendlight', 'attend_segments', implementation='attendlight-round-fit-v1',
    adaptation_notes=(
        'LLMTSCS AttendLight port: 24 lane tokens with three running segments and full-lane queue.',
        'Shared lane attention and phase self-attention, each four heads of width8; Dense32 then20/20/1.',
        'Two-stage node-major replay and tail cap; independent per-intersection exploration.',
        'Actual boundary next states/time-limit bootstrap and independent PyTorch RNG; not original NeurIPS runtime.',
    ))


class AttendLightObservationBuilder(BaselineObservationBuilder):
    def __init__(self, network, roadnet_path=None):
        # Reuse only topology/phase metadata from the ordinary canonical builder.
        super().__init__(network, replace(PROFILE, feature_kind='vehicle_count'), roadnet_path)
        self.view_schema_id = f'baseline-{PROFILE.schema_hash}'

    def build(self, snapshot, current_phase, signal_stage, phase_elapsed_s):
        observation = super().build(snapshot, current_phase, signal_stage, phase_elapsed_s)
        tokens = np.zeros((self.network.num_intersections, 24, 4), dtype=np.float32)
        for node, incoming in enumerate(self.codec.lane_ids):
            # Source exiting order: along W incoming direction (eastbound), then E/N/S.
            outgoing = tuple(lane for side in range(4)
                             for lane in self.codec.downstream_lanes[node][side * 3 + 1])
            if len(outgoing) != 12 or len(set(outgoing)) != 12:
                raise ValueError('AttendLight requires four distinct three-lane outgoing roads')
            for i, lane in enumerate(incoming + outgoing):
                if lane not in snapshot.lane_vehicles:
                    raise ValueError(f'AttendLight requires vehicle IDs for {lane}')
                tokens[node, i, 3] = snapshot.lane_waiting_count[lane]
                length = self.codec.lane_lengths[lane]
                for vehicle in snapshot.lane_vehicles[lane]:
                    if 'shadow' in vehicle:
                        continue  # Source AttendLight skips shadows only for running segments.
                    speed = snapshot.vehicle_speeds[vehicle]
                    distance = snapshot.vehicle_distances[vehicle]
                    if speed > .1:
                        if distance > length - 100:
                            tokens[node, i, 0] += 1
                        elif distance > length - 200:
                            tokens[node, i, 1] += 1
                        elif distance > length - 300:
                            tokens[node, i, 2] += 1
        names = tuple(f'lane{i}:{c}' for i in range(24) for c in ('run100','run200','run300','queue'))
        view = replace(observation.baseline_view, features=tokens.reshape(-1,96),
                       lane_features=tokens[:,:12].copy(), feature_names=names)
        return replace(observation, baseline_view=view)


class SourceAttention(nn.Module):
    """Separate source-shaped Q/K/V and output kernels; scaled dot-product softmax."""
    def __init__(self):
        super().__init__()
        for name in ('q', 'k', 'v'):
            self.register_parameter(name, nn.Parameter(torch.empty(32,4,8)))
            self.register_parameter(name+'_bias', nn.Parameter(torch.zeros(4,8)))
        self.output = nn.Parameter(torch.empty(4,8,32))
        self.output_bias = nn.Parameter(torch.zeros(32))
        # Glorot fan calculation on the source EinsumDense kernel shapes.
        for p in (self.q,self.k,self.v,self.output):
            receptive = math.prod(p.shape[:-2])
            bound = math.sqrt(6 / (receptive * (p.shape[-2]+p.shape[-1])))
            nn.init.uniform_(p, -bound, bound)

    def forward(self, query, values):
        q = torch.einsum('...td,dhk->...thk', query, self.q) + self.q_bias
        k = torch.einsum('...sd,dhk->...shk', values, self.k) + self.k_bias
        v = torch.einsum('...sd,dhk->...shk', values, self.v) + self.v_bias
        weights = (torch.einsum('...thk,...shk->...hts',q,k) / math.sqrt(8)).softmax(-1)
        attended = torch.einsum('...hts,...shk->...thk',weights,v)
        return torch.einsum('...thk,hkd->...td',attended,self.output)+self.output_bias


class AttendLightNetwork(nn.Module):
    def __init__(self, hidden_dim=20):
        super().__init__()
        self.lane_encoder=nn.Linear(4,32)
        self.lane_attention=SourceAttention()
        self.phase_attention=SourceAttention()
        self.head=nn.Sequential(nn.Linear(32,hidden_dim),nn.ReLU(),
                                nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1))
        self.register_buffer('phase_map',torch.tensor([
            [1,4,12,13,14,15,16,17],[7,10,18,19,20,21,22,23],
            [0,3,18,19,20,21,22,23],[6,9,12,13,14,15,16,17]]))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,state):
        lanes=torch.relu(self.lane_encoder(state['features'].reshape(*state['features'].shape[:-1],24,4)))
        groups=lanes[...,self.phase_map,:]
        phase=self.lane_attention(groups.mean(-2,keepdim=True),groups).squeeze(-2)
        phase=self.phase_attention(phase,phase)
        source_q=self.head(phase).squeeze(-1)
        mapping=state['source_action_to_local'][...,:4]
        return torch.zeros_like(source_q).scatter(-1,mapping,source_q)


@dataclass(frozen=True)
class AttendLightConfig(CoLightConfig):
    hidden_dim:int=20
    adam_epsilon:float=1e-8
    def to_dict(self):
        return {k:v for k,v in super().to_dict().items() if not k.startswith('attention_')}


class AttendLightLearner(MPLightLearner):
    def __init__(self,network,initial_observation,config=None,seed=0,device='cpu'):
        config=config or AttendLightConfig()
        if require_view(initial_observation).schema_id != f'baseline-{PROFILE.schema_hash}':
            raise ValueError('AttendLight requires its experiment observation builder')
        QLearner.__init__(self,PROFILE,network,initial_observation,config,seed,device)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=config.learning_rate,eps=config.adam_epsilon)
        self.target_history={}
        self.last_fit={}
        # QPolicy uses independent per-intersection exploration, like the source.

    def _validate_observation(self,observation):
        QLearner._validate_observation(self,observation)
        view=require_view(observation)
        if (view.features.shape[1]!=96 or view.lane_features.shape[1:]!=(12,4)
                or self.network.max_actions!=4 or not observation.action_mask.all()
                or np.any(view.source_action_to_local[:,:4]<0)
                or np.any(view.source_action_to_local[:,4:]!=-1)):
            raise ValueError('AttendLight requires 96 segment features and four L/T phases')
