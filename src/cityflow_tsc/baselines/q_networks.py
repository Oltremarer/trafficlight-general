"""Native PyTorch architecture ports; these are not upstream runtime replicas.

The canonical lane/phase structures follow LLMTSCS d5d4180. Generic FRAP
uses the same phase competition construction with the local phase-lane map.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .profiles import BaselineProfile


TensorState = Mapping[str, torch.Tensor]
CANONICAL_LT = (0, 1, 3, 4, 6, 7, 9, 10)


class DenseQNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, actions: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, actions))

    def forward(self, state: TensorState) -> torch.Tensor:
        return self.layers(state["features"])


class PressLightQNetwork(nn.Module):
    """A shared feature encoder and a Q branch for each CURRENT phase."""

    def __init__(self, feature_dim: int, hidden_dim: int, actions: int) -> None:
        super().__init__()
        self.shared = nn.Linear(feature_dim, hidden_dim)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, actions)) for _ in range(actions)
        ])

    def forward(self, state: TensorState) -> torch.Tensor:
        encoded = torch.sigmoid(self.shared(state["features"]))
        all_branches = torch.stack([branch(encoded) for branch in self.branches], dim=-2)
        selector = state["current_phase"][..., None, None]
        selector = selector.expand(*selector.shape[:-1], all_branches.shape[-1])
        return all_branches.gather(-2, selector).squeeze(-2)


class CanonicalPhaseQNetwork(nn.Module):
    """Keep fixed Q heads/PressLight branches in source order; return LOCAL Qs.

    Replay, action masks, exploration and the environment all keep local actions.
    Applying the permutation here covers both online and target-network updates.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, state: TensorState) -> torch.Tensor:
        source_to_local = state["source_action_to_local"]
        local_ids = torch.arange(state["action_mask"].shape[-1], device=source_to_local.device)
        local_to_source = (source_to_local.unsqueeze(-1) == local_ids).long().argmax(dim=-2)
        source_current = local_to_source.gather(-1, state["current_phase"].unsqueeze(-1)).squeeze(-1)
        source_q = self.model({**state, "current_phase": source_current})
        return source_q.gather(-1, local_to_source)


class PhaseCompetitionQNetwork(nn.Module):
    """Lane embeddings -> phase demands -> relation-gated pair competition.

    Linear layers on the final dimension implement the upstream 1x1 convolutions.
    Advanced MPLight has a separate running-vehicle embedding, rather than
    flattening its additional observations into an unrelated MLP.
    """

    def __init__(self, channels: int, hidden_dim: int, canonical: bool,
                 advanced: bool = False, phase_pairs: bool = False) -> None:
        super().__init__()
        if advanced and channels != 2:
            raise ValueError("AdvancedMPLight requires pressure and running channels")
        self.canonical = canonical
        self.channels = channels
        self.phase_pairs = phase_pairs
        self.value_embeddings = nn.ModuleList([nn.Linear(1, 4) for _ in range(channels)])
        self.phase_embedding = nn.Embedding(2, 4)
        lane_dim = 24 if advanced else 16
        self.lane_embedding = nn.Linear(4 * (channels + 1), lane_dim)
        self.relation_embedding = nn.Embedding(2, 4)
        self.demand_conv = nn.Linear(2 * lane_dim, hidden_dim)
        self.relation_conv = nn.Linear(4, hidden_dim)
        self.combine_conv = nn.Linear(hidden_dim, hidden_dim)
        self.pair_value = nn.Linear(hidden_dim, 1)
        # Single valid phase is a local extension; the original assumes 4/8.
        self.single_phase_value = nn.Linear(lane_dim, 1)

    def forward(self, state: TensorState) -> torch.Tensor:
        lane = state["lane_features"]
        lane_mask = state["lane_mask"]
        mapping = state["phase_lane_mask"]
        if self.canonical:
            idx = torch.as_tensor(CANONICAL_LT, device=lane.device)
            lane = lane.index_select(-2, idx)
            lane_mask = lane_mask.index_select(-1, idx)
            mapping = mapping.index_select(-1, idx)
            permitted = state["phase_encoding"]
        else:
            current = state["current_phase"][..., None, None]
            permitted = mapping.gather(-2, current.expand(*current.shape[:-1], mapping.shape[-1])).squeeze(-2)
            # A zero phase encoding denotes a yellow/all-red observation.
            permitted = permitted & (state["phase_encoding"].sum(dim=-1, keepdim=True) > 0)
        lane = lane * lane_mask[..., None]
        values = [torch.sigmoid(layer(lane[..., i:i + 1]))
                  for i, layer in enumerate(self.value_embeddings)]
        phase = torch.sigmoid(self.phase_embedding(permitted.long()))
        values = [phase] + values if self.phase_pairs else values + [phase]
        embedded = F.relu(self.lane_embedding(torch.cat(values, dim=-1)))
        embedded = embedded * lane_mask[..., None]
        served = mapping & lane_mask.unsqueeze(-2)
        demand = torch.einsum("bnal,bnld->bnad", served.to(embedded.dtype), embedded)
        actions = demand.shape[-2]
        left = demand.unsqueeze(-2).expand(*demand.shape[:-2], actions, actions, demand.shape[-1])
        right = demand.unsqueeze(-3).expand_as(left)
        shared = (served.unsqueeze(-2) & served.unsqueeze(-3)).sum(dim=-1)
        # LibSignal's two-demand pairs compete iff their union has three demands.
        relation = shared == 1 if self.phase_pairs else shared > 0
        competition = F.relu(self.demand_conv(torch.cat((left, right), dim=-1)))
        embedded_relation = self.relation_embedding(relation.long())
        if self.phase_pairs:
            embedded_relation = F.relu(embedded_relation)
        competition = competition * F.relu(self.relation_conv(embedded_relation))
        pairs = self.pair_value(F.relu(self.combine_conv(competition))).squeeze(-1)
        diagonal = torch.eye(actions, dtype=torch.bool, device=lane.device)
        pair_mask = (~diagonal) & state["action_mask"].unsqueeze(-2)
        q_values = (pairs * pair_mask).sum(dim=-1)
        no_opponent = ~pair_mask.any(dim=-1)
        return torch.where(no_opponent, self.single_phase_value(demand).squeeze(-1), q_values)


class NeighborAttention(nn.Module):
    """CoLight's separate query/key/value projections and mean over heads."""

    def __init__(self, hidden_dim: int, heads: int = 5) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = max(4, hidden_dim // 2)
        self.query = nn.Linear(hidden_dim, heads * self.head_dim)
        self.key = nn.Linear(hidden_dim, heads * self.head_dim)
        self.value = nn.Linear(hidden_dim, heads * self.head_dim)
        self.output = nn.Linear(self.head_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor, indices: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        b, n, _ = hidden.shape
        if indices.shape[-1] == 0:
            indices = torch.arange(n, device=hidden.device)[None, :, None].expand(b, -1, -1)
            valid = torch.ones_like(indices, dtype=torch.bool)
        else:
            # Isolated nodes fall back to self, never to a padded node zero.
            indices, valid = indices.clone(), valid.clone()
            isolated = ~valid.any(dim=-1)
            own = torch.arange(n, device=hidden.device)[None, :].expand(b, -1)
            indices[..., 0] = torch.where(isolated, own, indices[..., 0])
            valid[..., 0] |= isolated
        batch = torch.arange(b, device=hidden.device)[:, None, None]
        neighbors = hidden[batch, indices.clamp(0, n - 1)]
        q = F.relu(self.query(hidden)).reshape(b, n, self.heads, self.head_dim)
        k = F.relu(self.key(neighbors)).reshape(b, n, indices.shape[-1], self.heads, self.head_dim)
        v = F.relu(self.value(neighbors)).reshape_as(k)
        score = torch.einsum("bnhd,bnkhd->bnhk", q, k)
        score = score.masked_fill(~valid.unsqueeze(-2), -torch.inf)
        weight = score.softmax(dim=-1)
        message = torch.einsum("bnhk,bnkhd->bnhd", weight, v).mean(dim=-2)
        return F.relu(self.output(message))


class CoLightQNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, actions: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.attention = NeighborAttention(hidden_dim)
        self.action_layer = nn.Linear(hidden_dim, actions)

    def forward(self, state: TensorState) -> torch.Tensor:
        hidden = self.encoder(state["features"])
        hidden = self.attention(hidden, state["neighbor_index"], state["neighbor_mask"])
        return self.action_layer(hidden)


class IndependentQNetworks(nn.Module):
    """One parameter set per node; nodes are never treated as shared examples."""

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, state: TensorState) -> torch.Tensor:
        return torch.cat([model({key: value[:, i:i + 1] for key, value in state.items()})
                          for i, model in enumerate(self.models)], dim=1)


def build_q_network(profile: BaselineProfile, feature_dim: int, channels: int,
                    hidden_dim: int, actions: int, nodes: int) -> nn.Module:
    def one() -> nn.Module:
        if profile.algorithm in {"idqn", "dqn"}:
            return DenseQNetwork(feature_dim, hidden_dim, actions)
        if profile.algorithm == "presslight":
            if profile.layout == "canonical12":
                return CanonicalPhaseQNetwork(PressLightQNetwork(feature_dim, hidden_dim, 8))
            return PressLightQNetwork(feature_dim, hidden_dim, actions)
        if profile.algorithm in {"frap", "mplight", "advanced_mplight"}:
            return PhaseCompetitionQNetwork(channels, hidden_dim,
                                           profile.layout == "canonical12",
                                           profile.algorithm == "advanced_mplight",
                                           phase_pairs=profile.layout == "generic")
        if profile.algorithm == "colight":
            if profile.layout == "canonical12":
                return CanonicalPhaseQNetwork(CoLightQNetwork(feature_dim, hidden_dim, 8))
            return CoLightQNetwork(feature_dim, hidden_dim, actions)
        raise ValueError(f"not a Q-learning baseline: {profile.algorithm}")
    if not profile.parameter_sharing:
        if profile.algorithm == "colight":
            raise ValueError("this CoLight port requires shared graph parameters")
        return IndependentQNetworks([one() for _ in range(nodes)])
    return one()
