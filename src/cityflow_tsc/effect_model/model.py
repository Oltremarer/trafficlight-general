from __future__ import annotations

import torch
from torch import nn

from . import GROUPS, INPUT_DIMS


class RelationalLayer(nn.Module):
    def __init__(self, hidden=64, relations=8):
        super().__init__()
        self.messages = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(relations))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, adjacency):
        message = sum(torch.matmul(a, linear(h)) for a, linear in zip(adjacency, self.messages))
        return self.norm(h + torch.nn.functional.silu(message))


def head(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Dropout(.1),
                         nn.Linear(128, 64), nn.SiLU(), nn.Dropout(.1), nn.Linear(64, 36))


class EffectModel(nn.Module):
    """Shared-parameter graph/history encoder and signed fixed-horizon effects."""

    def __init__(self, static):
        super().__init__()
        for key in ("adjacency", "relative", "permission"):
            self.register_buffer(key, torch.as_tensor(static[key], dtype=torch.float32))
        self.projections = nn.ModuleList(nn.Linear(dim, 64) for dim in INPUT_DIMS)
        self.gru = nn.GRU(64, 64, num_layers=1, batch_first=True)
        self.graph = nn.ModuleList([RelationalLayer(), RelationalLayer()])
        self.action_encoder = nn.Sequential(nn.Linear(218, 64), nn.SiLU(), nn.Linear(64, 64))
        self.single_head = head(64 * 4 + 13)
        self.pair_head = head(64 * 6 + 26)
        self.pair_stage = False

    def encode(self, history, actions):
        """Only pre-root histories and prescribed control schedules are accepted."""
        projected = torch.cat([linear(history[:, :, a:b, :dim])
                               for (a, b, _), dim, linear in zip(GROUPS, INPUT_DIMS, self.projections)], dim=2)
        b, t, o, h = projected.shape
        _, final = self.gru(projected.transpose(1, 2).reshape(b * o, t, h))
        state = final[0].reshape(b, o, h)
        for layer in self.graph:
            state = layer(state, self.adjacency)
        return {"objects": state, "global": state.mean(dim=1), "actions": self.action_encoder(actions)}

    def query_parts(self, encoded, roots, nodes, targets, base_phase):
        state = encoded["objects"][roots]
        global_state = encoded["global"][roots, None, :].expand(-1, 272, -1)
        control_state = encoded["objects"][roots, 240 + nodes]
        action = encoded["actions"][roots, nodes, targets]
        intervention = torch.cat((control_state, action), dim=-1)[:, None, :].expand(-1, 272, -1)
        reference = base_phase[roots, nodes]
        relative = torch.cat((self.relative[nodes], self.permission[nodes, reference + 1, :, None],
                              self.permission[nodes, targets + 1, :, None]), dim=-1)
        return state, global_state, intervention, relative, reference != targets

    def single(self, encoded, roots, nodes, targets, base_phase):
        state, global_state, action, relative, changed = self.query_parts(encoded, roots, nodes, targets, base_phase)
        return self.single_head(torch.cat((state, global_state, action, relative), dim=-1)) * changed[:, None, None]

    def pair(self, encoded, roots, nodes, targets, base_phase):
        state, global_state, first, r1, c1 = self.query_parts(encoded, roots, nodes[:, 0], targets[:, 0], base_phase)
        _, _, second, r2, c2 = self.query_parts(encoded, roots, nodes[:, 1], targets[:, 1], base_phase)
        forward = torch.cat((state, global_state, first, second, r1, r2), dim=-1)
        reverse = torch.cat((state, global_state, second, first, r2, r1), dim=-1)
        result = (self.pair_head(forward) + self.pair_head(reverse)) * .5
        return result * (c1 & c2)[:, None, None]

    def start_pair_stage(self):
        self.pair_stage = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.pair_head.parameters():
            parameter.requires_grad_(True)
        nn.init.zeros_(self.pair_head[-1].weight)
        nn.init.zeros_(self.pair_head[-1].bias)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if self.pair_stage:
            # Frozen modules must remain in eval, including single-head dropout.
            for module in (self.projections, self.gru, self.graph, self.action_encoder, self.single_head):
                module.eval()
            self.pair_head.train(mode)
        return self


def effect_loss(prediction, target_physical, scale, total_scale):
    target = target_physical / scale
    pointwise = torch.nn.functional.smooth_l1_loss(prediction, target, beta=1., reduction="none")
    group_losses = []
    for a, b, _ in GROUPS:
        group = pointwise[:, a:b]
        nonzero = target_physical[:, a:b] != 0
        group_losses.append(.5 * group.mean() + .5 * group[nonzero].mean() if nonzero.any() else group.mean())
    total_prediction = (prediction * scale).sum(dim=(1, 2)) / total_scale
    total_target = target_physical.sum(dim=(1, 2)) / total_scale
    total = torch.nn.functional.smooth_l1_loss(total_prediction, total_target, beta=1.)
    return torch.stack(group_losses).mean() + .1 * total

