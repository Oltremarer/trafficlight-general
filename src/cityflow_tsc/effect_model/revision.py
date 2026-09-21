"""Approved loss revisions and full-request action semantics (no future inputs)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .model import EffectModel, effect_loss


def sequence_bank(base_phase, phase_masks):
    """T1: plan 0 follows reference; plans 1..4 hold phases 0..3 for 90 s."""
    base = np.asarray(base_phase, dtype=np.int64)
    reference = (base[:, None] + np.arange(8)) % 4
    requests = np.repeat(reference[:, None, :], 5, axis=1)
    for plan in range(1, 5):
        requests[:, plan, :3] = plan - 1
    features = np.empty((16, 5, 288), dtype=np.float32)
    for i in range(16):
        for plan in range(5):
            fields = [np.eye(4)[base[i]], np.eye(4)[requests[i, plan, 0]]]
            current = int(base[i])
            for target in requests[i, plan]:
                changed = target != current
                pre, post = (0 if changed else target + 1), target + 1
                fields.extend((np.eye(5)[pre], np.eye(5)[post], phase_masks[i, pre],
                               phase_masks[i, post], [float(changed)]))
                current = int(target)
            features[i, plan] = np.concatenate(fields)
    return features, requests


class RevisedEffectModel(EffectModel):
    def __init__(self, static, variant="A0", windows=36, full_sequence=False):
        super().__init__(static)
        self.variant, self.windows, self.full_sequence = variant, windows, full_sequence
        if full_sequence:
            self.action_encoder[0] = nn.Linear(288, 64)
        if windows != 36:
            self.single_head[-1] = nn.Linear(64, windows)
            self.pair_head[-1] = nn.Linear(64, windows)
        self.single_gate = None
        if variant == "A2":
            self.single_gate = nn.Sequential(nn.Linear(269, 128), nn.SiLU(), nn.Dropout(.1),
                                             nn.Linear(128, 64), nn.SiLU(), nn.Dropout(.1),
                                             nn.Linear(64, windows))
            for output in (self.single_head[-1], self.single_gate[-1]):
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)

    def encode(self, history, actions):
        result = super().encode(history, actions)
        if self.full_sequence:
            result["plan_changed"] = (actions != actions[:, :, :1]).any(dim=-1)
        return result

    def query_parts(self, encoded, roots, nodes, targets, base_phase):
        if not self.full_sequence:
            return super().query_parts(encoded, roots, nodes, targets, base_phase)
        state = encoded["objects"][roots]
        global_state = encoded["global"][roots, None, :].expand(-1, 272, -1)
        intervention = torch.cat((encoded["objects"][roots, 240 + nodes],
                                  encoded["actions"][roots, nodes, targets]), -1)[:, None].expand(-1, 272, -1)
        reference = base_phase[roots, nodes]
        first = torch.where(targets == 0, reference, targets - 1)
        relative = torch.cat((self.relative[nodes], self.permission[nodes, reference + 1, :, None],
                              self.permission[nodes, first + 1, :, None]), -1)
        return state, global_state, intervention, relative, encoded["plan_changed"][roots, nodes, targets]

    def single(self, encoded, roots, nodes, targets, base_phase, return_gate=False):
        state, glob, action, relative, changed = self.query_parts(encoded, roots, nodes, targets, base_phase)
        features = torch.cat((state, glob, action, relative), -1)
        value = self.single_head(features)
        logits = self.single_gate(features) if self.single_gate is not None else None
        prediction = value if logits is None else logits.sigmoid() * value
        prediction = prediction * changed[:, None, None]
        return (prediction, logits) if return_gate else prediction


def revision_loss(prediction_normalized, target_physical, scale, total_scale, variant, logits=None):
    if variant == "A0":
        return effect_loss(prediction_normalized, target_physical, scale, total_scale)
    local = (prediction_normalized - target_physical / scale).abs().mean()
    total = torch.nn.functional.smooth_l1_loss(
        (prediction_normalized * scale).sum((1, 2)) / total_scale,
        target_physical.sum((1, 2)) / total_scale, beta=1.)
    loss = local + .1 * total
    if variant == "A2":
        if logits is None:
            raise ValueError("A2 requires predicted gate logits, never a truth mask at inference")
        loss = loss + .01 * torch.nn.functional.binary_cross_entropy_with_logits(
            logits, (target_physical != 0).to(logits.dtype))
    return loss


def checkpoint_key(regret, joint_mae, epoch):
    return float(regret), float(joint_mae), int(epoch)
