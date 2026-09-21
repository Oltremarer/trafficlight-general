"""v8: position-supervised cumulative effects and same-root candidate contrasts."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .formal_model import FormalEffectModel

FAMILIES = ('A_C', 'A_D', 'A_L4', 'B_L4')
HORIZON_INDICES = (5, 17, 35, 47)
SCHEDULES = {
    'A_C': {'updates': 66000, 'batch': 64, 'warmup': 1100, 'validate': 1100, 'log': 220},
    'A_D': {'updates': 66000, 'batch': 64, 'warmup': 1100, 'validate': 1100, 'log': 220},
    'A_L4': {'updates': 132000, 'batch': 32, 'warmup': 2200, 'validate': 2200, 'log': 440},
    'B_L4': {'updates': 30000, 'batch': 128, 'warmup': 500, 'validate': 1000, 'log': 200},
}


def cumulative4(values):
    """Preserve sign and every location; accumulate before selecting horizons."""
    if values.shape[-2:] != (272, 48):
        raise ValueError('Expected 272 positions and 48 five-second windows')
    if torch.is_tensor(values):
        return values.to(torch.float64).cumsum(-1)[..., list(HORIZON_INDICES)]
    return np.asarray(values, dtype=np.float64).cumsum(-1)[..., list(HORIZON_INDICES)]


def local_head(inputs):
    result = nn.Sequential(nn.Linear(inputs, 128), nn.SiLU(), nn.Dropout(.1),
        nn.Linear(128, 64), nn.SiLU(), nn.Dropout(.1), nn.Linear(64, 4))
    nn.init.zeros_(result[-1].weight)
    nn.init.zeros_(result[-1].bias)
    return result


class LocalSingleModel(FormalEffectModel):
    def __init__(self, static):
        super().__init__(static)
        self.single_head = local_head(269)
        self.single_gate = local_head(269)
        for parameter in self.pair_head.parameters():
            parameter.requires_grad_(False)


class LocalPairModel(FormalEffectModel):
    def __init__(self, static, frozen_single_state):
        super().__init__(static)
        self.load_state_dict(frozen_single_state)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.pair_head = local_head(410)
        self.pair_gate = local_head(410)
        self.pair_stage = True
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'pair_gate'):
            self.pair_gate.train(mode)
        return self

    def pair(self, encoded, roots, nodes, targets, base_phase, return_gate=False):
        state, glob, first, r1, c1 = self.query_parts(encoded, roots, nodes[:, 0], targets[:, 0], base_phase)
        _, _, second, r2, c2 = self.query_parts(encoded, roots, nodes[:, 1], targets[:, 1], base_phase)
        forward = torch.cat((state, glob, first, second, r1, r2), -1)
        reverse = torch.cat((state, glob, second, first, r2, r1), -1)
        value = (self.pair_head(forward) + self.pair_head(reverse)) * .5
        logits = (self.pair_gate(forward) + self.pair_gate(reverse)) * .5
        prediction = value * logits.sigmoid() * (c1 & c2)[:, None, None]
        return (prediction, logits) if return_gate else prediction


def local_loss(prediction, target_physical, position_scale, total_scale, logits):
    if prediction.shape != target_physical.shape or prediction.shape[-2:] != (272, 4):
        raise ValueError('Local cumulative target shape mismatch')
    location = (prediction - target_physical / position_scale).abs().mean()
    error240 = ((prediction * position_scale - target_physical)[:, :, -1].sum(1) / total_scale[-1])
    total = nn.functional.smooth_l1_loss(error240, torch.zeros_like(error240), beta=1.)
    gate = nn.functional.binary_cross_entropy_with_logits(logits, (target_physical != 0).to(logits.dtype))
    return location + .1 * total + .01 * gate


def composed_contrast_loss(single_physical, true_single_physical, incidence, scale):
    """A-only target: never accepts true joint labels or pair effects."""
    if single_physical.shape != true_single_physical.shape or len(single_physical) != 64:
        raise ValueError('One complete root with 64 single queries required')
    p = incidence @ single_physical.sum((1, 2))
    y = incidence @ true_single_physical.sum((1, 2))
    i, j = torch.triu_indices(len(p), len(p), 1, device=p.device)
    return nn.functional.smooth_l1_loss((p[i] - p[j]) / scale, (y[i] - y[j]) / scale, beta=1.)


def root_batches(root_count, seed, maximum=66000):
    if root_count < 1 or maximum < 1:
        raise ValueError('Nonempty roots and positive update budget required')
    rng = np.random.default_rng(seed)
    update, cycle = 0, 0
    while update < maximum:
        cycle += 1
        for root in rng.permutation(root_count):
            update += 1
            yield update, cycle, np.arange(64, dtype=np.int64) + int(root) * 64
            if update == maximum:
                return


def pair_groups(roots):
    groups = []
    for ri, root in enumerate(roots):
        lookup = {(int(pi), int(a), int(b)): q for q, (pi, (a, b)) in enumerate(
            zip(root['pair_query_pair_index'], root['pair_actions']))}
        expected = {(p, a, b) for p in range(120) for a in range(1, 5) for b in range(1, 5)}
        if len(lookup) != 1920 or set(lookup) != expected:
            raise ValueError('Expected all 120 pairs and 16 action combinations')
        for pi in range(120):
            groups.append([ri * 1920 + lookup[pi, a, b] for a in range(1, 5) for b in range(1, 5)])
    return np.asarray(groups, dtype=np.int64)


def pair_batches(groups, seed, maximum=30000):
    if groups.ndim != 2 or groups.shape[1] != 16 or not len(groups) or len(groups) % 8:
        raise ValueError('Eight complete sixteen-action pair groups per batch required')
    rng = np.random.default_rng(seed)
    update, cycle = 0, 0
    while update < maximum:
        cycle += 1
        order = rng.permutation(len(groups))
        for offset in range(0, len(groups), 8):
            update += 1
            yield update, cycle, groups[order[offset:offset + 8]].ravel()
            if update == maximum:
                return


def compose(incidence, effects):
    """FP64 sparse factor summation; avoid dense 65 x 1920 multiplication."""
    result = np.zeros((len(incidence), *effects.shape[1:]), dtype=np.float64)
    for i, row in enumerate(incidence):
        active = np.flatnonzero(row)
        if len(active):
            result[i] = (effects[active] * row[active].reshape(-1, *([1] * (effects.ndim - 1)))).sum(0, dtype=np.float64)
    return result


def pair_indices(root, geometry, budget):
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    if budget == 120:
        return np.arange(1920)
    if budget != 24:
        raise ValueError('Only fixed budgets 0, 24 and 120 are approved')
    adjacent = np.flatnonzero(np.asarray(geometry)[:, -1] > .5)
    if len(adjacent) != 24:
        raise ValueError('Expected 24 adjacent pairs in the source 4x4 grid')
    return np.flatnonzero(np.isin(root['pair_query_pair_index'], adjacent))
