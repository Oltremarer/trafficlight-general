"""Fixed v3 factor models, physical multiscale losses, and actual sparse inference."""
from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn

from .revision import RevisedEffectModel, sequence_bank

HORIZONS = (30, 90, 180, 240)
PAIRS = np.asarray(list(itertools.combinations(range(16), 2)), dtype=np.int64)


class FormalEffectModel(RevisedEffectModel):
    def __init__(self, static):
        super().__init__(static, variant="A2", windows=48, full_sequence=True)
        # Coordinates/hops derive from the observed static road network, not labels.
        if "rank_geometry" in static:
            geometry = np.asarray(static["rank_geometry"], dtype=np.float32)
        else:
            rel = np.asarray(static["relative"])
            geometry = np.stack([np.r_[np.abs(rel[i, 240 + j, :2]),
                                       rel[i, 240 + j, 2:4],
                                       float(np.isclose(rel[i, 240 + j, 3], 1 / 8))]
                                 for i, j in PAIRS]).astype(np.float32)
        self.register_buffer("rank_geometry", torch.as_tensor(geometry))

    def train(self, mode=True):
        super().train(mode)
        if self.pair_stage:
            # The inherited model omitted this added head: freeze its Dropout too.
            self.single_gate.eval()
        return self


def create_model(static):
    return FormalEffectModel(static)


def tensor_scales(stats, kind, device):
    return {name: torch.as_tensor(np.array(stats[kind + "_" + name], copy=True),
                                  dtype=torch.float32, device=device)
            for name in ("s5", "sp", "sj")}


def multiscale_loss(prediction_normalized, target_physical, scales, variant, logits=None):
    """Natural element means, linear physical differences, fixed train-only scales."""
    error = prediction_normalized * scales["s5"] - target_physical
    l5 = (error / scales["s5"]).abs().mean()
    cumulative = error.cumsum(-1)[..., [h // 5 - 1 for h in HORIZONS]]
    lp = nn.functional.smooth_l1_loss(cumulative / scales["sp"],
                                     torch.zeros_like(cumulative), beta=1.)
    total = cumulative.sum(1) / scales["sj"]
    lj = nn.functional.smooth_l1_loss(total, torch.zeros_like(total), beta=1.)
    lj240 = nn.functional.smooth_l1_loss(total[:, -1], torch.zeros_like(total[:, -1]), beta=1.)
    if variant == "A_ref":
        loss = l5 + .1 * lj240
    elif variant in ("A_MS", "B1"):
        loss = .1 * l5 + lp + lj
    else:
        raise ValueError("Unknown fixed loss variant: " + variant)
    if variant != "B1":
        if logits is None:
            raise ValueError("A requires gate logits during training")
        loss = loss + .01 * nn.functional.binary_cross_entropy_with_logits(
            logits, (target_physical != 0).to(logits.dtype))
    return loss


class Ranker(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(197, 64), nn.SiLU(), nn.Linear(64, 1), nn.Softplus())


def ranking_features(model, encoded, pairs=None):
    """Symmetric, label-free 197D feature for all 120 pairs in fixed order."""
    pairs = PAIRS if pairs is None else np.asarray(pairs)
    index = torch.as_tensor(pairs, device=model.rank_geometry.device)
    state = encoded["objects"][:, 240:256]
    first, second = state[:, index[:, 0]], state[:, index[:, 1]]
    glob = encoded["global"][:, None].expand(-1, len(pairs), -1)
    lookup = {tuple(p): n for n, p in enumerate(PAIRS)}
    gi = torch.as_tensor([lookup[tuple(sorted(p))] for p in pairs], device=index.device)
    geometry = model.rank_geometry[gi][None].expand(len(state), -1, -1)
    return torch.cat((first + second, first * second, glob, geometry), -1)


def tie_argmin(scores, tolerance=1e-6):
    """FP64, absolute tolerance; first frozen candidate (reference first)."""
    if torch.is_tensor(scores):
        values = scores.to(torch.float64)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite candidate score")
        return int(torch.nonzero(values <= values.min() + tolerance)[0, 0])
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite candidate score")
    return int(np.flatnonzero(values <= values.min() + tolerance)[0])


def group_mean(records, key):
    """Mean policy roots within flow, flows within cohort, then equal cohorts."""
    cohorts = {}
    for row in records:
        cohort = str(row["cohort_id"])
        cohorts.setdefault(cohort, {}).setdefault(str(row["flow_id"]), []).append(float(row[key]))
    return float(np.mean([np.mean([np.mean(values) for values in flows.values()])
                          for flows in cohorts.values()]))


def root_metadata(root):
    return {key: root[key] for key in ("root_id", "split", "cohort_id", "flow_id", "policy", "time_s")}


def decision_metrics(predicted_scores, true_scores):
    p, y = np.asarray(predicted_scores, dtype=np.float64), np.asarray(true_scores, dtype=np.float64)
    selected = tie_argmin(p)
    optimum = float(y.min())
    available = float(y[0] - optimum)
    benefit = float(y[0] - y[selected])
    return {"selected": selected, "regret": float(y[selected] - optimum),
            "optimal_choice": bool(y[selected] <= optimum + 1e-6),
            "worse_than_reference": bool(y[selected] > y[0] + 1e-6),
            "excess_wait_vs_reference": float(max(0., y[selected] - y[0])),
            "benefit_vs_reference": benefit, "oracle_available_benefit": available,
            "benefit_capture": benefit / available if available else None,
            "predicted_delta_J": float(p[selected]), "true_delta_J": float(y[selected]),
            "true_optimal_delta_J": optimum}


def prepare_inputs(root, stats, device, rebuild_inputs=False):
    if rebuild_inputs:
        from .data import history_features, normalize_history
        history = normalize_history(history_features(root["raw_history"], root["root_context"],
                                                      root["input_static"]), stats)
        bank, _ = sequence_bank(root["base_phase"], root["phase_masks"])
    else:
        history, bank = root["normalized_history"], root["action_bank"]
    return (torch.as_tensor(np.asarray(history)[None], dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(bank)[None], dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(root["base_phase"])[None], dtype=torch.long, device=device))


@torch.no_grad()
def predict_root(model, root, stats, device, mode="Pred-S", ranker=None,
                 rebuild_inputs=False, return_fields=True):
    """One encoding; select pairs BEFORE detailed prediction, at most 128/query batch.

    No truth or future reference is read. Returned factor fields cover only selected
    pairs; pair_indices maps them to the fixed 1,920-query catalogue. Score-only
    mode materializes just the selected joint field, not 65 reconstructed fields.
    """
    model.eval()
    history, bank, base = prepare_inputs(root, stats, device, rebuild_inputs)
    encoded = model.encode(history, bank)
    single_scale = tensor_scales(stats, "single", device)["s5"]
    nodes = torch.as_tensor(root["single_nodes"], dtype=torch.long, device=device).reshape(-1)
    actions = torch.as_tensor(root["single_actions"], dtype=torch.long, device=device).reshape(-1)
    single = model.single(encoded, torch.zeros_like(nodes), nodes, actions, base) * single_scale
    s_incidence = torch.as_tensor(root["s_incidence"], device=device, dtype=torch.float64)
    scores = s_incidence @ single.sum((1, 2), dtype=torch.float64)
    rank_scores, selected_pairs = None, np.empty((0, 2), dtype=np.int64)
    selected_query_indices = np.empty(0, dtype=np.int64)
    pair, p_incidence = None, None
    if mode != "Pred-S":
        pair_nodes = np.asarray(root["pair_unique_nodes"])
        if mode in ("All-120", "Pred-SP"):
            pair_index = np.arange(120)
        elif mode == "Adj-24":
            pair_index = np.flatnonzero(model.rank_geometry[:, -1].cpu().numpy() > .5)
            if len(pair_index) != 24:
                raise ValueError("Expected 24 physical adjacent pairs in this grid")
        elif mode in ("Top-32", "Top-24", "Top-8"):
            if ranker is None:
                raise ValueError("Sparse learned selection requires the separately trained R")
            ranker.eval()
            rank_scores = ranker(ranking_features(model, encoded)).reshape(-1)
            # Stable descending sort preserves lexicographic pair order for ties.
            rank_values = rank_scores.cpu().numpy()
            if not np.isfinite(rank_values).all():
                raise FloatingPointError("Nonfinite rank scores")
            pair_index = np.argsort(-rank_values, kind="stable")[:int(mode.split("-")[1])]
        else:
            raise ValueError("Unknown inference mode: " + mode)
        selected_pairs = pair_nodes[pair_index]
        selected_query_indices = np.flatnonzero(np.isin(root["pair_query_pair_index"], pair_index))
        n = torch.as_tensor(np.asarray(root["pair_nodes"])[selected_query_indices], dtype=torch.long, device=device)
        a = torch.as_tensor(np.asarray(root["pair_actions"])[selected_query_indices], dtype=torch.long, device=device)
        scale = tensor_scales(stats, "pair", device)["s5"]
        chunks = []
        for offset in range(0, len(n), 128):
            ni, ai = n[offset:offset + 128], a[offset:offset + 128]
            chunks.append(model.pair(encoded, torch.zeros(len(ni), dtype=torch.long, device=device), ni, ai, base) * scale)
        pair = torch.cat(chunks)
        p_incidence = torch.as_tensor(np.asarray(root["p_incidence"])[:, selected_query_indices],
                                      device=device, dtype=torch.float64)
        scores = scores + p_incidence @ pair.sum((1, 2), dtype=torch.float64)
    selected = tie_argmin(scores)
    joint = None
    if return_fields:
        joint = s_incidence @ single.flatten(1).to(torch.float64)
        if pair is not None:
            joint.add_(p_incidence @ pair.flatten(1).to(torch.float64))
        joint = joint.reshape(-1, 272, 48)
        selected_field = joint[selected]
    else:
        selected_field = (single * s_incidence[selected, :, None, None]).sum(0)
        if pair is not None:
            selected_field = selected_field + (pair * p_incidence[selected, :, None, None]).sum(0)
    return {"single": single, "pair": pair, "pair_indices": selected_query_indices,
            "joint": joint, "scores": scores, "selected": selected, "selected_field": selected_field,
            "selected_pairs": selected_pairs, "rank_scores": rank_scores}


@torch.no_grad()
def validation(model, roots, stats, device, include_pairs=False):
    records = []
    for root in roots:
        output = predict_root(model, root, stats, device, "All-120" if include_pairs else "Pred-S")
        truth = torch.as_tensor(np.asarray(root["joint"]), device=device)
        record = root_metadata(root)
        record.update(decision_metrics(output["scores"].cpu().numpy(),
                                       truth.sum((1, 2), dtype=torch.float64).cpu().numpy()))
        record["joint_mae"] = float((output["joint"] - truth).abs().mean(dtype=torch.float64))
        records.append(record)
    return {"regret": group_mean(records, "regret"), "joint_mae": group_mean(records, "joint_mae"),
            "optimal_choice_rate": group_mean(records, "optimal_choice"), "records": records}
