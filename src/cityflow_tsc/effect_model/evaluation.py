from __future__ import annotations

import time

import numpy as np
import torch

from . import GROUPS
from .data import history_features, normalize_history, action_bank


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict_factors(model, root, scales, device, include_pairs, batch_size=128):
    """Encode once and reuse 48/1080 factors; never accept a future baseline."""
    model.eval()
    sync(device);start = time.perf_counter()
    history_array = normalize_history(history_features(root["raw_history"], root["root_context"],
                                                       root["input_static"]), root["input_stats"])
    action_array = action_bank(root["base_phase"], root["phase_masks"])
    processed = time.perf_counter()
    history = torch.as_tensor(history_array[None], device=device)
    actions = torch.as_tensor(action_array[None], device=device)
    base = torch.as_tensor(root["base_phase"][None], device=device)
    sync(device);converted = time.perf_counter()
    encoded = model.encode(history, actions)
    sync(device);encoded_at = time.perf_counter()
    outputs = []
    for kind, key in enumerate(("single", "pair") if include_pairs else ("single",)):
        nodes = torch.as_tensor(root[key + "_nodes"], device=device)
        targets = torch.as_tensor(root[key + "_actions"], device=device)
        chunks = []
        for offset in range(0, len(nodes), batch_size):
            n, a = nodes[offset:offset + batch_size], targets[offset:offset + batch_size]
            root_indices = torch.zeros(len(n), dtype=torch.long, device=device)
            if key == "single":
                prediction = model.single(encoded, root_indices, n[:, 0], a[:, 0], base)
            else:
                prediction = model.pair(encoded, root_indices, n, a, base)
            chunks.append(prediction * scales[kind])
        outputs.append(torch.cat(chunks))
    sync(device);predicted_at = time.perf_counter()
    timing = {"input_preprocessing_s": processed - start, "input_transfer_s": converted - processed,
              "encode_s": encoded_at - converted,
              "factor_prediction_s": predicted_at - encoded_at}
    return outputs[0], outputs[1] if include_pairs else None, timing


def compose(single, pair, root, device):
    si = torch.as_tensor(root["s_incidence"], device=device)
    result = si @ single.flatten(1)
    if pair is not None:
        pi = torch.as_tensor(root["p_incidence"], device=device)
        result = result + pi @ pair.flatten(1)
    return result.reshape(-1, 272, 36)


@torch.no_grad()
def validation_score(model, roots, scales, device, include_pairs):
    errors = []
    for root in roots:
        single, pair, _ = predict_factors(model, root, scales, device, include_pairs)
        prediction = compose(single, pair, root, device)
        target = torch.as_tensor(root["joint"], device=device)
        errors.append(float((prediction - target).abs().mean(dtype=torch.float64)))
    return float(np.mean(errors)), errors


def error_stats(prediction, target):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    error = p - y
    nz = y != 0
    numerator = np.abs(error).sum()
    denominator = np.abs(y).sum()
    return {"mae_vehicle_s": float(np.abs(error).mean()),
            "rmse_vehicle_s": float(np.sqrt(np.mean(error ** 2))),
            "effect_l1_relative": float(numerator / denominator) if denominator else None,
            "absolute_error_sum": float(numerator), "absolute_effect_sum": float(denominator),
            "nonzero_target_elements": int(nz.sum()), "zero_target_elements": int((~nz).sum()),
            "nonzero_mae": float(np.abs(error[nz]).mean()) if nz.any() else None,
            "zero_mae": float(np.abs(error[~nz]).mean()) if (~nz).any() else None}


def joint_metrics(prediction, root):
    target = root["joint"]
    result = {"all": error_stats(prediction, target), "groups": {}, "horizons": {}, "action_sizes": {}}
    for a, b, name in GROUPS:
        result["groups"][name] = error_stats(prediction[:, a:b], target[:, a:b])
    for horizon in (30, 60, 120, 180):
        # Sum windows for each location first; never cancel errors across locations.
        result["horizons"][str(horizon)] = error_stats(
            prediction[:, :, :horizon // 5].sum(axis=2, dtype=np.float64),
            target[:, :, :horizon // 5].sum(axis=2, dtype=np.float64))
    for size in sorted(set(root["joint_sizes"].tolist())):
        mask = root["joint_sizes"] == size
        result["action_sizes"][str(size)] = error_stats(prediction[mask], target[mask])
    true_scores = target.sum(axis=(1, 2), dtype=np.float64)
    scores = prediction.sum(axis=(1, 2), dtype=np.float64)
    chosen = int(np.argmin(scores))
    optimum = float(true_scores.min())
    true_ties = np.flatnonzero(np.isclose(true_scores, optimum, rtol=0., atol=1e-6))
    result["decision"] = {"candidate_count": len(scores), "chosen_branch_id": str(root["joint_ids"][chosen]),
                          "regret_vehicle_s": float(true_scores[chosen] - optimum),
                          "optimal_choice": bool(chosen in true_ties), "true_optimal_ties": len(true_ties),
                          "predicted_minimum_ties": int(np.isclose(scores, scores.min(), rtol=0., atol=1e-6).sum()),
                          "predicted_delta_J": float(scores[chosen]), "true_delta_J": float(true_scores[chosen]),
                          "true_optimal_delta_J": optimum}
    return result


@torch.no_grad()
def evaluate(model, roots, scales, device, include_pairs, output, seed, label):
    records = []
    for root in roots:
        single, pair, timing = predict_factors(model, root, scales, device, include_pairs)
        sync(device);start = time.perf_counter()
        joint = compose(single, pair, root, device)
        # Online choice needs only predicted differences: no Y0 or true outcomes.
        selected = int(joint.sum(dim=(1, 2), dtype=torch.float64).argmin())
        sync(device);timing["compose_and_select_s"] = time.perf_counter() - start
        start = time.perf_counter()
        joint_np = joint.cpu().numpy()
        single_np = single.cpu().numpy()
        pair_np = pair.cpu().numpy() if pair is not None else np.zeros_like(root["pair"])
        timing["prediction_readback_s"] = time.perf_counter() - start
        timing["online_total_s"] = sum(timing.values())
        record = {k: root[k] for k in ("root_id", "flow_id", "policy", "time_s", "split", "backlog")}
        record.update(seed=seed, model=label, single=error_stats(single_np, root["single"]),
                      pair=error_stats(pair_np, root["pair"]), joint=joint_metrics(joint_np, root), timing=timing)
        record["stored_simulator_joint_restore_rollout_s"] = root["simulator_joint_restore_rollout_s"]
        if record["joint"]["decision"]["chosen_branch_id"] != str(root["joint_ids"][selected]):
            raise RuntimeError("Online score and reported selection differ")
        np.savez_compressed(output / f'{root["root_id"]}_{label}.npz', single=single_np, pair=pair_np,
                            joint=joint_np, joint_ids=root["joint_ids"])
        records.append(record)
    return records


def oracle_metrics(roots, device):
    records = []
    for root in roots:
        single = torch.as_tensor(root["single"], device=device)
        pair = torch.as_tensor(root["pair"], device=device)
        for name, interaction in (("oracle_single", None), ("oracle_pair", pair)):
            prediction = compose(single, interaction, root, device).cpu().numpy()
            records.append({**{k: root[k] for k in ("root_id", "flow_id", "policy", "time_s", "split")},
                            "model": name, "joint": joint_metrics(prediction, root)})
    return records


def aggregate_records(records):
    """Equal roots within flow, then equal flows; also expose every root record."""
    summary = {}
    for split in sorted({r["split"] for r in records}):
        subset = [r for r in records if r["split"] == split]
        flows = {}
        for flow in sorted({r["flow_id"] for r in subset}):
            rows = [r for r in subset if r["flow_id"] == flow]
            flows[flow] = {"roots": len(rows),
                           "joint_mae_vehicle_s": float(np.mean([r["joint"]["all"]["mae_vehicle_s"] for r in rows])),
                           "regret_vehicle_s": float(np.mean([r["joint"]["decision"]["regret_vehicle_s"] for r in rows])),
                           "optimal_choice_rate": float(np.mean([r["joint"]["decision"]["optimal_choice"] for r in rows]))}
        summary[split] = {"by_flow": flows, **{key: float(np.mean([f[key] for f in flows.values()]))
                                             for key in ("joint_mae_vehicle_s", "regret_vehicle_s", "optimal_choice_rate")}}
    return summary
