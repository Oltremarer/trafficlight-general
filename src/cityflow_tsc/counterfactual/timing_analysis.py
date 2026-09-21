"""Read committed H240 timing outcomes once; report effects and same-panel decisions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .writer import atomic_json, atomic_npz


def waiting_windows_240(raw):
    values = np.concatenate((raw["lane_q"].astype(np.int64).sum(-1),
                             raw["intersection_nq"].astype(np.int64)[..., 1],
                             raw["boundary_pending"].astype(np.int64)), axis=-1)
    if values.shape[1:] != (240, 272):
        raise ValueError("Timing experiment requires H240, 272 locations")
    return values.reshape(len(values), 48, 5, 272).sum(2).transpose(0, 2, 1)


def panel_decision(outcomes, plan, panel):
    rows = {r["branch_id"]: r for r in plan["branches"]}
    base = outcomes[panel["baseline_id"]]
    ids = panel["candidate_ids"]
    true = np.asarray([outcomes[k].sum(dtype=np.int64) for k in ids])
    first_scores, pair_scores = [], []
    for bid in ids:
        row = rows[bid]
        if len(row["changed_intersections"]) < 2:
            first_scores.append(int(outcomes[bid].sum()))
            pair_scores.append(int(outcomes[bid].sum()))
        else:
            a, b = row["single_ids"]
            first = outcomes[a] + outcomes[b] - base
            interaction = outcomes[bid] - outcomes[a] - outcomes[b] + base
            first_scores.append(int(first.sum())); pair_scores.append(int((first + interaction).sum()))
    report = {}
    for label, scores in (("oracle_single", first_scores), ("oracle_pair", pair_scores)):
        selected = int(np.argmin(scores))
        report[label] = {"regret_vehicle_s": int(true[selected] - true.min()),
                         "chosen_branch_id": ids[selected], "optimal_ties": int((true == true.min()).sum()),
                         "optimal_choice": bool(true[selected] == true.min()),
                         "true_change_from_baseline_vehicle_s": int(true[selected] - true[0])}
    return report


def analyze_root(run, root_id):
    run = Path(run)
    output = run / "analysis" / root_id
    if (output / "summary.json").exists():
        return json.loads((output / "summary.json").read_text())
    plan = json.loads((run / "roots" / root_id / "branches.json").read_text())
    ids = [r["branch_id"] for r in plan["branches"]]
    index = {key: i for i, key in enumerate(ids)}
    values = np.empty((len(ids), 272, 48), dtype=np.int64)
    events = np.empty((len(ids), 240, 48, 2), dtype=np.int32)
    receiving = np.empty((len(ids), 240, 48), dtype=np.int16)
    speed_loss = np.empty((len(ids), 256, 48), dtype=np.float32)
    for path in sorted((run / "shards" / root_id).glob("*.npz")):
        with np.load(path) as raw:
            indices = [index[key] for key in raw["branch_ids"].tolist()]
            n = len(indices)
            values[indices] = waiting_windows_240(raw)
            events[indices] = raw["lane_events"].astype(np.int32).reshape(n, 48, 5, 240, 2).sum(2).transpose(0, 2, 1, 3)
            receiving[indices] = raw["receiving_unavailable"].astype(np.int16).reshape(n, 48, 5, 240).sum(2).transpose(0, 2, 1)
            sl = np.concatenate((raw["lane_speed_loss"], raw["intersection_speed_loss"]), axis=-1)
            speed_loss[indices] = sl.reshape(n, 48, 5, 256).sum(2).transpose(0, 2, 1)
    lookup = {key: values[i] for key, i in index.items()}
    reports = []
    all_effects, pair_metadata = [], []
    for panel_index, panel in enumerate(plan["panels"]):
        bi = index[panel["baseline_id"]]
        interactions, mechanisms = [], {"lane_enter_leave": [], "receiving_proxy": [], "speed_loss_proxy": []}
        for combo in panel["combinations"]:
            pi = index[combo["branch_id"]]; a, b = [index[k] for k in combo["single_ids"]]
            interactions.append(values[pi] - values[a] - values[b] + values[bi])
            for key, array in (("lane_enter_leave", events), ("receiving_proxy", receiving), ("speed_loss_proxy", speed_loss)):
                delta = array[pi].astype(np.float64) - array[a] - array[b] + array[bi]
                mechanisms[key].append({"absolute_interaction_sum": float(np.abs(delta).sum()),
                                         "absolute_interaction_by_window": np.abs(delta).sum(axis=(0, 2) if delta.ndim == 3 else 0).tolist()})
        effect = np.stack(interactions)
        magnitude = np.abs(effect).sum(axis=(0, 2))
        nonzero = np.count_nonzero(effect)
        reports.append({**{k: panel[k] for k in ("A", "B", "stratum", "hops", "timing", "A_window_s", "B_window_s")},
                        "combinations": len(effect), "absolute_interaction_vehicle_s": int(np.abs(effect).sum()),
                        "nonzero_elements": int(nonzero), "total_elements": int(effect.size),
                        "nonzero_fraction": float(nonzero / effect.size),
                        "absolute_interaction_by_window": np.abs(effect).sum(axis=(0, 1)).tolist(),
                        "top_location_indices": np.argsort(-magnitude, kind="stable")[:10].tolist(),
                        "cumulative_horizons": {str(h): {"absolute_interaction_vehicle_s": int(np.abs(effect[:, :, :h // 5].sum(2)).sum()),
                                                         "signed_global_interaction_vehicle_s": effect[:, :, :h // 5].sum((1, 2)).tolist()}
                                                for h in (30, 60, 90, 120, 180, 240)},
                        "decision": panel_decision(lookup, plan, panel), "mechanism_proxies": mechanisms})
        all_effects.extend(effect)
        pair_metadata.extend([[panel_index, c["phase_A"], c["phase_B"]] for c in panel["combinations"]])
    atomic_npz(output / "interaction_fields.npz", {"interaction_vehicle_s": np.stack(all_effects),
               "panel_index_phase_A_phase_B": np.asarray(pair_metadata, dtype=np.int64)})
    result = {"root_id": root_id, "horizon_s": 240, "panels": reports,
              "units": "waiting vehicle-seconds; receiving proxy lane-seconds; speed loss proxy integrated deficit-seconds",
              "limitations": ["Two-agent oracle pair reconstruction is exact by construction, not a higher-order generalization test",
                              "Only oracle factors here; no neural model performance",
                              "Receiving and speed-loss proxies explain co-occurring mechanics but do not prove causation",
                              "T1 is the pre-fixed formal setting, not chosen from these results"]}
    atomic_json(output / "summary.json", result)
    return result
