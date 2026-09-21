"""Locked-checkpoint v3 test evaluation; oracle factors never choose checkpoints."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .collect_counterfactual import stamp
from .counterfactual.writer import atomic_json, sha256
from .effect_model.formal_model import decision_metrics, group_mean, predict_root, root_metadata

MODES = ("Pred-S", "All-120", "Adj-24", "Top-32", "Top-24", "Top-8")
HORIZONS = (30, 60, 90, 120, 180, 240)


def compose(incidence, factors):
    """Sparse sums in physical FP64; baseline row remains identically zero."""
    output = np.zeros((len(incidence), *factors.shape[1:]), dtype=np.float64)
    for k, row in enumerate(incidence):
        indices = np.flatnonzero(row)
        if len(indices):
            output[k] = np.asarray(factors[indices]).sum(0, dtype=np.float64)
    return output


def errors(prediction, truth):
    p, y = np.asarray(prediction), np.asarray(truth)
    d = p.astype(np.float64) - y
    mask = y != 0
    numerator = float(np.abs(d).sum(dtype=np.float64))
    denominator = float(np.abs(y.astype(np.float64)).sum(dtype=np.float64))
    return {"mae": float(np.abs(d).mean()), "bias": float(d.mean()),
            "absolute_error_sum": numerator, "absolute_effect_sum": denominator,
            "effect_l1_relative": numerator / denominator if denominator else None,
            "zero_fraction": float((~mask).mean()),
            "zero_mae": float(np.abs(d[~mask]).mean()) if (~mask).any() else None,
            "nonzero_mae": float(np.abs(d[mask]).mean()) if mask.any() else None,
            "nonzero_sign_accuracy": float((np.sign(p[mask]) == np.sign(y[mask])).mean()) if mask.any() else None}


def field_metrics(prediction, truth):
    result = errors(prediction, truth)
    for start, end, name in ((0, 240, "lane"), (240, 256, "intersection"), (256, 272, "boundary")):
        result[name] = errors(prediction[:, start:end], truth[:, start:end])
    result["horizons"] = {}
    for h in HORIZONS:
        p = np.asarray(prediction)[:, :, :h // 5].sum(-1, dtype=np.float64)
        y = np.asarray(truth)[:, :, :h // 5].sum(-1, dtype=np.float64)
        result["horizons"][str(h)] = {"location": errors(p, y), "global": errors(p.sum(-1), y.sum(-1))}
    return result


def joint_metrics(prediction, root):
    truth = root["joint"]
    result = {"field": field_metrics(prediction, truth), "decisions": {}, "slices": {}}
    for h in HORIZONS:
        p = prediction[:, :, :h // 5].sum((1, 2), dtype=np.float64)
        y = truth[:, :, :h // 5].sum((1, 2), dtype=np.float64)
        result["decisions"][str(h)] = decision_metrics(p, y)
    result.update(result["decisions"]["240"])
    sizes = np.asarray(root["joint_sizes"])
    groups = np.asarray(root["joint_groups"])
    for label, mask in [(f"size_{s}", sizes == s) for s in (3, 4, 8, 16)] + [
            ("group_" + str(g), groups == g) for g in np.unique(groups) if str(g) not in ("baseline", "reference")]:
        if not mask.any():
            continue
        # Error excludes baseline. Slice decisions include reference plus this subset.
        indices = np.r_[0, np.flatnonzero(mask & (sizes > 0))]
        result["slices"][label] = {"field": field_metrics(prediction[mask], truth[mask]),
            "decision": decision_metrics(prediction[indices].sum((1, 2), dtype=np.float64),
                                          truth[indices].sum((1, 2), dtype=np.float64))}
    return result


def flattened(record, prefix=""):
    for key, value in record.items():
        name = prefix + key
        if isinstance(value, dict):
            yield from flattened(value, name + ".")
        elif isinstance(value, (float, int, bool)) or value is None:
            yield name, value


def aggregate(records):
    """No branch-level pseudoreplication; optional metrics retain their coverage."""
    keys = sorted({k for row in records for k, _ in flattened(row["metrics"])})
    values = [dict(flattened(row["metrics"])) for row in records]
    result = {}
    for key in keys:
        if key.endswith("selected"):
            continue
        valid = [{**r, "value": v[key]} for r, v in zip(records, values)
                 if key in v and v[key] is not None]
        result[key] = {"mean": group_mean(valid, "value") if valid else None,
                       "root_count": len(valid), "cohort_count": len({r["cohort_id"] for r in valid})}
    return result


def metadata(root):
    return {**root_metadata(root), **{k: root[k] for k in ("source_id", "profile") if k in root}}


def action_opportunity(root):
    """Describe decision information without dropping ties or zero-effect roots."""
    truth = np.asarray(root["joint"])
    scores = truth.sum((1, 2), dtype=np.float64)
    return {**metadata(root), "candidate_count": len(scores),
            "score_range_vehicle_seconds": float(scores.max() - scores.min()),
            "oracle_benefit_vehicle_seconds": float(scores[0] - scores.min()),
            "all_scores_tied": bool(scores.max() - scores.min() <= 1e-6),
            "all_joint_fields_zero": bool(not np.any(truth)),
            "nonreference_zero_field_fraction": float(np.mean(~np.any(truth[1:] != 0, axis=(1, 2))))}


def verify_checkpoint_lock(run):
    run = Path(run)
    lock = json.loads((run / "checkpoints_locked.json").read_text())
    files = {**lock["checkpoints"], "selected_A.json": lock["selected_A_sha256"],
             "derived/normalization.npz": lock["normalization_sha256"],
             "derived/pair_normalization.npz": lock["pair_normalization_sha256"]}
    for relative, expected in files.items():
        if sha256(run / relative) != expected:
            raise IOError("Locked model or statistics changed: " + relative)
    repair_path = run / "test_repair.json"
    selection = json.loads((run / "selection.json").read_text())
    from .effect_model.formal_data import TEST_REPAIR_SCHEMA
    if selection.get("schema") == TEST_REPAIR_SCHEMA and not repair_path.exists():
        raise IOError("Test-time repair provenance is missing")
    if repair_path.exists():
        repair = json.loads(repair_path.read_text())
        frozen = {**repair["reused_files"], "selection.json": repair["selection_sha256"],
                  "protocol.json": repair["protocol_sha256"]}
        for relative, expected in frozen.items():
            if sha256(run / relative) != expected:
                raise IOError("Frozen test-time repair input changed: " + relative)


def record_result(directory, records, root, method, seed, prediction, *, single=None, pair=None, pair_indices=None):
    metrics = joint_metrics(prediction, root)
    if single is not None:
        metrics["single"] = field_metrics(single, root["single"])
    if pair is not None:
        indices = np.arange(1920) if pair_indices is None else pair_indices
        metrics["pair"] = field_metrics(pair, root["pair"][indices])
    row = {**metadata(root), "method": method, "seed": seed, "metrics": metrics}
    records.append(row)
    atomic_json(directory / "roots" / root["root_id"] / f"{method}_seed_{seed}.json", row)
    return row


def evaluate(run):
    from .effect_model.formal_data import load_data
    from .train_formal_effects import load_a, load_b, load_ranker
    run = Path(run)
    locked = run / "checkpoints_locked.json"
    if not locked.exists():
        raise RuntimeError("All A/B/R checkpoints must be locked before test access")
    verify_checkpoint_lock(run)
    directory = run / "evaluation"; directory.mkdir(exist_ok=True)
    if (directory / "summary.json").exists():
        return json.loads((directory / "summary.json").read_text())
    started = time.perf_counter()
    roots, static, stats = load_data(run, include_pairs=True, splits=("test",))
    if len(roots) != 12 or len({r["cohort_id"] for r in roots}) != 3:
        raise ValueError("Expected exactly twelve test roots in three original cohorts")
    device = torch.device("cuda")
    records = []
    # Oracle methods use test truth only now; none feeds model/checkpoint selection.
    for root in roots:
        true_s = compose(root["s_incidence"], root["single"])
        true_p = compose(root["p_incidence"], root["pair"])
        for method, prediction in (("Zero", np.zeros_like(true_s)), ("True-S", true_s), ("True-SP", true_s + true_p)):
            record_result(directory, records, root, method, None, prediction)
    for seed in (42, 43, 44):
        for variant in ("A_ref", "A_MS"):
            model, meta = load_a(run, seed, device, variant=variant)
            for root in roots:
                output = predict_root(model, root, stats, device)
                record_result(directory, records, root, variant, seed, output["joint"].cpu().numpy(),
                              single=output["single"].cpu().numpy())
            del model
        model, meta = load_b(run, seed, device)
        ranker, rank_meta = load_ranker(run, seed, device)
        for root in roots:
            true_s = compose(root["s_incidence"], root["single"])
            true_p = compose(root["p_incidence"], root["pair"])
            for mode in MODES:
                output = predict_root(model, root, stats, device, mode, ranker)
                pair = output["pair"].cpu().numpy() if output["pair"] is not None else None
                single = output["single"].cpu().numpy()
                row = record_result(directory, records, root, mode, seed, output["joint"].cpu().numpy(),
                                    single=single, pair=pair, pair_indices=output["pair_indices"])
                indices = output["pair_indices"]
                # Post-hoc truth energy is evaluation only, never a selector input.
                total_energy = float(np.abs(root["pair"]).sum(dtype=np.float64))
                kept_energy = float(np.abs(root["pair"][indices]).sum(dtype=np.float64)) if len(indices) else 0.
                selected_pair_ids = np.unique(np.asarray(root["pair_query_pair_index"])[indices])
                z_total = float(np.asarray(root["rank_targets"]).sum(dtype=np.float64))
                z_kept = float(np.asarray(root["rank_targets"])[selected_pair_ids].sum(dtype=np.float64))
                row["metrics"]["sparsity"] = {"pairs": len(output["selected_pairs"]),
                    "detailed_pair_queries": len(indices), "true_abs_field_energy_kept": kept_energy,
                    "true_abs_field_energy_fraction": kept_energy / total_energy if total_energy else None,
                    "true_z_kept": z_kept, "true_z_omitted": z_total - z_kept,
                    "true_z_fraction": z_kept / z_total if z_total else None}
                atomic_json(directory / "roots" / root["root_id"] / f"{mode}_seed_{seed}.json", row)
                if mode == "All-120":
                    pred_s = compose(root["s_incidence"], single)
                    pred_p = compose(root["p_incidence"], pair)
                    record_result(directory, records, root, "Pred-S+True-P", seed, pred_s + true_p)
                    record_result(directory, records, root, "True-S+Pred-P", seed, true_s + pred_p)
                    distances = np.asarray(static["relative"])[root["pair_nodes"][:, 0], 240 + root["pair_nodes"][:, 1], 3] * 8
                    row["metrics"]["pair_distance"] = {name: field_metrics(pair[mask], root["pair"][mask])
                        for name, mask in (("adjacent", distances == 1), ("two_hop", distances == 2), ("far", distances > 2)) if mask.any()}
                    atomic_json(directory / "roots" / root["root_id"] / f"{mode}_seed_{seed}.json", row)
            atomic_json(directory / "progress.json", {"stage": "evaluating", "seed": seed, "root_id": root["root_id"],
                                                       "records": len(records), "updated_at": stamp()})
        del model, ranker
        torch.cuda.empty_cache()
    grouped = defaultdict(list)
    for row in records:
        grouped[row["method"], row["seed"]].append(row)
    summaries = []
    for (method, seed), rows in grouped.items():
        summary = {"method": method, "seed": seed, "metrics": aggregate(rows), "cohorts": {}}
        for cohort in sorted({r["cohort_id"] for r in rows}):
            summary["cohorts"][cohort] = aggregate([r for r in rows if r["cohort_id"] == cohort])
        summary["slices"] = {field: {str(value): aggregate([r for r in rows if r.get(field) == value])
            for value in sorted({r[field] for r in rows if field in r})}
            for field in ("source_id", "profile", "policy", "time_s")}
        summary["worst_regret_root"] = max(rows, key=lambda r: r["metrics"]["regret"])["root_id"]
        summaries.append(summary)
    seed_summary = {}
    for method in sorted({r["method"] for r in summaries}):
        rows = [r for r in summaries if r["method"] == method]
        seed_summary[method] = {key: {"mean": float(np.mean(v)),
            "training_seed_std": float(np.std(v, ddof=1)) if len(v) > 1 else None, "seeds": len(v)}
            for key in rows[0]["metrics"]
            if (v := [r["metrics"][key]["mean"] for r in rows if r["metrics"][key]["mean"] is not None])}
    # Per-root deltas are reported alongside averages, not converted to branch CIs.
    paired = []
    for seed in (42, 43, 44):
        by = {(r["root_id"], r["method"]): r for r in records if r["seed"] == seed}
        for root in roots:
            rid = root["root_id"]
            for first, second in (("A_MS", "A_ref"), ("All-120", "Pred-S"), ("Top-24", "Adj-24")):
                paired.append({**metadata(root), "seed": seed, "comparison": first + " minus " + second,
                    "regret_delta": by[rid, first]["metrics"]["regret"] - by[rid, second]["metrics"]["regret"]})
    result = {"stage": "complete", "finished_at": stamp(), "wall_s": time.perf_counter() - started,
              "checkpoint_lock_sha256": sha256(locked), "test_roots": 12, "test_cohorts": 3,
              "records": len(records), "summaries": summaries, "seed_summary": seed_summary,
              "paired_root_differences": paired, "oracle_caution": "True-SP residual is a 3+ joint test, not a learned model score",
              "uncertainty": "Seed SD measures training randomness; only three independent test cohorts, no branch-level CI"}
    result["action_opportunity"] = [action_opportunity(root) for root in roots]
    result["test_protocol"] = ("time-corrected test after inspecting original test; no model refitting"
                               if (run / "test_repair.json").exists() else "original_v3")
    atomic_json(directory / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    evaluate(args.run_dir)


if __name__ == "__main__":
    main()
