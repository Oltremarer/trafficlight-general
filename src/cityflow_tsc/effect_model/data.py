from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..counterfactual.writer import atomic_json, atomic_npz, sha256
from . import GROUPS, INPUT_DIMS


def root_split(root):
    if root["flow_id"] == "train_069" and root["time_s"] in (600, 1800, 3000):
        return "development_holdout"
    if root["flow_id"] in ("train_036", "train_012"):
        if root["time_s"] in (600, 1800):
            return "train"
        if root["time_s"] == 3000:
            return "validation"
    raise ValueError("Root is outside the authorized development split")


def waiting_windows(raw):
    """Output [branch, object, window], summing raw ticks in integer units."""
    q = raw["lane_q"].astype(np.int64).sum(axis=-1)
    nq = raw["intersection_nq"].astype(np.int64)[..., 1]
    b = raw["boundary_pending"].astype(np.int64)
    values = np.concatenate((q, nq, b), axis=-1)
    if values.shape[1:] != (180, 272):
        raise ValueError(f"Unexpected outcome shape: {values.shape}")
    return values.reshape(len(values), 36, 5, 272).sum(axis=2).transpose(0, 2, 1)


def effect_labels(outcomes, plan):
    """Compute integer single/pair effects before any scaling or float casting."""
    rows = plan["branches"]
    ids = {b["branch_id"]: i for i, b in enumerate(rows)}
    base = next(i for i, b in enumerate(rows) if not b["changed_intersections"])
    singles = [i for i, b in enumerate(rows) if len(b["changed_intersections"]) == 1]
    pairs = [i for i, b in enumerate(rows) if len(b["changed_intersections"]) == 2]
    joints = [i for i, b in enumerate(rows) if len(b["changed_intersections"]) >= 3]
    if len(singles) != 48 or len(pairs) != 1080:
        raise ValueError("Expected complete single and all-pair factors")
    y = np.asarray(outcomes, dtype=np.int64)
    single = y[singles] - y[base]
    pair = np.stack([y[i] - y[ids[rows[i]["single_ids"][0]]]
                     - y[ids[rows[i]["single_ids"][1]]] + y[base] for i in pairs])
    single_index = {rows[i]["branch_id"]: k for k, i in enumerate(singles)}
    pair_index = {rows[i]["branch_id"]: k for k, i in enumerate(pairs)}
    s_incidence = np.zeros((len(joints), len(singles)), dtype=np.float32)
    p_incidence = np.zeros((len(joints), len(pairs)), dtype=np.float32)
    for k, i in enumerate(joints):
        s_incidence[k, [single_index[j] for j in rows[i]["single_ids"]]] = 1
        p_incidence[k, [pair_index[j] for j in rows[i]["pair_ids"]]] = 1

    def queries(indices):
        nodes = [rows[i]["changed_intersections"] for i in indices]
        actions = [[rows[i]["requests"][0][j] for j in ns] for i, ns in zip(indices, nodes)]
        return np.asarray(nodes, dtype=np.int64), np.asarray(actions, dtype=np.int64)

    sn, sa = queries(singles)
    pn, pa = queries(pairs)
    return {"single": single.astype(np.float32), "pair": pair.astype(np.float32),
            "joint": (y[joints] - y[base]).astype(np.float32),
            "baseline": y[base].astype(np.float32), "single_nodes": sn, "single_actions": sa,
            "pair_nodes": pn, "pair_actions": pa, "s_incidence": s_incidence, "p_incidence": p_incidence,
            "joint_sizes": np.asarray([len(rows[i]["changed_intersections"]) for i in joints]),
            "joint_ids": np.asarray([rows[i]["branch_id"] for i in joints], dtype="U24")}


def build_static(geometry, roadnet):
    g = geometry
    lane_ids, nodes, boundaries = g["lane_ids"], g["intersection_ids"], g["boundary_road_ids"]
    if (len(lane_ids), len(nodes), len(boundaries)) != (240, 16, 16):
        raise ValueError("This fixed contract requires 240 lanes, 16 intersections, 16 boundaries")
    ni = {v: i for i, v in enumerate(nodes)}
    roads = {r["id"]: r for r in roadnet["roads"]}
    intersections = {n["id"]: n for n in roadnet["intersections"]}
    adj = np.zeros((8, 272, 272), dtype=np.float32)
    coords = np.zeros((272, 2), dtype=np.float32)
    static = np.zeros((272, 2), dtype=np.float32)
    incoming = np.zeros((16, 272), dtype=np.float32)
    outgoing = np.zeros_like(incoming)
    attached = np.zeros_like(incoming)
    anchors = [[] for _ in range(272)]

    def edge(rel, src, dst):
        adj[rel, dst, src] = 1

    for li, rid in enumerate(g["lane_road_ids"]):
        road = roads[rid]
        coords[li] = np.mean([[p["x"], p["y"]] for p in road["points"]], axis=0)
        static[li] = g["lane_lengths_m"][li], g["lane_speed_limits_mps"][li]
        a, b = road["startIntersection"], road["endIntersection"]
        if a in ni:
            i = ni[a]; edge(2, 240 + i, li); edge(3, li, 240 + i)
            outgoing[i, li] = 1; anchors[li].append(i)
        if b in ni:
            i = ni[b]; edge(0, li, 240 + i); edge(1, 240 + i, li)
            incoming[i, li] = 1; anchors[li].append(i)
        if rid in boundaries:
            bi = 256 + boundaries.index(rid)
            edge(4, bi, li); edge(5, li, bi)
    for i, iid in enumerate(nodes):
        point = intersections[iid]["point"]
        coords[240 + i] = point["x"], point["y"]
        anchors[240 + i] = [i]
        static[240 + i] = incoming[i].sum(), outgoing[i].sum()
    for i, rid in enumerate(boundaries):
        road = roads[rid]; point = intersections[road["startIntersection"]]["point"]
        coords[256 + i] = point["x"], point["y"]
        anchors[256 + i] = [ni[road["endIntersection"]]]
        attached[ni[road["endIntersection"]], 256 + i] = 1
        static[256 + i] = len(road["lanes"]), np.mean([x["maxSpeed"] for x in road["lanes"]])
    for _, _, src, dst in g["lanelinks"].values():
        edge(6, src, dst); edge(7, dst, src)
    adj /= np.maximum(adj.sum(axis=-1, keepdims=True), 1)
    hops = np.full((16, 16), 16, dtype=np.float32)
    np.fill_diagonal(hops, 0)
    for i, neighbors in enumerate(g["adjacency"]):
        hops[i, neighbors] = 1
    for k in range(16):
        hops = np.minimum(hops, hops[:, k:k+1] + hops[k:k+1, :])
    types = np.zeros((272, 3), dtype=np.float32)
    for k, (a, b, _) in enumerate(GROUPS):
        types[a:b, k] = 1
    relative = np.zeros((16, 272, 11), dtype=np.float32)
    for i in range(16):
        delta = (coords - coords[240 + i]) / 1000.0
        relative[i, :, :2] = delta
        relative[i, :, 2] = np.linalg.norm(delta, axis=-1)
        relative[i, :, 3] = [min(hops[i, anchors[o]]) / 8.0 for o in range(272)]
        relative[i, :, 4:6] = np.stack((incoming[i], outgoing[i]), axis=-1)
        relative[i, 240 + i, 6] = 1
        relative[i, :, 7] = attached[i]
        relative[i, :, 8:] = types
    masks = np.zeros((16, 5, 12), dtype=np.float32)
    permission = np.zeros((16, 5, 272), dtype=np.float32)
    for i, iid in enumerate(nodes):
        links = intersections[iid]["roadLinks"]
        if len(links) != 12:
            raise ValueError("Expected twelve local roadLink slots")
        for phase, active in enumerate(g["phase_available_local_roadlinks_0_to_4"][i]):
            masks[i, phase, active] = 1
            for j in active:
                link = links[j]
                for lane in link["laneLinks"]:
                    permission[i, phase, lane_ids.index(f'{link["startRoad"]}_{lane["startLaneIndex"]}')] = 1
                    permission[i, phase, lane_ids.index(f'{link["endRoad"]}_{lane["endLaneIndex"]}')] = 1
            permission[i, phase, 240 + i] = bool(active)
    return {"adjacency": adj, "relative": relative, "permission": permission,
            "phase_masks": masks, "object_static": static}


def action_bank(base_phase, masks):
    """Known control requests/permission schedule only; no realized future data."""
    result = np.empty((16, 4, 218), dtype=np.float32)
    for i, ref in enumerate(base_phase):
        for target in range(4):
            fields = [np.eye(4)[ref], np.eye(4)[target]]
            current = int(ref)
            for requested in [target] + [(int(ref) + k) % 4 for k in range(1, 6)]:
                changed = requested != current
                pre = 0 if changed else requested + 1
                post = requested + 1
                fields.extend((np.eye(5)[pre], np.eye(5)[post], masks[i, pre], masks[i, post], [float(changed)]))
                current = requested
            result[i, target] = np.concatenate(fields)
    return result


def history_features(raw, root, static):
    times = raw["time_s"]
    if len(times) != 150 or not np.array_equal(times, np.arange(root["time_s"] - 149, root["time_s"] + 1)):
        raise ValueError("History must contain only the 150 recorded ticks ending at the root")
    ends = np.arange(4, 150, 5)
    x = np.zeros((30, 272, 20), dtype=np.float32)
    n = raw["lane_n"][ends].astype(np.float64)
    x[:, :240, :3] = n
    x[:, :240, 3:6] = raw["lane_q"][ends]
    x[:, :240, 6:9] = np.divide(raw["lane_speed_sum"][ends], n, out=np.zeros_like(n), where=n > 0)
    x[:, :240, 9:12] = n > 0
    tail_mask = n.sum(axis=-1) > 0
    x[:, :240, 12:14] = raw["lane_tail"][ends] * tail_mask[..., None]
    x[:, :240, 14] = tail_mask
    x[:, :240, 15] = raw["receiving_unavailable"].reshape(30, 5, 240).mean(axis=1)
    x[:, :240, 16:18] = raw["lane_events"].astype(np.int64).reshape(30, 5, 240, 2).sum(axis=1)
    x[:, 240:256, :2] = raw["intersection_nq"][ends]
    # At historical endpoints use the following tick's interval-start signal;
    # at the root use authoritative signal_context, never a post-root tick.
    phases = raw["phase_used"][np.minimum(ends + 1, 149)].copy()
    elapsed = raw["phase_elapsed_start_s"][np.minimum(ends + 1, 149)].copy()
    phases[-1] = np.asarray(root["signal_context"]["current_phase"]) + 1
    elapsed[-1] = root["signal_context"]["phase_elapsed_s"]
    x[:, 240:256, 2:7] = np.eye(5)[phases]
    x[:, 240:256, 7] = elapsed
    x[:, 240:256, 8] = phases == 0
    x[:, 256:, 0] = raw["boundary_pending"][ends]
    for col, key in ((1, "boundary_generated"), (2, "boundary_admitted")):
        x[:, 256:, col] = raw[key].astype(np.int64).reshape(30, 5, 16).sum(axis=1)
    for (a, b, _), dim in zip(GROUPS, INPUT_DIMS):
        x[:, a:b, dim-2:dim] = static["object_static"][a:b]
    return x


def fit_input_stats(histories):
    """Input: training roots once each, not repeated for their action queries."""
    x = np.asarray(histories, dtype=np.float64)
    mean = np.zeros((3, 20), dtype=np.float32)
    std = np.ones((3, 20), dtype=np.float32)
    for group, ((a, b, _), dim) in enumerate(zip(GROUPS, INPUT_DIMS)):
        for f in range(dim):
            if (group == 0 and f in (9, 10, 11, 14)) or (group == 1 and f in (2, 3, 4, 5, 6, 8)):
                continue
            v = x[:, :, a:b, f]
            mask = np.ones_like(v, dtype=bool)
            if group == 0 and f in (6, 7, 8):
                mask = x[:, :, a:b, f + 3] > 0
            if group == 0 and f in (12, 13):
                mask = x[:, :, a:b, 14] > 0
            values = v[mask]
            if len(values):
                mean[group, f] = values.mean()
                sigma = values.std()
                std[group, f] = sigma if sigma > 1e-6 else 1.0
    return {"mean": mean, "std": std}


def normalize_history(history, stats):
    x = history.copy()
    for group, ((a, b, _), dim) in enumerate(zip(GROUPS, INPUT_DIMS)):
        x[:, a:b, :dim] = (x[:, a:b, :dim] - stats["mean"][group, :dim]) / stats["std"][group, :dim]
    x[:, :240, 6:9] *= history[:, :240, 9:12]
    x[:, :240, 12:14] *= history[:, :240, 14:15]
    return x


def p95_nonzero(values):
    values = np.abs(np.asarray(values, dtype=np.float64)).ravel()
    values = values[values > 0]
    return max(1.0, float(np.percentile(values, 95))) if len(values) else 1.0


def fit_output_scales(train_roots):
    scales = np.ones((2, 272, 1), dtype=np.float32)
    totals = np.ones(2, dtype=np.float32)
    for kind, key in enumerate(("single", "pair")):
        for a, b, _ in GROUPS:
            values = np.concatenate([root[key][:, a:b].ravel() for root in train_roots])
            scales[kind, a:b] = p95_nonzero(values)
        totals[kind] = p95_nonzero(np.concatenate([r[key].sum(axis=(1, 2), dtype=np.float64) for r in train_roots]))
    return scales, totals


def prepare_data(dataset, output, progress):
    start = time.perf_counter()
    summary = json.loads((dataset / "summary.json").read_text())
    if summary["stage"] != "complete" or summary["event_semantics"] != "route-completion-v2":
        raise ValueError("Require the completed route-aware dataset")
    selection = json.loads((dataset / "selection.json").read_text())
    geometry = json.loads((dataset / "static/geometry.json").read_text())
    road_path = Path(selection["tasks"][0]["manifest"]["scenario"]["roadnet_path"])
    static = build_static(geometry, json.loads(road_path.read_text()))
    atomic_npz(output / "data/static.npz", static)
    roots = []
    sources = []
    for root_path in sorted((dataset / "roots").glob("*/root.json")):
        root = json.loads(root_path.read_text());directory = root_path.parent
        split = root_split(root)
        plan = json.loads((directory / "branches.json").read_text())
        rows = plan["branches"];lookup = {r["branch_id"]: i for i, r in enumerate(rows)}
        outcomes = np.empty((len(rows), 272, 36), dtype=np.int64)
        simulator_times = np.zeros((len(rows), 2), dtype=np.float64)
        seen = set()
        for shard_path in sorted((dataset / "shards" / root["root_id"]).glob("*.npz")):
            with np.load(shard_path, allow_pickle=False) as raw:
                ids = raw["branch_ids"].tolist()
                if seen.intersection(ids):
                    raise ValueError("Duplicate source branch")
                seen.update(ids)
                outcomes[[lookup[i] for i in ids]] = waiting_windows(raw)
                simulator_times[[lookup[i] for i in ids]] = raw["timing_restore_rollout_s"]
        if seen != set(lookup):
            raise ValueError("Missing source branch outcomes")
        labels = effect_labels(outcomes, plan)
        with np.load(directory / "history.npz", allow_pickle=False) as history:
            raw_history = {key: history[key] for key in (
                "time_s", "lane_n", "lane_q", "lane_speed_sum", "lane_tail", "receiving_unavailable",
                "lane_events", "intersection_nq", "phase_used", "phase_elapsed_start_s",
                "boundary_pending", "boundary_generated", "boundary_admitted")}
        features = history_features(raw_history, root, static)
        base = np.asarray(root["signal_context"]["current_phase"], dtype=np.int64)
        arrays = {**labels, "history": features, "base_phase": base, "action_bank": action_bank(base, static["phase_masks"])}
        checksum = atomic_npz(output / "data" / f'{root["root_id"]}.npz', arrays)
        info = {k: root[k] for k in ("root_id", "flow_id", "policy", "time_s", "backlog")}
        info.update(split=split, joint_count=len(labels["joint"]), derived_sha256=checksum)
        info["simulator_joint_restore_rollout_s"] = simulator_times[
            [i for i, row in enumerate(rows) if len(row["changed_intersections"]) >= 3]].sum(axis=0).tolist()
        roots.append({**info, **arrays, "raw_history": raw_history,
                      "root_context": {"time_s": root["time_s"], "signal_context": root["signal_context"]},
                      "input_static": {"object_static": static["object_static"]},
                      "phase_masks": static["phase_masks"]})
        sources.append({**info, "root_sha256": sha256(root_path), "plan_sha256": sha256(directory / "branches.json"),
                        "history_sha256": sha256(directory / "history.npz")})
        progress(stage="preparing_data", roots_prepared=len(roots), root_id=root["root_id"])
    split_counts = {s: sum(r["split"] == s for r in roots) for s in ("train", "validation", "development_holdout")}
    if split_counts != {"train": 8, "validation": 4, "development_holdout": 6}:
        raise ValueError(f"Incorrect root split: {split_counts}")
    joint_counts = {s: sum(r["joint_count"] for r in roots if r["split"] == s) for s in split_counts}
    if joint_counts != {"train": 2017, "validation": 1007, "development_holdout": 1515}:
        raise ValueError(f"Incorrect joint split counts: {joint_counts}")
    train = [r for r in roots if r["split"] == "train"]
    input_stats = fit_input_stats([r["history"] for r in train])
    scales, total_scales = fit_output_scales(train)
    atomic_npz(output / "data/normalization.npz", {**input_stats, "output_scales": scales, "total_scales": total_scales})
    for root in roots:
        root["normalized_history"] = normalize_history(root["history"], input_stats)
        root["input_stats"] = input_stats
    manifest = {"source_dataset": str(dataset), "source_protocol_sha256": sha256(dataset / "protocol.json"),
                "source_summary_sha256": sha256(dataset / "summary.json"), "roots": sources,
                "split_root_counts": split_counts, "split_joint_counts": joint_counts,
                "data_preparation_s": time.perf_counter() - start,
                "input_statistics": "training histories once per root; valid speed/tail only",
                "scale_groups": {name: {key: float(scales[k, a, 0]) for k, key in enumerate(("single", "pair"))}
                                 for a, b, name in GROUPS}, "total_scales": total_scales.tolist()}
    atomic_json(output / "split_and_data.json", manifest)
    return roots, static, scales, total_scales, manifest
