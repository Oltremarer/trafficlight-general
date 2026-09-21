"""Committed-only, disk-backed labels for the fixed 60-root T1 experiment.

Preparation is staged: the single/joint data and its training statistics are
immutable before A training starts; pair data adds a separate statistics file.
Neither preparation nor loading opens test roots unless explicitly requested.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..counterfactual.writer import atomic_json, sha256
from . import GROUPS
from .data import build_static, fit_input_stats, history_features, normalize_history
from .revision import sequence_bank


SCHEMA = "formal-t1-effect-v3"
TEST_REPAIR_SCHEMA = "formal-effect-v3-test-time-repair-20260912"
SINGLE_REVISION_SCHEMA = "formal-single-coverage-v4-20260915"
HORIZONS = (30, 90, 180, 240)
HISTORY_KEYS = ("time_s", "lane_n", "lane_q", "lane_speed_sum", "lane_tail",
                "receiving_unavailable", "lane_events", "intersection_nq",
                "phase_used", "phase_elapsed_start_s", "boundary_pending",
                "boundary_generated", "boundary_admitted")
PAIR_NODES = np.asarray(list(itertools.combinations(range(16), 2)), dtype=np.int64)


def _write_numpy(path, value, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        if compressed:
            np.savez_compressed(stream, **value)
        else:
            np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def waiting_windows(raw):
    """Sum disjoint physical waiting locations before differencing/casting."""
    lane = np.asarray(raw["lane_q"]).astype(np.int64).sum(axis=-1)
    junction = np.asarray(raw["intersection_nq"]).astype(np.int64)[..., 1]
    boundary = np.asarray(raw["boundary_pending"]).astype(np.int64)
    values = np.concatenate((lane, junction, boundary), axis=-1)
    if values.shape[1:] != (240, 272):
        raise ValueError(f"Expected [branch,240,272] physical waiting: {values.shape}")
    return values.reshape(len(values), 48, 5, 272).sum(axis=2, dtype=np.int64).transpose(0, 2, 1)


def plan_arrays(plan):
    """Return fixed factor queries and incidences, with reference candidate first."""
    rows = plan["branches"]
    lookup = {row["branch_id"]: row for row in rows}
    baseline = [row for row in rows if not row["changed_intersections"]]
    singles = [row for row in rows if len(row["changed_intersections"]) == 1]
    pairs = [row for row in rows if len(row["changed_intersections"]) == 2]
    candidates = plan["candidate_ids"]
    if (len(rows), len(lookup), len(baseline), len(singles), len(pairs), len(candidates)) != (2049, 2049, 1, 64, 1920, 65):
        raise ValueError("Expected 2049 unique T1 branches, 64 singles, 1920 pairs and 65 candidates")
    if candidates[0] != baseline[0]["branch_id"] or len(set(candidates)) != 65:
        raise ValueError("The unique reference must be candidate zero")
    if any(len(lookup[bid]["changed_intersections"]) not in (3, 4, 8, 16) for bid in candidates[1:]):
        raise ValueError("Main decision menu may only add the frozen 3+ joint actions")
    single_index = {row["branch_id"]: i for i, row in enumerate(singles)}
    pair_index = {row["branch_id"]: i for i, row in enumerate(pairs)}
    single_incidence = np.zeros((65, 64), dtype=np.float32)
    pair_incidence = np.zeros((65, 1920), dtype=np.float32)
    for i, bid in enumerate(candidates):
        row = lookup[bid]
        single_incidence[i, [single_index[sid] for sid in row["single_ids"]]] = 1
        pair_incidence[i, [pair_index[pid] for pid in row["pair_ids"]]] = 1

    def queries(factors):
        nodes = np.asarray([row["changed_intersections"] for row in factors], dtype=np.int64)
        actions = np.asarray([[row["plan_ids"][node] for node in row["changed_intersections"]]
                              for row in factors], dtype=np.int64)
        if np.any((actions < 1) | (actions > 4)):
            raise ValueError("Factors use T1 plan IDs 1..4, not first-phase IDs 0..3")
        return nodes, actions

    single_nodes, single_actions = queries(singles)
    pair_nodes, pair_actions = queries(pairs)
    expected_single = {(node, action) for node in range(16) for action in range(1, 5)}
    if set(map(tuple, np.concatenate((single_nodes, single_actions), axis=1))) != expected_single:
        raise ValueError("Incomplete single plan menu")
    pair_lookup = {tuple(nodes): i for i, nodes in enumerate(PAIR_NODES.tolist())}
    pair_query_pair_index = np.asarray([pair_lookup[tuple(nodes)] for nodes in pair_nodes], dtype=np.int64)
    expected_pairs = {(i, j, a, b) for i, j in PAIR_NODES for a in range(1, 5) for b in range(1, 5)}
    if set(map(tuple, np.concatenate((pair_nodes, pair_actions), axis=1))) != expected_pairs:
        raise ValueError("Incomplete pair plan menu")
    groups = ["reference"]
    for bid in candidates[1:]:
        roles = [role for role in lookup[bid]["roles"] if role.startswith("joint_")]
        if len(roles) != 1:
            raise ValueError("Each frozen joint candidate must have one sampling group")
        groups.append(roles[0].split("_", 2)[2])
    return {
        "single_nodes": single_nodes, "single_actions": single_actions,
        "single_queries": np.concatenate((single_nodes, single_actions), axis=1),
        "pair_nodes": pair_nodes, "pair_actions": pair_actions,
        "pair_queries": np.concatenate((pair_nodes, pair_actions), axis=1),
        "pair_unique_nodes": PAIR_NODES.copy(), "pair_query_pair_index": pair_query_pair_index,
        "s_incidence": single_incidence, "p_incidence": pair_incidence,
        "single_ids": np.asarray([row["branch_id"] for row in singles], dtype="U24"),
        "pair_ids": np.asarray([row["branch_id"] for row in pairs], dtype="U24"),
        "pair_single_indices": np.asarray([[single_index[sid] for sid in row["single_ids"]] for row in pairs], dtype=np.int64),
        "joint_ids": np.asarray(candidates, dtype="U24"),
        "joint_sizes": np.asarray([len(lookup[bid]["changed_intersections"]) for bid in candidates], dtype=np.int64),
        "joint_groups": np.asarray(groups, dtype="U12"),
        "joint_plans": np.asarray([lookup[bid]["plan_ids"] for bid in candidates], dtype=np.int64),
    }


def committed_outcomes(run, root_id, required_ids):
    """Yield one required committed shard at a time, never retain raw shards.

    Commit metadata is the discovery source. Uncommitted payloads and entirely
    irrelevant pair shards are not opened during initial A preparation.
    """
    required, seen = set(required_ids), set()
    for metadata_path in sorted((Path(run) / "shards" / root_id).glob("*.npz.json")):
        metadata = json.loads(metadata_path.read_text())
        ids = metadata["branch_ids"]
        if not required.intersection(ids):
            continue
        path = Path(str(metadata_path)[:-5])
        if not path.is_file():
            raise FileNotFoundError(f"Committed payload missing: {path}")
        if metadata.get("root_id") != root_id or metadata.get("branch_count") != len(ids):
            raise ValueError(f"Committed shard identity/count mismatch: {metadata_path}")
        if not metadata.get("sha256") or len(set(ids)) != len(ids):
            raise ValueError(f"Malformed committed shard: {metadata_path}")
        selected = [i for i, bid in enumerate(ids) if bid in required]
        selected_ids = [ids[i] for i in selected]
        if seen.intersection(selected_ids):
            raise ValueError(f"Duplicate committed source branch in {metadata_path}")
        with np.load(path, allow_pickle=False) as raw:
            if raw["branch_ids"].tolist() != ids:
                raise ValueError(f"Commit/payload branch IDs disagree: {path}")
            values = waiting_windows(raw)
        if len(values) != len(ids):
            raise ValueError(f"Commit/payload branch count differs: {path}")
        seen.update(selected_ids)
        yield selected_ids, values[selected], {
            "path": str(path.relative_to(run)), "sha256": metadata["sha256"],
            "branch_ids": selected_ids,
        }
    if seen != required:
        raise FileNotFoundError(f"{root_id}: {len(required - seen)} required branches are not committed")


class _Histogram:
    """Exact linear P95 of nonzero integer magnitudes without dense copies."""
    def __init__(self):
        self.counts = Counter()

    def update(self, values):
        values = np.abs(np.asarray(values, dtype=np.int64)).ravel()
        keys, counts = np.unique(values[values != 0], return_counts=True)
        self.counts.update({int(key): int(count) for key, count in zip(keys, counts)})

    def percentile(self):
        count = sum(self.counts.values())
        if not count:
            return 1.0
        index = .95 * (count - 1)
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        consumed = 0
        lower = upper = None
        for value, frequency in sorted(self.counts.items()):
            if lower is None and lo < consumed + frequency:
                lower = value
            if hi < consumed + frequency:
                upper = value
                break
            consumed += frequency
        return max(1.0, float(lower + (upper - lower) * (index - lo)))

    def to_list(self):
        return [[key, value] for key, value in sorted(self.counts.items())]

    def merge(self, values):
        self.counts.update({int(key): int(count) for key, count in values})


class ScaleAccumulator:
    """Exact train-only P95 scales; cumulative sums remain signed until abs."""
    def __init__(self):
        self.local = [_Histogram() for _ in GROUPS]
        self.position = [[_Histogram() for _ in HORIZONS] for _ in GROUPS]
        self.total = [_Histogram() for _ in HORIZONS]

    def update(self, values):
        values = np.asarray(values, dtype=np.int64)
        if values.shape[1:] != (272, 48):
            raise ValueError("Scale labels must have 272 locations and 48 windows")
        cumulative = values.cumsum(axis=-1, dtype=np.int64)[:, :, np.asarray(HORIZONS) // 5 - 1]
        for group, (a, b, _) in enumerate(GROUPS):
            self.local[group].update(values[:, a:b])
            for h in range(4):
                self.position[group][h].update(cumulative[:, a:b, h])
        totals = cumulative.sum(axis=1, dtype=np.int64)
        for h in range(4):
            self.total[h].update(totals[:, h])

    def arrays(self, prefix):
        local, position = np.ones((272, 1), dtype=np.float32), np.ones((272, 4), dtype=np.float32)
        for group, (a, b, _) in enumerate(GROUPS):
            local[a:b] = self.local[group].percentile()
            position[a:b] = [hist.percentile() for hist in self.position[group]]
        return {prefix + "_s5": local, prefix + "_sp": position,
                prefix + "_sj": np.asarray([hist.percentile() for hist in self.total], dtype=np.float32)}

    def to_dict(self):
        return {"local": [hist.to_list() for hist in self.local],
                "position": [[hist.to_list() for hist in group] for group in self.position],
                "total": [hist.to_list() for hist in self.total]}

    def merge(self, value):
        for hist, counts in zip(self.local, value["local"]):
            hist.merge(counts)
        for group, groups in zip(self.position, value["position"]):
            for hist, counts in zip(group, groups):
                hist.merge(counts)
        for hist, counts in zip(self.total, value["total"]):
            hist.merge(counts)


def _root_metadata(task, root):
    info = {key: root.get(key, task.get(key)) for key in (
        "root_id", "split", "source_id", "cohort_id", "source_sha256", "cohort_sha256",
        "group_id", "flow_id", "policy", "time_s", "profile", "demand_scale", "manifest_path")}
    if any(info[key] is None for key in ("root_id", "split", "source_id", "cohort_id", "flow_id", "policy", "time_s")):
        raise ValueError("Frozen formal root is missing split/source/cohort metadata")
    if info["root_id"] != task["root_id"] or info["split"] != task["split"]:
        raise ValueError("Root does not match frozen task identity/split")
    return info


def _derive_initial(run, task, static):
    rid = task["root_id"]
    target, source = run / "derived/roots" / rid, run / "roots" / rid
    marker = target / "initial.complete.json"
    if marker.exists():
        return json.loads(marker.read_text())
    root = json.loads((source / "root.json").read_text())
    plan = json.loads((source / "branches.json").read_text())
    arrays = plan_arrays(plan)
    ids = [*arrays["single_ids"].tolist(), *arrays["joint_ids"].tolist()]
    outcome, sources = {}, []
    for shard_ids, values, metadata in committed_outcomes(run, rid, ids):
        sources.append(metadata)
        outcome.update(zip(shard_ids, values))
    baseline = outcome[arrays["joint_ids"][0]]
    single_int = np.stack([outcome[bid] - baseline for bid in arrays["single_ids"]])
    joint_int = np.stack([outcome[bid] - baseline for bid in arrays["joint_ids"]])
    _write_numpy(target / "baseline.npy", baseline)
    _write_numpy(target / "single_int64.npy", single_int)
    _write_numpy(target / "single.npy", single_int.astype(np.float32))
    _write_numpy(target / "joint.npy", joint_int.astype(np.float32))
    scales = ScaleAccumulator()
    scales.update(single_int)
    atomic_json(target / "single_scale_counts.json", scales.to_dict())
    with np.load(source / "history.npz", allow_pickle=False) as history:
        raw_history = {key: history[key] for key in HISTORY_KEYS}
    base = np.asarray(root["signal_context"]["current_phase"], dtype=np.int64)
    arrays.update(history=history_features(raw_history, root, static), base_phase=base,
                  action_bank=sequence_bank(base, static["phase_masks"])[0])
    _write_numpy(target / "inputs.npz", arrays, compressed=True)
    info = {**_root_metadata(task, root), "schema": SCHEMA, "initial_branch_count": 129,
            "source_shards": sources, "root_sha256": sha256(source / "root.json"),
            "plan_sha256": sha256(source / "branches.json"), "history_sha256": sha256(source / "history.npz")}
    atomic_json(marker, info)
    return info


def _derive_pairs(run, task):
    rid = task["root_id"]
    target = run / "derived/roots" / rid
    marker = target / "pair.complete.json"
    if marker.exists():
        return json.loads(marker.read_text())
    arrays = _read_npz(target / "inputs.npz")
    baseline = np.load(target / "baseline.npy", mmap_mode="r", allow_pickle=False)
    singles = np.load(target / "single_int64.npy", mmap_mode="r", allow_pickle=False)
    ids = arrays["pair_ids"].tolist()
    index = {bid: i for i, bid in enumerate(ids)}
    temporary = target / "pair.npy.tmp"
    pair = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(1920, 272, 48))
    scales, rank_targets, sources = ScaleAccumulator(), np.zeros(120, dtype=np.int64), []
    for shard_ids, outcomes, metadata in committed_outcomes(run, rid, ids):
        rows = np.asarray([index[bid] for bid in shard_ids], dtype=np.int64)
        si = arrays["pair_single_indices"][rows]
        # Exact integer inclusion/exclusion, not a residual of model predictions.
        effects = outcomes - baseline - singles[si[:, 0]] - singles[si[:, 1]]
        pair[rows] = effects.astype(np.float32)
        scales.update(effects)
        totals = np.abs(effects.sum(axis=(1, 2), dtype=np.int64))
        np.maximum.at(rank_targets, arrays["pair_query_pair_index"][rows], totals)
        sources.append(metadata)
    pair.flush()
    del pair
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, target / "pair.npy")
    _write_numpy(target / "rank_targets.npy", rank_targets)
    atomic_json(target / "pair_scale_counts.json", scales.to_dict())
    info = {"root_id": rid, "schema": SCHEMA, "pair_count": 1920, "source_shards": sources}
    atomic_json(marker, info)
    return info


def _selected_tasks(selection, splits):
    splits = tuple(splits)
    if not splits or not set(splits).issubset({"train", "validation", "test"}):
        raise ValueError("Explicit train/validation/test split names required")
    tasks = selection["tasks"]
    counts = Counter(task["split"] for task in tasks)
    if selection.get("schema") == TEST_REPAIR_SCHEMA:
        if splits != ("test",):
            raise ValueError("Test-time repair is test-only; training/validation access is forbidden")
        if (counts != {"test": 12} or len({t["root_id"] for t in tasks}) != 12
                or len({t["cohort_id"] for t in tasks}) != 3):
            raise ValueError("Test-time repair requires twelve distinct test roots in three cohorts")
    elif selection.get("schema") == SINGLE_REVISION_SCHEMA:
        expected = selection.get("split_roots", {})
        if (set(expected) != {"train", "validation", "test"}
                or any(not isinstance(n, int) or n <= 0 for n in expected.values())
                or counts != expected or len({t["root_id"] for t in tasks}) != sum(expected.values())):
            raise ValueError("Single revision requires the exact declared distinct split roots")
    elif counts != {"train": 36, "validation": 12, "test": 12} or len({task["root_id"] for task in tasks}) != 60:
        raise ValueError("Frozen selection must contain the approved 36/12/12 distinct roots")
    groups = {}
    for task in tasks:
        group = (task.get("source_sha256", task["source_id"]), task["cohort_id"])
        if group in groups and groups[group] != task["split"]:
            raise ValueError("Related demand cohorts cannot cross splits")
        groups[group] = task["split"]
    return [task for task in tasks if task["split"] in splits]


def _static_data(run, selection):
    path = run / "derived/static.npz"
    if path.exists():
        return _read_npz(path)
    geometry = json.loads((run / "static/geometry.json").read_text())
    road_path = Path(selection["tasks"][0]["manifest"]["scenario"]["roadnet_path"])
    static = build_static(geometry, json.loads(road_path.read_text()))
    features = []
    for i, j in PAIR_NODES:
        relative = static["relative"][i, 240 + j]
        features.append([abs(relative[0]), abs(relative[1]), relative[2], relative[3], float(relative[3] == .125)])
    static.update(rank_geometry=np.asarray(features, dtype=np.float32), pair_unique_nodes=PAIR_NODES.copy())
    _write_numpy(path, static, compressed=True)
    return static


def _fit_statistics(run, tasks, include_pairs):
    train = [task for task in tasks if task["split"] == "train"]
    selection_path = run / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    expected_train = (selection["split_roots"]["train"]
                      if selection.get("schema") == SINGLE_REVISION_SCHEMA else 36)
    initial_path = run / "derived/normalization.npz"
    if not initial_path.exists():
        if len(train) != expected_train:
            raise ValueError("Initial scales require all declared training roots, never validation/test substitution")
        histories, scales = [], ScaleAccumulator()
        for task in train:
            path = run / "derived/roots" / task["root_id"]
            histories.append(_read_npz(path / "inputs.npz")["history"])
            scales.merge(json.loads((path / "single_scale_counts.json").read_text()))
        stats = {**fit_input_stats(histories), **scales.arrays("single"), "horizons_s": np.asarray(HORIZONS)}
        _write_numpy(initial_path, stats, compressed=True)
    pair_path = run / "derived/pair_normalization.npz"
    if include_pairs and not pair_path.exists():
        if len(train) != 36:
            raise ValueError("Pair scales require all 36 training roots")
        scales, ranks = ScaleAccumulator(), _Histogram()
        for task in train:
            path = run / "derived/roots" / task["root_id"]
            scales.merge(json.loads((path / "pair_scale_counts.json").read_text()))
            ranks.update(np.load(path / "rank_targets.npy", mmap_mode="r", allow_pickle=False))
        _write_numpy(pair_path, {**scales.arrays("pair"), "rank_scale": np.asarray(ranks.percentile(), dtype=np.float32)}, compressed=True)


def prepare_dataset(run, include_pairs=False, splits=("train", "validation")):
    """Derive requested committed roots and return the updated data manifest.

    Call once in the orchestration process before spawning training workers.
    A later include_pairs=True call adds pair files without touching A scales.
    Test preparation is explicit and must be called only after checkpoint lock.
    """
    import fcntl
    run = Path(run)
    if "test" in splits and not (run / "checkpoints_locked.json").is_file():
        raise ValueError("Test labels remain unopened until checkpoints_locked.json exists")
    (run / "derived").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (run / "derived/prepare.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        selection = json.loads((run / "selection.json").read_text())
        if selection.get("schema") == SINGLE_REVISION_SCHEMA and include_pairs:
            raise ValueError("Single revision does not collect or prepare pair labels")
        tasks = _selected_tasks(selection, splits)
        static = _static_data(run, selection)
        manifest_path = run / "derived/manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
            "schema": SCHEMA, "source_run": str(run), "source_selection_sha256": sha256(run / "selection.json"), "roots": []}
        if manifest["source_selection_sha256"] != sha256(run / "selection.json"):
            raise ValueError("Frozen selection changed after data derivation")
        records = {info["root_id"]: info for info in manifest["roots"]}
        for i, task in enumerate(tasks):
            info = _derive_initial(run, task, static)
            if include_pairs:
                _derive_pairs(run, task)
            records[task["root_id"]] = {**info, "pairs_prepared": (run / "derived/roots" / task["root_id"] / "pair.complete.json").exists()}
            atomic_json(run / "derived/progress.json", {"stage": "pairs" if include_pairs else "initial", "roots_prepared": i + 1,
                        "roots_requested": len(tasks), "root_id": task["root_id"], "elapsed_s": time.perf_counter() - started})
        _fit_statistics(run, tasks, include_pairs)
        manifest.update(roots=list(records.values()), horizons_s=list(HORIZONS), windows=48,
                        single_statistics_sha256=sha256(run / "derived/normalization.npz"),
                        statistics_scope=(f"{selection.get('split_roots', {}).get('train', 36)} training roots only; "
                                          "exact nonzero P95; signed sums before absolute value"),
                        last_preparation_s=time.perf_counter() - started)
        if (run / "derived/pair_normalization.npz").exists():
            manifest["pair_statistics_sha256"] = sha256(run / "derived/pair_normalization.npz")
        atomic_json(manifest_path, manifest)
    return manifest


def load_data(run, include_pairs=False, splits=("train", "validation"), with_raw_history=False):
    """Read already prepared roots as per-root read-only memmaps; no fitting.

    Test loads additionally expose the pre-root raw input for isolated online
    timing. Merely listing the frozen selection does not open any test labels.
    """
    run = Path(run)
    if "test" in splits and not (run / "checkpoints_locked.json").is_file():
        raise ValueError("Test labels remain unopened until checkpoints_locked.json exists")
    selection = json.loads((run / "selection.json").read_text())
    tasks = _selected_tasks(selection, splits)
    static = _read_npz(run / "derived/static.npz")
    stats = _read_npz(run / "derived/normalization.npz")
    if include_pairs:
        stats.update(_read_npz(run / "derived/pair_normalization.npz"))
    roots = []
    for task in tasks:
        rid = task["root_id"]
        path = run / "derived/roots" / rid
        info = json.loads((path / "initial.complete.json").read_text())
        arrays = _read_npz(path / "inputs.npz")
        root = {**info, **arrays,
                "single": np.load(path / "single.npy", mmap_mode="r", allow_pickle=False),
                "joint": np.load(path / "joint.npy", mmap_mode="r", allow_pickle=False)}
        root["normalized_history"] = normalize_history(root["history"], stats)
        root["input_stats"] = {key: stats[key] for key in ("mean", "std")}
        root["input_static"] = {"object_static": static["object_static"]}
        root["phase_masks"] = static["phase_masks"]
        if include_pairs:
            if not (path / "pair.complete.json").exists():
                raise FileNotFoundError(f"Pair derivation is not committed: {rid}")
            root["pair"] = np.load(path / "pair.npy", mmap_mode="r", allow_pickle=False)
            root["rank_targets"] = np.load(path / "rank_targets.npy", mmap_mode="r", allow_pickle=False)
        if with_raw_history or task["split"] == "test":
            source = run / "roots" / rid
            with np.load(source / "history.npz", allow_pickle=False) as raw:
                root["raw_history"] = {key: raw[key] for key in HISTORY_KEYS}
            context = json.loads((source / "root.json").read_text())
            root["root_context"] = {"time_s": context["time_s"], "signal_context": context["signal_context"]}
        roots.append(root)
    return roots, static, stats
