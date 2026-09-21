from __future__ import annotations

import hashlib
import itertools
import json
import random

import numpy as np

from . import SCHEMA

MASTER_SEED = 20260909
ROOT_TIMES = (600, 1200, 1800, 2400, 3000)


def stable_seed(*parts):
    raw = json.dumps([MASTER_SEED, *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_flows(conditions):
    cells = {}
    for row in conditions:
        c = row["generator_config"]
        cells.setdefault((row["split"], c["source_id"], c["profile"]), []).append(row)
    selected = []
    for key in sorted(cells):
        rows = sorted(cells[key], key=lambda r: r["flow_id"])
        rng = random.Random(stable_seed("flow_selection", key))
        if key[0] == "train":
            pairs = [p for p in itertools.combinations(rows, 2)
                     if p[0]["generator_config"]["demand_scale"] != p[1]["generator_config"]["demand_scale"]]
            if not pairs:
                raise ValueError(f"No different-scale flow pair in {key}")
            selected.extend(rng.choice(pairs))
        else:
            selected.append(rng.choice(rows))
    order = {"train": 0, "validation": 1, "test": 2}
    selected.sort(key=lambda r: (order[r["split"]], r["flow_id"]))
    counts = {s: sum(r["split"] == s for r in selected) for s in order}
    if counts != {"train": 24, "validation": 6, "test": 6}:
        raise ValueError(f"Unexpected source/profile layout: {counts}")
    return selected


def continuation(a0, first):
    return np.asarray([first] + [[(a + k) % 4 for a in a0] for k in range(1, 6)], dtype=np.uint8)


def make_branches(root_id, a0, full, adjacency, panel):
    a0 = tuple(int(a) for a in a0)
    n = len(a0)
    rng = random.Random(stable_seed("branches", root_id))
    options = [[a for a in range(4) if a != b] for b in a0]
    rows = {}

    def add(values, role):
        key = tuple(values)
        if key in rows:
            if role not in rows[key]["roles"]:
                rows[key]["roles"].append(role)
            return False
        changed = [i for i, (a, b) in enumerate(zip(key, a0)) if a != b]
        requests = continuation(a0, key).tolist()
        rows[key] = {
            "branch_id": digest([SCHEMA, root_id, requests])[:24],
            "requests": requests, "changed_intersections": changed,
            "roles": [role], "compositional_holdout": len(changed) >= 3,
        }
        return True

    add(a0, "baseline")
    for i in range(n):
        for a in options[i]:
            values = list(a0); values[i] = a
            add(values, "single")
    pairs = list(itertools.combinations(range(n), 2))
    chosen_pairs = pairs if full else sorted(rng.sample(pairs, 32))
    for i, j in chosen_pairs:
        for a, b in itertools.product(options[i], options[j]):
            values = list(a0); values[i], values[j] = a, b
            add(values, "pair")
    if full:
        for size in (3, 4, 8, n):
            for mode, count in (("connected", 8), ("uniform", 8)) if size != n else (("all", 16),):
                accepted = 0
                while accepted < count:
                    if mode == "connected":
                        nodes = {rng.randrange(n)}
                        while len(nodes) < size:
                            frontier = sorted(set().union(*(set(adjacency[i]) for i in nodes)) - nodes)
                            if not frontier:
                                raise ValueError("Cannot grow requested connected action set")
                            nodes.add(rng.choice(frontier))
                        chosen = sorted(nodes)
                    else:
                        chosen = sorted(rng.sample(range(n), size))
                    values = list(a0)
                    for i in chosen:
                        values[i] = rng.choice(options[i])
                    accepted += add(values, f"joint_{size}_{mode}")
        for actions in itertools.product(range(4), repeat=4):
            values = list(a0)
            for i, a in zip(panel, actions):
                values[i] = a
            add(values, "panel_exact")
    for key, row in rows.items():
        singles = []
        pair_refs = []
        for i in row["changed_intersections"]:
            single = list(a0); single[i] = key[i]
            singles.append(rows[tuple(single)]["branch_id"])
        for i, j in itertools.combinations(row["changed_intersections"], 2):
            pair = list(a0); pair[i], pair[j] = key[i], key[j]
            reference = rows.get(tuple(pair))
            pair_refs.append(None if reference is None else reference["branch_id"])
        row["baseline_id"] = rows[a0]["branch_id"]
        row["single_ids"] = singles
        row["pair_ids"] = pair_refs
    return list(rows.values()), chosen_pairs


def full_root_index(roots, ordinal, policy):
    use_max = (ordinal % 2 == 0) == (policy == "max_pressure")
    values = sorted(r["backlog"] for r in roots)
    target = values[-1] if use_max else values[2]
    return next(i for i, row in enumerate(roots) if row["backlog"] == target)
