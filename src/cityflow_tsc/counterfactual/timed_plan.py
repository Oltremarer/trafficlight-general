"""Full absolute-time requests for the approved T0/T1/T2 experiments."""
from __future__ import annotations

import itertools
import random

import numpy as np

from .plan import digest, stable_seed


def reference_requests(base):
    return (np.asarray(base, dtype=np.int64)[None] + np.arange(8)[:, None]) % 4


def override(reference, node, phase, start, end):
    if start % 30 or end % 30 or not 0 <= start < end <= 240:
        raise ValueError("Intervention must use complete 30-second request intervals")
    result = np.asarray(reference).copy()
    result[start // 30:end // 30, node] = phase
    return result


def hop_distances(adjacency):
    n = len(adjacency)
    h = np.full((n, n), n, dtype=np.int64)
    np.fill_diagonal(h, 0)
    for i, js in enumerate(adjacency):
        h[i, list(js)] = 1
    for k in range(n):
        h = np.minimum(h, h[:, k:k + 1] + h[k:k + 1])
    return h


def select_pairs(root_id, adjacency):
    h = hop_distances(adjacency)
    rng = random.Random(stable_seed("timing_pairs_v2", root_id))
    groups = {"adjacent": [], "two_hop": [], "far": []}
    for i, j in itertools.combinations(range(len(adjacency)), 2):
        groups["adjacent" if h[i, j] == 1 else "two_hop" if h[i, j] == 2 else "far"].append((i, j))
    chosen = []
    for group, count in (("adjacent", 8), ("two_hop", 4), ("far", 4)):
        for i, j in sorted(rng.sample(groups[group], count)):
            # This bidirectional grid uses a preassigned lower-index A -> higher-index B direction.
            chosen.append({"A": i, "B": j, "stratum": group, "hops": int(h[i, j])})
    return chosen


class BranchBook:
    def __init__(self, root_id, base):
        self.root_id, self.reference = root_id, reference_requests(base)
        self.rows = {}
        self.baseline = self.add(self.reference, "baseline")

    def add(self, requests, role, singles=(), pairs=()):
        requests = np.asarray(requests, dtype=np.uint8)
        key = requests.tobytes()
        if key in self.rows:
            row = self.rows[key]
            if role not in row["roles"]:
                row["roles"].append(role)
            return row["branch_id"]
        changed = np.flatnonzero(np.any(requests != self.reference, axis=0)).tolist()
        row = {"branch_id": digest(["timed-full-request-v2", self.root_id, requests.tolist()])[:24],
               "requests": requests.tolist(), "changed_intersections": changed,
               "roles": [role], "baseline_id": getattr(self, "baseline", None),
               "single_ids": list(singles), "pair_ids": list(pairs),
               "compositional_holdout": len(changed) >= 3}
        self.rows[key] = row
        return row["branch_id"]

    def branches(self):
        result = list(self.rows.values())
        result[0]["baseline_id"] = self.baseline
        return result


def timing_plan(root_id, base, adjacency):
    book = BranchBook(root_id, base)
    selected = select_pairs(root_id, adjacency)
    panels = []
    for pair in selected:
        i, j = pair["A"], pair["B"]
        for name, first, second in (("T0", (0, 30), (0, 30)),
                                    ("T1", (0, 90), (0, 90)), ("T2", (0, 30), (60, 90))):
            options_a = list(range(4)) if name == "T1" else [a for a in range(4) if a != book.reference[first[0] // 30, i]]
            options_b = list(range(4)) if name == "T1" else [a for a in range(4) if a != book.reference[second[0] // 30, j]]
            singles_a, singles_b = {}, {}
            for a in options_a:
                singles_a[a] = book.add(override(book.reference, i, a, *first), "single")
            for b in options_b:
                singles_b[b] = book.add(override(book.reference, j, b, *second), "single")
            combinations = []
            for a, b in itertools.product(options_a, options_b):
                requests = override(override(book.reference, i, a, *first), j, b, *second)
                sid = [singles_a[a], singles_b[b]]
                bid = book.add(requests, "pair", singles=sid)
                combinations.append({"phase_A": a, "phase_B": b, "branch_id": bid, "single_ids": sid})
            candidates = [book.baseline, *singles_a.values(), *singles_b.values(), *[c["branch_id"] for c in combinations]]
            panels.append({**pair, "timing": name, "A_window_s": list(first), "B_window_s": list(second),
                           "baseline_id": book.baseline, "single_A": {str(k): v for k, v in singles_a.items()},
                           "single_B": {str(k): v for k, v in singles_b.items()},
                           "combinations": combinations, "candidate_ids": candidates})
    return {"root_id": root_id, "branch_count": len(book.rows), "branch_upper_bound_before_dedup": 16 * 57,
            "horizon_s": 240, "formal_setting": "T1", "selected_pairs": selected,
            "panels": panels, "branches": book.branches()}


def formal_t1_plan(root_id, base, adjacency):
    book = BranchBook(root_id, base)
    singles, pairs = {}, {}
    for i in range(16):
        for phase in range(4):
            singles[i, phase] = book.add(override(book.reference, i, phase, 0, 90), "single")
    for i, j in itertools.combinations(range(16), 2):
        for a, b in itertools.product(range(4), repeat=2):
            requests = override(override(book.reference, i, a, 0, 90), j, b, 0, 90)
            pairs[i, a, j, b] = book.add(requests, "pair", [singles[i, a], singles[j, b]])
    rng = random.Random(stable_seed("formal_t1_joint_v2", root_id))
    joints = []
    for size in (3, 4, 8, 16):
        modes = (("connected", 8), ("uniform", 8)) if size != 16 else (("all", 16),)
        for mode, count in modes:
            accepted = 0
            while accepted < count:
                if mode == "connected":
                    nodes = {rng.randrange(16)}
                    while len(nodes) < size:
                        frontier = sorted(set().union(*(set(adjacency[i]) for i in nodes)) - nodes)
                        nodes.add(rng.choice(frontier))
                    nodes = sorted(nodes)
                else:
                    nodes = sorted(rng.sample(range(16), size))
                phases = {i: rng.randrange(4) for i in nodes}
                requests = book.reference.copy()
                for i in nodes:
                    requests = override(requests, i, phases[i], 0, 90)
                before = len(book.rows)
                bid = book.add(requests, f"joint_{size}_{mode}", [singles[i, phases[i]] for i in nodes],
                               [pairs[i, phases[i], j, phases[j]] for i, j in itertools.combinations(nodes, 2)])
                if len(book.rows) > before:
                    accepted += 1; joints.append(bid)
    rows = book.branches()
    if len(rows) != 2049 or len(joints) != 64:
        raise ValueError("T1 requires 1 baseline + 64 singles + 1920 pairs + 64 joints")
    for row in rows:
        row["plan_ids"] = [0 if i not in row["changed_intersections"] else row["requests"][0][i] + 1 for i in range(16)]
    return {"root_id": root_id, "branch_count": len(rows), "horizon_s": 240, "formal_setting": "T1",
            "candidate_ids": [book.baseline, *joints], "branches": rows}
