"""ATT accounting and source-labelled published CoLight reference values."""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from .metrics import MetricCollector
from .baselines.trajectory import write_json


class CoLightMetrics(MetricCollector):
    """Keep native metrics; add the LLMTSCS per-intersection incoming-lane ATT.

    Compatibility includes its empty-previous-set departure branch. This field
    is deliberately named llmtscs_att_s, not silently substituted for engine ATT.
    """

    def __init__(self, network, waiting_speed_threshold=0.1, *, lane_change=False):
        self.network = network
        self.lane_change = lane_change
        super().__init__(waiting_speed_threshold)

    def configure(self, scenario, step_s, initial):
        super().configure(scenario, step_s, initial)
        if self.lane_change and self._lifecycle is not None:
            self._lifecycle.active_count_source = (
                'len(engine.get_vehicles(False)): real vehicles, excluding lane-change shadows')

    def reset(self):
        super().reset()
        self.previous = [set() for _ in self.network.intersections]
        self.times = [{} for _ in self.network.intersections]
        self.last_time = 0.0

    def observe(self, snapshot, elapsed_s):
        # CityFlow counts temporary lane-change shadows in get_vehicle_count(),
        # but both get_vehicles lists already contain only real, stable IDs.
        # Keep the raw snapshot unchanged and account for physical vehicles here.
        physical = snapshot
        if self.lane_change and snapshot.active_vehicle_ids is not None:
            physical = replace(snapshot, active_vehicle_count=len(snapshot.active_vehicle_ids))
        super().observe(physical, elapsed_s)
        self.last_time = snapshot.time_s
        for i, intersection in enumerate(self.network.intersections):
            current = {v for lane in intersection.incoming_lanes
                       for v in snapshot.lane_vehicles.get(lane, ())}
            previous = self.previous[i]
            for vehicle in current - previous:
                self.times[i].setdefault(vehicle, [snapshot.time_s, None])
            left = previous - current if previous else current - previous
            for vehicle in left:
                self.times[i][vehicle][1] = snapshot.time_s
            self.previous[i] = current

    def summary(self, engine_average_travel_time_s):
        result = super().summary(engine_average_travel_time_s)
        durations = {}
        for intersection in self.times:
            for vehicle, (enter, leave) in intersection.items():
                if "shadow" in vehicle:
                    continue
                duration = (self.last_time if leave is None else leave) - enter
                durations[vehicle] = durations.get(vehicle, 0.0) + duration
        result["llmtscs_att_s"] = float(np.mean(list(durations.values()))) if durations else 0.0
        result["llmtscs_att_vehicle_count"] = float(len(durations))
        return result


# File hashes previously verified against the official LLMTSCS dataset blobs.
ROADNET_HASHES = {
    "Jinan": "55abc036ac4ac48705301f4cf46ea80d8d88a82abb9ed68404bedb96eb3cf81d",
    "Hangzhou": "11e2fe89f632e43e81f56ea87a308d544b66f9c5af8a410e16149668d6d376c1",
}
FLOW_HASHES = {
    "Jinan1": "233739633ef0b637125cb304dfffff9488503bac6e6861ca39243cc8ffdcebd5",
    "Jinan2": "d0931c1b759479f9e748d69c16414020bfba03d555dcc6cdf7223a0c18cb9e69",
    "Jinan3": "4245107cc7ce91b9699519f2c1258be4397e83291080a43cb0b93479557739cd",
    "Hangzhou1": "595a6140e6649efe5be1b274eb8b03e8c4029e517c9ac2ad53c9ea556a414213",
    "Hangzhou2": "e9b9e31e9a9a5f59d1a668b242319c3ef69f1ab724e2f2b618ed4886cc1a7a9c",
}


def identify_dataset(roadnet_hash, flow_hash):
    for name, digest in FLOW_HASHES.items():
        city = "Jinan" if name.startswith("Jinan") else "Hangzhou"
        if flow_hash == digest and roadnet_hash == ROADNET_HASHES[city]:
            return name
    return "custom"


DATASETS = tuple(FLOW_HASHES)


def references(dataset):
    """Published numbers only. Shared labels are not assertions of identical files."""
    if dataset not in DATASETS:
        return []
    index = DATASETS.index(dataset)
    rows = []
    def add(paper, table, label, value, url, setting, relation="same label; files not cross-verified"):
        rows.append(dict(paper=paper, table=table, paper_dataset=label, paper_att_s=value,
                         url=url, setting=setting, data_relation=relation))
    add("LLMLight (arXiv v5)", "2", dataset,
        [279.60, 274.77, 266.39, 322.85, 342.90][index],
        "https://arxiv.org/html/2312.16044v5#S4.T2",
        "target-flow training; paper green/yellow/red=30/3/2; public code decision/yellow/red=30/5/0",
        "official public roadnet/flow hashes match; original run seeds/runtime unknown")
    add("FuzzyLight (arXiv v2)", "2 (noise-free)", dataset,
        [281.58, 257.13, 261.34, 301.70, 339.26][index],
        "https://arxiv.org/html/2501.15820v2",
        "50 rounds; last ten tests; paper yellow/red=3/2; ATT aggregation requires alignment")
    add("Traffic-R1", "5 (appendix)", dataset,
        [272.44, 250.41, 248.84, 294.61, 335.32][index],
        "https://aclanthology.org/2026.acl-long.995.pdf",
        "RL target-flow training; paper green/yellow/red=15/3/2")
    add("Traffic-R1", "2", dataset,
        [472.44, 450.41, 498.84, 494.61, 435.32][index],
        "https://aclanthology.org/2026.acl-long.995.pdf",
        "ZERO-SHOT transfer; different training regime; paper green/yellow/red=15/3/2")
    add("Astra", "1", dataset,
        [340.36, 290.74, 293.32, 363.71, 458.70][index],
        "https://liuzhidan.github.io/files/2026-KDD-Astra.pdf",
        "8-phase control, unlike this four-phase implementation; yellow=2")
    if dataset in {"Jinan2", "Hangzhou1"}:
        jinan = dataset == "Jinan2"
        add("CoLLMLight", "1", "Jinan" if jinan else "Hangzhou",
            474.4 if jinan else 530.2,
            "https://proceedings.iclr.cc/paper_files/paper/2026/file/6d7a9f292360193eb530d693f7941c73-Paper-Conference.pdf",
            "ZERO-SHOT Syn-Train -> real data; green/yellow/red=30/3/2",
            "candidate correspondence by city/flow description; file equality not verified")
        add("FutureLight", "2", "Jinan1 (real)" if jinan else "Hangzhou1 (real)",
            271.2 if jinan else 310.7, "https://www.vldb.org/pvldb/vol19/p2344-xu.pdf",
            "80 rounds, last ten tests; Jinan real flow has 4365 departures (not LLMLight Jinan1)",
            "candidate correspondence by city/flow description; file equality not verified")
    if dataset == "Hangzhou1":
        add("AMM", "1", "Hangzhou 4x4", 689.89, "https://arxiv.org/abs/2501.02548",
            "8 phases, control=20s, adaptation task; not a matched four-phase experiment")
    return rows


def aggregate_seed_runs(runs, tail_rounds=10):
    """Average rounds inside a training seed FIRST, then SD over independent seeds."""
    if tail_rounds <= 0 or not runs:
        raise ValueError("need positive tail_rounds and completed runs")
    seeds = [run["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate training seed is not an independent repetition")
    values = []
    for run in runs:
        if run.get("status") != "completed" or not run["rounds"]:
            raise ValueError("only completed training runs may be aggregated")
        tail = run["rounds"][-tail_rounds:]
        keys = set.intersection(*(set(record["evaluation"]) for record in tail))
        values.append({"seed": run["seed"], "rounds_averaged": len(tail),
                       "means": {key: float(np.mean([r["evaluation"][key] for r in tail]))
                                 for key in sorted(keys)}})
    keys = set.intersection(*(set(record["means"]) for record in values))
    metrics = {}
    for key in sorted(keys):
        samples = [record["means"][key] for record in values]
        metrics[key] = {"mean": float(np.mean(samples)),
                        "std": float(np.std(samples, ddof=1)) if len(samples) > 1 else None}
    return {"unit": "independent training seeds; average last rounds within each seed",
            "n_training_seeds": len(values), "tail_rounds_requested": tail_rounds,
            "per_seed": values, "metrics": metrics}


def write_comparison(output: Path, protocol, aggregate, *, reference_rows=None):
    """Write references beside observed results, never manufacture a matched ranking."""
    output.mkdir(parents=True, exist_ok=True)
    metrics = aggregate["metrics"]
    local = metrics["llmtscs_att_s"]
    rows = references(protocol["dataset"]) if reference_rows is None else reference_rows
    for row in rows:
        row.update(our_llmtscs_att_s=local["mean"], our_seed_std_s=local["std"],
                   our_engine_att_s=metrics["average_travel_time_s"]["mean"],
                   comparison="external reference; protocol differences must be considered")
    write_json(output / "paper_comparison.json", {"protocol": protocol, "aggregate": aggregate,
                                                  "paper_references": rows})
    fields = ["paper", "table", "paper_dataset", "paper_att_s", "our_llmtscs_att_s",
              "our_seed_std_s", "our_engine_att_s", "setting", "data_relation", "comparison", "url"]
    with (output / "paper_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    std = "not estimated (one training seed)" if local["std"] is None else f'{local["std"]:.2f}'
    text = [f'# {protocol.get("baseline", "CoLight")} results and published references', "",
            f'Dataset: {protocol["dataset"]}; independent training seeds: {aggregate["n_training_seeds"]}.',
            f'Our LLMTSCS-compatible ATT: {local["mean"]:.2f} s; seed SD: {std}.',
            f'Our CityFlow engine ATT: {metrics["average_travel_time_s"]["mean"]:.2f} s.',
            "", "Paper values are external references, not matched-condition comparisons or reproduction claims.",
            "", "| Paper / table | Dataset | Published ATT (s) | Conditions / data relation |",
            "|---|---|---:|---|"]
    for row in rows:
        text.append(f'| [{row["paper"]}]({row["url"]}) / {row["table"]} | {row["paper_dataset"]} | '
                    f'{row["paper_att_s"]:.2f} | {row["setting"]}; {row["data_relation"]} |')
    if not rows:
        text += ["", "Unrecognized dataset hashes: no automatic paper-dataset correspondence."]
    (output / "paper_comparison.md").write_text("\n".join(text) + "\n", encoding="utf-8")
