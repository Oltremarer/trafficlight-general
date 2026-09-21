#!/usr/bin/env python3
"""Generate the fixed, validation-only candidate set for the second sweep.

The first 24 rows cover a wider region.  The remaining rows densely sample
neighbourhoods of the two previous validation leaders (Jinan T04 and Hangzhou
T28).  The same frozen set is evaluated in each city: that keeps the first
screen fair and lets the result itself say whether the optima are city-specific.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List


FIELDS = (
    "candidate_id",
    "region",
    "encoder_hidden_dim",
    "movement_latent_dim",
    "history_length",
    "adapter_learning_rate",
    "pretrained_learning_rate",
    "frozen_pretrained_epochs",
    "reward_loss_weight",
    "node_consistency_loss_weight",
    "movement_consistency_loss_weight",
    "reconstruction_loss_weight",
    "temporal_decay",
    "batch_size",
    "weight_decay",
    "gradient_clip_norm",
    "sampling_strategy",
    "seed",
)


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _format(row: Dict[str, object]) -> Dict[str, str]:
    result = {}
    for field in FIELDS:
        value = row[field]
        result[field] = f"{value:.8g}" if isinstance(value, float) else str(value)
    return result


def _wide(rng: random.Random, candidate_id: str) -> Dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "region": "wide_coverage",
        "encoder_hidden_dim": rng.choice((128, 160, 192, 224, 256, 288, 320, 384)),
        "movement_latent_dim": rng.choice((64, 80, 96, 112, 128, 144, 160, 192)),
        "history_length": rng.choice((1, 2, 3, 4, 5, 6, 7)),
        "adapter_learning_rate": _log_uniform(rng, 1e-4, 8e-4),
        "pretrained_learning_rate": rng.choice(
            (0.0, 3e-6, 7e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5)
        ),
        "frozen_pretrained_epochs": rng.choice((0, 1, 2, 3, 5, 7, 10, 12)),
        "reward_loss_weight": rng.choice((0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3)),
        "node_consistency_loss_weight": rng.choice((0.5, 1.0, 1.5, 2.0, 3.0, 4.0)),
        "movement_consistency_loss_weight": rng.choice((0.5, 1.0, 2.0, 3.0, 4.0, 6.0)),
        "reconstruction_loss_weight": rng.choice((0.1, 0.25, 0.5, 0.75, 1.0)),
        "temporal_decay": rng.choice((0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)),
        "batch_size": rng.choice((48, 64, 80, 96, 128)),
        "weight_decay": _log_uniform(rng, 1e-6, 3e-4),
        "gradient_clip_norm": rng.choice((3.0, 5.0, 10.0, 20.0)),
        "sampling_strategy": "flow_policy_balanced",
        "seed": 173,
    }


def _local(
    rng: random.Random, candidate_id: str, region: str, base: Dict[str, object]
) -> Dict[str, object]:
    row = dict(base)
    row.update(
        {
            "candidate_id": candidate_id,
            "region": region,
            "encoder_hidden_dim": rng.choice(
                (192, 224, 256, 256, 256, 288, 320)
            ),
            "movement_latent_dim": rng.choice((96, 112, 128, 128, 128, 144, 160)),
            "history_length": rng.choice((
                max(1, int(base["history_length"]) - 2),
                max(1, int(base["history_length"]) - 1),
                int(base["history_length"]),
                int(base["history_length"]),
                min(7, int(base["history_length"]) + 1),
                min(7, int(base["history_length"]) + 2),
            )),
            "adapter_learning_rate": _log_uniform(rng, 2e-4, 6e-4),
            "pretrained_learning_rate": rng.choice((0.0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5)),
            "frozen_pretrained_epochs": rng.choice((0, 1, 2, 3, 4, 5, 6, 8)),
            "reward_loss_weight": rng.choice((0.0, 0.05, 0.1, 0.15, 0.2)),
            "node_consistency_loss_weight": rng.choice((1.0, 1.5, 2.0, 2.5, 3.0)),
            "movement_consistency_loss_weight": rng.choice((1.0, 1.5, 2.0, 2.5, 3.0)),
            "reconstruction_loss_weight": rng.choice((0.25, 0.5, 0.75)),
            "temporal_decay": rng.choice((0.75, 0.8, 0.85, 0.9)),
            "batch_size": rng.choice((64, 80, 96, 128)),
            "weight_decay": _log_uniform(rng, 3e-6, 1e-4),
            "gradient_clip_norm": rng.choice((5.0, 10.0, 20.0)),
            "sampling_strategy": "flow_policy_balanced",
            "seed": 173,
        }
    )
    return row


def generate(seed: int = 20260825) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for index in range(24):
        rows.append(_wide(rng, f"C{index + 1:03d}"))
    jinan_t04 = {
        "history_length": 3,
        "encoder_hidden_dim": 256,
        "movement_latent_dim": 128,
    }
    hangzhou_t28 = {
        "history_length": 5,
        "encoder_hidden_dim": 256,
        "movement_latent_dim": 128,
    }
    for index in range(48):
        rows.append(_local(rng, f"C{len(rows) + 1:03d}", "dense_jinan_T04", jinan_t04))
    for index in range(48):
        rows.append(_local(rng, f"C{len(rows) + 1:03d}", "dense_hangzhou_T28", hangzhou_t28))
    if len(rows) != 120:
        raise AssertionError("expanded plan must contain 120 candidates")
    return [_format(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(generate(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
