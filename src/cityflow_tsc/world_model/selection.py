"""Validation-only ranking for structured CityFlow prediction models.

The ranking intentionally uses named traffic features.  It must never depend on
JSON object order: summary files are serialized with sorted keys, while the
feature tensor has a semantic, fixed order.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


CORE_COUNT_FEATURE_NAMES = (
    "incoming_vehicle_count",
    "incoming_queue_count",
    "outgoing_vehicle_count",
    "outgoing_queue_count",
)
RATE_FEATURE_NAMES = (
    "recent_arrival_rate_veh_s",
    "recent_departure_rate_veh_s",
)


def _mean_named_normalized_rmse(
    step: Mapping[str, Any], feature_names: Sequence[str]
) -> float:
    features = step.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("evaluation step has no feature-level metrics")
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise ValueError(f"evaluation is missing named features: {missing}")
    return sum(float(features[name]["normalized_rmse"]) for name in feature_names) / len(
        feature_names
    )


def selection_breakdown(evaluation: Mapping[str, Any]) -> Dict[str, float]:
    """Return a state-only validation score and auditable named components.

    The scoring function emphasizes traffic quantities that an intersection
    operator can interpret.  It includes all features so an improvement in
    counts cannot hide a collapse in rates, speed, or occupancy, while the
    last two rollout steps explicitly penalize long-horizon drift.
    """

    steps = evaluation.get("per_horizon")
    if not isinstance(steps, list) or not steps:
        raise ValueError("evaluation must contain non-empty per_horizon metrics")

    horizon_weights = [float(index + 1) for index in range(len(steps))]
    denominator = sum(horizon_weights)
    overall = sum(
        weight * float(step["normalized_feature_rmse"])
        for weight, step in zip(horizon_weights, steps)
    ) / denominator
    core = sum(
        weight * _mean_named_normalized_rmse(step, CORE_COUNT_FEATURE_NAMES)
        for weight, step in zip(horizon_weights, steps)
    ) / denominator
    rates = sum(
        weight * _mean_named_normalized_rmse(step, RATE_FEATURE_NAMES)
        for weight, step in zip(horizon_weights, steps)
    ) / denominator
    long_steps = steps[-min(2, len(steps)) :]
    long_core = sum(
        _mean_named_normalized_rmse(step, CORE_COUNT_FEATURE_NAMES)
        for step in long_steps
    ) / len(long_steps)
    score = 0.40 * core + 0.20 * rates + 0.20 * overall + 0.20 * long_core
    return {
        "selection_score": float(score),
        "weighted_overall_normalized_rmse": float(overall),
        "weighted_core_count_normalized_rmse": float(core),
        "weighted_rate_normalized_rmse": float(rates),
        "long_horizon_core_count_normalized_rmse": float(long_core),
    }


def selection_from_summary(summary: Mapping[str, Any]) -> Dict[str, float]:
    evaluation = summary.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("summary has no validation evaluation")
    return selection_breakdown(evaluation)
