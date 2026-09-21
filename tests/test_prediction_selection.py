import pytest

from cityflow_tsc.world_model.prediction_training import PredictionTrainConfig
from cityflow_tsc.world_model.selection import selection_breakdown


def _feature(value: float):
    return {"normalized_rmse": value}


def test_selection_uses_named_vehicle_queue_features_not_json_order() -> None:
    features = {
        # Alphabetical order would put this first, but it is not a core count.
        "downstream_occupancy_ratio": _feature(100.0),
        "incoming_vehicle_count": _feature(1.0),
        "incoming_queue_count": _feature(2.0),
        "outgoing_vehicle_count": _feature(3.0),
        "outgoing_queue_count": _feature(4.0),
        "recent_arrival_rate_veh_s": _feature(5.0),
        "recent_departure_rate_veh_s": _feature(7.0),
    }
    result = selection_breakdown(
        {"per_horizon": [{"normalized_feature_rmse": 9.0, "features": features}]}
    )
    assert result["weighted_core_count_normalized_rmse"] == pytest.approx(2.5)
    assert result["weighted_rate_normalized_rmse"] == pytest.approx(6.0)
    assert result["selection_score"] == pytest.approx(4.5)


def test_adapter_only_finetune_is_a_valid_config() -> None:
    config = PredictionTrainConfig(
        epochs=4,
        frozen_pretrained_epochs=4,
        pretrained_learning_rate=0.0,
    )
    assert config.frozen_pretrained_epochs == 4
