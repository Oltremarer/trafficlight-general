from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - covered on ML runtime
    raise RuntimeError(
        "The World Model requires the optional ML dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc

from ..dqn import network_schema_sha256
from ..observations import QueuePressureObservationBuilder
from ..types import NetworkSpec
from .data import FeatureStatistics
from .model import GraphWorldModel, WorldModelConfig


CHECKPOINT_VERSION = "cityflow-tsc-graph-world-model-v1"


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_world_model_checkpoint(
    path: Path,
    model: GraphWorldModel,
    network: NetworkSpec,
    roadnet_sha256: str,
    statistics: FeatureStatistics,
    control: Dict[str, Any],
    training_provenance: Sequence[Dict[str, Any]],
    validation_provenance: Sequence[Dict[str, Any]],
    training_summary: Dict[str, Any],
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "statistics": statistics.to_dict(),
        "feature_names": list(QueuePressureObservationBuilder.feature_names),
        "observation_schema_id": QueuePressureObservationBuilder.schema_id,
        "roadnet_sha256": roadnet_sha256,
        "network_schema_sha256": network_schema_sha256(network),
        "intersection_ids": list(network.intersection_ids),
        "max_movements": network.max_movements,
        "max_actions": network.max_actions,
        "engine_phase_ids": [
            list(item.engine_phase_ids) for item in network.intersections
        ],
        "control": control,
        "training_provenance": list(training_provenance),
        "validation_provenance": list(validation_provenance),
        "training_summary": training_summary,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return checkpoint_sha256(target)


def load_world_model_checkpoint(
    path: Path,
    network: NetworkSpec,
    roadnet_sha256: str,
    device: str = "cpu",
    expected_control: Optional[Dict[str, Any]] = None,
) -> Tuple[GraphWorldModel, FeatureStatistics, Dict[str, Any], str]:
    target = Path(path)
    try:
        payload = torch.load(target, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 2.0 compatibility
        payload = torch.load(target, map_location=device)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported World Model checkpoint version")
    expected = {
        "feature_names": list(QueuePressureObservationBuilder.feature_names),
        "observation_schema_id": QueuePressureObservationBuilder.schema_id,
        "roadnet_sha256": roadnet_sha256,
        "network_schema_sha256": network_schema_sha256(network),
        "intersection_ids": list(network.intersection_ids),
        "max_movements": network.max_movements,
        "max_actions": network.max_actions,
        "engine_phase_ids": [
            list(item.engine_phase_ids) for item in network.intersections
        ],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"checkpoint {key} does not match the runtime data contract"
            )
    if expected_control is not None and payload.get("control") != expected_control:
        raise ValueError(
            "checkpoint control does not match the runtime timing configuration"
        )
    config = WorldModelConfig(**payload["model_config"])
    model = GraphWorldModel(
        config=config,
        neighbor_index=torch.as_tensor(network.neighbor_index, dtype=torch.long),
        neighbor_mask=torch.as_tensor(network.neighbor_mask, dtype=torch.bool),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    statistics = FeatureStatistics.from_dict(payload["statistics"])
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state_dict", "statistics", "model_config"}
    }
    return model, statistics, metadata, checkpoint_sha256(target)
