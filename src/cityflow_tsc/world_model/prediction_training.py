from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ModuleNotFoundError as exc:  # pragma: no cover - covered on ML runtime
    raise RuntimeError(
        "Traffic prediction training requires the optional RL dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc

from ..trajectory import sha256_file
from .latent_prediction import (
    DirectObservationModel,
    PretrainedLatentTrafficModel,
    TrafficPredictionConfig,
)
from .rich_data import RichFeatureStatistics, RichTrajectorySequenceDataset
from .selection import CORE_COUNT_FEATURE_NAMES


@dataclass(frozen=True)
class PredictionTrainConfig:
    epochs: int = 30
    batch_size: int = 64
    adapter_learning_rate: float = 3e-4
    pretrained_learning_rate: float = 3e-5
    frozen_pretrained_epochs: int = 5
    weight_decay: float = 1e-5
    reward_loss_weight: float = 0.1
    checkpoint_reward_loss_weight: float = 0.0
    consistency_loss_weight: float = 2.0
    movement_consistency_loss_weight: float = 2.0
    reconstruction_loss_weight: float = 0.5
    temporal_decay: float = 0.8
    gradient_clip_norm: float = 10.0
    num_workers: int = 0
    sampling_strategy: str = "shuffle"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not 0 <= self.frozen_pretrained_epochs <= self.epochs:
            raise ValueError("frozen_pretrained_epochs must be in [0, epochs]")
        if self.adapter_learning_rate <= 0 or self.pretrained_learning_rate < 0:
            raise ValueError("adapter learning rate must be positive and core rate non-negative")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("optimizer values are invalid")
        if any(
            value < 0
            for value in (
                self.reward_loss_weight,
                self.checkpoint_reward_loss_weight,
                self.consistency_loss_weight,
                self.movement_consistency_loss_weight,
                self.reconstruction_loss_weight,
            )
        ):
            raise ValueError("loss weights cannot be negative")
        if not 0 < self.temporal_decay <= 1:
            raise ValueError("temporal_decay must be in (0, 1]")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.sampling_strategy not in {"shuffle", "flow_policy_balanced"}:
            raise ValueError("unsupported sampling strategy")


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, target)


def _move(batch: Dict[str, "torch.Tensor"], device: "torch.device"):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _weighted_state_loss(
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    mask: "torch.Tensor",
    temporal_decay: float,
) -> "torch.Tensor":
    horizon = prediction.shape[1]
    weights = torch.pow(
        torch.as_tensor(temporal_decay, device=prediction.device),
        torch.arange(horizon, device=prediction.device),
    )
    squared = (prediction - target) ** 2
    per_step = (squared * mask.float()).sum(dim=(0, 2, 3, 4)) / mask.sum(
        dim=(0, 2, 3, 4)
    ).clamp(min=1)
    return (per_step * weights).sum() / weights.sum()


def _two_hot(values: "torch.Tensor", model: PretrainedLatentTrafficModel):
    config = model.config
    symlog = torch.sign(values) * torch.log1p(torch.abs(values))
    symlog = symlog.clamp(config.reward_min, config.reward_max)
    scale = (config.reward_bins - 1) / (config.reward_max - config.reward_min)
    position = (symlog - config.reward_min) * scale
    lower = torch.floor(position).long().clamp(0, config.reward_bins - 1)
    upper = torch.clamp(lower + 1, max=config.reward_bins - 1)
    offset = (position - lower.float()).unsqueeze(-1)
    result = torch.zeros(
        *values.shape,
        config.reward_bins,
        device=values.device,
        dtype=values.dtype,
    )
    result.scatter_(-1, lower.unsqueeze(-1), 1.0 - offset)
    result.scatter_add_(-1, upper.unsqueeze(-1), offset)
    return result


def _target_latents(
    model: PretrainedLatentTrafficModel,
    batch: Dict[str, "torch.Tensor"],
) -> Dict[str, "torch.Tensor"]:
    node_targets = []
    movement_targets = []
    with torch.no_grad():
        for step in range(batch["actions"].shape[1]):
            target_batch = {
                "history_features": batch["target_histories"][:, step],
                "history_valid_mask": batch["target_history_valid_mask"][:, step],
                "history_movement_mask": batch[
                    "target_history_movement_mask"
                ][:, step],
                "history_current_phase": batch[
                    "target_history_current_phase"
                ][:, step],
                "history_signal_stage": batch[
                    "target_history_signal_stage"
                ][:, step],
                "history_phase_elapsed_s": batch[
                    "target_history_phase_elapsed_s"
                ][:, step],
                "history_time_s": batch["target_history_time_s"][:, step],
                "movement_static_features": batch["movement_static_features"],
                "demand_features": batch["demand_features"],
                "neighbor_index": batch["neighbor_index"],
                "neighbor_mask": batch["neighbor_mask"],
            }
            node_latent, movement_latent = model.encode(target_batch)
            node_targets.append(node_latent)
            movement_targets.append(movement_latent)
    return {
        "node": torch.stack(node_targets, dim=1),
        "movement": torch.stack(movement_targets, dim=1),
    }


def _masked_mse(
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    mask: "torch.Tensor",
) -> "torch.Tensor":
    weights = mask.float()
    while weights.ndim < prediction.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(prediction)
    return (((prediction - target) ** 2) * weights).sum() / weights.sum().clamp(
        min=1.0
    )


def prediction_loss(
    model: "torch.nn.Module",
    batch: Dict[str, "torch.Tensor"],
    config: PredictionTrainConfig,
) -> Dict[str, "torch.Tensor"]:
    prediction = model.rollout(batch)
    mask = batch["target_valid_mask"] & batch[
        "target_valid_mask"
    ].any(dim=-1, keepdim=True)
    state_loss = _weighted_state_loss(
        prediction["features"],
        batch["target_features"],
        mask,
        config.temporal_decay,
    )
    if isinstance(model, PretrainedLatentTrafficModel):
        target_distribution = _two_hot(batch["rewards"], model)
        reward_loss = -(
            target_distribution
            * F.log_softmax(prediction["reward_logits"], dim=-1)
        ).sum(dim=-1).mean()
        targets = _target_latents(model, batch)
        node_consistency_loss = F.mse_loss(
            prediction["node_latents"], targets["node"]
        )
        movement_consistency_loss = _masked_mse(
            prediction["movement_latents"],
            targets["movement"],
            batch["target_history_movement_mask"][:, :, -1],
        )
        reconstruction_loss = _masked_mse(
            prediction["reconstruction"],
            batch["history_features"][:, -1],
            batch["history_valid_mask"][:, -1],
        )
        consistency_loss = node_consistency_loss + movement_consistency_loss
    else:
        reward_loss = F.mse_loss(prediction["rewards"], batch["rewards"])
        consistency_loss = torch.zeros((), device=state_loss.device)
        node_consistency_loss = torch.zeros((), device=state_loss.device)
        movement_consistency_loss = torch.zeros((), device=state_loss.device)
        reconstruction_loss = torch.zeros((), device=state_loss.device)
    total = (
        state_loss
        + config.reward_loss_weight * reward_loss
        + config.consistency_loss_weight * node_consistency_loss
        + config.movement_consistency_loss_weight * movement_consistency_loss
        + config.reconstruction_loss_weight * reconstruction_loss
    )
    return {
        "loss": total,
        "state_loss": state_loss,
        "reward_loss": reward_loss,
        "consistency_loss": consistency_loss,
        "node_consistency_loss": node_consistency_loss,
        "movement_consistency_loss": movement_consistency_loss,
        "reconstruction_loss": reconstruction_loss,
    }


def _optimizer(
    model: "torch.nn.Module", config: PredictionTrainConfig
) -> "torch.optim.Optimizer":
    if isinstance(model, PretrainedLatentTrafficModel):
        core_ids = {
            id(parameter)
            for module in (model.dynamics, model.reward_head)
            for parameter in module.parameters()
        }
        core_ids.add(id(model.task_embedding))
        core = [parameter for parameter in model.parameters() if id(parameter) in core_ids]
        adapters = [
            parameter for parameter in model.parameters() if id(parameter) not in core_ids
        ]
        return torch.optim.AdamW(
            [
                {"params": adapters, "lr": config.adapter_learning_rate},
                {"params": core, "lr": 0.0, "name": "pretrained_core"},
            ],
            weight_decay=config.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.adapter_learning_rate,
        weight_decay=config.weight_decay,
    )


def _run_epoch(
    model: "torch.nn.Module",
    loader: DataLoader,
    device: "torch.device",
    config: PredictionTrainConfig,
    optimizer: Optional["torch.optim.Optimizer"],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "state_loss": 0.0,
        "reward_loss": 0.0,
        "consistency_loss": 0.0,
        "node_consistency_loss": 0.0,
        "movement_consistency_loss": 0.0,
        "reconstruction_loss": 0.0,
    }
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                # The compact direct baseline has no pretrained bf16 weights
                # to preserve.  Keep its optimization in fp32 so a numerical
                # stability control is not itself confounded by mixed
                # precision overflow.
                enabled=(
                    device.type == "cuda"
                    and isinstance(model, PretrainedLatentTrafficModel)
                ),
            ):
                losses = prediction_loss(model, batch, config)
            nonfinite = [
                key for key, value in losses.items() if not torch.isfinite(value).all()
            ]
            if nonfinite:
                raise FloatingPointError(
                    "non-finite prediction loss during "
                    f"{'training' if training else 'validation'}: {nonfinite}"
                )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
    if not batches:
        raise ValueError("cannot train or validate on an empty dataset")
    return {key: value / batches for key, value in totals.items()}


def train_prediction_model(
    model: "torch.nn.Module",
    train_dataset: RichTrajectorySequenceDataset,
    validation_dataset: Optional[RichTrajectorySequenceDataset],
    config: PredictionTrainConfig,
    device: str,
    progress_path: Optional[Path] = None,
) -> Dict[str, Any]:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {device!r}, but CUDA is unavailable")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    sampler = None
    if config.sampling_strategy == "flow_policy_balanced":
        sampler = WeightedRandomSampler(
            train_dataset.sampling_weights(config.sampling_strategy),
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator if sampler is None else None,
        num_workers=config.num_workers,
        pin_memory=target.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=target.type == "cuda",
        )
        if validation_dataset is not None
        else None
    )
    model.to(target)
    optimizer = _optimizer(model, config)
    history: List[Dict[str, Any]] = []
    best_epoch = 0
    best_validation_selection_loss = float("inf")
    best_validation_state_loss = float("inf")
    best_state = None
    for epoch in range(config.epochs):
        if isinstance(model, PretrainedLatentTrafficModel):
            core_lr = (
                0.0
                if epoch < config.frozen_pretrained_epochs
                else config.pretrained_learning_rate
            )
            optimizer.param_groups[1]["lr"] = core_lr
            for parameter in optimizer.param_groups[1]["params"]:
                parameter.requires_grad_(core_lr > 0.0)
        train_metrics = _run_epoch(model, train_loader, target, config, optimizer)
        validation_metrics = (
            _run_epoch(model, validation_loader, target, config, None)
            if validation_loader is not None
            else None
        )
        history.append(
            {
                "epoch": epoch + 1,
                "pretrained_core_learning_rate": (
                    float(optimizer.param_groups[1]["lr"])
                    if isinstance(model, PretrainedLatentTrafficModel)
                    else None
                ),
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        selection_metrics = validation_metrics if validation_metrics is not None else train_metrics
        validation_state_loss = float(selection_metrics["state_loss"])
        selection_loss = validation_state_loss
        selection_loss += (
            config.checkpoint_reward_loss_weight
            * float(selection_metrics["reward_loss"])
        )
        if selection_loss < best_validation_selection_loss:
            best_validation_selection_loss = selection_loss
            best_validation_state_loss = validation_state_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if progress_path is not None:
            _write_json_atomic(
                progress_path,
                {
                    "status": "training",
                    "completed_epochs": epoch + 1,
                    "total_epochs": config.epochs,
                    "latest": history[-1],
                    "best_epoch": best_epoch,
                    "best_validation_state_loss": best_validation_state_loss,
                    "best_validation_selection_loss": best_validation_selection_loss,
                },
            )
    if best_state is None:
        raise RuntimeError("training produced no model state")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    summary = {
        "config": asdict(config),
        "model_kind": model.model_kind,
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset)
        if validation_dataset is not None
        else 0,
        "selection_metric": (
            "validation_state_loss"
            if config.checkpoint_reward_loss_weight == 0.0
            else "validation_state_loss_plus_weighted_reward_loss"
        ),
        "checkpoint_reward_loss_weight": config.checkpoint_reward_loss_weight,
        "best_epoch": best_epoch,
        "best_validation_state_loss": best_validation_state_loss,
        "best_validation_selection_loss": best_validation_selection_loss,
        "history": history,
    }
    if progress_path is not None:
        _write_json_atomic(
            progress_path,
            {
                "status": "training_complete_best_state_restored",
                "completed_epochs": config.epochs,
                "total_epochs": config.epochs,
                "best_epoch": best_epoch,
                "best_validation_state_loss": best_validation_state_loss,
                "best_validation_selection_loss": best_validation_selection_loss,
            },
        )
    return summary


@torch.no_grad()
def evaluate_prediction_model(
    model: "torch.nn.Module",
    dataset: RichTrajectorySequenceDataset,
    device: str,
    batch_size: int,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    target = torch.device(device)
    model.to(target).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    horizon = dataset.rollout_horizon
    feature_count = len(feature_names)
    abs_sum = np.zeros((horizon, feature_count), dtype=np.float64)
    square_sum = np.zeros((horizon, feature_count), dtype=np.float64)
    normalized_abs_sum = np.zeros((horizon, feature_count), dtype=np.float64)
    normalized_square_sum = np.zeros((horizon, feature_count), dtype=np.float64)
    counts = np.zeros((horizon, feature_count), dtype=np.float64)
    reward_abs = np.zeros(horizon, dtype=np.float64)
    reward_square = np.zeros(horizon, dtype=np.float64)
    reward_count = np.zeros(horizon, dtype=np.float64)
    reward_available = True
    core_indices = tuple(feature_names.index(name) for name in CORE_COUNT_FEATURE_NAMES)

    for raw_batch in loader:
        batch = _move(raw_batch, target)
        prediction = model.rollout(batch)
        if not torch.isfinite(prediction["features"]).all():
            raise FloatingPointError(
                f"non-finite {model.model_kind} rollout during evaluation"
            )
        normalized_errors = (
            prediction["features"] - batch["target_features"]
        ).float()
        predicted_features = dataset.statistics.denormalize_features_tensor(
            prediction["features"]
        )
        true_features = dataset.statistics.denormalize_features_tensor(
            batch["target_features"]
        )
        errors = (predicted_features - true_features).float()
        if not torch.isfinite(errors).all():
            raise FloatingPointError(
                f"non-finite denormalized {model.model_kind} feature error"
            )
        mask = batch["target_valid_mask"].float()
        abs_sum += (errors.abs() * mask).sum(dim=(0, 2, 3)).cpu().numpy()
        square_sum += ((errors ** 2) * mask).sum(dim=(0, 2, 3)).cpu().numpy()
        normalized_abs_sum += (
            normalized_errors.abs() * mask
        ).sum(dim=(0, 2, 3)).cpu().numpy()
        normalized_square_sum += (
            (normalized_errors ** 2) * mask
        ).sum(dim=(0, 2, 3)).cpu().numpy()
        counts += mask.sum(dim=(0, 2, 3)).cpu().numpy()

        if "rewards" in prediction:
            predicted_rewards = dataset.statistics.denormalize_rewards_tensor(
                prediction["rewards"]
            )
            true_rewards = dataset.statistics.denormalize_rewards_tensor(batch["rewards"])
            reward_errors = (predicted_rewards - true_rewards).float()
            if not torch.isfinite(reward_errors).all():
                raise FloatingPointError(
                    f"non-finite {model.model_kind} reward error"
                )
            reward_abs += reward_errors.abs().sum(dim=(0, 2)).cpu().numpy()
            reward_square += (reward_errors ** 2).sum(dim=(0, 2)).cpu().numpy()
            reward_count += np.prod(reward_errors.shape[::2])
        else:
            reward_available = False

    safe_counts = np.maximum(counts, 1.0)
    feature_mae = abs_sum / safe_counts
    feature_rmse = np.sqrt(square_sum / safe_counts)
    normalized_feature_mae = normalized_abs_sum / safe_counts
    normalized_feature_rmse = np.sqrt(normalized_square_sum / safe_counts)
    safe_reward_count = np.maximum(reward_count, 1.0)
    reward_mae = reward_abs / safe_reward_count
    reward_rmse = np.sqrt(reward_square / safe_reward_count)
    per_horizon = []
    for step in range(horizon):
        per_horizon.append(
            {
                "horizon": step + 1,
                "normalized_feature_mae": float(
                    normalized_abs_sum[step].sum() / safe_counts[step].sum()
                ),
                "normalized_feature_rmse": float(
                    np.sqrt(
                        normalized_square_sum[step].sum()
                        / safe_counts[step].sum()
                    )
                ),
                "core_count_mae": float(
                    abs_sum[step, core_indices].sum()
                    / safe_counts[step, core_indices].sum()
                ),
                "core_count_rmse": float(
                    np.sqrt(
                        square_sum[step, core_indices].sum()
                        / safe_counts[step, core_indices].sum()
                    )
                ),
                "overall_feature_mae": float(abs_sum[step].sum() / safe_counts[step].sum()),
                "overall_feature_rmse": float(
                    np.sqrt(square_sum[step].sum() / safe_counts[step].sum())
                ),
                "reward_mae": float(reward_mae[step]) if reward_available else None,
                "reward_rmse": float(reward_rmse[step]) if reward_available else None,
                "features": {
                    name: {
                        "mae": float(feature_mae[step, index]),
                        "rmse": float(feature_rmse[step, index]),
                        "normalized_mae": float(
                            normalized_feature_mae[step, index]
                        ),
                        "normalized_rmse": float(
                            normalized_feature_rmse[step, index]
                        ),
                        "count": int(counts[step, index]),
                    }
                    for index, name in enumerate(feature_names)
                },
            }
        )
    return {
        "model_kind": model.model_kind,
        "windows": len(dataset),
        "cityflow_role": "ground_truth_targets",
        "core_count_feature_names": list(CORE_COUNT_FEATURE_NAMES),
        "reward_available": reward_available,
        "per_horizon": per_horizon,
    }


def save_prediction_checkpoint(
    path: Path,
    model: "torch.nn.Module",
    model_config: TrafficPredictionConfig,
    statistics: RichFeatureStatistics,
    training_summary: Dict[str, Any],
    train_provenance: Sequence[Dict[str, Any]],
    validation_provenance: Sequence[Dict[str, Any]],
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": "cityflow-rich-prediction-v2",
        "architecture": (
            "node-movement-latent-v1"
            if isinstance(model, PretrainedLatentTrafficModel)
            else "direct-observation-v1"
        ),
        "model_kind": model.model_kind,
        "model_config": model_config.to_dict(),
        "model_state_dict": model.state_dict(),
        "statistics": statistics.to_dict(),
        "training_summary": training_summary,
        "train_provenance": list(train_provenance),
        "validation_provenance": list(validation_provenance),
        "pretrained_report": (
            model.pretrained_report
            if isinstance(model, PretrainedLatentTrafficModel)
            else None
        ),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return sha256_file(target)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    _write_json_atomic(path, value)
