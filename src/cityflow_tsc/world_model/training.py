from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - covered on ML runtime
    raise RuntimeError(
        "The World Model requires the optional ML dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc

from .data import TrajectoryTransitionDataset
from .model import GraphWorldModel


@dataclass(frozen=True)
class WorldModelTrainConfig:
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 10.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("training epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("training optimizer values are invalid")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


def _move(batch: Dict[str, "torch.Tensor"], device: "torch.device"):
    return {key: value.to(device) for key, value in batch.items()}


def _loss(
    model: GraphWorldModel, batch: Dict[str, "torch.Tensor"]
) -> Dict[str, "torch.Tensor"]:
    prediction = model(
        batch["features"],
        batch["valid_mask"],
        batch["movement_mask"],
        batch["current_phase"],
        batch["signal_stage"],
        batch["phase_elapsed_s"],
        batch["actions"],
    )
    state_mask = (
        batch["valid_mask"]
        & batch["next_valid_mask"]
        & batch["movement_mask"][..., None]
    )
    squared = (prediction["next_features"] - batch["next_features"]) ** 2
    state_loss = (squared * state_mask.float()).sum() / state_mask.sum().clamp(min=1)
    reward_loss = F.mse_loss(prediction["rewards"], batch["rewards"])
    total = state_loss + model.config.reward_loss_weight * reward_loss
    return {
        "loss": total,
        "state_loss": state_loss,
        "reward_loss": reward_loss,
    }


def _run_loader(
    model: GraphWorldModel,
    loader: DataLoader,
    device: "torch.device",
    optimizer: Optional["torch.optim.Optimizer"],
    gradient_clip_norm: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "state_loss": 0.0, "reward_loss": 0.0}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            losses = _loss(model, batch)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
    if not batches:
        raise ValueError("cannot train or validate on an empty transition dataset")
    return {key: value / batches for key, value in totals.items()}


def train_world_model(
    model: GraphWorldModel,
    train_dataset: TrajectoryTransitionDataset,
    validation_dataset: Optional[TrajectoryTransitionDataset],
    config: WorldModelTrainConfig,
    device: str,
) -> Dict[str, Any]:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {device!r}, but CUDA is unavailable")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )
    model.to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: List[Dict[str, Any]] = []
    for epoch in range(config.epochs):
        train_metrics = _run_loader(
            model,
            train_loader,
            target,
            optimizer,
            config.gradient_clip_norm,
        )
        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = _run_loader(
                model,
                validation_loader,
                target,
                None,
                config.gradient_clip_norm,
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
    model.eval()
    return {
        "config": asdict(config),
        "train_transitions": len(train_dataset),
        "validation_transitions": len(validation_dataset)
        if validation_dataset is not None
        else 0,
        "history": history,
    }
