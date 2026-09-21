from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - covered on ML runtime
    raise RuntimeError(
        "The World Model requires the optional ML dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc


@dataclass(frozen=True)
class WorldModelConfig:
    hidden_dim: int = 128
    max_actions: int = 4
    reward_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.max_actions <= 0:
            raise ValueError("World Model dimensions must be positive")
        if self.reward_loss_weight <= 0:
            raise ValueError("reward_loss_weight must be positive")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class GraphWorldModel(nn.Module):
    """Predict next primitive traffic counts and reward under a joint action."""

    primitive_feature_count = 4

    def __init__(
        self,
        config: WorldModelConfig,
        neighbor_index: "torch.Tensor",
        neighbor_mask: "torch.Tensor",
    ) -> None:
        super().__init__()
        if neighbor_index.ndim != 2 or neighbor_index.shape != neighbor_mask.shape:
            raise ValueError("neighbor tensors must have matching [N, D] shapes")
        self.config = config
        hidden = config.hidden_dim
        self.register_buffer("neighbor_index", neighbor_index.long())
        self.register_buffer("neighbor_mask", neighbor_mask.bool())
        self.movement_encoder = nn.Sequential(
            nn.Linear(self.primitive_feature_count * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.phase_embedding = nn.Embedding(config.max_actions + 1, hidden)
        self.stage_embedding = nn.Embedding(3, hidden)
        self.elapsed_encoder = nn.Sequential(nn.Linear(1, hidden), nn.Tanh())
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.action_embedding = nn.Embedding(config.max_actions, hidden)
        self.transition_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.primitive_feature_count),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _graph_context(
        self,
        features: "torch.Tensor",
        valid_mask: "torch.Tensor",
        movement_mask: "torch.Tensor",
        current_phase: "torch.Tensor",
        signal_stage: "torch.Tensor",
        phase_elapsed_s: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if features.ndim != 4 or features.shape[-1] != self.primitive_feature_count:
            raise ValueError("features must have shape [B, N, M, 4]")
        encoded_input = torch.cat(
            [features * valid_mask.float(), valid_mask.float()], dim=-1
        )
        movement = self.movement_encoder(encoded_input)
        movement = movement * movement_mask[..., None].float()
        divisor = movement_mask.sum(dim=2, keepdim=True).clamp(min=1).float()
        pooled = movement.sum(dim=2) / divisor

        unknown_phase = torch.full_like(current_phase, self.config.max_actions)
        phase = torch.where(
            (current_phase >= 0) & (current_phase < self.config.max_actions),
            current_phase,
            unknown_phase,
        )
        stage = signal_stage.clamp(min=0, max=2)
        elapsed = torch.log1p(torch.clamp(phase_elapsed_s, min=0.0))[..., None]
        node = self.node_encoder(
            torch.cat(
                [
                    pooled,
                    self.phase_embedding(phase),
                    self.stage_embedding(stage),
                    self.elapsed_encoder(elapsed),
                ],
                dim=-1,
            )
        )

        safe_index = self.neighbor_index.clamp(min=0)
        neighbors = node[:, safe_index]
        neighbor_weights = self.neighbor_mask[None, ..., None].float()
        neighbor_sum = (neighbors * neighbor_weights).sum(dim=2)
        neighbor_count = neighbor_weights.sum(dim=2).clamp(min=1.0)
        neighbor_mean = neighbor_sum / neighbor_count
        graph = self.graph_encoder(torch.cat([node, neighbor_mean], dim=-1))
        return movement, graph

    def forward(
        self,
        features: "torch.Tensor",
        valid_mask: "torch.Tensor",
        movement_mask: "torch.Tensor",
        current_phase: "torch.Tensor",
        signal_stage: "torch.Tensor",
        phase_elapsed_s: "torch.Tensor",
        actions: "torch.Tensor",
    ) -> Dict[str, "torch.Tensor"]:
        movement, graph = self._graph_context(
            features,
            valid_mask,
            movement_mask,
            current_phase,
            signal_stage,
            phase_elapsed_s,
        )
        action = self.action_embedding(actions)
        expanded_graph = graph[:, :, None, :].expand_as(movement)
        expanded_action = action[:, :, None, :].expand_as(movement)
        delta = self.transition_head(
            torch.cat([movement, expanded_graph, expanded_action], dim=-1)
        )
        next_features = features + delta
        next_features = next_features * valid_mask.float()
        rewards = self.reward_head(torch.cat([graph, action], dim=-1)).squeeze(-1)
        return {
            "next_features": next_features,
            "rewards": rewards,
        }

    def rollout_scores(
        self,
        features: "torch.Tensor",
        valid_mask: "torch.Tensor",
        movement_mask: "torch.Tensor",
        current_phase: "torch.Tensor",
        signal_stage: "torch.Tensor",
        phase_elapsed_s: "torch.Tensor",
        action_sequences: "torch.Tensor",
        decision_interval_s: float,
        transition_time_s: float,
        discount: float,
    ) -> "torch.Tensor":
        """Roll out candidate joint-action sequences and return predicted returns."""

        if action_sequences.ndim != 3:
            raise ValueError("action_sequences must have shape [B, H, N]")
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        scores = torch.zeros(features.shape[0], dtype=features.dtype, device=features.device)
        state = features
        phase = current_phase
        stage = signal_stage
        elapsed = phase_elapsed_s
        green_after_change = max(float(decision_interval_s - transition_time_s), 0.0)
        for step in range(action_sequences.shape[1]):
            actions = action_sequences[:, step]
            prediction = self(
                state,
                valid_mask,
                movement_mask,
                phase,
                stage,
                elapsed,
                actions,
            )
            scores = scores + (discount ** step) * prediction["rewards"].sum(dim=1)
            same_phase = actions == phase
            elapsed = torch.where(
                same_phase,
                elapsed + float(decision_interval_s),
                torch.full_like(elapsed, green_after_change),
            )
            phase = actions
            stage = torch.zeros_like(stage)
            state = prediction["next_features"]
        return scores
