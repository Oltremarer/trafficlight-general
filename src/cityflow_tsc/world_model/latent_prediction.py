from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - covered on ML runtime
    raise RuntimeError(
        "Latent traffic models require the optional RL dependencies. "
        "Install this project with: python -m pip install -e '.[rl]'"
    ) from exc


class SimNorm(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, values: "torch.Tensor") -> "torch.Tensor":
        shape = values.shape
        values = values.reshape(*shape[:-1], -1, self.dim)
        return F.softmax(values, dim=-1).reshape(shape)


class NormedLinear(nn.Linear):
    """TD-MPC2-compatible linear, layer norm, and Mish block."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.act = activation if activation is not None else nn.Mish(inplace=False)

    def forward(self, values: "torch.Tensor") -> "torch.Tensor":
        return self.act(self.ln(F.linear(values, self.weight, self.bias)))


def _tdmpc_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    output_activation: Optional[nn.Module],
) -> nn.Sequential:
    return nn.Sequential(
        NormedLinear(input_dim, hidden_dim),
        NormedLinear(hidden_dim, hidden_dim),
        NormedLinear(hidden_dim, output_dim, output_activation)
        if output_activation is not None
        else nn.Linear(hidden_dim, output_dim),
    )


def _gather_neighbors(
    values: "torch.Tensor",
    neighbor_index: "torch.Tensor",
    neighbor_mask: "torch.Tensor",
) -> "torch.Tensor":
    batch, nodes, hidden = values.shape
    safe = neighbor_index.clamp(min=0)
    gathered = torch.gather(
        values,
        1,
        safe.reshape(batch, -1)[..., None].expand(-1, -1, hidden),
    ).reshape(batch, nodes, safe.shape[-1], hidden)
    weights = neighbor_mask[..., None].float()
    return (gathered * weights).sum(dim=2) / weights.sum(dim=2).clamp(min=1.0)


@dataclass(frozen=True)
class TrafficPredictionConfig:
    feature_count: int
    static_feature_count: int
    demand_feature_count: int
    max_actions: int
    encoder_hidden_dim: int = 128
    latent_dim: int = 512
    movement_latent_dim: int = 64
    action_dim: int = 6
    task_dim: int = 96
    tdmpc_hidden_dim: int = 512
    reward_bins: int = 101
    reward_min: float = -10.0
    reward_max: float = 10.0
    simnorm_dim: int = 8
    # The direct model rolls its own predictions back into its observation
    # history.  Unlike the latent model's SimNorm state, that recursion used
    # to be unbounded and could make the signed-log inverse overflow at test
    # time.  These are deliberately expressed in normalized units so they do
    # not encode a city-specific raw vehicle-count limit.
    direct_max_normalized_delta: float = 3.0
    direct_normalized_state_limit: float = 12.0
    direct_normalized_reward_limit: float = 10.0

    def __post_init__(self) -> None:
        integer_values = (
            self.feature_count,
            self.static_feature_count,
            self.demand_feature_count,
            self.max_actions,
            self.encoder_hidden_dim,
            self.latent_dim,
            self.movement_latent_dim,
            self.action_dim,
            self.task_dim,
            self.tdmpc_hidden_dim,
            self.reward_bins,
            self.simnorm_dim,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("traffic prediction dimensions must be positive")
        if self.latent_dim % self.simnorm_dim:
            raise ValueError("latent_dim must be divisible by simnorm_dim")
        if self.reward_min >= self.reward_max:
            raise ValueError("reward_min must be smaller than reward_max")
        if any(
            value <= 0
            for value in (
                self.direct_max_normalized_delta,
                self.direct_normalized_state_limit,
                self.direct_normalized_reward_limit,
            )
        ):
            raise ValueError("direct-model stability limits must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActionSemanticEncoder(nn.Module):
    """Map discrete local and neighbor phases to TD-MPC2's six action channels."""

    def __init__(self, max_actions: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.max_actions = int(max_actions)
        self.layers = nn.Sequential(
            nn.Linear(2 * max_actions + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        actions: "torch.Tensor",
        current_phase: "torch.Tensor",
        phase_movement_mask: "torch.Tensor",
        movement_mask: "torch.Tensor",
        neighbor_index: "torch.Tensor",
        neighbor_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        if actions.ndim != 2:
            raise ValueError("actions must have shape [B, N]")
        batch, nodes = actions.shape
        local = F.one_hot(actions, self.max_actions).float()
        safe = neighbor_index.clamp(min=0)
        neighbor_actions = torch.gather(
            actions,
            1,
            safe.reshape(batch, -1),
        ).reshape(batch, nodes, safe.shape[-1])
        neighbor_one_hot = F.one_hot(
            neighbor_actions, self.max_actions
        ).float()
        weights = neighbor_mask[..., None].float()
        neighbor_mean = (neighbor_one_hot * weights).sum(dim=2) / weights.sum(
            dim=2
        ).clamp(min=1.0)

        selected = self.selected_movement_mask(
            actions, phase_movement_mask, movement_mask
        )
        valid_movements = movement_mask.float()
        served_fraction = selected.sum(dim=-1, keepdim=True) / valid_movements.sum(
            dim=-1, keepdim=True
        ).clamp(min=1.0)
        changed = (actions != current_phase).float()[..., None]
        has_neighbor = neighbor_mask.any(dim=-1, keepdim=True).float()
        return self.layers(
            torch.cat(
                [local, neighbor_mean, served_fraction, changed, has_neighbor],
                dim=-1,
            )
        )

    @staticmethod
    def selected_movement_mask(
        actions: "torch.Tensor",
        phase_movement_mask: "torch.Tensor",
        movement_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return the exact movements served by each selected local phase."""
        selected = torch.gather(
            phase_movement_mask,
            2,
            actions[..., None, None].expand(
                -1, -1, 1, phase_movement_mask.shape[-1]
            ),
        ).squeeze(2)
        return selected.float() * movement_mask.float()

    @classmethod
    def movement_features(
        cls,
        actions: "torch.Tensor",
        current_phase: "torch.Tensor",
        phase_movement_mask: "torch.Tensor",
        movement_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Keep phase semantics per movement instead of reducing them to a ratio."""
        selected = cls.selected_movement_mask(
            actions, phase_movement_mask, movement_mask
        )
        changed = (
            (actions != current_phase).float()[..., None]
            * movement_mask.float()
        )
        return torch.stack([selected, changed], dim=-1)


class TrafficHistoryEncoder(nn.Module):
    def __init__(self, config: TrafficPredictionConfig) -> None:
        super().__init__()
        hidden = config.encoder_hidden_dim
        self.config = config
        self.movement_encoder = nn.Sequential(
            nn.Linear(
                config.feature_count * 2 + config.static_feature_count,
                hidden,
            ),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.phase_embedding = nn.Embedding(config.max_actions + 1, hidden)
        self.stage_embedding = nn.Embedding(3, hidden)
        self.elapsed_encoder = nn.Sequential(nn.Linear(1, hidden), nn.Tanh())
        self.time_encoder = nn.Sequential(nn.Linear(3, hidden), nn.Tanh())
        self.demand_encoder = nn.Sequential(
            nn.Linear(config.demand_feature_count, hidden), nn.ReLU()
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden * 6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.history_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

    def forward(
        self,
        history_features: "torch.Tensor",
        history_valid_mask: "torch.Tensor",
        history_movement_mask: "torch.Tensor",
        history_current_phase: "torch.Tensor",
        history_signal_stage: "torch.Tensor",
        history_phase_elapsed_s: "torch.Tensor",
        history_time_s: "torch.Tensor",
        movement_static_features: "torch.Tensor",
        demand_features: "torch.Tensor",
        neighbor_index: "torch.Tensor",
        neighbor_mask: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        if history_features.ndim != 5:
            raise ValueError("history_features must have shape [B, T, N, M, F]")
        batch, history, nodes, movements, _ = history_features.shape
        static = movement_static_features[:, None].expand(
            -1, history, -1, -1, -1
        )
        movement = self.movement_encoder(
            torch.cat(
                [
                    history_features * history_valid_mask.float(),
                    history_valid_mask.float(),
                    static,
                ],
                dim=-1,
            )
        )
        movement = movement * history_movement_mask[..., None].float()
        pooled = movement.sum(dim=3) / history_movement_mask.sum(
            dim=3, keepdim=True
        ).clamp(min=1).float()

        unknown = torch.full_like(history_current_phase, self.config.max_actions)
        phase = torch.where(
            (history_current_phase >= 0)
            & (history_current_phase < self.config.max_actions),
            history_current_phase,
            unknown,
        )
        stage = history_signal_stage.clamp(min=0, max=2)
        elapsed = torch.log1p(
            torch.clamp(history_phase_elapsed_s, min=0.0)
        )[..., None]
        angle = 2.0 * torch.pi * history_time_s / 3600.0
        time_features = torch.stack(
            [
                torch.log1p(torch.clamp(history_time_s, min=0.0)) / 10.0,
                torch.sin(angle),
                torch.cos(angle),
            ],
            dim=-1,
        )
        demand = self.demand_encoder(demand_features)[:, None, None].expand(
            -1, history, nodes, -1
        )
        encoded_time = self.time_encoder(time_features)[:, :, None].expand(
            -1, -1, nodes, -1
        )
        node = self.node_encoder(
            torch.cat(
                [
                    pooled,
                    self.phase_embedding(phase),
                    self.stage_embedding(stage),
                    self.elapsed_encoder(elapsed),
                    encoded_time,
                    demand,
                ],
                dim=-1,
            )
        )

        graph_steps = []
        for time_index in range(history):
            current = node[:, time_index]
            neighbors = _gather_neighbors(current, neighbor_index, neighbor_mask)
            graph_steps.append(
                self.graph_encoder(torch.cat([current, neighbors], dim=-1))
            )
        graph = torch.stack(graph_steps, dim=2)
        recurrent, _ = self.history_gru(graph.reshape(batch * nodes, history, -1))
        recurrent = recurrent[:, -1].reshape(batch, nodes, -1)
        context = self.output(torch.cat([graph[:, :, -1], recurrent], dim=-1))
        return movement[:, -1], context, movement


class DirectObservationModel(nn.Module):
    """Predict in observation space; no persistent learned rollout state."""

    model_kind = "direct_observation"

    def __init__(self, config: TrafficPredictionConfig) -> None:
        super().__init__()
        hidden = config.encoder_hidden_dim
        self.config = config
        self.encoder = TrafficHistoryEncoder(config)
        self.action_encoder = ActionSemanticEncoder(
            config.max_actions, config.action_dim, hidden
        )
        self.action_projection = nn.Linear(config.action_dim, hidden)
        self.transition = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.feature_count),
        )
        self.reward = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def _one_step(self, batch: Dict[str, "torch.Tensor"]):
        movement, context, _ = self.encoder(
            batch["history_features"],
            batch["history_valid_mask"],
            batch["history_movement_mask"],
            batch["history_current_phase"],
            batch["history_signal_stage"],
            batch["history_phase_elapsed_s"],
            batch["history_time_s"],
            batch["movement_static_features"],
            batch["demand_features"],
            batch["neighbor_index"],
            batch["neighbor_mask"],
        )
        action = self.action_projection(self.action_encoder(
            batch["step_actions"],
            batch["history_current_phase"][:, -1],
            batch["phase_movement_mask"],
            batch["history_movement_mask"][:, -1],
            batch["neighbor_index"],
            batch["neighbor_mask"],
        ))
        expanded_context = context[:, :, None].expand_as(movement)
        expanded_action = action[:, :, None].expand_as(movement)
        raw_delta = self.transition(
            torch.cat([movement, expanded_context, expanded_action], dim=-1)
        )
        # Keep the residual update locally linear while preventing recursive
        # multi-step rollout from leaving the finite, normalized observation
        # domain.  This remains a direct-observation model: no latent state is
        # introduced or carried across rollout steps.
        delta_limit = self.config.direct_max_normalized_delta
        delta = delta_limit * torch.tanh(raw_delta / delta_limit)
        current = batch["history_features"][:, -1]
        next_features = torch.clamp(
            current + delta,
            min=-self.config.direct_normalized_state_limit,
            max=self.config.direct_normalized_state_limit,
        )
        next_features = next_features * batch["history_valid_mask"][:, -1].float()
        raw_reward = self.reward(torch.cat([context, action], dim=-1)).squeeze(-1)
        reward_limit = self.config.direct_normalized_reward_limit
        reward = reward_limit * torch.tanh(raw_reward / reward_limit)
        return next_features, reward

    def rollout(self, batch: Dict[str, "torch.Tensor"]):
        work = {key: value for key, value in batch.items()}
        predictions = []
        rewards = []
        horizon = batch["actions"].shape[1]
        for step in range(horizon):
            work["step_actions"] = batch["actions"][:, step]
            next_features, reward = self._one_step(work)
            predictions.append(next_features)
            rewards.append(reward)
            work = _advance_history(work, next_features, batch, step)
        return {
            "features": torch.stack(predictions, dim=1),
            "rewards": torch.stack(rewards, dim=1),
        }


class PretrainedLatentTrafficModel(nn.Module):
    """TD-MPC2 node dynamics with traffic-specific persistent movement slots."""

    model_kind = "pretrained_latent"

    def __init__(self, config: TrafficPredictionConfig) -> None:
        super().__init__()
        hidden = config.encoder_hidden_dim
        self.config = config
        self.encoder = TrafficHistoryEncoder(config)
        self.to_latent = nn.Sequential(
            nn.Linear(hidden, config.latent_dim), SimNorm(config.simnorm_dim)
        )
        self.movement_history_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.to_movement_latent = nn.Sequential(
            nn.Linear(hidden, config.movement_latent_dim),
            nn.LayerNorm(config.movement_latent_dim),
            nn.Tanh(),
        )
        self.action_encoder = ActionSemanticEncoder(
            config.max_actions, config.action_dim, hidden
        )
        self.task_embedding = nn.Parameter(torch.zeros(config.task_dim))
        td_input = config.latent_dim + config.action_dim + config.task_dim
        self.dynamics = _tdmpc_mlp(
            td_input,
            config.tdmpc_hidden_dim,
            config.latent_dim,
            SimNorm(config.simnorm_dim),
        )
        self.reward_head = _tdmpc_mlp(
            td_input,
            config.tdmpc_hidden_dim,
            config.reward_bins,
            None,
        )
        self.movement_query = nn.Sequential(
            nn.Linear(config.static_feature_count, hidden), nn.ReLU()
        )
        self.movement_node_context = nn.Sequential(
            nn.Linear(config.latent_dim * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.movement_action_context = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.movement_natural_dynamics = nn.Sequential(
            nn.Linear(config.movement_latent_dim + hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.movement_latent_dim),
        )
        self.movement_service_dynamics = nn.Sequential(
            nn.Linear(config.movement_latent_dim + hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.movement_latent_dim),
        )
        self.movement_transition_norm = nn.LayerNorm(config.movement_latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(
                config.latent_dim + config.movement_latent_dim + hidden,
                config.tdmpc_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(config.tdmpc_hidden_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.feature_count),
        )
        self._pretrained_report: Optional[Dict[str, Any]] = None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load_tdmpc2_checkpoint(self, path: Path) -> Dict[str, Any]:
        checkpoint = Path(path).expanduser().resolve()
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch 2.0 compatibility
            payload = torch.load(checkpoint, map_location="cpu")
        source = payload.get("model", payload)
        mappings = {
            "dynamics": "_dynamics.",
            "reward_head": "_reward.",
        }
        loaded_parameters = 0
        loaded_tensors = 0
        for module_name, source_prefix in mappings.items():
            module = getattr(self, module_name)
            target_state = module.state_dict()
            source_state = {
                key[len(source_prefix) :]: value
                for key, value in source.items()
                if key.startswith(source_prefix)
            }
            if set(source_state) != set(target_state):
                missing = sorted(set(target_state) - set(source_state))
                extra = sorted(set(source_state) - set(target_state))
                raise ValueError(
                    f"TD-MPC2 {module_name} keys do not match: missing={missing}, extra={extra}"
                )
            for key, target in target_state.items():
                if tuple(source_state[key].shape) != tuple(target.shape):
                    raise ValueError(
                        f"TD-MPC2 {module_name}.{key} has shape "
                        f"{tuple(source_state[key].shape)}, expected {tuple(target.shape)}"
                    )
                loaded_parameters += int(target.numel())
                loaded_tensors += 1
            module.load_state_dict(source_state, strict=True)

        task_weights = source.get("_task_emb.weight")
        if task_weights is None or tuple(task_weights.shape[1:]) != (
            self.config.task_dim,
        ):
            raise ValueError("TD-MPC2 checkpoint has no compatible task embedding")
        with torch.no_grad():
            self.task_embedding.copy_(task_weights.float().mean(dim=0))
        loaded_parameters += int(self.task_embedding.numel())
        loaded_tensors += 1
        report = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": self._sha256(checkpoint),
            "loaded_tensors": loaded_tensors,
            "loaded_parameters": loaded_parameters,
            "source_task_count": int(task_weights.shape[0]),
            "source": "nicklashansen/tdmpc2 multitask MT80-5M",
            "license": "MIT",
        }
        self._pretrained_report = report
        return report

    @property
    def pretrained_report(self) -> Dict[str, Any]:
        if self._pretrained_report is None:
            raise RuntimeError("TD-MPC2 checkpoint has not been loaded")
        return dict(self._pretrained_report)

    def encode(
        self, batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        _, context, movement_history = self.encoder(
            batch["history_features"],
            batch["history_valid_mask"],
            batch["history_movement_mask"],
            batch["history_current_phase"],
            batch["history_signal_stage"],
            batch["history_phase_elapsed_s"],
            batch["history_time_s"],
            batch["movement_static_features"],
            batch["demand_features"],
            batch["neighbor_index"],
            batch["neighbor_mask"],
        )
        batch_size, history, nodes, movements, hidden = movement_history.shape
        sequences = movement_history.permute(0, 2, 3, 1, 4).reshape(
            batch_size * nodes * movements, history, hidden
        )
        recurrent, _ = self.movement_history_gru(sequences)
        movement_context = recurrent[:, -1].reshape(
            batch_size, nodes, movements, hidden
        )
        movement_mask = batch["history_movement_mask"][:, -1, ..., None].float()
        movement_latent = self.to_movement_latent(movement_context) * movement_mask
        return self.to_latent(context), movement_latent

    def _action(
        self, batch: Dict[str, "torch.Tensor"], step: int
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        actions = batch["actions"][:, step]
        movement_mask = batch["history_movement_mask"][:, -1]
        node_action = self.action_encoder(
            actions,
            batch["rolling_phase"],
            batch["phase_movement_mask"],
            movement_mask,
            batch["neighbor_index"],
            batch["neighbor_mask"],
        )
        movement_action = self.action_encoder.movement_features(
            actions,
            batch["rolling_phase"],
            batch["phase_movement_mask"],
            movement_mask,
        )
        return node_action, movement_action

    def _transition_node(
        self, latent: "torch.Tensor", action: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        task = self.task_embedding[None, None].expand(
            latent.shape[0], latent.shape[1], -1
        )
        values = torch.cat([latent, action, task], dim=-1)
        return self.dynamics(values), self.reward_head(values)

    def _transition_movements(
        self,
        movement_latent: "torch.Tensor",
        node_latent: "torch.Tensor",
        next_node_latent: "torch.Tensor",
        movement_action: "torch.Tensor",
        static: "torch.Tensor",
        movement_mask: "torch.Tensor",
        neighbor_index: "torch.Tensor",
        neighbor_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        neighbor_latent = _gather_neighbors(
            node_latent, neighbor_index, neighbor_mask
        )
        node_context = self.movement_node_context(
            torch.cat([node_latent, next_node_latent, neighbor_latent], dim=-1)
        )[:, :, None].expand(-1, -1, movement_latent.shape[2], -1)
        static_context = self.movement_query(static)
        action_context = self.movement_action_context(movement_action)
        natural_delta = self.movement_natural_dynamics(
            torch.cat(
                [movement_latent, node_context, static_context], dim=-1
            )
        )
        service_delta = self.movement_service_dynamics(
            torch.cat(
                [
                    movement_latent,
                    node_context,
                    static_context,
                    action_context,
                ],
                dim=-1,
            )
        )
        served = movement_action[..., :1]
        next_latent = self.movement_transition_norm(
            movement_latent + natural_delta + served * service_delta
        )
        return next_latent * movement_mask[..., None].float()

    def decode(
        self,
        node_latent: "torch.Tensor",
        movement_latent: "torch.Tensor",
        static: "torch.Tensor",
    ) -> "torch.Tensor":
        query = self.movement_query(static)
        expanded = node_latent[:, :, None].expand(-1, -1, static.shape[2], -1)
        return self.decoder(
            torch.cat([expanded, movement_latent, query], dim=-1)
        )

    def reward_value(self, logits: "torch.Tensor") -> "torch.Tensor":
        bins = torch.linspace(
            self.config.reward_min,
            self.config.reward_max,
            self.config.reward_bins,
            device=logits.device,
            dtype=logits.dtype,
        )
        symlog_value = (F.softmax(logits, dim=-1) * bins).sum(dim=-1)
        return torch.sign(symlog_value) * torch.expm1(torch.abs(symlog_value))

    def rollout(self, batch: Dict[str, "torch.Tensor"]):
        work = {key: value for key, value in batch.items()}
        node_latent, movement_latent = self.encode(work)
        static = batch["movement_static_features"]
        decoded = self.decode(node_latent, movement_latent, static)
        reconstruction = decoded
        current_features = batch["history_features"][:, -1]
        movement_mask = batch["history_movement_mask"][:, -1]
        predictions = []
        reward_logits = []
        node_latents = []
        movement_latents = []
        work["rolling_phase"] = batch["history_current_phase"][:, -1]
        for step in range(batch["actions"].shape[1]):
            node_action, movement_action = self._action(work, step)
            next_node_latent, logits = self._transition_node(
                node_latent, node_action
            )
            next_movement_latent = self._transition_movements(
                movement_latent,
                node_latent,
                next_node_latent,
                movement_action,
                static,
                movement_mask,
                batch["neighbor_index"],
                batch["neighbor_mask"],
            )
            next_decoded = self.decode(
                next_node_latent, next_movement_latent, static
            )
            next_features = current_features + next_decoded - decoded
            next_features = next_features * movement_mask[..., None].float()
            predictions.append(next_features)
            reward_logits.append(logits)
            node_latents.append(next_node_latent)
            movement_latents.append(next_movement_latent)
            node_latent = next_node_latent
            movement_latent = next_movement_latent
            decoded = next_decoded
            current_features = next_features
            work["rolling_phase"] = batch["actions"][:, step]
        return {
            "features": torch.stack(predictions, dim=1),
            "reward_logits": torch.stack(reward_logits, dim=1),
            "rewards": torch.stack(
                [self.reward_value(item) for item in reward_logits], dim=1
            ),
            "reconstruction": reconstruction,
            "latents": torch.stack(node_latents, dim=1),
            "node_latents": torch.stack(node_latents, dim=1),
            "movement_latents": torch.stack(movement_latents, dim=1),
        }


def _advance_history(
    work: Dict[str, "torch.Tensor"],
    next_features: "torch.Tensor",
    original: Dict[str, "torch.Tensor"],
    step: int,
) -> Dict[str, "torch.Tensor"]:
    updated = {key: value for key, value in work.items()}
    phase = original["actions"][:, step]
    stage = original["target_signal_stage"][:, step]
    elapsed = original["target_phase_elapsed_s"][:, step]
    time_s = original["target_time_s"][:, step]
    updated["history_features"] = torch.cat(
        [work["history_features"][:, 1:], next_features[:, None]], dim=1
    )
    updated["history_valid_mask"] = torch.cat(
        [
            work["history_valid_mask"][:, 1:],
            original["target_valid_mask"][:, step : step + 1],
        ],
        dim=1,
    )
    updated["history_movement_mask"] = torch.cat(
        [
            work["history_movement_mask"][:, 1:],
            original["target_valid_mask"][:, step : step + 1].any(dim=-1),
        ],
        dim=1,
    )
    for key, value in (
        ("history_current_phase", phase),
        ("history_signal_stage", stage),
        ("history_phase_elapsed_s", elapsed),
        ("history_time_s", time_s),
    ):
        updated[key] = torch.cat([work[key][:, 1:], value[:, None]], dim=1)
    return updated
