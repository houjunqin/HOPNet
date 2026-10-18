import inspect
import os
import time
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm

from models.HOPNet.model import HOPNet
from models.HOPNet.metrics import masked_mae, masked_mape, masked_rmse


def log_string(log_dir: str, string: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "log"), "a+", encoding="utf-8") as log:
        log.write(string + "\n")
        log.flush()
    print(string)


class EarlyStopping:
    def __init__(self, patience: int = 15, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Trainer:
    """
    Trainer for the feature-wise embedding + ConvCAnext CHONet.

    The model embeds every dynamic input variable and every graph-edge
    variable independently, then applies ConvCAnext, fine/coarse graph
    message passing, temporal convolutions, and a direct forecast head.

    Expected shapes:
        x:                  [B, input_length, N, input_dim]
        y:                  [B, target_length, N, input_dim or output_dim]
        model output:       [B, horizon, N, output_dim]

    The model predicts the complete horizon in one forward pass. The model's
    rollout() method is used only as a validation/test compatibility wrapper;
    it remains non-autoregressive.
    """

    REQUIRED_GRAPH_ATTRIBUTES = (
        "edge_index",
        "edge_attr",
    )

    def __init__(self, datasets, **kwargs):
        self.datasets = datasets
        self.cfg_model = {
            **kwargs.get("data", {}),
            **kwargs.get("HOPNet", {}),
        }

        self.log_dir = self.cfg_model.get("log", "logs/HOPNet/")
        self.model_file = os.path.join(self.log_dir, "best_model.pth")

        self.input_dim = int(self.cfg_model.get("input_dim", 10))
        self.output_dim = int(self.cfg_model.get("output_dim", 1))
        self.input_feature_idx = list(self.cfg_model.get("input_feature_idx", range(self.input_dim)))
        self.output_feature_idx = list(self.cfg_model.get("output_feature_idx", range(self.output_dim)))
        self.horizon = int(self.cfg_model.get("horizon", 12))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Keep model parameters and graph tensors in FP32.
        # AMP performs temporary lower-precision computation only.
        self.use_amp = bool(
            self.cfg_model.get("use_amp", True)
        ) and self.device.type == "cuda"

        amp_dtype_name = str(
            self.cfg_model.get("amp_dtype", "bfloat16")
        ).lower()

        if amp_dtype_name in {"bf16", "bfloat16"}:
            self.amp_dtype = torch.bfloat16
        elif amp_dtype_name in {"fp16", "float16", "half"}:
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32
            self.use_amp = False

        # Gradient scaling is mainly required for FP16.
        # BF16 has a much wider exponent range and normally does not need it.
        self.use_grad_scaler = (
            self.use_amp
            and self.amp_dtype == torch.float16
        )

        try:
            self.gradscaler = amp.GradScaler(
                "cuda",
                enabled=self.use_grad_scaler,
            )
        except TypeError:
            self.gradscaler = amp.GradScaler(
                enabled=self.use_grad_scaler
            )

        self.rollout_every = int(
            self.cfg_model.get("rollout_every", 5)
        )
        self.rollout_batches = int(
            self.cfg_model.get("rollout_batches", 2)
        )
        self.max_grad_norm = float(
            self.cfg_model.get("max_grad_norm", 1.0)
        )

        self.warmup_epochs = int(
            self.cfg_model.get("loss_warmup_epochs", 5)
        )
        self.lambda_phys = float(
            self.cfg_model.get("lambda_phys", 0.01)
        )
        self.aux_ramp_epochs = max(
            1,
            int(self.cfg_model.get("aux_ramp_epochs", 10)),
        )

        # Direct multi-horizon training controls.
        self.horizon_final_weight = float(
            self.cfg_model.get("horizon_final_weight", 1.25)
        )
        self.adaptive_phys_weight = bool(
            self.cfg_model.get("adaptive_phys_weight", True)
        )
        self.phys_target_ratio = float(
            self.cfg_model.get("phys_target_ratio", 0.03)
        )
        self.phys_lambda_min = float(
            self.cfg_model.get("phys_lambda_min", 0.0)
        )
        self.phys_lambda_max = float(
            self.cfg_model.get("phys_lambda_max", 20.0)
        )

        # Use one validation quantity consistently for scheduling,
        # checkpointing, and early stopping.
        self.monitor_metric = str(
            self.cfg_model.get("monitor_metric", "val_mae")
        ).lower()
        self.minimum_epochs = int(
            self.cfg_model.get("minimum_epochs", 8)
        )

        # Speed controls for the direct model.
        self.physics_every_n_batches = max(
            1,
            int(
                self.cfg_model.get(
                    "physics_every_n_batches",
                    4,
                )
            ),
        )
        self.compute_validation_physics = bool(
            self.cfg_model.get(
                "compute_validation_physics",
                False,
            )
        )
        self.use_physics_loss = bool(
            self.cfg_model.get("use_physics_loss", False)
        )
        self.lambda_residual = float(
            self.cfg_model.get("lambda_residual", 0.01)
        )
        self.lambda_forecast_smooth = float(
            self.cfg_model.get(
                "lambda_forecast_smooth",
                0.002,
            )
        )

        os.makedirs(self.log_dir, exist_ok=True)

        # ConvCAnext normalization and edge embeddings depend on graph
        # metadata. Build graphs first, infer a consistent edge dimension,
        # and choose fixed or dynamic node normalization before creating the
        # model.
  
        self.model = HOPNet(
            self.cfg_model
        ).to(
            device=self.device,
            dtype=torch.float32,
        )

        total_parameters = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        optimizer_kwargs = {
            "lr": self.cfg_model.get("learning_rate", 1e-4),
            "weight_decay": self.cfg_model.get("weight_decay", 1e-4),
            "betas": (0.9, 0.95),
        }

        if self.device.type == "cuda" and "fused" in inspect.signature(
            torch.optim.AdamW
        ).parameters:
            optimizer_kwargs["fused"] = bool(
                self.cfg_model.get("fused_optimizer", True)
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **optimizer_kwargs,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(
                self.cfg_model.get("scheduler_factor", 0.5)
            ),
            patience=int(
                self.cfg_model.get("scheduler_patience", 3)
            ),
            threshold=float(
                self.cfg_model.get("scheduler_threshold", 2e-3)
            ),
            threshold_mode=str(
                self.cfg_model.get(
                    "scheduler_threshold_mode",
                    "rel",
                )
            ),
            cooldown=int(
                self.cfg_model.get("scheduler_cooldown", 1)
            ),
            min_lr=float(
                self.cfg_model.get("min_lr", 1e-6)
            ),
        )

        log_string(
            self.log_dir,
            (
                "HOPNet configuration | "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"horizon={self.horizon}, "
                f"amp={self.use_amp}, "
                f"amp_dtype={self.amp_dtype}, "
                f"d_model={self.cfg_model.get('d_model', 32)}, "
                f"state_embed_dim="
                f"{self.cfg_model.get('state_embed_dim', 4)}, "
                f"temporal_layers="
                f"{self.cfg_model.get('temporal_layers', 1)}, "
                f"node_counts={self.cfg_model.get('num_nodes', [])}, "
                f"Model parameters | total={total_parameters:,}, "
                f"trainable={trainable_parameters:,}"
            ),
        )
 
    def _prepare_graph(self, graph):
        """
        Keep persistent graph data in stable dtypes.

        Floating graph attributes stay FP32. Integer topology tensors stay
        int64. AMP will cast eligible operations internally during forward().
        """
        float_attrs = (
            "edge_attr",
            "coarse_edge_attr",
            "pos",
            "static_node",
        )
        long_attrs = (
            "edge_index",
            "coarse_edge_index",
            "level_assignment",
        )

        for name in float_attrs:
            if hasattr(graph, name):
                value = getattr(graph, name)
                if torch.is_tensor(value):
                    setattr(
                        graph,
                        name,
                        value.to(
                            device=self.device,
                            dtype=torch.float32,
                            non_blocking=True,
                        ),
                    )

        for name in long_attrs:
            if hasattr(graph, name):
                value = getattr(graph, name)
                if torch.is_tensor(value):
                    setattr(
                        graph,
                        name,
                        value.to(
                            device=self.device,
                            dtype=torch.long,
                            non_blocking=True,
                        ),
                    )

        return graph

    def _output_positions_in_history(self):
        """
        Return positions of output features inside history_state.

        Example
        -------
        input_feature_idx  = [0, 1, 2, 5]
        output_feature_idx = [0, 2]
        -> positions [0, 2]
        """
        positions = []

        for output_idx in self.output_feature_idx:
            if output_idx not in self.input_feature_idx:
                raise ValueError(
                    f"output feature {output_idx} is not present in "
                    f"input_feature_idx={self.input_feature_idx}. "
                    "Persistence and physics losses require every output "
                    "feature to be present in the history input."
                )

            positions.append(
                self.input_feature_idx.index(output_idx)
            )

        return positions

    def _scaler_feature_count(self, scaler):
        """
        Infer how many channels the scaler was fitted on.
        Supports common scaler attributes such as mean/std, min/max,
        data_min/data_max, scale, and center.
        """
        candidate_attrs = (
            "mean",
            "std",
            "min",
            "max",
            "data_min",
            "data_max",
            "scale",
            "center",
        )

        for name in candidate_attrs:
            if not hasattr(scaler, name):
                continue

            value = getattr(scaler, name)

            if value is None:
                continue

            if torch.is_tensor(value):
                if value.ndim == 0:
                    continue
                return int(value.shape[-1])

            try:
                array = np.asarray(value)
            except Exception:
                continue

            if array.ndim > 0:
                return int(array.shape[-1])

        return None

    def _inverse_selected(
        self,
        values: torch.Tensor,
        scaler,
        feature_idx,
        full_dim: int | None = None,
    ) -> torch.Tensor:
        """
        Inverse-transform selected output channels with the scaler statistics
        that actually correspond to those channels.

        This handles three common dataset layouts:

        1. scaler fitted on all raw dataset channels:
              scaler_dim == raw feature count
           -> output_feature_idx are used directly.

        2. scaler fitted only on model input channels:
              scaler_dim == len(input_feature_idx)
           -> each output feature is mapped to its position inside
              input_feature_idx.

        3. scaler fitted only on output channels:
              scaler_dim == len(output_feature_idx)
           -> selected values can be inverse-transformed directly.
        """
        values = values.float()
        feature_idx = list(feature_idx)

        if values.size(-1) != len(feature_idx):
            raise ValueError(
                f"Prediction has {values.size(-1)} channels, but "
                f"feature_idx={feature_idx} contains "
                f"{len(feature_idx)} channels."
            )

        scaler_dim = self._scaler_feature_count(scaler)

        # --------------------------------------------------
        # Case 1: scaler is output-only.
        # --------------------------------------------------
        if scaler_dim == len(feature_idx):
            transformed = scaler.inverse_transform(values)

            if not torch.is_tensor(transformed):
                transformed = torch.as_tensor(
                    transformed,
                    device=values.device,
                    dtype=values.dtype,
                )

            return transformed.to(
                device=values.device,
                dtype=values.dtype,
            )

        # --------------------------------------------------
        # Case 2: scaler is fitted on model input features.
        # Example:
        # input_feature_idx  = [0,2,3,5,7,9]  -> scaler_dim=6
        # output_feature_idx = [0]            -> scaler column 0
        # --------------------------------------------------
        if scaler_dim == len(self.input_feature_idx):
            scaler_positions = []

            for idx in feature_idx:
                if idx not in self.input_feature_idx:
                    raise ValueError(
                        f"Output feature {idx} is not included in "
                        f"input_feature_idx={self.input_feature_idx}, "
                        "but the scaler appears to have been fitted on "
                        "the input feature subset."
                    )

                scaler_positions.append(
                    self.input_feature_idx.index(idx)
                )

            padded = torch.zeros(
                *values.shape[:-1],
                scaler_dim,
                device=values.device,
                dtype=values.dtype,
            )

            padded[..., scaler_positions] = values
            transformed = scaler.inverse_transform(padded)

            if not torch.is_tensor(transformed):
                transformed = torch.as_tensor(
                    transformed,
                    device=values.device,
                    dtype=values.dtype,
                )
            else:
                transformed = transformed.to(
                    device=values.device,
                    dtype=values.dtype,
                )

            return transformed[..., scaler_positions]

        # --------------------------------------------------
        # Case 3: scaler is fitted on all raw channels.
        # --------------------------------------------------
        if scaler_dim is not None:
            max_feature = max(feature_idx)

            if max_feature < scaler_dim:
                padded = torch.zeros(
                    *values.shape[:-1],
                    scaler_dim,
                    device=values.device,
                    dtype=values.dtype,
                )
                padded[..., feature_idx] = values

                transformed = scaler.inverse_transform(
                    padded
                )

                if not torch.is_tensor(transformed):
                    transformed = torch.as_tensor(
                        transformed,
                        device=values.device,
                        dtype=values.dtype,
                    )
                else:
                    transformed = transformed.to(
                        device=values.device,
                        dtype=values.dtype,
                    )

                return transformed[..., feature_idx]

        # --------------------------------------------------
        # Last-resort direct attempt for unusual scalers.
        # --------------------------------------------------
        try:
            transformed = scaler.inverse_transform(values)
        except Exception as exc:
            raise RuntimeError(
                "Could not map prediction channels to scaler statistics. "
                f"scaler_dim={scaler_dim}, "
                f"input_feature_idx={self.input_feature_idx}, "
                f"output_feature_idx={self.output_feature_idx}, "
                f"prediction_channels={values.size(-1)}."
            ) from exc

        if not torch.is_tensor(transformed):
            transformed = torch.as_tensor(
                transformed,
                device=values.device,
                dtype=values.dtype,
            )

        return transformed.to(
            device=values.device,
            dtype=values.dtype,
        )

    def _select_history_state(self, x: torch.Tensor) -> torch.Tensor:
        return x[
            ..., self.input_feature_idx
        ].to(dtype=torch.float32)

    def _select_target_state(self, y: torch.Tensor) -> torch.Tensor:
        return y[
            ..., self.output_feature_idx
        ].to(dtype=torch.float32)

    def _align_prediction_and_target(
        self,
        prediction: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        Align direct prediction and target to a common forecast horizon.
        """
        if prediction.ndim != 4:
            raise ValueError(
                "HOPNet prediction must have shape "
                "[B,H,N,C]."
            )

        target = self._select_target_state(y)

        if target.ndim != 4:
            raise ValueError(
                "Target must have shape [B,H,N,C]."
            )

        effective_horizon = min(
            self.horizon,
            prediction.size(1),
            target.size(1),
        )

        if effective_horizon < 1:
            raise ValueError(
                "Prediction and target have no overlapping horizon."
            )

        return (
            prediction[:, :effective_horizon],
            target[:, :effective_horizon],
            effective_horizon,
        )

    @staticmethod
    def continuity_residual(
        eta_t: torch.Tensor,
        eta_tp1: torch.Tensor,
        edge_index: torch.Tensor,
        dt: float = 1.0,
        normalize_degree: bool = True,
        clamp_residual: float | None = 10.0,
    ) -> torch.Tensor:
        """
        Compute one-step graph continuity residual.

        Args:
            eta_t: [B, N, C]
            eta_tp1: [B, N, C]
            edge_index: [2, E]
        """
        src, dst = edge_index
        batch_size, num_nodes, channels = eta_t.shape

        flux = eta_t[:, src] - eta_t[:, dst]
        divergence = torch.zeros(
            batch_size,
            num_nodes,
            channels,
            device=eta_t.device,
            dtype=eta_t.dtype,
        )
        divergence.index_add_(1, dst, flux)

        if normalize_degree:
            degree = torch.zeros(
                num_nodes,
                device=eta_t.device,
                dtype=eta_t.dtype,
            )
            degree.index_add_(
                0,
                dst,
                torch.ones(
                    edge_index.size(1),
                    device=eta_t.device,
                    dtype=eta_t.dtype,
                ),
            )
            divergence = divergence / degree.clamp(min=1.0).view(
                1,
                num_nodes,
                1,
            )

        residual = (eta_tp1 - eta_t) / dt + 0.1 * divergence

        if clamp_residual is not None:
            residual = residual.clamp(
                min=-clamp_residual,
                max=clamp_residual,
            )

        return residual

    def sequence_continuity_residual(
        self,
        last_history_state: torch.Tensor,
        forecast_state: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute residuals for all direct forecast transitions:

            last observed -> step 1 -> step 2 -> ... -> step H

        Args:
            last_history_state: [B, N, C]
            forecast_state: [B, H, N, C]

        Returns:
            residual: [B, H, N, C]
        """
        previous_state = torch.cat(
            [
                last_history_state.unsqueeze(1),
                forecast_state[:, :-1],
            ],
            dim=1,
        )

        batch_size, horizon, num_nodes, channels = forecast_state.shape

        residual = self.continuity_residual(
            eta_t=previous_state.reshape(
                batch_size * horizon,
                num_nodes,
                channels,
            ),
            eta_tp1=forecast_state.reshape(
                batch_size * horizon,
                num_nodes,
                channels,
            ),
            edge_index=edge_index,
            dt=self.cfg_model.get("dt", 1.0),
            normalize_degree=True,
            clamp_residual=self.cfg_model.get(
                "phys_residual_clip",
                10.0,
            ),
        )

        return residual.view(
            batch_size,
            horizon,
            num_nodes,
            channels,
        )

    def _horizon_weighted_smooth_l1(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """
        Smooth-L1 loss with slightly larger weight on later horizons.

        prediction/target:
            [B, H, N, C]

        horizon_final_weight=1.0 gives uniform weighting. Values around
        1.2-1.4 improve long-horizon accuracy without sacrificing the
        first forecast step too aggressively.
        """
        element_loss = F.smooth_l1_loss(
            prediction,
            target,
            beta=beta,
            reduction="none",
        )
        per_horizon = element_loss.mean(dim=(0, 2, 3))

        horizon_count = prediction.size(1)
        weights = torch.linspace(
            1.0,
            self.horizon_final_weight,
            steps=horizon_count,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        weights = weights / weights.mean().clamp(min=1e-6)

        return (per_horizon * weights).mean()

    def compute_losses(
        self,
        pred,
        x,
        y,
        scaler,
        subgraph,
        epoch: int,
        compute_physics: bool = True,
    ):
        history_state_norm = self._select_history_state(x)

        pred_state_norm, target_state_norm, _ = (
            self._align_prediction_and_target(
                pred["state"],
                y,
            )
        )

        full_dim = max(
            int(x.size(-1)),
            int(y.size(-1)),
            max(self.input_feature_idx) + 1,
            max(self.output_feature_idx) + 1,
        )

        # Physical-space losses are used for reporting and optional physics.
        # Use feature-aware inverse transformation.
        pred_state = self._inverse_selected(
            pred_state_norm.detach(),
            scaler,
            self.output_feature_idx,
            full_dim,
        )
        target_state = self._inverse_selected(
            target_state_norm.detach(),
            scaler,
            self.output_feature_idx,
            full_dim,
        )

        state_huber_beta = float(
            self.cfg_model.get("state_huber_beta", 0.1)
        )

        loss_state_norm = self._horizon_weighted_smooth_l1(
            pred_state_norm,
            target_state_norm,
            beta=state_huber_beta,
        )
        loss_state = self._horizon_weighted_smooth_l1(
            pred_state,
            target_state,
            beta=state_huber_beta,
        )

        # Persistence must select the predicted variables from history by
        # feature identity, not simply the first output_dim channels.
        output_positions = self._output_positions_in_history()

        persistence = history_state_norm[
            :, -1, :, output_positions
        ].unsqueeze(1)
        persistence = persistence.expand_as(
            pred_state_norm
        )

        loss_residual = F.smooth_l1_loss(
            pred_state_norm,
            persistence,
            beta=state_huber_beta,
        )

        if pred_state_norm.size(1) > 1:
            forecast_difference = (
                pred_state_norm[:, 1:]
                - pred_state_norm[:, :-1]
            )
            loss_forecast_smooth = (
                forecast_difference.square().mean()
            )
        else:
            loss_forecast_smooth = (
                loss_state_norm.new_zeros(())
            )

        compute_physics = (
            compute_physics
            and self.use_physics_loss
        )

        if compute_physics:
            with amp.autocast(
                device_type=self.device.type,
                enabled=False,
            ):
                last_history_norm = history_state_norm[
                    :, -1, :, output_positions
                ].float()

                last_history_state = self._inverse_selected(
                    last_history_norm,
                    scaler,
                    self.output_feature_idx,
                    full_dim,
                )

                pred_state_phys = self._inverse_selected(
                    pred_state_norm.float(),
                    scaler,
                    self.output_feature_idx,
                    full_dim,
                )

                residual = self.sequence_continuity_residual(
                    last_history_state=last_history_state,
                    forecast_state=pred_state_phys,
                    edge_index=subgraph.edge_index,
                )

                loss_phys = F.smooth_l1_loss(
                    residual,
                    torch.zeros_like(residual),
                    beta=float(
                        self.cfg_model.get(
                            "phys_huber_beta",
                            1.0,
                        )
                    ),
                )

                residual_norm = self.sequence_continuity_residual(
                    last_history_state=last_history_norm,
                    forecast_state=pred_state_norm.float(),
                    edge_index=subgraph.edge_index,
                )

                loss_phys_norm = F.smooth_l1_loss(
                    residual_norm,
                    torch.zeros_like(residual_norm),
                    beta=float(
                        self.cfg_model.get(
                            "phys_huber_beta",
                            1.0,
                        )
                    ),
                )
        else:
            loss_phys = loss_state.new_zeros(())
            loss_phys_norm = loss_state_norm.new_zeros(())

        if epoch <= self.warmup_epochs:
            auxiliary_scale = 0.0
        else:
            auxiliary_scale = min(
                1.0,
                (
                    epoch - self.warmup_epochs
                ) / self.aux_ramp_epochs,
            )

        if not compute_physics:
            effective_lambda_phys = (
                loss_state_norm.new_zeros(())
            )

        elif self.adaptive_phys_weight:
            effective_lambda_phys = (
                self.phys_target_ratio
                * loss_state_norm.detach()
                / loss_phys_norm.detach().clamp(
                    min=1e-8
                )
            )

            effective_lambda_phys = (
                effective_lambda_phys.clamp(
                    min=self.phys_lambda_min,
                    max=self.phys_lambda_max,
                )
            )

        else:
            effective_lambda_phys = (
                loss_state_norm.new_tensor(
                    self.lambda_phys
                )
            )

        total_loss = (
            loss_state
            + auxiliary_scale
            * effective_lambda_phys
            * loss_phys
        )

        total_loss_norm = (
            loss_state_norm
            + self.lambda_residual
            * loss_residual
            + self.lambda_forecast_smooth
            * loss_forecast_smooth
            + auxiliary_scale
            * effective_lambda_phys
            * loss_phys_norm
        )

        return {
            "total_loss_norm": total_loss_norm,
            "total_loss": total_loss,
            "loss_state_norm": loss_state_norm,
            "loss_state": loss_state,
            "loss_phys_norm": loss_phys_norm,
            "loss_phys": loss_phys,
            "loss_residual": loss_residual,
            "loss_forecast_smooth": loss_forecast_smooth,
            "effective_lambda_phys": effective_lambda_phys,
        }

    def train(self):
        log_string(
            self.log_dir,
            "====================== START TRAINING ======================",
        )

        training_history = []
        start_total_time = time.time()
        best_monitor_value = float("inf")

        early_stopping = EarlyStopping(
            patience=self.cfg_model.get("patience", 15),
            delta=self.cfg_model.get("early_stop_delta", 1e-4),
        )

        for epoch in range(1, self.cfg_model.get("epochs", 50) + 1):
            epoch_start = time.time()
            self.model.train()

            train_total_loss = []
            train_total_loss_norm = []
            train_state_loss = []
            train_state_loss_norm = []
            train_phys_loss = []
            train_phys_loss_norm = []

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            for subgraph_idx in range(1, len(self.datasets) + 1):
                sub = self.datasets[subgraph_idx]
                scaler = sub["scaler"]
                train_loader = sub["X_train_loader"]
                subgraph = self._prepare_graph(
                    sub["coastal_graph"]
                )

                progress = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch} | Train Sub {subgraph_idx}",
                    leave=False,
                )

                for batch_idx, (x, y, _) in enumerate(progress):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    history_state = self._select_history_state(x)
                    self.optimizer.zero_grad(set_to_none=True)

                    with amp.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    ):
                        # Persistent tensors must remain FP32.
                        # Autocast handles temporary BF16/FP16 execution.
                        if history_state.dtype != torch.float32:
                            history_state = history_state.float()

                        pred = self.model(
                            history_state=history_state,
                            graph=subgraph,
                        )

                        losses = self.compute_losses(
                            pred=pred,
                            x=x,
                            y=y,
                            scaler=scaler,
                            subgraph=subgraph,
                            epoch=epoch,
                            compute_physics=(
                                batch_idx
                                % self.physics_every_n_batches
                                == 0
                            ),
                        )

                    total_loss_norm = losses["total_loss_norm"]

                    # Train with the complete configured objective, not only
                    # the state term.
                    self.gradscaler.scale(total_loss_norm).backward()
                    self.gradscaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    self.gradscaler.step(self.optimizer)
                    self.gradscaler.update()

                    train_total_loss.append(
                        losses["total_loss"].item()
                    )
                    train_total_loss_norm.append(
                        total_loss_norm.item()
                    )
                    train_state_loss.append(
                        losses["loss_state"].item()
                    )
                    train_state_loss_norm.append(
                        losses["loss_state_norm"].item()
                    )
                    train_phys_loss.append(
                        losses["loss_phys"].item()
                    )
                    train_phys_loss_norm.append(
                        losses["loss_phys_norm"].item()
                    )

                    progress.set_postfix(
                        loss=f"{total_loss_norm.item():.4f}",
                    )

            avg_train_loss = float(np.mean(train_total_loss))
            avg_train_loss_norm = float(
                np.mean(train_total_loss_norm)
            )
            avg_train_state_loss = float(
                np.mean(train_state_loss)
            )
            avg_train_state_loss_norm = float(
                np.mean(train_state_loss_norm)
            )
            avg_train_phys_loss = float(
                np.mean(train_phys_loss)
            )
            avg_train_phys_loss_norm = float(
                np.mean(train_phys_loss_norm)
            )

            self.model.eval()

            val_loss = []
            val_loss_norm = []
            val_state_mae = []
            val_state_norm_mae = []
            val_state_mape = []
            val_state_rmse = []
            val_phys_loss = []
            val_phys_loss_norm = []
            val_rollout_mae = []
            val_rollout_mape = []
            val_rollout_rmse = []

            for subgraph_idx in range(1, len(self.datasets) + 1):
                sub = self.datasets[subgraph_idx]

                metrics = self.validate(
                    val_loader=sub["X_val_loader"],
                    scaler=sub["scaler"],
                    subgraph=sub["coastal_graph"],
                    epoch=epoch,
                    do_rollout=(
                        self.rollout_every > 0
                        and epoch % self.rollout_every == 0
                    ),
                )

                val_loss.append(metrics["val_loss"])
                val_loss_norm.append(metrics["val_loss_norm"])
                val_state_mae.append(metrics["val_state_mae"])
                val_state_norm_mae.append(
                    metrics["val_state_norm_mae"]
                )
                val_state_mape.append(metrics["val_state_mape"])
                val_state_rmse.append(metrics["val_state_rmse"])
                val_phys_loss.append(metrics["val_phys_loss"])
                val_phys_loss_norm.append(
                    metrics["val_phys_loss_norm"]
                )

                for key, meter in (
                    ("val_rollout_mae", val_rollout_mae),
                    ("val_rollout_mape", val_rollout_mape),
                    ("val_rollout_rmse", val_rollout_rmse),
                ):
                    value = metrics[key]
                    if not np.isnan(value):
                        meter.append(value)

            avg_val_loss = float(np.mean(val_loss))
            avg_val_loss_norm = float(np.mean(val_loss_norm))
            avg_val_state_mae = float(np.mean(val_state_mae))
            avg_val_state_norm_mae = float(
                np.mean(val_state_norm_mae)
            )
            avg_val_state_mape = float(np.mean(val_state_mape))
            avg_val_state_rmse = float(np.mean(val_state_rmse))
            avg_val_phys_loss = (
                float(np.mean(val_phys_loss))
                if self.compute_validation_physics
                else 0.0
            )
            avg_val_phys_loss_norm = (
                float(np.mean(val_phys_loss_norm))
                if self.compute_validation_physics
                else 0.0
            )

            avg_val_rollout_mae = (
                float(np.mean(val_rollout_mae))
                if val_rollout_mae
                else np.nan
            )
            avg_val_rollout_mape = (
                float(np.mean(val_rollout_mape))
                if val_rollout_mape
                else np.nan
            )
            avg_val_rollout_rmse = (
                float(np.mean(val_rollout_rmse))
                if val_rollout_rmse
                else np.nan
            )

            monitor_values = {
                "val_mae": avg_val_state_mae,
                "val_rmse": avg_val_state_rmse,
                "val_loss": avg_val_loss,
                "val_loss_norm": avg_val_loss_norm,
            }
            if self.monitor_metric not in monitor_values:
                raise ValueError(
                    "monitor_metric must be one of: "
                    "'val_mae', 'val_rmse', 'val_loss', "
                    "'val_loss_norm'."
                )

            monitor_value = monitor_values[self.monitor_metric]
            self.lr_scheduler.step(monitor_value)

            epoch_time = time.time() - epoch_start
            gpu_mem_alloc = (
                torch.cuda.max_memory_allocated(self.device) / 1_000_000
                if self.device.type == "cuda"
                else 0.0
            )

            training_history.append(
                {
                    "Epoch": epoch,
                    "Monitor_Metric": self.monitor_metric,
                    "Monitor_Value": monitor_value,
                    "Avg_Train_Loss": avg_train_loss,
                    "Avg_Train_Loss_Norm": avg_train_loss_norm,
                    "Avg_Train_State_Loss": avg_train_state_loss,
                    "Avg_Train_State_Loss_Norm": avg_train_state_loss_norm,
                    "Avg_Train_Phys_Loss": avg_train_phys_loss,
                    "Avg_Train_Phys_Loss_Norm": avg_train_phys_loss_norm,
                    "Avg_Val_Loss": avg_val_loss,
                    "Avg_Val_Loss_Norm": avg_val_loss_norm,
                    "Avg_Val_State_MAE": avg_val_state_mae,
                    "Avg_Val_State_Norm_MAE": avg_val_state_norm_mae,
                    "Avg_Val_State_MAPE": avg_val_state_mape,
                    "Avg_Val_State_RMSE": avg_val_state_rmse,
                    "Avg_Val_Phys_Loss": avg_val_phys_loss,
                    "Avg_Val_Phys_Loss_Norm": avg_val_phys_loss_norm,
                    "Avg_Val_Rollout_MAE": avg_val_rollout_mae,
                    "Avg_Val_Rollout_MAPE": avg_val_rollout_mape,
                    "Avg_Val_Rollout_RMSE": avg_val_rollout_rmse,
                    "LR": self.optimizer.param_groups[0]["lr"],
                    "Time_sec": epoch_time,
                    "GPU_Mem_MB": gpu_mem_alloc,
                }
            )

            log_string(
                self.log_dir,
                (
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Train Loss Norm: {avg_train_loss_norm:.4f} | "
                    f"Train State: {avg_train_state_loss:.4f} | "
                    f"Train Phys: {avg_train_phys_loss:.4f}\n"
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val MAE: {avg_val_state_mae:.4f} | "
                    f"Val MAPE: {avg_val_state_mape:.4f} | "
                    f"Val RMSE: {avg_val_state_rmse:.4f} | "
                    f"Val Phys: {avg_val_phys_loss:.4f}\n"
                    f"Val Rollout MAE: {avg_val_rollout_mae:.4f} | "
                    f"Val Rollout MAPE: {avg_val_rollout_mape:.4f} | "
                    f"Val Rollout RMSE: {avg_val_rollout_rmse:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                ),
            )

            if monitor_value < best_monitor_value:
                best_monitor_value = monitor_value
                torch.save(self.model.state_dict(), self.model_file)
                log_string(
                    self.log_dir,
                    (
                        f"Saved best model at epoch {epoch:03d} with "
                        f"{self.monitor_metric}: "
                        f"{best_monitor_value:.4f}"
                    ),
                )
            else:
                log_string(
                    self.log_dir,
                    (
                        f"No improvement in {self.monitor_metric}: "
                        f"{monitor_value:.4f} "
                        f"(Best: {best_monitor_value:.4f})"
                    ),
                )

            if epoch >= self.minimum_epochs:
                early_stopping(monitor_value)
                if early_stopping.early_stop:
                    log_string(
                        self.log_dir,
                        (
                            f"Early stopping triggered at epoch {epoch} "
                            f"using {self.monitor_metric}"
                        ),
                    )
                    break

        train_excel_path = os.path.join(
            self.log_dir,
            "training_results.xlsx",
        )
        with pd.ExcelWriter(
            train_excel_path,
            mode="w",
            engine="openpyxl",
        ) as writer:
            pd.DataFrame(training_history).to_excel(
                writer,
                sheet_name="Epoch_History",
                index=False,
            )

        log_string(
            self.log_dir,
            f"Training results saved to {train_excel_path}",
        )
        log_string(
            self.log_dir,
            (
                "Training Runtime: "
                f"{(time.time() - start_total_time) / 60:.2f} minutes"
            ),
        )

    @torch.no_grad()
    def validate(
        self,
        val_loader,
        scaler,
        subgraph,
        epoch,
        do_rollout=False,
    ):
        self.model.eval()

        val_loss_meter = []
        val_loss_norm_meter = []
        val_state_mae_meter = []
        val_state_norm_mae_meter = []
        val_state_mape_meter = []
        val_state_rmse_meter = []
        val_phys_meter = []
        val_phys_norm_meter = []
        val_rollout_mae_meter = []
        val_rollout_mape_meter = []
        val_rollout_rmse_meter = []

        for batch_idx, (x, y, _) in enumerate(val_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            history_state = self._select_history_state(x)
            full_dim = max(
                int(x.size(-1)),
                int(y.size(-1)),
                max(self.input_feature_idx) + 1,
                max(self.output_feature_idx) + 1,
            )

            with amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                pred = self.model(
                    history_state=history_state,
                    graph=subgraph,
                )
                losses = self.compute_losses(
                    pred=pred,
                    x=x,
                    y=y,
                    scaler=scaler,
                    subgraph=subgraph,
                    epoch=epoch,
                    compute_physics=self.compute_validation_physics,
                )

            pred_state_norm, target_state_norm, effective_horizon = (
                self._align_prediction_and_target(
                    pred["state"],
                    y,
                )
            )
            pred_state = self._inverse_selected(
                pred_state_norm,
                scaler,
                self.output_feature_idx,
                full_dim,
            )
            target_state = self._inverse_selected(
                target_state_norm,
                scaler,
                self.output_feature_idx,
                full_dim,
            )

            val_loss_meter.append(losses["total_loss"].item())
            val_loss_norm_meter.append(
                losses["total_loss_norm"].item()
            )
            val_state_mae_meter.append(
                masked_mae(pred_state, target_state, 0.0).item()
            )
            val_state_norm_mae_meter.append(
                masked_mae(
                    pred_state_norm,
                    target_state_norm,
                    0.0,
                ).item()
            )
            val_state_mape_meter.append(
                masked_mape(pred_state, target_state, 0.0).item()
            )
            val_state_rmse_meter.append(
                masked_rmse(pred_state, target_state, 0.0).item()
            )
            val_phys_meter.append(losses["loss_phys"].item())
            val_phys_norm_meter.append(
                losses["loss_phys_norm"].item()
            )

            if do_rollout and batch_idx < self.rollout_batches:
                # True autoregressive rollout when the model implements it.
                if hasattr(self.model, "rollout"):
                    rollout_norm = self.model.rollout(
                        history_state=history_state,
                        graph=subgraph,
                        horizon=effective_horizon,
                    )["state"]

                    rollout_norm = rollout_norm[
                        :, :effective_horizon
                    ]

                    rollout_state = self._inverse_selected(
                        rollout_norm,
                        scaler,
                        self.output_feature_idx,
                        full_dim,
                    )
                else:
                    rollout_state = pred_state

                val_rollout_mae_meter.append(
                    masked_mae(
                        rollout_state,
                        target_state,
                        0.0,
                    ).item()
                )
                val_rollout_mape_meter.append(
                    masked_mape(
                        rollout_state,
                        target_state,
                        0.0,
                    ).item()
                )
                val_rollout_rmse_meter.append(
                    masked_rmse(
                        rollout_state,
                        target_state,
                        0.0,
                    ).item()
                )

        return {
            "val_loss": float(np.mean(val_loss_meter)),
            "val_loss_norm": float(np.mean(val_loss_norm_meter)),
            "val_state_mae": float(np.mean(val_state_mae_meter)),
            "val_state_norm_mae": float(
                np.mean(val_state_norm_mae_meter)
            ),
            "val_state_mape": float(np.mean(val_state_mape_meter)),
            "val_state_rmse": float(np.mean(val_state_rmse_meter)),
            "val_phys_loss": float(np.mean(val_phys_meter)),
            "val_phys_loss_norm": float(
                np.mean(val_phys_norm_meter)
            ),
            "val_rollout_mae": (
                float(np.mean(val_rollout_mae_meter))
                if val_rollout_mae_meter
                else np.nan
            ),
            "val_rollout_mape": (
                float(np.mean(val_rollout_mape_meter))
                if val_rollout_mape_meter
                else np.nan
            ),
            "val_rollout_rmse": (
                float(np.mean(val_rollout_rmse_meter))
                if val_rollout_rmse_meter
                else np.nan
            ),
        }

    def _load_best_model(self) -> None:
        try:
            state_dict = torch.load(
                self.model_file,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(
                self.model_file,
                map_location=self.device,
            )

        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def test(self, start_time):
        log_string(
            self.log_dir,
            "\n" + "=" * 20 + " FINAL TEST MODE " + "=" * 20,
        )

        self._load_best_model()
        self.model.eval()

        test_summary = []
        horizon_metrics = []
        outputs = {}

        for subgraph_idx in range(1, len(self.datasets) + 1):
            sub = self.datasets[subgraph_idx]
            scaler = sub["scaler"]
            subgraph = self._prepare_graph(
                sub.get(
                    "coastal_graph",
                    sub.get("graph"),
                )
            )
            test_loader = self._get_loader(
                sub,
                "X_test_loader",
            )

            preds_state = []
            trues_state = []
            rollout_preds_state = []
            rollout_trues_state = []

            for x, y, _ in test_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                history_state = self._select_history_state(x)
                full_dim = max(
                    int(x.size(-1)),
                    int(y.size(-1)),
                    max(self.input_feature_idx) + 1,
                    max(self.output_feature_idx) + 1,
                )

                with amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    pred = self.model(
                        history_state=history_state,
                        graph=subgraph,
                    )

                pred_state_norm, target_state_norm, effective_horizon = (
                    self._align_prediction_and_target(
                        pred["state"],
                        y,
                    )
                )
                pred_state = self._inverse_selected(
                    pred_state_norm,
                    scaler,
                    self.output_feature_idx,
                    full_dim,
                )
                target_state = self._inverse_selected(
                    target_state_norm,
                    scaler,
                    self.output_feature_idx,
                    full_dim,
                )

                preds_state.append(pred_state.cpu())
                trues_state.append(target_state.cpu())

                if hasattr(self.model, "rollout"):
                    rollout_norm = self.model.rollout(
                        history_state=history_state,
                        graph=subgraph,
                        horizon=effective_horizon,
                    )["state"][:, :effective_horizon]

                    rollout_state = self._inverse_selected(
                        rollout_norm,
                        scaler,
                        self.output_feature_idx,
                        full_dim,
                    )
                else:
                    rollout_state = pred_state

                rollout_preds_state.append(rollout_state.cpu())
                rollout_trues_state.append(target_state.cpu())

            test_pred_state = torch.cat(preds_state, dim=0)
            test_true_state = torch.cat(trues_state, dim=0)
            rollout_pred_state = torch.cat(
                rollout_preds_state,
                dim=0,
            )
            rollout_true_state = torch.cat(
                rollout_trues_state,
                dim=0,
            )

            test_state_mae = masked_mae(
                test_pred_state,
                test_true_state,
                0.0,
            ).item()
            test_state_rmse = masked_rmse(
                test_pred_state,
                test_true_state,
                0.0,
            ).item()
            test_state_mape = masked_mape(
                test_pred_state,
                test_true_state,
                0.0,
            ).item()

            rollout_state_mae = masked_mae(
                rollout_pred_state,
                rollout_true_state,
                0.0,
            ).item()
            rollout_state_rmse = masked_rmse(
                rollout_pred_state,
                rollout_true_state,
                0.0,
            ).item()
            rollout_state_mape = masked_mape(
                rollout_pred_state,
                rollout_true_state,
                0.0,
            ).item()

            log_string(
                self.log_dir,
                (
                    f"Sub {subgraph_idx} Overall Test -> "
                    f"State MAE: {test_state_mae:.4f}, "
                    f"State MAPE: {test_state_mape:.4f}, "
                    f"State RMSE: {test_state_rmse:.4f}\n"
                    f"Rollout State MAE: {rollout_state_mae:.4f}, "
                    f"Rollout State MAPE: {rollout_state_mape:.4f}, "
                    f"Rollout State RMSE: {rollout_state_rmse:.4f}"
                ),
            )

            test_summary.append(
                {
                    "Subgraph": subgraph_idx,
                    "Test State MAE": test_state_mae,
                    "Test State RMSE": test_state_rmse,
                    "Test State MAPE": test_state_mape,
                    "Rollout State MAE": rollout_state_mae,
                    "Rollout State MAPE": rollout_state_mape,
                    "Rollout State RMSE": rollout_state_rmse,
                }
            )

            num_horizons = min(
                self.horizon,
                rollout_pred_state.size(1),
                rollout_true_state.size(1),
            )
            for horizon_idx in range(num_horizons):
                horizon_pred = rollout_pred_state[:, horizon_idx]
                horizon_true = rollout_true_state[:, horizon_idx]

                horizon_metrics.append(
                    {
                        "Subgraph": subgraph_idx,
                        "Horizon_Step": horizon_idx + 1,
                        "State MAE": masked_mae(
                            horizon_pred,
                            horizon_true,
                            0.0,
                        ).item(),
                        "State RMSE": masked_rmse(
                            horizon_pred,
                            horizon_true,
                            0.0,
                        ).item(),
                        "State MAPE": masked_mape(
                            horizon_pred,
                            horizon_true,
                            0.0,
                        ).item(),
                    }
                )

            outputs[subgraph_idx] = {
                "preds_state": test_pred_state,
                "trues_state": test_true_state,
                "rollout_preds_state": rollout_pred_state,
                "rollout_trues_state": rollout_true_state,
            }

        test_excel_path = os.path.join(
            self.log_dir,
            "training_results.xlsx",
        )
        writer_kwargs = {
            "path": test_excel_path,
            "engine": "openpyxl",
        }

        if os.path.exists(test_excel_path):
            writer_kwargs.update(
                mode="a",
                if_sheet_exists="replace",
            )
        else:
            writer_kwargs.update(mode="w")

        with pd.ExcelWriter(**writer_kwargs) as writer:
            pd.DataFrame(test_summary).to_excel(
                writer,
                sheet_name="Overall_Summary",
                index=False,
            )
            pd.DataFrame(horizon_metrics).to_excel(
                writer,
                sheet_name="Horizon_Step_Analysis",
                index=False,
            )

        log_string(
            self.log_dir,
            f"Final test metrics saved to {test_excel_path}",
        )
        log_string(
            self.log_dir,
            (
                "Total Runtime: "
                f"{(time.time() - start_time) / 60:.2f} minutes"
            ),
        )

        return outputs

    @staticmethod
    def _loader_keys_for_split(sub: Dict, split: str):
        candidates = {
            "train": ("X_train_loader_wave", "X_train_loader"),
            "val": ("X_val_loader_wave", "X_val_loader"),
            "test": ("X_test_loader",),
            "all": (
                "sub_loader",
                "X_train_loader_wave",
                "X_train_loader",
                "X_val_loader_wave",
                "X_val_loader",
                "X_test_loader",
            ),
        }[split]

        if split == "all":
            return [name for name in candidates if name in sub]

        return [next(name for name in candidates if name in sub)]

    @torch.no_grad()
    def collect_predictions(
        self,
        split: str = "test",
        load_best: bool = True,
        save_npz: bool = True,
    ):
        if load_best:
            self._load_best_model()

        self.model.eval()

        split = split.lower()
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(
                "split must be one of: 'train', 'val', 'test', 'all'"
            )

        tensor_keys = [
            "preds_state",
            "trues_state",
            "rollout_preds_state",
            "rollout_trues_state",
        ]

        def combine_subgraphs_by_node(by_subgraph, loader_key):
            combined = {}

            for key in tensor_keys:
                tensors = []

                for subgraph_idx in sorted(by_subgraph):
                    if loader_key not in by_subgraph[subgraph_idx]:
                        continue

                    tensor = by_subgraph[subgraph_idx][loader_key][key]
                    if tensor is not None:
                        tensors.append(tensor)

                if not tensors:
                    combined[key] = None
                    continue

                min_samples = min(
                    tensor.size(0) for tensor in tensors
                )
                min_horizon = min(
                    tensor.size(1) for tensor in tensors
                )
                tensors = [
                    tensor[:min_samples, :min_horizon]
                    for tensor in tensors
                ]

                if tensors[0].ndim != 4:
                    raise ValueError(
                        f"Expected a 4D direct forecast for {key}, but "
                        f"received {tuple(tensors[0].shape)}."
                    )

                # [samples, horizon, nodes, channels]
                combined[key] = torch.cat(tensors, dim=2)

            return combined

        outputs = {
            "by_subgraph": {},
            "combined": {},
        }
        all_loader_keys = set()

        for subgraph_idx in range(1, len(self.datasets) + 1):
            sub = self.datasets[subgraph_idx]
            scaler = sub["scaler"]
            subgraph = self._prepare_graph(
                sub.get(
                    "coastal_graph",
                    sub.get("graph"),
                )
            )

            outputs["by_subgraph"][subgraph_idx] = {}

            try:
                loader_keys = self._loader_keys_for_split(sub, split)
            except StopIteration as exc:
                raise KeyError(
                    f"No loader found for split='{split}' in subgraph "
                    f"{subgraph_idx}."
                ) from exc

            for loader_key in loader_keys:
                all_loader_keys.add(loader_key)
                loader: Iterable = sub[loader_key]
                sub_outputs = {key: [] for key in tensor_keys}
                metadata = []

                for batch_idx, (x, y, _) in enumerate(loader):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    history_state = self._select_history_state(x)
                    full_dim = max(
                        int(x.size(-1)),
                        int(y.size(-1)),
                        max(self.input_feature_idx) + 1,
                        max(self.output_feature_idx) + 1,
                    )

                    with amp.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    ):
                        pred = self.model(
                            history_state=history_state,
                            graph=subgraph,
                        )

                    pred_state_norm, target_state_norm, effective_horizon = (
                        self._align_prediction_and_target(
                            pred["state"],
                            y,
                        )
                    )
                    pred_state = self._inverse_selected(
                        pred_state_norm,
                        scaler,
                        self.output_feature_idx,
                        full_dim,
                    )
                    true_state = self._inverse_selected(
                        target_state_norm,
                        scaler,
                        self.output_feature_idx,
                        full_dim,
                    )

                    if hasattr(self.model, "rollout"):
                        rollout_norm = self.model.rollout(
                            history_state=history_state,
                            graph=subgraph,
                            horizon=effective_horizon,
                        )["state"][:, :effective_horizon]

                        rollout_state = self._inverse_selected(
                            rollout_norm,
                            scaler,
                            self.output_feature_idx,
                            full_dim,
                        )
                    else:
                        rollout_state = pred_state

                    sub_outputs["preds_state"].append(
                        pred_state.cpu()
                    )
                    sub_outputs["trues_state"].append(
                        true_state.cpu()
                    )
                    sub_outputs["rollout_preds_state"].append(
                        rollout_state.cpu()
                    )
                    sub_outputs["rollout_trues_state"].append(
                        true_state.cpu()
                    )

                    metadata.append(
                        {
                            "subgraph": subgraph_idx,
                            "loader": loader_key,
                            "batch": batch_idx,
                        }
                    )

                packed = {
                    key: (
                        torch.cat(sub_outputs[key], dim=0)
                        if sub_outputs[key]
                        else None
                    )
                    for key in tensor_keys
                }
                packed["meta"] = metadata
                outputs["by_subgraph"][subgraph_idx][
                    loader_key
                ] = packed

        for loader_key in sorted(all_loader_keys):
            outputs["combined"][loader_key] = (
                combine_subgraphs_by_node(
                    outputs["by_subgraph"],
                    loader_key,
                )
            )

        if save_npz:
            save_dir = os.path.join(
                self.log_dir,
                f"predictions_{split}",
            )
            os.makedirs(save_dir, exist_ok=True)

            for (
                subgraph_idx,
                subgraph_outputs,
            ) in outputs["by_subgraph"].items():
                for loader_key, data in subgraph_outputs.items():
                    safe_loader_key = (
                        loader_key.replace("X_", "")
                        .replace("_loader", "")
                    )
                    save_path = os.path.join(
                        save_dir,
                        (
                            f"subgraph_{subgraph_idx}_"
                            f"{safe_loader_key}.npz"
                        ),
                    )

                    np.savez_compressed(
                        save_path,
                        **{
                            key: data[key].numpy()
                            for key in tensor_keys
                            if data[key] is not None
                        },
                    )
                    log_string(
                        self.log_dir,
                        f"Saved subgraph predictions to {save_path}",
                    )

            for loader_key, data in outputs["combined"].items():
                safe_loader_key = (
                    loader_key.replace("X_", "")
                    .replace("_loader", "")
                )
                save_path = os.path.join(
                    save_dir,
                    f"combined_{safe_loader_key}_by_node.npz",
                )

                np.savez_compressed(
                    save_path,
                    **{
                        key: data[key].numpy()
                        for key in tensor_keys
                        if data[key] is not None
                    },
                )
                log_string(
                    self.log_dir,
                    f"Saved combined predictions to {save_path}",
                )

        return outputs