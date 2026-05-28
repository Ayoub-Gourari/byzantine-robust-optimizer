import logging
import numpy as np
import os
import copy
import torch

from collections import defaultdict
from typing import Optional, Union, Callable, Any, Tuple


class TorchWorker(object):
    """A worker for distributed training.

    Compute gradients locally and store the gradient.
    """

    def __init__(
        self,
        data_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_func: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
        local_steps: int = 1,
    ):
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        if local_steps < 1:
            raise ValueError(f"local_steps must be >= 1. Got {local_steps}.")
        self.local_steps = local_steps

        self.running = {}
        self.metrics = {}
        self.state = defaultdict(dict)

    def add_metric(
        self, name: str, callback: Callable[[torch.Tensor, torch.Tensor], float]
    ):
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")

        self.metrics[name] = callback

    def add_metrics(self, metrics: dict):
        for name in metrics:
            self.add_metric(name, metrics[name])

    def __str__(self) -> str:
        return "TorchWorker"

    def train_epoch_start(self) -> None:
        self.running["train_loader_iterator"] = iter(self.data_loader)
        self.model.train()

    def compute_gradient(self) -> Tuple[float, int]:
        results = {"loss": 0, "length": 0, "metrics": defaultdict(float)}
        accumulated_grads = defaultdict(lambda: None)

        for _ in range(self.local_steps):
            data, target = self.running["train_loader_iterator"].__next__()
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, target)
            loss.backward()

            batch_length = len(target)
            results["loss"] += loss.item() * batch_length
            results["length"] += batch_length
            for name, metric in self.metrics.items():
                results["metrics"][name] += metric(output, target) * batch_length

            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if accumulated_grads[p] is None:
                        accumulated_grads[p] = torch.clone(p.grad).detach()
                    else:
                        accumulated_grads[p].add_(p.grad)

        self.optimizer.zero_grad()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if accumulated_grads[p] is None:
                    continue
                p.grad = accumulated_grads[p].div(self.local_steps)
        self._save_grad()

        self.running["data"] = data
        self.running["target"] = target

        for name, metric in self.metrics.items():
            results["metrics"][name] /= results["length"]
        results["loss"] /= results["length"]
        return results

    def get_gradient(self) -> torch.Tensor:
        return self._get_saved_grad()

    def finalize_private_state(self) -> None:
        pass

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for p in self.model.parameters():
            end = beg + len(p.grad.view(-1))
            x = gradient[beg:end].reshape_as(p.grad.data)
            p.grad.data = x.clone().detach()
            beg = end

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                param_state["saved_grad"] = torch.clone(p.grad).detach()

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["saved_grad"].data.view(-1))
        return torch.cat(layer_gradients)


class WorkerWithMomentum(TorchWorker):
    """
    Note that we use `WorkerWithMomentum` instead of using multiple `torch.optim.Optimizer`
    because we need to explicitly update the `momentum_buffer`.
    """

    def __init__(self, momentum, momentum_mode="classic", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        if momentum_mode not in ["classic", "ema"]:
            raise ValueError(f"Unknown momentum_mode: {momentum_mode}.")
        self.momentum_mode = momentum_mode

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(p.grad).detach()
                    if self.momentum_mode == "ema":
                        param_state["momentum_buffer"].mul_(1 - self.momentum)
                else:
                    param_state["momentum_buffer"].mul_(self.momentum)
                    if self.momentum_mode == "classic":
                        param_state["momentum_buffer"].add_(p.grad)
                    else:
                        param_state["momentum_buffer"].add_(
                            p.grad, alpha=1 - self.momentum
                        )

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["momentum_buffer"].data.view(-1))
        return torch.cat(layer_gradients)


class ResidualTrackingWorker(TorchWorker):
    """Client-side residual tracker for private residual-tracking FedAvg."""

    def __init__(
        self,
        residual_clip_tau,
        residual_alpha,
        residual_center_mode="ema",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.residual_clip_tau = residual_clip_tau
        self.residual_alpha = residual_alpha
        if residual_center_mode not in ["ema", "buffer"]:
            raise ValueError(
                f"Unknown residual_center_mode: {residual_center_mode}."
            )
        self.residual_center_mode = residual_center_mode
        self.private_center = None
        self.raw_prev_update_center = None
        self.raw_ema_center = None
        self.latest_diagnostics = {}

    def _clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.residual_clip_tau / v_norm.clamp_min(1e-12).item())
        return v * scale

    def _flatten_current_gradient(self):
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                layer_gradients.append(p.grad.detach().view(-1))
        return torch.cat(layer_gradients)

    def _save_grad(self) -> None:
        raw_update = self._flatten_current_gradient()
        if self.private_center is None:
            self.private_center = torch.zeros_like(raw_update)

        center_before = self.private_center
        residual = raw_update - center_before
        clipped_residual = self._clip(residual)
        clipped_residual_norm = torch.norm(clipped_residual).item()
        residual_norm = torch.norm(residual).item()
        raw_update_norm = torch.norm(raw_update).item()
        center_norm = torch.norm(center_before).item()

        if self.raw_prev_update_center is None:
            raw_prev_center = torch.zeros_like(raw_update)
        else:
            raw_prev_center = self.raw_prev_update_center

        if self.raw_ema_center is None:
            raw_ema_center = torch.zeros_like(raw_update)
        else:
            raw_ema_center = self.raw_ema_center

        raw_prev_residual_norm = torch.norm(raw_update - raw_prev_center).item()
        raw_ema_residual_norm = torch.norm(raw_update - raw_ema_center).item()

        if self.residual_center_mode == "ema":
            self.private_center = (
                center_before + self.residual_alpha * clipped_residual.detach()
            )
        else:
            self.private_center = raw_update.detach().clone()
        if self.raw_ema_center is None:
            self.raw_ema_center = raw_update.detach().clone()
        else:
            self.raw_ema_center = (
                (1 - self.residual_alpha) * raw_ema_center
                + self.residual_alpha * raw_update.detach()
            )
        self.raw_prev_update_center = raw_update.detach().clone()
        self.state["residual_tracking"]["saved_residual"] = clipped_residual.detach()
        self.latest_diagnostics = {
            "raw_update_norm": raw_update_norm,
            "private_center_norm": center_norm,
            "next_private_center_norm": torch.norm(self.private_center).item(),
            "residual_norm": residual_norm,
            "clipped_residual_norm": clipped_residual_norm,
            "residual_to_raw_norm_ratio": residual_norm / max(raw_update_norm, 1e-12),
            "raw_prev_residual_to_raw_norm_ratio": raw_prev_residual_norm
            / max(raw_update_norm, 1e-12),
            "raw_ema_residual_to_raw_norm_ratio": raw_ema_residual_norm
            / max(raw_update_norm, 1e-12),
            "clip_fraction": float(residual_norm > self.residual_clip_tau),
        }

    def _get_saved_grad(self) -> torch.Tensor:
        return self.state["residual_tracking"]["saved_residual"].data

    def __str__(self) -> str:
        return "ResidualTrackingWorker"


class AnchorResetResidualTrackingWorker(TorchWorker):
    """Client residual tracker that exposes old centers and stages center updates."""

    def __init__(
        self,
        residual_clip_tau,
        residual_alpha,
        residual_center_mode="ema",
        residual_center_beta=0.9,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.residual_clip_tau = residual_clip_tau
        self.residual_alpha = residual_alpha
        self.residual_center_beta = residual_center_beta
        normalized_mode = residual_center_mode.replace("-", "_")
        if normalized_mode not in ["ema", "clipped_residual_ema"]:
            raise ValueError(
                "Unknown residual_center_mode for anchor resets: "
                f"{residual_center_mode}."
            )
        if not 0 <= residual_center_beta <= 1:
            raise ValueError(
                f"residual_center_beta must be in [0, 1]. Got {residual_center_beta}."
            )
        self.residual_center_mode = normalized_mode
        self.private_center = None
        self.pending_private_center = None
        self.latest_diagnostics = {}

    def _clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.residual_clip_tau / v_norm.clamp_min(1e-12).item())
        return v * scale

    def _flatten_current_gradient(self):
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                layer_gradients.append(p.grad.detach().view(-1))
        return torch.cat(layer_gradients)

    def _save_grad(self) -> None:
        raw_update = self._flatten_current_gradient()
        if self.private_center is None:
            self.private_center = torch.zeros_like(raw_update)

        center_before = self.private_center
        residual = raw_update - center_before
        clipped_residual = self._clip(residual)
        clipped_residual_norm = torch.norm(clipped_residual).item()
        residual_norm = torch.norm(residual).item()
        raw_update_norm = torch.norm(raw_update).item()
        center_norm = torch.norm(center_before).item()

        if self.residual_center_mode == "ema":
            next_private_center = (
                self.residual_center_beta * center_before
                + (1 - self.residual_center_beta) * raw_update.detach()
            )
        else:
            next_private_center = (
                center_before + self.residual_alpha * clipped_residual.detach()
            )
        self.pending_private_center = next_private_center.detach()

        self.state["residual_tracking_anchor"]["payload"] = {
            "clipped_residual": clipped_residual.detach(),
            "client_center": center_before.detach(),
            "client_center_old": center_before.detach(),
            "client_center_new": self.pending_private_center.detach(),
            "raw_update_norm": raw_update_norm,
            "residual_norm": residual_norm,
            "clipped_residual_norm": clipped_residual_norm,
            "residual_was_clipped": float(residual_norm > self.residual_clip_tau),
        }
        self.latest_diagnostics = {
            "raw_update_norm": raw_update_norm,
            "private_center_norm": center_norm,
            "next_private_center_norm": torch.norm(self.pending_private_center).item(),
            "old_center_norm": center_norm,
            "new_center_norm": torch.norm(self.pending_private_center).item(),
            "residual_norm": residual_norm,
            "clipped_residual_norm": clipped_residual_norm,
            "residual_to_raw_norm_ratio": residual_norm / max(raw_update_norm, 1e-12),
            "clip_fraction": float(residual_norm > self.residual_clip_tau),
            "center_update_mode_id": float(
                0 if self.residual_center_mode == "ema" else 1
            ),
        }

    def _get_saved_grad(self):
        return self.state["residual_tracking_anchor"]["payload"]

    def finalize_private_state(self) -> None:
        if self.pending_private_center is not None:
            self.private_center = self.pending_private_center
            self.pending_private_center = None

    def __str__(self) -> str:
        return "AnchorResetResidualTrackingWorker"


class ByzantineWorker(TorchWorker):
    def configure(self, simulator):
        # call configure after defining DistribtuedSimulator
        self.simulator = simulator
        simulator.register_omniscient_callback(self.omniscient_callback)

    def compute_gradient(self) -> Tuple[float, int]:
        # Use self.simulator to get all other workers
        # Note that the byzantine worker does not modify the states directly.
        return super().compute_gradient()

    def get_gradient(self) -> torch.Tensor:
        # Use self.simulator to get all other workers
        return super().get_gradient()

    def omniscient_callback(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return "ByzantineWorker"
