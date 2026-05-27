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

    def __init__(self, momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(p.grad).detach()
                else:
                    param_state["momentum_buffer"].mul_(self.momentum).add_(p.grad)

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["momentum_buffer"].data.view(-1))
        return torch.cat(layer_gradients)


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
