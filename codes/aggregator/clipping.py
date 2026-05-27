import logging
import torch
import types
from .base import _BaseAggregator
from .base import _BaseAsyncAggregator


debug_logger = logging.getLogger("debug")


class Clipping(_BaseAggregator):
    def __init__(
        self,
        tau,
        n_iter=1,
        center_update="current",
        center_momentum=0.9,
        center_source="aggregate",
    ):
        self.tau = tau
        self.n_iter = n_iter
        self.center_update = center_update
        self.center_momentum = center_momentum
        self.center_source = center_source
        super(Clipping, self).__init__()
        self.momentum = None
        self.latest_diagnostics = {}

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs):
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])

        center_before = torch.clone(self.momentum).detach()
        raw_norms = torch.tensor(
            [torch.norm(v).item() for v in inputs],
            device=inputs[0].device,
        )
        centered_norms = torch.tensor(
            [torch.norm(v - center_before).item() for v in inputs],
            device=inputs[0].device,
        )
        mean_update = torch.stack(inputs, dim=0).mean(dim=0)
        mean_update_norm = torch.norm(mean_update).item()
        center_norm = torch.norm(center_before).item()
        center_eps = center_before.norm().clamp_min(1e-12)
        cosines = torch.tensor(
            [
                torch.dot(v, center_before).item()
                / (torch.norm(v).clamp_min(1e-12).item() * center_eps.item())
                for v in inputs
            ],
            device=inputs[0].device,
        )

        clip_fractions = []
        refined_center = torch.clone(self.momentum).detach()
        for _ in range(self.n_iter):
            clip_fractions.append(
                sum(float(torch.norm(v - refined_center) > self.tau) for v in inputs)
                / len(inputs)
            )
            refined_center = (
                sum(self.clip(v - refined_center) for v in inputs) / len(inputs)
                + refined_center
            )

        aggregate = torch.clone(refined_center).detach()
        if self.center_update == "current":
            self.momentum = aggregate
        elif self.center_update == "ema":
            center_target = aggregate if self.center_source == "aggregate" else mean_update
            self.momentum = (
                self.center_momentum * center_before
                + (1 - self.center_momentum) * center_target
            )
        else:
            raise NotImplementedError(self.center_update)

        self.latest_diagnostics = {
            "center_norm": center_norm,
            "raw_grad_norm_mean": raw_norms.mean().item(),
            "raw_grad_norm_max": raw_norms.max().item(),
            "centered_grad_distance_mean": centered_norms.mean().item(),
            "centered_grad_distance_max": centered_norms.max().item(),
            "centered_to_raw_norm_ratio": (
                centered_norms.mean() / raw_norms.mean().clamp_min(1e-12)
            ).item(),
            "mean_update_norm": mean_update_norm,
            "aggregate_norm": torch.norm(aggregate).item(),
            "next_center_norm": torch.norm(self.momentum).item(),
            "mean_cosine_with_center": cosines.mean().item(),
            "min_cosine_with_center": cosines.min().item(),
            "center_to_mean_update_norm_ratio": center_norm / max(mean_update_norm, 1e-12),
            "clip_fraction_mean": sum(clip_fractions) / len(clip_fractions),
        }

        return aggregate.detach()

    def __str__(self):
        return (
            "Clipping (tau={}, n_iter={}, center_update={}, center_momentum={}, center_source={})"
        ).format(
            self.tau,
            self.n_iter,
            self.center_update,
            self.center_momentum,
            self.center_source,
        )


class AnchorClipping(Clipping):
    def __init__(self, node, weights, opt, model, tau, n_iter=1):
        super(AnchorClipping, self).__init__(tau, n_iter)
        self._anchor_buffer = self._vectorize_model(model)
        self.opt = self._wrap_step(opt, model)
        assert n_iter == 1
        assert len(weights.shape) == 1
        self.node = node
        self.weights = weights

    def _vectorize_model(self, model):
        """"""
        state_dict = model.state_dict()
        return torch.cat([state_dict[k].data.view(-1) for k in state_dict])

    def _wrap_step(self, opt: torch.optim.Optimizer, model: torch.nn.Module):
        """Wrap the step function of opt to track the change."""
        debug_logger.info(f"Wrap the step function of opt")

        if hasattr(opt, "_core_step") or hasattr(opt, "anchorclipping"):
            raise NotImplementedError

        # Cache the old class
        opt._core_step = types.MethodType(opt.__class__.step, opt)
        opt.anchorclipping = self

        # Update the anchor vector y everytime the `step` is called.
        def anchor_clipping_step(self, closure=None):
            state_dict = model.state_dict()
            flattened = torch.cat([state_dict[k].data.view(-1) for k in state_dict])
            self._core_step(closure=closure)
            after_state_dict = model.state_dict()
            after_flattened = torch.cat(
                [state_dict[k].data.view(-1) for k in after_state_dict]
            )
            # debug_logger.info((after_flattened - flattened)[:5])
            self.anchorclipping._anchor_buffer.add_(after_flattened - flattened)

        opt.step = types.MethodType(anchor_clipping_step, opt)

    def __call__(self, inputs):
        assert len(inputs) == 1 + len(self.node.edges)
        clipped = self._anchor_buffer + self.clip(inputs[0] - self._anchor_buffer)
        s = self.weights[self.node.index] * clipped
        for e, inp in zip(self.node.edges, inputs[1:]):
            theothernode = e.theother(self.node)
            clipped = self._anchor_buffer + self.clip(inp - self._anchor_buffer)
            s += self.weights[theothernode.index] * clipped
        return s

    def __str__(self):
        return "AnchorClipping(tau={}, n_iter={})".format(self.tau, self.n_iter)


class AsyncCenteredClipping(_BaseAsyncAggregator):
    """
    Comparing to Clipping, AsyncCenteredClipping does not average the clipped gradient but use
    fraction $1 / n$ where `n` is the number of total gradients.

    """

    def __init__(self, tau, n_iter=1):
        self.tau = tau
        self.n_iter = n_iter
        super(AsyncCenteredClipping, self).__init__()

        self.momentum = 0

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs):
        n = len(inputs)
        filtered = list(filter(lambda x: x is not None, inputs))

        for _ in range(self.n_iter):
            self.momentum = (
                sum(self.clip(v - self.momentum) for v in filtered) / n + self.momentum
            )

        return torch.clone(self.momentum).detach()

    def __str__(self):
        return "AsyncCenteredClipping (tau={}, n_iter={})".format(self.tau, self.n_iter)
