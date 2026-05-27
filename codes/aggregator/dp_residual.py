import torch

from .base import _BaseAggregator


class CentralDPResidualTracking(_BaseAggregator):
    """Aggregate clipped residual increments and maintain a noisy public tracker."""

    def __init__(self, residual_clip_tau, residual_alpha, noise_multiplier):
        self.residual_clip_tau = residual_clip_tau
        self.residual_alpha = residual_alpha
        self.noise_multiplier = noise_multiplier
        self.public_tracker = None
        self.latest_diagnostics = {}
        super().__init__()

    def __call__(self, inputs):
        if self.public_tracker is None:
            self.public_tracker = torch.zeros_like(inputs[0])

        n_clients = len(inputs)
        clipped_residuals = torch.stack(inputs, dim=0)
        clipped_residual_norms = torch.norm(clipped_residuals, dim=1)
        mean_clipped_residual = clipped_residuals.mean(dim=0)

        noise_std = (
            self.noise_multiplier
            * self.residual_alpha
            * self.residual_clip_tau
            / n_clients
        )
        noise = torch.randn_like(mean_clipped_residual) * noise_std
        noisy_delta = self.residual_alpha * mean_clipped_residual + noise

        self.public_tracker = self.public_tracker + noisy_delta.detach()
        self.latest_diagnostics = {
            "dp_residual_clipped_norm_mean": clipped_residual_norms.mean().item(),
            "dp_residual_clipped_norm_max": clipped_residual_norms.max().item(),
            "dp_residual_clipped_to_tau_ratio_mean": (
                clipped_residual_norms.mean()
                / max(self.residual_clip_tau, 1e-12)
            ).item(),
            "dp_residual_mean_norm": torch.norm(mean_clipped_residual).item(),
            "dp_residual_noise_std": noise_std,
            "dp_residual_noise_norm": torch.norm(noise).item(),
            "dp_residual_noisy_delta_norm": torch.norm(noisy_delta).item(),
            "dp_public_tracker_norm": torch.norm(self.public_tracker).item(),
        }

        return self.public_tracker.detach()

    def __str__(self):
        return (
            "CentralDPResidualTracking(residual_clip_tau={}, residual_alpha={}, noise_multiplier={})"
        ).format(
            self.residual_clip_tau,
            self.residual_alpha,
            self.noise_multiplier,
        )


class CentralDPFedAvg(_BaseAggregator):
    """Standard central-DP FedAvg with clipped client updates and Gaussian noise."""

    def __init__(self, clip_tau, noise_multiplier):
        self.clip_tau = clip_tau
        self.noise_multiplier = noise_multiplier
        self.latest_diagnostics = {}
        super().__init__()

    def _clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.clip_tau / v_norm.clamp_min(1e-12).item())
        return v * scale

    def __call__(self, inputs):
        n_clients = len(inputs)
        raw_updates = torch.stack(inputs, dim=0)
        raw_update_norms = torch.norm(raw_updates, dim=1)
        clipped_updates = torch.stack([self._clip(v) for v in inputs], dim=0)
        clipped_update_norms = torch.norm(clipped_updates, dim=1)
        mean_clipped_update = clipped_updates.mean(dim=0)

        noise_std = self.noise_multiplier * self.clip_tau / n_clients
        noise = torch.randn_like(mean_clipped_update) * noise_std
        noisy_average = mean_clipped_update + noise

        self.latest_diagnostics = {
            "dp_fedavg_raw_update_norm_mean": raw_update_norms.mean().item(),
            "dp_fedavg_raw_update_norm_max": raw_update_norms.max().item(),
            "dp_fedavg_clipped_update_norm_mean": clipped_update_norms.mean().item(),
            "dp_fedavg_clipped_update_norm_max": clipped_update_norms.max().item(),
            "dp_fedavg_clipped_to_raw_norm_ratio": (
                clipped_update_norms.mean() / raw_update_norms.mean().clamp_min(1e-12)
            ).item(),
            "dp_fedavg_clip_fraction_mean": (
                (raw_update_norms > self.clip_tau).float().mean().item()
            ),
            "dp_fedavg_mean_clipped_update_norm": torch.norm(mean_clipped_update).item(),
            "dp_fedavg_noise_std": noise_std,
            "dp_fedavg_noise_norm": torch.norm(noise).item(),
            "dp_fedavg_noisy_average_norm": torch.norm(noisy_average).item(),
        }

        return noisy_average.detach()

    def __str__(self):
        return "CentralDPFedAvg(clip_tau={}, noise_multiplier={})".format(
            self.clip_tau,
            self.noise_multiplier,
        )
