import torch

from .base import _BaseAggregator


def full_participation_add_remove_sensitivity(clip_bound, n_clients, scale=1.0):
    return scale * clip_bound / n_clients


def gaussian_noise_std(noise_multiplier, sensitivity):
    return noise_multiplier * sensitivity


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

        sensitivity = full_participation_add_remove_sensitivity(
            self.residual_clip_tau,
            n_clients,
            scale=self.residual_alpha,
        )
        noise_std = gaussian_noise_std(self.noise_multiplier, sensitivity)
        noise = torch.randn_like(mean_clipped_residual) * noise_std
        noisy_delta = self.residual_alpha * mean_clipped_residual + noise

        self.public_tracker = self.public_tracker + noisy_delta.detach()
        self.latest_diagnostics = {
            "C_res": self.residual_clip_tau,
            "sigma_res": self.noise_multiplier,
            "alpha": self.residual_alpha,
            "dp_residual_sensitivity": sensitivity,
            "is_anchor_round": 0.0,
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
            "server_center_norm": torch.norm(self.public_tracker).item(),
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


class ResidualTrackingDPFedAvgWithAnchorResets(_BaseAggregator):
    """Residual-tracking DP-FedAvg with periodic DP resets from client centers."""

    def __init__(
        self,
        residual_clip_tau,
        anchor_clip_tau,
        residual_alpha,
        residual_noise_multiplier,
        anchor_noise_multiplier,
        anchor_period,
    ):
        if anchor_period < 1:
            raise ValueError(f"anchor_period must be >= 1. Got {anchor_period}.")
        self.residual_clip_tau = residual_clip_tau
        self.anchor_clip_tau = anchor_clip_tau
        self.residual_alpha = residual_alpha
        self.residual_noise_multiplier = residual_noise_multiplier
        self.anchor_noise_multiplier = anchor_noise_multiplier
        self.anchor_period = anchor_period
        self.public_tracker = None
        self.round_index = 0
        self.latest_diagnostics = {}
        super().__init__()

    @property
    def noise_multiplier(self):
        return self.residual_noise_multiplier

    @noise_multiplier.setter
    def noise_multiplier(self, value):
        self.residual_noise_multiplier = value
        self.anchor_noise_multiplier = value

    def _clip(self, v, tau):
        v_norm = torch.norm(v)
        scale = min(1, tau / v_norm.clamp_min(1e-12).item())
        return v * scale

    def __call__(self, inputs):
        if not inputs:
            raise ValueError("ResidualTrackingDPFedAvgWithAnchorResets needs inputs.")
        if not isinstance(inputs[0], dict):
            raise TypeError(
                "ResidualTrackingDPFedAvgWithAnchorResets expects each client input "
                "to be a dict with clipped_residual and client_center tensors."
            )

        n_clients = len(inputs)
        clipped_residuals = torch.stack(
            [inp["clipped_residual"] for inp in inputs], dim=0
        )
        client_centers = torch.stack([inp["client_center_old"] for inp in inputs], dim=0)
        new_client_centers = torch.stack(
            [inp["client_center_new"] for inp in inputs], dim=0
        )
        device = clipped_residuals.device

        if self.public_tracker is None:
            self.public_tracker = torch.zeros_like(clipped_residuals[0])

        raw_update_norms = torch.tensor(
            [inp["raw_update_norm"] for inp in inputs],
            dtype=torch.float32,
            device=device,
        )
        residual_norms = torch.tensor(
            [inp["residual_norm"] for inp in inputs],
            dtype=torch.float32,
            device=device,
        )
        residual_clip_flags = torch.tensor(
            [inp["residual_was_clipped"] for inp in inputs],
            dtype=torch.float32,
            device=device,
        )
        clipped_residual_norms = torch.norm(clipped_residuals, dim=1)
        center_norms = torch.norm(client_centers, dim=1)
        new_center_norms = torch.norm(new_client_centers, dim=1)
        anchor_clip_flags = (center_norms > self.anchor_clip_tau).float()
        anchor_new_clip_flags = (new_center_norms > self.anchor_clip_tau).float()
        mean_clipped_residual = clipped_residuals.mean(dim=0)

        # Round 0 has deterministic zero centers, so adding anchor noise there only
        # injects tracker noise without releasing data-dependent center information.
        is_anchor_round = (
            self.round_index > 0 and self.round_index % self.anchor_period == 0
        )
        residual_sensitivity = full_participation_add_remove_sensitivity(
            self.residual_clip_tau,
            n_clients,
            scale=self.residual_alpha,
        )
        anchor_sensitivity = full_participation_add_remove_sensitivity(
            self.anchor_clip_tau,
            n_clients,
        )
        residual_noise_std = gaussian_noise_std(
            self.residual_noise_multiplier, residual_sensitivity
        )
        anchor_noise_std = gaussian_noise_std(
            self.anchor_noise_multiplier, anchor_sensitivity
        )
        residual_noise = torch.randn_like(mean_clipped_residual) * residual_noise_std
        noisy_delta = self.residual_alpha * mean_clipped_residual + residual_noise
        residual_noise_norm = torch.norm(residual_noise).item()
        residual_noisy_delta_norm = torch.norm(noisy_delta).item()
        anchor_noise_norm = 0.0

        if is_anchor_round:
            clipped_centers = torch.stack(
                [self._clip(center, self.anchor_clip_tau) for center in client_centers],
                dim=0,
            )
            mean_clipped_center = clipped_centers.mean(dim=0)
            anchor_noise = torch.randn_like(mean_clipped_center) * anchor_noise_std
            anchor = mean_clipped_center + anchor_noise
            self.public_tracker = (anchor + noisy_delta).detach()
            anchor_noise_norm = torch.norm(anchor_noise).item()
            anchor_clipped_norms = torch.norm(clipped_centers, dim=1)
            anchor_norm = torch.norm(anchor).item()
        else:
            self.public_tracker = (self.public_tracker + noisy_delta).detach()
            anchor_clipped_norms = torch.zeros_like(center_norms)
            anchor_norm = 0.0

        self.latest_diagnostics = {
            "round_index": self.round_index,
            "is_anchor_round": float(is_anchor_round),
            "C_res": self.residual_clip_tau,
            "C_anchor": self.anchor_clip_tau,
            "sigma_res": self.residual_noise_multiplier,
            "sigma_anchor": self.anchor_noise_multiplier,
            "alpha": self.residual_alpha,
            "anchor_period": self.anchor_period,
            "dp_residual_sensitivity": residual_sensitivity,
            "dp_anchor_sensitivity": anchor_sensitivity,
            "dp_anchor_from_preupdate_centers": 1.0 if is_anchor_round else 0.0,
            "server_center_norm": torch.norm(self.public_tracker).item(),
            "raw_update_norm_mean": raw_update_norms.mean().item(),
            "residual_norm_mean": residual_norms.mean().item(),
            "old_center_norm_mean": center_norms.mean().item(),
            "new_center_norm_mean": new_center_norms.mean().item(),
            "anchor_old_center_norm_mean": center_norms.mean().item(),
            "anchor_new_center_norm_mean": new_center_norms.mean().item(),
            "residual_to_raw_norm_ratio_mean": (
                residual_norms / raw_update_norms.clamp_min(1e-12)
            )
            .mean()
            .item(),
            "residual_clipping_frequency": residual_clip_flags.mean().item(),
            "anchor_clipping_frequency": (
                anchor_clip_flags.mean().item() if is_anchor_round else 0.0
            ),
            "anchor_old_center_clipping_frequency": (
                anchor_clip_flags.mean().item() if is_anchor_round else 0.0
            ),
            "anchor_new_center_clipping_frequency": (
                anchor_new_clip_flags.mean().item() if is_anchor_round else 0.0
            ),
            "dp_residual_clipped_norm_mean": clipped_residual_norms.mean().item(),
            "dp_residual_clipped_norm_max": clipped_residual_norms.max().item(),
            "dp_residual_mean_norm": torch.norm(mean_clipped_residual).item(),
            "dp_residual_noise_std": residual_noise_std,
            "dp_residual_noise_norm": residual_noise_norm,
            "dp_residual_noisy_delta_norm": residual_noisy_delta_norm,
            "dp_anchor_center_norm_mean": center_norms.mean().item(),
            "dp_anchor_center_norm_max": center_norms.max().item(),
            "dp_anchor_clipped_center_norm_mean": anchor_clipped_norms.mean().item(),
            "dp_anchor_noise_std": anchor_noise_std if is_anchor_round else 0.0,
            "dp_anchor_noise_norm": anchor_noise_norm,
            "dp_anchor_norm": anchor_norm,
            "dp_public_tracker_norm": torch.norm(self.public_tracker).item(),
        }
        self.round_index += 1

        return self.public_tracker.detach()

    def __str__(self):
        return (
            "ResidualTrackingDPFedAvgWithAnchorResets("
            "residual_clip_tau={}, anchor_clip_tau={}, residual_alpha={}, "
            "residual_noise_multiplier={}, anchor_noise_multiplier={}, "
            "anchor_period={})"
        ).format(
            self.residual_clip_tau,
            self.anchor_clip_tau,
            self.residual_alpha,
            self.residual_noise_multiplier,
            self.anchor_noise_multiplier,
            self.anchor_period,
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

        sensitivity = full_participation_add_remove_sensitivity(
            self.clip_tau, n_clients
        )
        noise_std = gaussian_noise_std(self.noise_multiplier, sensitivity)
        noise = torch.randn_like(mean_clipped_update) * noise_std
        noisy_average = mean_clipped_update + noise

        self.latest_diagnostics = {
            "C_std": self.clip_tau,
            "sigma": self.noise_multiplier,
            "dp_fedavg_sensitivity": sensitivity,
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
