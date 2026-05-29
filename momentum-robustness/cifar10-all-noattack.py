"""
"""
import argparse
import numpy as np
import os
import torch
from torchvision import datasets

from torch.nn.modules.loss import CrossEntropyLoss
from codes.sampler import DistributedSampler
from codes.simulators.simulator import (
    ParallelTrainer,
    DistributedEvaluator,
)
from codes.simulators.worker import (
    AnchorResetResidualTrackingWorker,
    ResidualTrackingWorker,
    WorkerWithMomentum,
)
from codes.simulators.server import TorchServer

from codes.tasks.cifar10 import cifar10, get_resnet
from codes.utils import top1_accuracy, initialize_logger
from codes.aggregator.clipping import Clipping
from codes.aggregator.base import Mean
from codes.aggregator.dp_residual import (
    CentralDPFedAvg,
    CentralDPResidualTracking,
    ResidualTrackingDPFedAvgWithAnchorResets,
    full_participation_add_remove_sensitivity,
    gaussian_noise_std,
)

try:
    import wandb
except ImportError:  # pragma: no cover - handled at runtime when wandb logging is requested
    wandb = None


EXP_ID = __file__[:-3]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"
DATA_DIR = ROOT_DIR + "datasets/"
EXP_DIR = ROOT_DIR + f"outputs/{EXP_ID}/"

parser = argparse.ArgumentParser(description="")
parser.add_argument("--use-cuda", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.1, help="Server/client learning rate.")
parser.add_argument(
    "--epochs",
    type=int,
    default=70,
    help="Number of training epochs.",
)

parser.add_argument("--attack", type=str, default="NA", help="Select from BF and LF.")
parser.add_argument(
    "--agg",
    type=str,
    default="cp",
    help=(
        "Aggregator. Use dp-fedavg, dp-residual, or dp-residual-anchor for "
        "central-DP baselines."
    ),
)
parser.add_argument(
    "--model",
    type=str,
    default="resnet20",
    help="CIFAR ResNet model to use.",
)
parser.add_argument("--momentum", type=float, default=0, help="momentum")
parser.add_argument(
    "--momentum-mode",
    type=str,
    default="ema",
    choices=["classic", "ema"],
    help="Use original momentum accumulation or normalized EMA worker momentum.",
)
parser.add_argument(
    "--local-steps",
    type=int,
    default=1,
    help="Number of local minibatch gradients to average per worker before aggregation.",
)
parser.add_argument(
    "--clip-tau",
    type=float,
    default=100.0,
    help="Centered clipping radius for the no-attack experiment.",
)
parser.add_argument(
    "--inner-iterations", type=int, default=1, help="[HP]: number of inner iterations."
)
parser.add_argument(
    "--center-update",
    type=str,
    default="current",
    choices=["current", "ema"],
    help="How to update the center used by centered clipping.",
)
parser.add_argument(
    "--center-momentum",
    type=float,
    default=0.9,
    help="EMA coefficient when using --center-update ema.",
)
parser.add_argument(
    "--center-source",
    type=str,
    default="aggregate",
    choices=["aggregate", "mean"],
    help="What target the EMA center should track.",
)
parser.add_argument(
    "--center-scale",
    type=float,
    default=1.0,
    help="Scalar multiplier applied to the center before measuring distances and clipping.",
)
parser.add_argument(
    "--per-client-center-momentum",
    type=float,
    default=0.9,
    help="EMA coefficient for per-client historical centers used in diagnostics.",
)
parser.add_argument(
    "--residual-alpha",
    type=float,
    default=1.0,
    help="Client tracker step size for residual-tracking DP methods.",
)
parser.add_argument(
    "--residual-center-mode",
    type=str,
    default="ema",
    choices=["ema", "buffer", "clipped-residual-ema", "clipped_residual_ema"],
    help=(
        "Client-side center update. For dp-residual this preserves the legacy "
        "ema/buffer behavior. For dp-residual-anchor, use ema for "
        "c_i = beta c_i + (1 - beta) g_i or clipped-residual-ema for "
        "c_i = c_i + alpha clip(g_i - c_i)."
    ),
)
parser.add_argument(
    "--residual-center-beta",
    type=float,
    default=0.9,
    help="EMA beta for --agg dp-residual-anchor when --residual-center-mode ema.",
)
parser.add_argument(
    "--dp-noise-multiplier",
    type=float,
    default=0.0,
    help=(
        "Central Gaussian noise multiplier for DP-FedAvg and residual releases "
        "with add/remove adjacency."
    ),
)
parser.add_argument(
    "--dp-anchor-noise-multiplier",
    type=float,
    default=None,
    help=(
        "Central Gaussian noise multiplier for anchor resets. Defaults to "
        "--dp-noise-multiplier."
    ),
)
parser.add_argument(
    "--anchor-clip-tau",
    type=float,
    default=None,
    help="Clipping radius C_anchor for --agg dp-residual-anchor.",
)
parser.add_argument(
    "--anchor-period",
    type=int,
    default=100,
    help=(
        "Reset server center every anchor_period rounds for dp-residual-anchor. "
        "Use -1 to disable anchor resets."
    ),
)
parser.add_argument(
    "--target-epsilon",
    type=float,
    default=None,
    help="Target epsilon for the full-participation central-DP accountant.",
)
parser.add_argument(
    "--target-delta",
    type=float,
    default=1e-5,
    help="Target delta for the full-participation central-DP accountant.",
)
parser.add_argument(
    "--privacy-dry-run",
    action="store_true",
    default=False,
    help="Print the DP privacy table and exit before model/data/training setup.",
)
parser.add_argument(
    "--privacy-report-anchor-periods",
    type=str,
    default="10,20,50,-1",
    help="Comma-separated anchor periods for --privacy-dry-run comparison rows.",
)
parser.add_argument(
    "--privacy-report-fedavg-clip-tau",
    type=float,
    default=None,
    help=(
        "DP-FedAvg clipping radius used in --privacy-dry-run tables. Defaults "
        "to --clip-tau."
    ),
)
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument(
    "--wandb-project",
    type=str,
    default="Federated_Centered_Clipping",
    help="W&B project name.",
)
parser.add_argument(
    "--wandb-entity",
    type=str,
    default="ae-gourari-cole-polytechnique",
    help="W&B entity name.",
)
parser.add_argument(
    "--wandb-run-name",
    type=str,
    default=None,
    help="Optional W&B run name.",
)

args = parser.parse_args()


N_WORKERS = 14
N_BYZ = 0
BATCH_SIZE = 32
TEST_BATCH_SIZE = 128
MAX_BATCHES_PER_EPOCH = 9999999
EPOCHS = args.epochs
MOMENTUM = args.momentum
ANCHOR_CLIP_TAU = (
    args.anchor_clip_tau if args.anchor_clip_tau is not None else args.clip_tau
)
ANCHOR_NOISE_MULTIPLIER = (
    args.dp_anchor_noise_multiplier
    if args.dp_anchor_noise_multiplier is not None
    else args.dp_noise_multiplier
)
ANCHOR_EFFECTIVE_ALPHA = 1.0 if args.agg == "dp-residual-anchor" else args.residual_alpha
ANCHOR_CENTER_LABEL = (
    "prevmom" if args.agg == "dp-residual-anchor" else args.residual_center_mode
)
DP_NOISE_LABEL = (
    f"eps{args.target_epsilon}-delta{args.target_delta}"
    if args.target_epsilon is not None
    else f"sigma{args.dp_noise_multiplier}"
)
ANCHOR_LABEL = (
    ""
    if args.agg != "dp-residual-anchor"
    else (
        f"_Canchor{ANCHOR_CLIP_TAU}"
        f"_sanchor{ANCHOR_NOISE_MULTIPLIER}"
        f"_aperiod{args.anchor_period}"
        f"_beta{args.residual_center_beta}"
    )
)

# LOG_DIR = EXP_DIR + "log"
LOG_DIR = (
    EXP_DIR
    + ("debug/" if args.debug else "")
    + (
        f"{args.attack}_{args.agg}_tau{args.clip_tau}_m{args.momentum}"
        f"_mom{args.momentum_mode}"
        f"_model{args.model}"
        f"_lr{args.lr}"
        f"_center{args.center_update}-{args.center_source}-beta{args.center_momentum}"
        f"-scale{args.center_scale}"
        f"_rcenter{ANCHOR_CENTER_LABEL}-alpha{ANCHOR_EFFECTIVE_ALPHA}"
        f"{ANCHOR_LABEL}"
        f"_{DP_NOISE_LABEL}"
        f"_local{args.local_steps}"
        f"_seed{args.seed}"
    )
)

assert args.attack == "NA"


def get_sampler_callback(rank):
    def sampler_callback(x):
        return DistributedSampler(
            num_replicas=N_WORKERS,
            rank=rank,
            shuffle=True,
            dataset=x,
        )

    return sampler_callback


def _get_aggregator():
    if args.agg == "avg":
        return Mean()

    if args.agg == "cp":
        return Clipping(
            tau=args.clip_tau,
            n_iter=args.inner_iterations,
            center_update=args.center_update,
            center_momentum=args.center_momentum,
            center_source=args.center_source,
            center_scale=args.center_scale,
        )

    if args.agg == "dp-residual":
        return CentralDPResidualTracking(
            residual_clip_tau=args.clip_tau,
            residual_alpha=args.residual_alpha,
            noise_multiplier=args.dp_noise_multiplier,
        )

    if args.agg == "dp-residual-anchor":
        return ResidualTrackingDPFedAvgWithAnchorResets(
            residual_clip_tau=args.clip_tau,
            anchor_clip_tau=ANCHOR_CLIP_TAU,
            residual_alpha=args.residual_alpha,
            residual_noise_multiplier=args.dp_noise_multiplier,
            anchor_noise_multiplier=ANCHOR_NOISE_MULTIPLIER,
            anchor_period=args.anchor_period,
        )

    if args.agg == "dp-fedavg":
        return CentralDPFedAvg(
            clip_tau=args.clip_tau,
            noise_multiplier=args.dp_noise_multiplier,
        )

    raise NotImplementedError(args.agg)


def initialize_worker(
    worker_rank, model, optimizer, loss_func, device, kwargs
):
    train_loader = cifar10(
        data_dir=DATA_DIR,
        train=True,
        download=True,
        batch_size=BATCH_SIZE,
        sampler_callback=get_sampler_callback(worker_rank),
        dataset_cls=datasets.CIFAR10,
        drop_last=True,  # Exclude the influence of non-full batch.
        **kwargs,
    )
    worker_kwargs = {
        "local_steps": args.local_steps,
        "data_loader": train_loader,
        "model": model,
        "loss_func": loss_func,
        "device": device,
        "optimizer": optimizer,
        **kwargs,
    }

    if args.agg == "dp-residual":
        residual_center_mode = (
            "ema"
            if args.residual_center_mode.replace("-", "_") == "clipped_residual_ema"
            else args.residual_center_mode
        )
        return ResidualTrackingWorker(
            residual_clip_tau=args.clip_tau,
            residual_alpha=args.residual_alpha,
            residual_center_mode=residual_center_mode,
            **worker_kwargs,
        )

    if args.agg == "dp-residual-anchor":
        return AnchorResetResidualTrackingWorker(
            residual_clip_tau=args.clip_tau,
            residual_alpha=args.residual_alpha,
            residual_center_mode=args.residual_center_mode,
            residual_center_beta=args.residual_center_beta,
            momentum=MOMENTUM,
            momentum_mode=args.momentum_mode,
            **worker_kwargs,
        )

    return WorkerWithMomentum(
        momentum=MOMENTUM,
        momentum_mode=args.momentum_mode,
        **worker_kwargs,
    )


def maybe_init_wandb():
    if not args.wandb:
        return None

    if wandb is None:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb` before using --wandb."
        )

    run_name = args.wandb_run_name or (
        (
            f"noattack-{args.agg}-tau{args.clip_tau}-m{args.momentum}"
            f"-mom{args.momentum_mode}"
            f"-model{args.model}"
            f"-lr{args.lr}"
            f"-center{args.center_update}-{args.center_source}-beta{args.center_momentum}"
            f"-scale{args.center_scale}"
            f"-rcenter{ANCHOR_CENTER_LABEL}-alpha{ANCHOR_EFFECTIVE_ALPHA}"
            f"{ANCHOR_LABEL}"
            f"-{DP_NOISE_LABEL}"
            f"-local{args.local_steps}"
            f"-seed{args.seed}"
        )
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "attack": args.attack,
            "agg": args.agg,
            "model": args.model,
            "lr": args.lr,
            "clip_tau": args.clip_tau,
            "inner_iterations": args.inner_iterations,
            "momentum": args.momentum,
            "momentum_mode": args.momentum_mode,
            "local_steps": args.local_steps,
            "center_update": args.center_update,
            "center_momentum": args.center_momentum,
            "center_source": args.center_source,
            "center_scale": args.center_scale,
            "per_client_center_momentum": args.per_client_center_momentum,
            "residual_alpha": ANCHOR_EFFECTIVE_ALPHA,
            "residual_center_mode": ANCHOR_CENTER_LABEL,
            "residual_center_beta": args.residual_center_beta,
            "dp_noise_multiplier": args.dp_noise_multiplier,
            "dp_anchor_noise_multiplier": ANCHOR_NOISE_MULTIPLIER,
            "anchor_clip_tau": ANCHOR_CLIP_TAU,
            "anchor_period": args.anchor_period,
            "requested_dp_noise_multiplier": args.dp_noise_multiplier,
            "requested_dp_anchor_noise_multiplier": ANCHOR_NOISE_MULTIPLIER,
            "target_epsilon": args.target_epsilon,
            "target_delta": args.target_delta,
            "dp_adjacency": "add_remove",
            "seed": args.seed,
            "n_workers": N_WORKERS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        },
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("ardp/*", step_metric="train/global_step")
    wandb.define_metric("privacy/*", step_metric="train/global_step")
    wandb.define_metric("validation/*", step_metric="train/global_step")
    return run


def compute_full_participation_noise_multiplier(
    target_epsilon, target_delta, total_rounds
):
    if target_epsilon is None:
        return None
    if target_epsilon <= 0:
        raise ValueError(f"target_epsilon must be > 0. Got {target_epsilon}.")
    if not 0 < target_delta < 1:
        raise ValueError(f"target_delta must be in (0, 1). Got {target_delta}.")
    if total_rounds <= 0:
        raise ValueError(f"total_rounds must be > 0. Got {total_rounds}.")

    log_inv_delta = np.log(1.0 / target_delta)
    sqrt_rho = np.sqrt(log_inv_delta + target_epsilon) - np.sqrt(log_inv_delta)
    rho = max(float(sqrt_rho**2), 1e-16)
    return np.sqrt(total_rounds / (2.0 * rho))


def compute_epsilon_from_rho(rho, target_delta):
    if rho == float("inf"):
        return float("inf")
    if not 0 < target_delta < 1:
        raise ValueError(f"target_delta must be in (0, 1). Got {target_delta}.")
    return rho + 2.0 * np.sqrt(rho * np.log(1.0 / target_delta))


def compute_composed_full_participation_epsilon(mechanisms, target_delta):
    total_rho = 0.0
    for mechanism in mechanisms:
        releases = mechanism["releases"]
        noise_multiplier = mechanism["noise_multiplier"]
        if releases <= 0:
            continue
        if noise_multiplier <= 0:
            return float("inf")
        total_rho += releases / (2.0 * noise_multiplier**2)
    return compute_epsilon_from_rho(total_rho, target_delta)


def compute_full_participation_epsilon(noise_multiplier, target_delta, total_rounds):
    if total_rounds <= 0:
        raise ValueError(f"total_rounds must be > 0. Got {total_rounds}.")

    return compute_composed_full_participation_epsilon(
        [
            {
                "name": "single",
                "releases": total_rounds,
                "noise_multiplier": noise_multiplier,
            }
        ],
        target_delta,
    )


def estimate_rounds_per_epoch(dataset_size=50000):
    samples_per_worker = int(np.ceil(dataset_size * 1.0 / N_WORKERS))
    return samples_per_worker // BATCH_SIZE


def parse_anchor_periods(value):
    periods = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        periods.append(int(item))
    if not periods:
        raise ValueError("At least one anchor period is required.")
    return periods


def count_anchor_rounds(total_rounds, anchor_period):
    if anchor_period == -1:
        return 0
    if anchor_period < 1:
        raise ValueError(f"anchor_period must be -1 or >= 1. Got {anchor_period}.")
    if total_rounds <= 1:
        return 0
    return (total_rounds - 1) // anchor_period


def dp_fedavg_mechanisms(total_rounds, clip_tau, noise_multiplier):
    sensitivity = full_participation_add_remove_sensitivity(clip_tau, N_WORKERS)
    return [
        {
            "name": "dp_fedavg",
            "releases": total_rounds,
            "sensitivity": sensitivity,
            "noise_multiplier": noise_multiplier,
            "noise_std": gaussian_noise_std(noise_multiplier, sensitivity),
        }
    ]


def ardp_mechanisms(
    total_rounds,
    c_res,
    c_anchor,
    anchor_period,
    residual_sigma,
    anchor_sigma,
):
    anchor_releases = count_anchor_rounds(total_rounds, anchor_period)
    residual_sensitivity = full_participation_add_remove_sensitivity(
        c_res, N_WORKERS
    )
    anchor_sensitivity = (
        0.0
        if anchor_releases == 0
        else full_participation_add_remove_sensitivity(c_anchor, N_WORKERS)
    )
    return [
        {
            "name": "residual",
            "releases": total_rounds,
            "sensitivity": residual_sensitivity,
            "noise_multiplier": residual_sigma,
            "noise_std": gaussian_noise_std(residual_sigma, residual_sensitivity),
        },
        {
            "name": "anchor",
            "releases": anchor_releases,
            "sensitivity": anchor_sensitivity,
            "noise_multiplier": anchor_sigma,
            "noise_std": (
                0.0
                if anchor_releases == 0
                else gaussian_noise_std(anchor_sigma, anchor_sensitivity)
            ),
        },
    ]


def common_sigma_for_mechanisms(target_epsilon, target_delta, release_count):
    return compute_full_participation_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        total_rounds=release_count,
    )


def privacy_row(
    method,
    total_rounds,
    c_res,
    c_anchor,
    anchor_period,
    residual_sigma,
    anchor_sigma,
    mechanisms,
):
    epsilon = compute_composed_full_participation_epsilon(
        mechanisms=mechanisms,
        target_delta=args.target_delta,
    )
    residual_rounds = mechanisms[0]["releases"] if mechanisms else 0
    anchor_rounds = (
        next((m["releases"] for m in mechanisms if m["name"] == "anchor"), 0)
        if mechanisms
        else 0
    )
    residual_sensitivity = mechanisms[0]["sensitivity"] if mechanisms else 0.0
    anchor_sensitivity = (
        next((m["sensitivity"] for m in mechanisms if m["name"] == "anchor"), 0.0)
        if mechanisms
        else 0.0
    )
    return {
        "method": method,
        "rounds_res": residual_rounds,
        "rounds_anchor": anchor_rounds,
        "C_res": c_res,
        "C_anchor": c_anchor,
        "sigma_res": residual_sigma,
        "sigma_anchor": anchor_sigma,
        "epsilon": epsilon,
        "delta": args.target_delta,
        "residual_sensitivity": residual_sensitivity,
        "anchor_sensitivity": anchor_sensitivity,
        "anchor_period": anchor_period,
        "total_rounds": total_rounds,
    }


def format_privacy_value(value):
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if value == float("inf"):
        return "inf"
    return f"{value:.6g}"


def print_privacy_table(rows, rounds_per_epoch, total_rounds):
    print(
        "Privacy dry run:"
        f" rounds_per_epoch={rounds_per_epoch}, total_rounds={total_rounds},"
        f" epochs={EPOCHS}, N={N_WORKERS}, delta={args.target_delta}"
    )
    header = [
        "method",
        "anchor_period",
        "rounds_res",
        "rounds_anchor",
        "C_res",
        "C_anchor",
        "sigma_res",
        "sigma_anchor",
        "epsilon",
        "delta",
        "res_sens",
        "anchor_sens",
    ]
    print(" ".join(f"{item:>14}" for item in header))
    for row in rows:
        values = [
            row["method"],
            row["anchor_period"],
            row["rounds_res"],
            row["rounds_anchor"],
            row["C_res"],
            row["C_anchor"],
            row["sigma_res"],
            row["sigma_anchor"],
            row["epsilon"],
            row["delta"],
            row["residual_sensitivity"],
            row["anchor_sensitivity"],
        ]
        print(" ".join(f"{format_privacy_value(value):>14}" for value in values))


def privacy_metrics_from_mechanisms(
    mechanisms,
    epsilon,
    target_epsilon,
    target_delta,
    c_res,
    c_anchor,
    sigma_res,
    sigma_anchor,
):
    residual = mechanisms[0] if mechanisms else {}
    anchor = next((m for m in mechanisms if m["name"] == "anchor"), {})
    return {
        "privacy/target_epsilon": (
            float("nan") if target_epsilon is None else target_epsilon
        ),
        "privacy/epsilon": epsilon,
        "privacy/delta": target_delta,
        "privacy/residual_rounds": residual.get("releases", 0),
        "privacy/anchor_rounds": anchor.get("releases", 0),
        "privacy/sigma_res": sigma_res,
        "privacy/sigma_anchor": sigma_anchor,
        "privacy/C_res": c_res,
        "privacy/C_anchor": 0.0 if c_anchor is None else c_anchor,
        "privacy/residual_sensitivity": residual.get("sensitivity", 0.0),
        "privacy/anchor_sensitivity": anchor.get("sensitivity", 0.0),
    }


def run_privacy_dry_run():
    rounds_per_epoch = estimate_rounds_per_epoch()
    total_rounds = EPOCHS * rounds_per_epoch
    anchor_periods = parse_anchor_periods(args.privacy_report_anchor_periods)
    rows = build_privacy_rows(total_rounds, anchor_periods)
    print_privacy_table(rows, rounds_per_epoch, total_rounds)


def build_privacy_rows(total_rounds, anchor_periods):
    fedavg_clip_tau = (
        args.privacy_report_fedavg_clip_tau
        if args.privacy_report_fedavg_clip_tau is not None
        else args.clip_tau
    )
    rows = []

    if args.target_epsilon is not None:
        fedavg_sigma = common_sigma_for_mechanisms(
            args.target_epsilon,
            args.target_delta,
            total_rounds,
        )
    else:
        fedavg_sigma = args.dp_noise_multiplier
    rows.append(
        privacy_row(
            "DP-FedAvg",
            total_rounds,
            fedavg_clip_tau,
            None,
            "-",
            fedavg_sigma,
            None,
            dp_fedavg_mechanisms(total_rounds, fedavg_clip_tau, fedavg_sigma),
        )
    )

    for anchor_period in anchor_periods:
        anchor_releases = count_anchor_rounds(total_rounds, anchor_period)
        if args.target_epsilon is not None:
            ardp_sigma = common_sigma_for_mechanisms(
                args.target_epsilon,
                args.target_delta,
                total_rounds + anchor_releases,
            )
            residual_sigma = ardp_sigma
            anchor_sigma = ardp_sigma
        else:
            residual_sigma = args.dp_noise_multiplier
            anchor_sigma = ANCHOR_NOISE_MULTIPLIER
        rows.append(
            privacy_row(
                "ARDP",
                total_rounds,
                args.clip_tau,
                ANCHOR_CLIP_TAU,
                anchor_period,
                residual_sigma,
                anchor_sigma,
                ardp_mechanisms(
                    total_rounds,
                    args.clip_tau,
                    ANCHOR_CLIP_TAU,
                    anchor_period,
                    residual_sigma,
                    anchor_sigma,
                ),
            )
        )
    return rows


def get_dp_accountant_mechanisms(total_rounds, residual_sigma, anchor_sigma):
    if args.agg == "dp-fedavg":
        return dp_fedavg_mechanisms(total_rounds, args.clip_tau, residual_sigma)

    if args.agg == "dp-residual":
        sensitivity = full_participation_add_remove_sensitivity(
            args.clip_tau, N_WORKERS, scale=args.residual_alpha
        )
        return [
            {
                "name": "residual",
                "releases": total_rounds,
                "sensitivity": sensitivity,
                "noise_multiplier": residual_sigma,
                "noise_std": gaussian_noise_std(residual_sigma, sensitivity),
            }
        ]

    if args.agg == "dp-residual-anchor":
        return ardp_mechanisms(
            total_rounds,
            args.clip_tau,
            ANCHOR_CLIP_TAU,
            args.anchor_period,
            residual_sigma,
            anchor_sigma,
        )

    return []


class PerClientHistoryDiagnostics:
    def __init__(self, center_momentum):
        self.center_momentum = center_momentum
        self.prev_updates = None
        self.ema_centers = None
        self.latest_diagnostics = {}

    def update(self, updates):
        if not updates:
            self.latest_diagnostics = {}
            return self.latest_diagnostics

        raw_norms = torch.tensor(
            [torch.norm(update).item() for update in updates],
            device=updates[0].device,
        )
        raw_norm_mean = raw_norms.mean().clamp_min(1e-12)

        if self.prev_updates is None:
            prev_centers = [torch.zeros_like(update) for update in updates]
        else:
            prev_centers = self.prev_updates

        if self.ema_centers is None:
            ema_centers = [torch.zeros_like(update) for update in updates]
        else:
            ema_centers = self.ema_centers

        prev_residual_norms = torch.tensor(
            [torch.norm(update - center).item() for update, center in zip(updates, prev_centers)],
            device=updates[0].device,
        )
        ema_residual_norms = torch.tensor(
            [torch.norm(update - center).item() for update, center in zip(updates, ema_centers)],
            device=updates[0].device,
        )
        prev_cosines = torch.tensor(
            [
                torch.dot(update, center).item()
                / (
                    torch.norm(update).clamp_min(1e-12).item()
                    * torch.norm(center).clamp_min(1e-12).item()
                )
                for update, center in zip(updates, prev_centers)
            ],
            device=updates[0].device,
        )
        ema_cosines = torch.tensor(
            [
                torch.dot(update, center).item()
                / (
                    torch.norm(update).clamp_min(1e-12).item()
                    * torch.norm(center).clamp_min(1e-12).item()
                )
                for update, center in zip(updates, ema_centers)
            ],
            device=updates[0].device,
        )

        self.latest_diagnostics = {
            "per_client_prev_centered_grad_distance_mean": prev_residual_norms.mean().item(),
            "per_client_prev_centered_to_raw_norm_ratio": (
                prev_residual_norms.mean() / raw_norm_mean
            ).item(),
            "per_client_prev_mean_cosine_with_center": prev_cosines.mean().item(),
            "per_client_prev_min_cosine_with_center": prev_cosines.min().item(),
            "per_client_ema_centered_grad_distance_mean": ema_residual_norms.mean().item(),
            "per_client_ema_centered_to_raw_norm_ratio": (
                ema_residual_norms.mean() / raw_norm_mean
            ).item(),
            "per_client_ema_mean_cosine_with_center": ema_cosines.mean().item(),
            "per_client_ema_min_cosine_with_center": ema_cosines.min().item(),
            "per_client_prev_center_norm_mean": torch.tensor(
                [torch.norm(center).item() for center in prev_centers],
                device=updates[0].device,
            ).mean().item(),
            "per_client_ema_center_norm_mean": torch.tensor(
                [torch.norm(center).item() for center in ema_centers],
                device=updates[0].device,
            ).mean().item(),
        }

        if self.ema_centers is None:
            self.ema_centers = [update.detach().clone() for update in updates]
        else:
            self.ema_centers = [
                self.center_momentum * center
                + (1 - self.center_momentum) * update.detach()
                for center, update in zip(self.ema_centers, updates)
            ]
        self.prev_updates = [update.detach().clone() for update in updates]
        return self.latest_diagnostics


def make_wandb_post_batch_hook():
    def hook(trainer, epoch, batch_idx):
        if wandb is None or wandb.run is None:
            return

        diagnostics = getattr(trainer.aggregator, "latest_diagnostics", None)
        history_tracker = getattr(trainer, "per_client_history_tracker", None)
        per_client_diagnostics = {}
        if history_tracker is not None:
            per_client_diagnostics = history_tracker.update(
                getattr(trainer, "latest_worker_gradients", None)
            )
        worker_diagnostics = [
            getattr(worker, "latest_diagnostics", None)
            for worker in trainer.workers
            if getattr(worker, "latest_diagnostics", None)
        ]

        if not diagnostics and not per_client_diagnostics and not worker_diagnostics:
            return

        payload = {
            "train/epoch": epoch,
            "train/batch_idx": batch_idx,
            "train/global_step": trainer.global_step,
        }
        if diagnostics:
            payload.update({f"train/{key}": value for key, value in diagnostics.items()})
            payload.update(
                {
                    key: value
                    for key, value in diagnostics.items()
                    if key.startswith("ardp/")
                }
            )
        if per_client_diagnostics:
            payload.update(
                {
                    "train/per_client_prev_centered_grad_distance_mean": per_client_diagnostics[
                        "per_client_prev_centered_grad_distance_mean"
                    ],
                    "train/per_client_prev_centered_to_raw_norm_ratio": per_client_diagnostics[
                        "per_client_prev_centered_to_raw_norm_ratio"
                    ],
                    "train/per_client_prev_mean_cosine_with_center": per_client_diagnostics[
                        "per_client_prev_mean_cosine_with_center"
                    ],
                    "train/per_client_prev_min_cosine_with_center": per_client_diagnostics[
                        "per_client_prev_min_cosine_with_center"
                    ],
                    "train/per_client_prev_center_norm_mean": per_client_diagnostics[
                        "per_client_prev_center_norm_mean"
                    ],
                    "train/per_client_ema_centered_grad_distance_mean": per_client_diagnostics[
                        "per_client_ema_centered_grad_distance_mean"
                    ],
                    "train/per_client_ema_centered_to_raw_norm_ratio": per_client_diagnostics[
                        "per_client_ema_centered_to_raw_norm_ratio"
                    ],
                    "train/per_client_ema_mean_cosine_with_center": per_client_diagnostics[
                        "per_client_ema_mean_cosine_with_center"
                    ],
                    "train/per_client_ema_min_cosine_with_center": per_client_diagnostics[
                        "per_client_ema_min_cosine_with_center"
                    ],
                    "train/per_client_ema_center_norm_mean": per_client_diagnostics[
                        "per_client_ema_center_norm_mean"
                    ],
                }
            )
        if worker_diagnostics:
            for key in worker_diagnostics[0]:
                values = torch.tensor(
                    [diagnostic[key] for diagnostic in worker_diagnostics],
                    dtype=torch.float32,
                )
                payload[f"train/dp_client_{key}_mean"] = values.mean().item()
                payload[f"train/dp_client_{key}_max"] = values.max().item()
                if key == "residual_to_raw_norm_ratio":
                    payload.setdefault(
                        "train/residual_to_raw_norm_ratio_mean",
                        values.mean().item(),
                    )
                if key == "stochastic_grad_norm":
                    payload.setdefault(
                        "train/stochastic_grad_norm_mean",
                        values.mean().item(),
                    )
                if key == "effective_update_norm":
                    payload.setdefault(
                        "train/effective_update_norm_mean",
                        values.mean().item(),
                    )
                if key == "raw_update_norm":
                    payload.setdefault(
                        "train/worker_raw_update_norm_mean",
                        values.mean().item(),
                    )
                if key == "residual_norm":
                    payload.setdefault(
                        "train/worker_residual_norm_mean",
                        values.mean().item(),
                    )
                if key == "clip_fraction":
                    payload.setdefault(
                        "train/residual_clipping_frequency",
                        values.mean().item(),
                    )
        wandb.log(payload, step=trainer.global_step)

    return hook


def log_wandb_validation(eval_log, epoch, global_step):
    if wandb is None or wandb.run is None:
        return

    wandb.log(
        {
            "validation/epoch": epoch,
            "validation/loss": eval_log["Loss"],
            "validation/top1": eval_log["top1"],
            "validation/accuracy": eval_log["top1"],
            "validation/global_step": global_step,
        },
        step=global_step,
    )


def main(args):
    initialize_logger(LOG_DIR)
    run = maybe_init_wandb()

    if args.use_cuda and not torch.cuda.is_available():
        print("=> There is no cuda device!!!!")
        device = "cpu"
    else:
        device = torch.device("cuda" if args.use_cuda else "cpu")

    kwargs = {"pin_memory": True} if args.use_cuda else {}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = get_resnet(model=args.model, use_cuda=args.use_cuda, gn=False).to(device)
    # NOTE: no momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_func = CrossEntropyLoss().to(device)

    metrics = {"top1": top1_accuracy}

    server_opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    server = TorchServer(server_opt)

    post_batch_hooks = [make_wandb_post_batch_hook()] if args.wandb else []
    aggregator = _get_aggregator()
    trainer = ParallelTrainer(
        server=server,
        aggregator=aggregator,
        pre_batch_hooks=[],
        post_batch_hooks=post_batch_hooks,
        max_batches_per_epoch=MAX_BATCHES_PER_EPOCH,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
    )
    trainer.per_client_history_tracker = (
        None
        if args.agg in ["dp-residual", "dp-residual-anchor"]
        else PerClientHistoryDiagnostics(
            center_momentum=args.per_client_center_momentum
        )
    )

    test_loader = cifar10(
        data_dir=DATA_DIR,
        train=False,
        download=True,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        **kwargs,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        server_opt, milestones=[75], gamma=0.1
    )

    evaluator = DistributedEvaluator(
        model=model,
        data_loader=test_loader,
        loss_func=loss_func,
        device=device,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
    )

    for worker_rank in range(N_WORKERS):
        worker = initialize_worker(
            worker_rank=worker_rank,
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            device=device,
            kwargs={},
        )
        trainer.add_worker(worker)

    rounds_per_epoch = min(MAX_BATCHES_PER_EPOCH, len(trainer.workers[0].data_loader))
    total_rounds = EPOCHS * rounds_per_epoch
    if args.target_epsilon is not None:
        if args.agg not in ["dp-fedavg", "dp-residual", "dp-residual-anchor"]:
            raise ValueError(
                "--target-epsilon is only supported for --agg dp-fedavg, "
                "dp-residual, and dp-residual-anchor."
            )
        calibration_rounds = total_rounds
        if args.agg == "dp-residual-anchor":
            calibration_rounds += count_anchor_rounds(
                total_rounds, args.anchor_period
            )
        computed_noise_multiplier = compute_full_participation_noise_multiplier(
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            total_rounds=calibration_rounds,
        )
        if args.agg == "dp-residual-anchor":
            trainer.aggregator.residual_noise_multiplier = computed_noise_multiplier
            trainer.aggregator.anchor_noise_multiplier = computed_noise_multiplier
        else:
            trainer.aggregator.noise_multiplier = computed_noise_multiplier
        effective_residual_noise_multiplier = computed_noise_multiplier
        effective_anchor_noise_multiplier = computed_noise_multiplier
    else:
        effective_residual_noise_multiplier = args.dp_noise_multiplier
        effective_anchor_noise_multiplier = ANCHOR_NOISE_MULTIPLIER
    accountant_mechanisms = get_dp_accountant_mechanisms(
        total_rounds=total_rounds,
        residual_sigma=effective_residual_noise_multiplier,
        anchor_sigma=effective_anchor_noise_multiplier,
    )
    effective_epsilon = compute_composed_full_participation_epsilon(
        mechanisms=accountant_mechanisms,
        target_delta=args.target_delta,
    )

    trainer.accountant_summary = {
        "rounds_per_epoch": rounds_per_epoch,
        "total_rounds": total_rounds,
        "target_epsilon": args.target_epsilon,
        "target_delta": args.target_delta,
        "effective_noise_multiplier": effective_residual_noise_multiplier,
        "effective_residual_noise_multiplier": effective_residual_noise_multiplier,
        "effective_anchor_noise_multiplier": effective_anchor_noise_multiplier,
        "effective_epsilon": effective_epsilon,
        "mechanisms": accountant_mechanisms,
    }
    mechanism_summary = ", ".join(
        (
            f"{mechanism['name']}:releases={mechanism['releases']},"
            f"sensitivity={mechanism['sensitivity']:.6g},"
            f"sigma={mechanism['noise_multiplier']:.6g},"
            f"noise_std={mechanism['noise_std']:.6g}"
        )
        for mechanism in accountant_mechanisms
    )
    print(
        "DP accountant:"
        f" rounds_per_epoch={rounds_per_epoch}, total_rounds={total_rounds},"
        f" mechanisms=[{mechanism_summary}],"
        f" epsilon={effective_epsilon:.6f}, target_delta={args.target_delta}"
    )
    privacy_payload = privacy_metrics_from_mechanisms(
        mechanisms=accountant_mechanisms,
        epsilon=effective_epsilon,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        c_res=args.clip_tau,
        c_anchor=ANCHOR_CLIP_TAU if args.agg == "dp-residual-anchor" else None,
        sigma_res=effective_residual_noise_multiplier,
        sigma_anchor=effective_anchor_noise_multiplier,
    )
    if args.wandb and wandb.run is not None:
        wandb.config.update(
            {
                "rounds_per_epoch": rounds_per_epoch,
                "total_rounds": total_rounds,
                "effective_noise_multiplier": effective_residual_noise_multiplier,
                "effective_residual_noise_multiplier": effective_residual_noise_multiplier,
                "effective_anchor_noise_multiplier": effective_anchor_noise_multiplier,
                "dp_noise_multiplier": effective_residual_noise_multiplier,
                "dp_anchor_noise_multiplier": effective_anchor_noise_multiplier,
                "effective_epsilon": effective_epsilon,
                "dp_accountant_mechanisms": accountant_mechanisms,
                **privacy_payload,
            },
            allow_val_change=True,
        )
        wandb.log(privacy_payload, step=0)

    initial_eval_log = evaluator.evaluate(0)
    if args.wandb:
        log_wandb_validation(initial_eval_log, epoch=0, global_step=trainer.global_step)

    for epoch in range(1, EPOCHS + 1):
        trainer.train(epoch)
        eval_log = evaluator.evaluate(epoch)
        trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))
        scheduler.step()
        if args.wandb:
            log_wandb_validation(eval_log, epoch=epoch, global_step=trainer.global_step)
            wandb.log({"train/lr": scheduler.get_last_lr()[0]}, step=trainer.global_step)
        print(f"E={epoch}; Learning rate = {scheduler.get_lr()[0]:}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    if args.privacy_dry_run:
        run_privacy_dry_run()
    else:
        main(args)
