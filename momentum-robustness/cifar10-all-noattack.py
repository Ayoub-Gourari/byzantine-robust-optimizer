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
from codes.simulators.worker import WorkerWithMomentum
from codes.simulators.server import TorchServer

from codes.tasks.cifar10 import cifar10, get_resnet20
from codes.utils import top1_accuracy, initialize_logger
from codes.aggregator.clipping import Clipping
from codes.aggregator.base import Mean

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

parser.add_argument("--attack", type=str, default="NA", help="Select from BF and LF.")
parser.add_argument("--agg", type=str, default="cp", help="Aggregator.")
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
EPOCHS = 100
LR = 0.1
MOMENTUM = args.momentum

# LOG_DIR = EXP_DIR + "log"
LOG_DIR = (
    EXP_DIR
    + ("debug/" if args.debug else "")
    + (
        f"{args.attack}_{args.agg}_tau{args.clip_tau}_m{args.momentum}"
        f"_mom{args.momentum_mode}"
        f"_center{args.center_update}-{args.center_source}-beta{args.center_momentum}"
        f"-scale{args.center_scale}"
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
    return WorkerWithMomentum(
        momentum=MOMENTUM,
        momentum_mode=args.momentum_mode,
        local_steps=args.local_steps,
        data_loader=train_loader,
        model=model,
        loss_func=loss_func,
        device=device,
        optimizer=optimizer,
        **kwargs,
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
            f"-center{args.center_update}-{args.center_source}-beta{args.center_momentum}"
            f"-scale{args.center_scale}"
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
            "clip_tau": args.clip_tau,
            "inner_iterations": args.inner_iterations,
            "momentum": args.momentum,
            "momentum_mode": args.momentum_mode,
            "local_steps": args.local_steps,
            "center_update": args.center_update,
            "center_momentum": args.center_momentum,
            "center_source": args.center_source,
            "center_scale": args.center_scale,
            "seed": args.seed,
            "n_workers": N_WORKERS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
        },
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("validation/*", step_metric="train/global_step")
    return run


def make_wandb_post_batch_hook():
    def hook(trainer, epoch, batch_idx):
        if wandb is None or wandb.run is None:
            return

        diagnostics = getattr(trainer.aggregator, "latest_diagnostics", None)
        if not diagnostics:
            return

        payload = {
            "train/epoch": epoch,
            "train/batch_idx": batch_idx,
            "train/global_step": trainer.global_step,
            "train/raw_grad_norm_mean": diagnostics["raw_grad_norm_mean"],
            "train/centered_grad_distance_mean": diagnostics[
                "centered_grad_distance_mean"
            ],
            "train/raw_grad_norm_max": diagnostics["raw_grad_norm_max"],
            "train/centered_grad_distance_max": diagnostics[
                "centered_grad_distance_max"
            ],
            "train/centered_to_raw_norm_ratio": diagnostics[
                "centered_to_raw_norm_ratio"
            ],
            "train/center_norm": diagnostics["center_norm"],
            "train/scaled_center_norm": diagnostics["scaled_center_norm"],
            "train/next_center_norm": diagnostics["next_center_norm"],
            "train/mean_update_norm": diagnostics["mean_update_norm"],
            "train/aggregate_norm": diagnostics["aggregate_norm"],
            "train/mean_cosine_with_center": diagnostics["mean_cosine_with_center"],
            "train/min_cosine_with_center": diagnostics["min_cosine_with_center"],
            "train/center_to_mean_update_norm_ratio": diagnostics[
                "center_to_mean_update_norm_ratio"
            ],
            "train/clip_fraction_mean": diagnostics["clip_fraction_mean"],
        }
        wandb.log(payload, step=trainer.global_step)

    return hook


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

    model = get_resnet20(use_cuda=args.use_cuda, gn=False).to(device)
    # NOTE: no momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_func = CrossEntropyLoss().to(device)

    metrics = {"top1": top1_accuracy}

    server_opt = torch.optim.SGD(model.parameters(), lr=LR)
    server = TorchServer(server_opt)

    post_batch_hooks = [make_wandb_post_batch_hook()] if args.wandb else []
    trainer = ParallelTrainer(
        server=server,
        aggregator=_get_aggregator(),
        pre_batch_hooks=[],
        post_batch_hooks=post_batch_hooks,
        max_batches_per_epoch=MAX_BATCHES_PER_EPOCH,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
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
        server_opt, milestones=[75], gamma=LR
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

    for epoch in range(1, EPOCHS + 1):
        trainer.train(epoch)
        eval_log = evaluator.evaluate(epoch)
        trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))
        scheduler.step()
        if args.wandb:
            wandb.log(
                {
                    "validation/epoch": epoch,
                    "validation/loss": eval_log["Loss"],
                    "validation/top1": eval_log["top1"],
                    "validation/global_step": trainer.global_step,
                    "train/lr": scheduler.get_last_lr()[0],
                },
                step=trainer.global_step,
            )
        print(f"E={epoch}; Learning rate = {scheduler.get_lr()[0]:}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main(args)
