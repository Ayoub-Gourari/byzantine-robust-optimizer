"""Centralized CIFAR10/ResNet20 residual-clipping diagnostic runner."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from codes.tasks.cifar10 import cifar10, get_resnet
from codes.utils import top1_accuracy

try:
    import wandb
except ImportError:  # pragma: no cover - handled when W&B is requested.
    wandb = None


OPTIMIZER_MODES = (
    "sgd_momentum",
    "clipped_momentum",
    "residual_clipped_momentum",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Centralized ResNet/CIFAR10 comparison for unclipped, clipped, "
            "and residual-clipped EMA momentum SGD."
        )
    )
    parser.add_argument("--optimizer-mode", choices=OPTIMIZER_MODES, required=True)
    parser.add_argument("--model", type=str, default="resnet20")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--clip-c", type=float, default=None)
    parser.add_argument("--clip-c-res", type=float, default=None)
    parser.add_argument("--lr-milestones", type=str, default="75")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "datasets"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "outputs" / "central_residual_clipping"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-download", action="store_false", dest="download", default=True)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="non_private_residual_clipping")
    parser.add_argument("--wandb-entity", type=str, default="ae-gourari-cole-polytechnique")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def parse_milestones(value):
    if value is None or value.strip() == "":
        return []
    return sorted(int(item.strip()) for item in value.split(",") if item.strip())


def current_lr(base_lr, epoch, milestones, gamma):
    drops = sum((epoch - 1) >= milestone for milestone in milestones)
    return base_lr * (gamma ** drops)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_list_global_norm(values):
    if not values:
        return torch.tensor(0.0)
    total = torch.zeros((), device=values[0].device)
    for value in values:
        total = total + value.detach().pow(2).sum()
    return total.sqrt()


def tensor_list_dot(left, right):
    if not left:
        return torch.tensor(0.0)
    total = torch.zeros((), device=left[0].device)
    for lhs, rhs in zip(left, right):
        total = total + (lhs.detach() * rhs.detach()).sum()
    return total


def tensor_list_sub(left, right):
    return [lhs - rhs for lhs, rhs in zip(left, right)]


def tensor_list_add(left, right):
    return [lhs + rhs for lhs, rhs in zip(left, right)]


def clip_by_global_norm(values, threshold):
    norm = tensor_list_global_norm(values)
    if threshold is None:
        return [value.detach().clone() for value in values], norm.item(), 1.0
    scale = min(1.0, float(threshold) / (norm.item() + 1e-12))
    return [value * scale for value in values], norm.item(), scale


def collect_gradients(params, weight_decay):
    data_grads = []
    update_grads = []
    for param in params:
        if param.grad is None:
            grad = torch.zeros_like(param)
        else:
            grad = param.grad.detach().clone()
        data_grads.append(grad.detach().clone())
        if weight_decay != 0:
            grad = grad.add(param.detach(), alpha=weight_decay)
        update_grads.append(grad.detach().clone())
    return update_grads, data_grads


def clipping_error_norm(center, update, target):
    error = tensor_list_sub(tensor_list_add(center, update), target)
    return tensor_list_global_norm(error).item()


class MomentumClipper:
    def __init__(self, params, mode, beta, clip_c, clip_c_res):
        if mode == "clipped_momentum" and clip_c is None:
            raise ValueError("--clip-c is required for clipped_momentum.")
        if mode == "residual_clipped_momentum" and clip_c_res is None:
            raise ValueError("--clip-c-res is required for residual_clipped_momentum.")
        self.params = list(params)
        self.mode = mode
        self.beta = beta
        self.clip_c = clip_c
        self.clip_c_res = clip_c_res
        self.momentum = [torch.zeros_like(param) for param in self.params]

    def step(self, grads, lr):
        momentum_before = [value.detach().clone() for value in self.momentum]
        grad_norm = tensor_list_global_norm(grads).item()
        data = self._base_diagnostics(grads, momentum_before, grad_norm)

        comparison_threshold = self._comparison_threshold()
        if comparison_threshold is not None:
            standard_at_threshold, _, standard_scale = clip_by_global_norm(
                grads, comparison_threshold
            )
            residuals = tensor_list_sub(grads, momentum_before)
            residual_at_threshold, _, residual_scale = clip_by_global_norm(
                residuals, comparison_threshold
            )
            standard_error = tensor_list_global_norm(
                tensor_list_sub(standard_at_threshold, grads)
            ).item()
            residual_error = clipping_error_norm(
                momentum_before, residual_at_threshold, grads
            )
            data.update(
                {
                    "comparison_clip_threshold": comparison_threshold,
                    "comparison_standard_clip_fraction": standard_scale,
                    "comparison_residual_clip_fraction": residual_scale,
                    "comparison_standard_clipping_error_norm": standard_error,
                    "comparison_residual_clipping_error_norm": residual_error,
                    "comparison_residual_error_less_than_standard": float(
                        residual_error < standard_error
                    ),
                }
            )

        if self.mode == "sgd_momentum":
            self.momentum = [
                self.beta * momentum + (1 - self.beta) * grad
                for momentum, grad in zip(momentum_before, grads)
            ]
        elif self.mode == "clipped_momentum":
            clipped_grad, _, scale = clip_by_global_norm(grads, self.clip_c)
            data.update(
                {
                    "standard_clip_fraction": scale,
                    "standard_clipping_error_norm": tensor_list_global_norm(
                        tensor_list_sub(clipped_grad, grads)
                    ).item(),
                }
            )
            self.momentum = [
                self.beta * momentum + (1 - self.beta) * grad
                for momentum, grad in zip(momentum_before, clipped_grad)
            ]
        elif self.mode == "residual_clipped_momentum":
            residuals = tensor_list_sub(grads, momentum_before)
            clipped_residuals, _, scale = clip_by_global_norm(
                residuals, self.clip_c_res
            )
            data.update(
                {
                    "residual_clip_fraction": scale,
                    "residual_clipping_error_norm": clipping_error_norm(
                        momentum_before, clipped_residuals, grads
                    ),
                }
            )
            self.momentum = [
                momentum + (1 - self.beta) * residual
                for momentum, residual in zip(momentum_before, clipped_residuals)
            ]
        else:  # pragma: no cover - argparse prevents this.
            raise ValueError(f"Unknown optimizer_mode={self.mode}")

        with torch.no_grad():
            for param, momentum in zip(self.params, self.momentum):
                param.add_(momentum, alpha=-lr)

        data["next_momentum_norm"] = tensor_list_global_norm(self.momentum).item()
        return data

    def _base_diagnostics(self, grads, momentum_before, grad_norm):
        momentum_norm = tensor_list_global_norm(momentum_before).item()
        residuals = tensor_list_sub(grads, momentum_before)
        residual_norm = tensor_list_global_norm(residuals).item()
        cosine = tensor_list_dot(grads, momentum_before).item() / (
            grad_norm * momentum_norm + 1e-12
        )
        return {
            "grad_norm": grad_norm,
            "momentum_norm": momentum_norm,
            "residual_norm": residual_norm,
            "residual_to_grad_norm_ratio": residual_norm / (grad_norm + 1e-12),
            "cos_grad_momentum": cosine,
        }

    def _comparison_threshold(self):
        if self.mode == "clipped_momentum":
            return self.clip_c
        if self.mode == "residual_clipped_momentum":
            return self.clip_c_res
        return self.clip_c if self.clip_c is not None else self.clip_c_res


class RunningDiagnostics:
    def __init__(self):
        self.values = {}

    def update(self, diagnostics):
        for key, value in diagnostics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                self.values.setdefault(key, []).append(float(value))

    def summarize(self):
        summary = {}
        for key, values in self.values.items():
            array = np.asarray(values, dtype=np.float64)
            summary[f"{key}_mean"] = float(array.mean())
            summary[f"{key}_median"] = float(np.median(array))
        return summary


def maybe_init_wandb(args, run_name):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Install it before using --wandb.")

    entity = args.wandb_entity if args.wandb_entity else None
    run = wandb.init(
        project=args.wandb_project,
        entity=entity,
        name=args.wandb_run_name or run_name,
        config={**vars(args), "num_training_workers": 1},
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("validation/*", step_metric="train/global_step")
    return run


def write_jsonl(handle, payload):
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()


def log_payload(args, payload, metrics_file):
    write_jsonl(metrics_file, payload)
    if args.wandb and wandb is not None and wandb.run is not None:
        step = payload.get("train/global_step", payload.get("validation/global_step"))
        wandb.log(payload, step=step)


def train_one_epoch(
    args,
    model,
    train_loader,
    loss_func,
    device,
    optimizer,
    epoch,
    lr,
    global_step,
    metrics_file,
    tracker,
):
    model.train()
    params = optimizer.params
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break

        data = data.to(device, non_blocking=args.use_cuda)
        target = target.to(device, non_blocking=args.use_cuda)
        model.zero_grad(set_to_none=True)
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()

        grads, data_grads = collect_gradients(params, args.weight_decay)
        diagnostics = optimizer.step(grads, lr)
        diagnostics["data_grad_norm"] = tensor_list_global_norm(data_grads).item()
        tracker.update(diagnostics)

        global_step += 1
        if batch_idx % args.log_interval == 0:
            payload = {
                "train/global_step": global_step,
                "train/epoch": epoch,
                "train/batch_idx": batch_idx,
                "train/loss": loss.item(),
                "train/accuracy": top1_accuracy(output.detach(), target),
                "train/lr": lr,
            }
            payload.update({f"train/{key}": value for key, value in diagnostics.items()})
            log_payload(args, payload, metrics_file)
            print(
                " ".join(
                    [
                        f"E={epoch}",
                        f"B={batch_idx}",
                        f"step={global_step}",
                        f"loss={loss.item():.4f}",
                        f"acc={payload['train/accuracy']:.2f}",
                        f"grad_norm={diagnostics['grad_norm']:.4f}",
                        f"residual_ratio={diagnostics['residual_to_grad_norm_ratio']:.4f}",
                    ]
                ),
                flush=True,
            )

    return global_step


@torch.no_grad()
def evaluate(args, model, data_loader, loss_func, device):
    model.eval()
    total_loss = 0.0
    total_correct_pct_weighted = 0.0
    total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.max_test_batches is not None and batch_idx >= args.max_test_batches:
            break
        data = data.to(device, non_blocking=args.use_cuda)
        target = target.to(device, non_blocking=args.use_cuda)
        output = model(data)
        loss = loss_func(output, target)
        length = target.size(0)
        total_loss += loss.item() * length
        total_correct_pct_weighted += top1_accuracy(output, target) * length
        total += length
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct_pct_weighted / max(total, 1),
        "length": total,
    }


def validate_args(args):
    if not 0 <= args.beta < 1:
        raise ValueError(f"--beta must be in [0, 1). Got {args.beta}.")
    if args.optimizer_mode == "clipped_momentum" and args.clip_c is None:
        raise ValueError("--clip-c is required for clipped_momentum.")
    if args.optimizer_mode == "residual_clipped_momentum" and args.clip_c_res is None:
        raise ValueError("--clip-c-res is required for residual_clipped_momentum.")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be >= 1.")


def main():
    args = parse_args()
    validate_args(args)
    set_seeds(args.seed)

    if args.use_cuda and not torch.cuda.is_available():
        print("Requested CUDA, but no CUDA device is available; using CPU.")
        args.use_cuda = False
    device = torch.device("cuda" if args.use_cuda else "cpu")
    pin_memory = bool(args.use_cuda)

    run_name = args.run_name or (args.wandb_run_name or f"{args.optimizer_mode}-seed{args.seed}")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"

    run = maybe_init_wandb(args, run_name)
    milestones = parse_milestones(args.lr_milestones)

    train_loader = cifar10(
        data_dir=args.data_dir,
        train=True,
        download=args.download,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.loader_workers,
        pin_memory=pin_memory,
    )
    test_loader = cifar10(
        data_dir=args.data_dir,
        train=False,
        download=args.download,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.loader_workers,
        pin_memory=pin_memory,
    )

    model = get_resnet(model=args.model, use_cuda=args.use_cuda, gn=False).to(device)
    loss_func = CrossEntropyLoss().to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = MomentumClipper(
        params=params,
        mode=args.optimizer_mode,
        beta=args.beta,
        clip_c=args.clip_c,
        clip_c_res=args.clip_c_res,
    )
    tracker = RunningDiagnostics()
    global_step = 0
    best_accuracy = 0.0
    best_epoch = 0

    with metrics_path.open("w") as metrics_file:
        initial_eval = evaluate(args, model, test_loader, loss_func, device)
        payload = {
            "train/global_step": global_step,
            "validation/global_step": global_step,
            "validation/epoch": 0,
            "validation/loss": initial_eval["loss"],
            "validation/accuracy": initial_eval["accuracy"],
            "validation/top1": initial_eval["accuracy"],
            "validation/length": initial_eval["length"],
        }
        log_payload(args, payload, metrics_file)

        for epoch in range(1, args.epochs + 1):
            lr = current_lr(args.lr, epoch, milestones, args.lr_gamma)
            global_step = train_one_epoch(
                args=args,
                model=model,
                train_loader=train_loader,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                epoch=epoch,
                lr=lr,
                global_step=global_step,
                metrics_file=metrics_file,
                tracker=tracker,
            )
            eval_log = evaluate(args, model, test_loader, loss_func, device)
            if eval_log["accuracy"] > best_accuracy:
                best_accuracy = eval_log["accuracy"]
                best_epoch = epoch
            payload = {
                "train/global_step": global_step,
                "validation/global_step": global_step,
                "validation/epoch": epoch,
                "validation/loss": eval_log["loss"],
                "validation/accuracy": eval_log["accuracy"],
                "validation/top1": eval_log["accuracy"],
                "validation/best_accuracy": best_accuracy,
                "validation/best_epoch": best_epoch,
                "validation/length": eval_log["length"],
                "train/lr": lr,
            }
            log_payload(args, payload, metrics_file)
            print(
                f"E={epoch} validation_acc={eval_log['accuracy']:.2f} "
                f"best={best_accuracy:.2f}@{best_epoch} lr={lr}",
                flush=True,
            )

    summary = {
        "run_name": run_name,
        "optimizer_mode": args.optimizer_mode,
        "best_validation_accuracy": best_accuracy,
        "best_validation_epoch": best_epoch,
        "global_step": global_step,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "clip_c": args.clip_c,
        "clip_c_res": args.clip_c_res,
        "weight_decay": args.weight_decay,
        "diagnostics": tracker.summarize(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.wandb and wandb is not None and wandb.run is not None:
        wandb.run.summary.update(summary)
    if run is not None:
        run.finish()

    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
