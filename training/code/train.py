#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

from dataset import PairedYuvDataset
from iqa_metrics_exact_refalgo_y import evaluate_selective_prefilter_y
from loss import CharbonnierLoss, EdgeConsistencyLoss, MsssimLoss, softclip01
from model import PrefilterNet, load_prefilter_state
from utils import calculate_psnr, calculate_ssim, ensure_dir, read_csv_rows, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FP32 PrefilterNet.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None, help="'auto' or a checkpoint path. Overrides config.train.resume.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def resolve_resume_path(run_dir: Path, resume_cfg: str | None) -> Path | None:
    if not resume_cfg or str(resume_cfg).lower() in {"none", "null", ""}:
        return None
    if resume_cfg == "auto":
        candidate = run_dir / "checkpoints" / "latest.pt"
        return candidate if candidate.is_file() else None
    path = Path(resume_cfg)
    return path if path.is_file() else None


def build_optimizer(model: PrefilterNet, train_cfg: dict) -> Adam:
    return Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )


def build_scheduler(optimizer: Adam, train_cfg: dict, total_iters: int) -> CosineAnnealingLR:
    scheduler_cfg = train_cfg.get("scheduler", {})
    t_max = int(scheduler_cfg.get("T_max", total_iters))
    eta_min = float(scheduler_cfg.get("eta_min", train_cfg.get("min_lr", 1e-6)))
    return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


def update_learning_rate(
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    current_iter: int,
    warmup_iter: int,
) -> None:
    if current_iter > 1:
        scheduler.step()

    if warmup_iter > 0 and current_iter < warmup_iter:
        for group in optimizer.param_groups:
            initial_lr = group.get("initial_lr", group["lr"])
            group["lr"] = initial_lr / warmup_iter * current_iter


def build_train_loss(
    train_cfg: dict,
) -> tuple[CharbonnierLoss | None, MsssimLoss | None, EdgeConsistencyLoss | None]:
    fidelity_cfg = train_cfg.get("fidelity_loss", {})
    perceptual_cfg = train_cfg.get("perceptual_loss", {})
    edge_aux_cfg = train_cfg.get("edge_aux_loss", {})

    fidelity_loss = None
    if fidelity_cfg.get("enabled", True):
        fidelity_loss = CharbonnierLoss(
            loss_weight=float(fidelity_cfg.get("loss_weight", 1.0)),
            eps=float(fidelity_cfg.get("eps", 1e-12)),
            roi_weight=float(fidelity_cfg.get("roi_weight", 20.0)),
            non_roi_weight=float(fidelity_cfg.get("non_roi_weight", 1.0)),
        )

    perceptual_loss = None
    if perceptual_cfg.get("enabled", True):
        perceptual_loss = MsssimLoss(
            loss_weight=float(perceptual_cfg.get("loss_weight", 0.16)),
            data_range=float(perceptual_cfg.get("data_range", 1.0)),
            win_size=int(perceptual_cfg.get("win_size", 11)),
            win_sigma=float(perceptual_cfg.get("win_sigma", 1.5)),
            weights=perceptual_cfg.get("weights"),
        )

    edge_aux_loss = None
    if edge_aux_cfg.get("enabled", False):
        edge_aux_loss = EdgeConsistencyLoss(
            loss_weight=float(edge_aux_cfg.get("loss_weight", 0.05)),
            match_weight=float(edge_aux_cfg.get("match_weight", 1.0)),
            retain_weight=float(edge_aux_cfg.get("retain_weight", 0.25)),
            retain_ratio=float(edge_aux_cfg.get("retain_ratio", 0.90)),
            mask_quantile=float(edge_aux_cfg.get("mask_quantile", 0.90)),
            mask_gamma=float(edge_aux_cfg.get("mask_gamma", 1.5)),
            eps=float(edge_aux_cfg.get("eps", 1e-6)),
        )

    return fidelity_loss, perceptual_loss, edge_aux_loss


def build_validation_cfg(config: dict[str, Any]) -> dict[str, Any]:
    validation_cfg = dict(config.get("validation", {}) or {})
    selective_cfg = dict(validation_cfg.get("selective_metric", {}) or {})
    selective_cfg.setdefault("enabled", True)
    selective_cfg.setdefault("mask_mode", "detail_gain")
    selective_cfg.setdefault("s_thr", 0.25)
    validation_cfg["selective_metric"] = selective_cfg
    validation_cfg.setdefault("primary_metric", "selective_score")
    validation_cfg.setdefault("primary_higher_is_better", True)
    return validation_cfg


def compute_train_loss(
    inputs: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    roi: torch.Tensor | None,
    only_train_y: bool,
    fidelity_loss: CharbonnierLoss | None,
    perceptual_loss: MsssimLoss | None,
    edge_aux_loss: EdgeConsistencyLoss | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if only_train_y:
        pred = preds[:, :1]
        gt = targets[:, :1]
        source = inputs[:, :1]
    else:
        pred = preds
        gt = targets
        source = inputs

    total_loss = pred.new_tensor(0.0)
    logs: dict[str, float] = {}
    if fidelity_loss is not None:
        l_fidelity = fidelity_loss(pred, gt, roi=roi)
        total_loss = total_loss + l_fidelity
        logs["l_fidelity"] = float(l_fidelity.detach().item())
    if perceptual_loss is not None:
        l_perceptual = perceptual_loss(softclip01(pred), gt)
        total_loss = total_loss + l_perceptual
        logs["l_perceptual"] = float(l_perceptual.detach().item())
    if edge_aux_loss is not None:
        l_edge_aux = edge_aux_loss(pred, source=inputs, target=targets)
        total_loss = total_loss + l_edge_aux
        logs["l_edge_aux"] = float(l_edge_aux.detach().item())
    logs["loss"] = float(total_loss.detach().item())
    return total_loss, logs


@torch.no_grad()
def validate(
    model: PrefilterNet,
    loader: DataLoader,
    fidelity_loss: CharbonnierLoss | None,
    perceptual_loss: MsssimLoss | None,
    edge_aux_loss: EdgeConsistencyLoss | None,
    device: torch.device,
    only_train_y: bool,
    validation_cfg: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    selective_sums: dict[str, float] = {}
    count = 0
    selective_cfg = validation_cfg.get("selective_metric", {})
    use_selective_metric = bool(selective_cfg.get("enabled", True))
    mask_mode = str(selective_cfg.get("mask_mode", "detail_gain"))
    s_thr = float(selective_cfg.get("s_thr", 0.25))

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        roi = batch.get("roi")
        if roi is not None:
            roi = roi.to(device)

        preds = model(inputs)
        loss, _ = compute_train_loss(
            inputs,
            preds,
            targets,
            roi,
            only_train_y,
            fidelity_loss,
            perceptual_loss,
            edge_aux_loss,
        )
        loss_sum += float(loss.item())

        for pred, target in zip(preds, targets):
            psnr_sum += calculate_psnr(pred[:1], target[:1])
            ssim_sum += calculate_ssim(pred[:1], target[:1])
            count += 1
        if use_selective_metric:
            for pred, target, src in zip(preds, targets, inputs):
                if only_train_y:
                    pred_eval = pred[:1]
                    target_eval = target[:1]
                    src_eval = src[:1]
                else:
                    pred_eval = pred
                    target_eval = target
                    src_eval = src
                selective_metrics = evaluate_selective_prefilter_y(
                    pred_eval.detach().cpu().numpy(),
                    target_eval.detach().cpu().numpy(),
                    src_eval.detach().cpu().numpy(),
                    mask_mode=mask_mode,
                    s_thr=s_thr,
                )
                for key, value in selective_metrics.items():
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        selective_sums[key] = selective_sums.get(key, 0.0) + float(value)

    model.train()
    if count == 0:
        return {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    metrics = {
        "loss": loss_sum / max(len(loader), 1),
        "psnr": psnr_sum / count,
        "ssim": ssim_sum / count,
    }
    for key, total in selective_sums.items():
        metrics[key] = total / count
    return metrics


def is_better_metric(candidate: float, best: float, higher_is_better: bool) -> bool:
    return candidate >= best if higher_is_better else candidate <= best


def initial_best_value(higher_is_better: bool) -> float:
    return float("-inf") if higher_is_better else float("inf")


def format_validation_log(metrics: dict[str, float]) -> str:
    text = (
        f"loss={metrics['loss']:.6f} "
        f"psnr={metrics['psnr']:.4f} "
        f"ssim={metrics['ssim']:.4f}"
    )
    if "selective_score" in metrics:
        text += (
            f" selective={metrics['selective_score']:.4f}"
            f" bgc={metrics.get('bg_completion', 0.0):.4f}"
            f" edgec={metrics.get('edge_source_completion', 0.0):.4f}"
            f" eret={metrics.get('edge_retention_ratio', 0.0):.4f}"
            f" eover={metrics.get('edge_oversmooth_vs_src', 0.0):.4f}"
        )
    return text


def maybe_load_pretrain(model: PrefilterNet, model_cfg: dict, device: torch.device) -> None:
    pretrain_path = model_cfg.get("pretrain_path") or model_cfg.get("pretrain_network_g")
    if not pretrain_path:
        return

    path = Path(pretrain_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pretrain checkpoint not found: {path}")

    strict = bool(model_cfg.get("strict_load", model_cfg.get("strict_load_g", True)))
    state = torch.load(path, map_location=device)
    missing, unexpected = load_prefilter_state(model, state, strict=strict)
    if missing or unexpected:
        print(f"[WARN] Pretrain load mismatch: missing={missing}, unexpected={unexpected}")
    print(f"[INFO] Loaded pretrain weights from {path}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    logger_cfg = config.get("logger", {})
    validation_cfg = build_validation_cfg(config)

    set_random_seed(int(train_cfg.get("seed", 123)))
    torch.backends.cudnn.benchmark = True

    device = select_device(train_cfg.get("device", "cuda"))
    data_root = Path(data_cfg["root"]).resolve()
    run_dir = ensure_dir(Path(config["output_root"]) / config["experiment_name"])
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    train_ds = PairedYuvDataset(
        data_root / "train",
        crop_size=train_cfg.get("crop_size"),
        training=True,
        hflip=train_cfg.get("hflip", True),
    )
    val_root = data_root / "val"
    val_manifest = val_root / "manifest.csv"
    has_val = val_manifest.is_file() and len(read_csv_rows(val_manifest)) > 0
    val_ds = PairedYuvDataset(val_root, crop_size=None, training=False, hflip=False) if has_val else None

    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(train_cfg.get("val_batch_size", 1)),
            shuffle=False,
            num_workers=max(0, int(train_cfg["num_workers"]) // 2),
        )

    num_iter_per_epoch = math.ceil(len(train_ds) / batch_size)
    total_iters = int(train_cfg.get("total_iter", 0))
    total_epochs = train_cfg.get("epochs")
    if total_iters <= 0:
        if total_epochs is None:
            raise ValueError("train.total_iter or train.epochs must be set")
        total_iters = int(total_epochs) * num_iter_per_epoch
    if total_epochs is None:
        total_epochs = math.ceil(total_iters / num_iter_per_epoch)
    total_epochs = int(total_epochs)

    model = PrefilterNet(**{k: v for k, v in model_cfg.items() if k not in {"pretrain_path", "pretrain_network_g", "strict_load", "strict_load_g"}}).to(device)
    fidelity_loss, perceptual_loss, edge_aux_loss = build_train_loss(train_cfg)
    if fidelity_loss is not None:
        fidelity_loss = fidelity_loss.to(device)
    if perceptual_loss is not None:
        perceptual_loss = perceptual_loss.to(device)
    if edge_aux_loss is not None:
        edge_aux_loss = edge_aux_loss.to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, total_iters)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(train_cfg.get("mixed_precision", False) and device.type == "cuda"),
    )

    resume_cfg = args.resume if args.resume is not None else train_cfg.get("resume", None)
    resume_path = resolve_resume_path(run_dir, resume_cfg)
    start_epoch = 0
    current_iter = 0
    primary_metric_name = str(validation_cfg.get("primary_metric", "selective_score"))
    primary_higher_is_better = bool(validation_cfg.get("primary_higher_is_better", True))
    best_metric_value = initial_best_value(primary_higher_is_better)
    best_psnr = float("-inf")
    only_train_y = bool(model_cfg.get("only_train_y", True))

    if resume_path is not None:
        state = torch.load(resume_path, map_location=device)
        missing, unexpected = load_prefilter_state(model, state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Resume checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))
        current_iter = int(state.get("current_iter", state.get("global_step", 0)))
        primary_metric_name = str(state.get("best_metric_name", primary_metric_name))
        best_metric_value = float(state.get("best_metric_value", state.get("best_psnr", best_metric_value)))
        best_psnr = float(state.get("best_psnr", best_psnr))
        print(f"[INFO] Resumed from {resume_path} @ epoch={start_epoch}, iter={current_iter}")
    else:
        maybe_load_pretrain(model, model_cfg, device)

    config_snapshot = run_dir / "train_config.yaml"
    with config_snapshot.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    metrics_log = run_dir / "metrics.jsonl"
    print(
        "[INFO] Training statistics: "
        f"num_train={len(train_ds)} batch_size={batch_size} "
        f"iters_per_epoch={num_iter_per_epoch} total_epochs={total_epochs} total_iters={total_iters}"
    )

    print_freq = int(logger_cfg.get("print_freq", train_cfg.get("print_freq", 1000)))
    save_latest_freq = int(logger_cfg.get("save_latest_freq", train_cfg.get("save_latest_freq", 500)))
    save_checkpoint_freq = int(logger_cfg.get("save_checkpoint_freq", train_cfg.get("save_checkpoint_freq", total_iters)))
    val_freq = int(logger_cfg.get("val_freq", train_cfg.get("val_freq", 1000)))
    warmup_iter = int(train_cfg.get("warmup_iter", 3000))

    epoch = start_epoch
    while current_iter < total_iters:
        epoch += 1
        model.train()
        for batch in train_loader:
            if current_iter >= total_iters:
                break

            current_iter += 1
            update_learning_rate(optimizer, scheduler, current_iter, warmup_iter)

            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            roi = batch.get("roi")
            if roi is not None:
                roi = roi.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                preds = model(inputs)
                loss, loss_logs = compute_train_loss(
                    inputs,
                    preds,
                    targets,
                    roi,
                    only_train_y,
                    fidelity_loss,
                    perceptual_loss,
                    edge_aux_loss,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if current_iter % print_freq == 0:
                log_msg = (
                    f"[ITER {current_iter:06d}] "
                    f"epoch={epoch:04d} "
                    f"loss={loss_logs['loss']:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6e}"
                )
                if "l_fidelity" in loss_logs:
                    log_msg += f" l_fidelity={loss_logs['l_fidelity']:.6f}"
                if "l_perceptual" in loss_logs:
                    log_msg += f" l_perceptual={loss_logs['l_perceptual']:.6f}"
                if "l_edge_aux" in loss_logs:
                    log_msg += f" l_edge_aux={loss_logs['l_edge_aux']:.6f}"
                print(log_msg)

            checkpoint = {
                "epoch": epoch,
                "current_iter": current_iter,
                "best_metric_name": primary_metric_name,
                "best_metric_value": best_metric_value,
                "best_psnr": best_psnr,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
            }

            if current_iter % save_checkpoint_freq == 0:
                save_checkpoint(ckpt_dir / f"iter_{current_iter:06d}.pt", checkpoint)
                print(f"[INFO] Saved checkpoint @ iter {current_iter}")

            if current_iter % save_latest_freq == 0:
                save_checkpoint(ckpt_dir / "latest.pt", checkpoint)

            if val_loader is not None and current_iter % val_freq == 0:
                val_metrics = validate(
                    model,
                    val_loader,
                    fidelity_loss,
                    perceptual_loss,
                    edge_aux_loss,
                    device,
                    only_train_y,
                    validation_cfg,
                )
                payload = {
                    "epoch": epoch,
                    "current_iter": current_iter,
                    "train_loss": loss_logs["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_psnr": val_metrics["psnr"],
                    "val_ssim": val_metrics["ssim"],
                    "primary_metric_name": primary_metric_name,
                    "primary_metric_value": val_metrics.get(primary_metric_name),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                for key in (
                    "selective_score",
                    "bg_completion",
                    "edge_source_completion",
                    "edge_retention_ratio",
                    "edge_oversmooth_vs_src",
                    "bg_hf_error",
                    "edge_preserve_error",
                    "edge_over_smooth_ratio",
                    "edge_gmsd",
                    "bg_grad_energy_ratio",
                    "edge_grad_energy_ratio",
                    "structure_alignment_error",
                ):
                    if key in val_metrics:
                        payload[key] = val_metrics[key]
                with metrics_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                print(f"[VAL {current_iter:06d}] {format_validation_log(val_metrics)}")
                if val_metrics["psnr"] >= best_psnr:
                    best_psnr = val_metrics["psnr"]
                current_primary_value = float(val_metrics.get(primary_metric_name, best_metric_value))
                if is_better_metric(current_primary_value, best_metric_value, primary_higher_is_better):
                    best_metric_value = current_primary_value
                    checkpoint["best_metric_name"] = primary_metric_name
                    checkpoint["best_metric_value"] = best_metric_value
                    checkpoint["best_psnr"] = best_psnr
                    save_checkpoint(ckpt_dir / "best.pt", checkpoint)

    final_checkpoint = {
        "epoch": epoch,
        "current_iter": current_iter,
        "best_metric_name": primary_metric_name,
        "best_metric_value": best_metric_value,
        "best_psnr": best_psnr,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
    }
    save_checkpoint(ckpt_dir / "latest.pt", final_checkpoint)
    print(
        f"[DONE] Training finished. "
        f"Best {primary_metric_name}={best_metric_value:.4f}, "
        f"Best PSNR={best_psnr:.4f}. Checkpoints: {ckpt_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
