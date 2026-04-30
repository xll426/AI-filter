#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import PairedYuvDataset
from model import PrefilterNet
from model_int_qat import (
    DistillationLoss,
    IntQATConfig,
    build_deploy_fp32_reference,
    DeployPrefilterIntQAT,
)
from train import (
    build_scheduler,
    build_train_loss,
    build_validation_cfg,
    compute_train_loss,
    format_validation_log,
    initial_best_value,
    is_better_metric,
    maybe_load_pretrain,
    resolve_resume_path,
    save_checkpoint,
    select_device,
    update_learning_rate,
    validate,
)
from utils import ensure_dir, read_csv_rows, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integer-aware QAT finetuning for deploy PrefilterNet.")
    parser.add_argument("--config", default="configs/int_qat_xlx_clean_roi_512_edge_aux.yaml")
    parser.add_argument("--resume", default=None, help="'auto' or a QAT checkpoint path. Overrides config.train.resume.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_fp32_source_model(model_cfg: dict[str, Any], device: torch.device) -> PrefilterNet:
    filtered_cfg = {
        k: v
        for k, v in model_cfg.items()
        if k
        not in {
            "pretrain_path",
            "pretrain_network_g",
            "strict_load",
            "strict_load_g",
            "weight_bits",
            "bias_bits",
            "per_channel_shift",
            "weight_range_penalty",
            "bias_range_penalty",
            "bias_l1_weight",
            "min_shift",
            "max_shift",
        }
    }
    fp32_model = PrefilterNet(**filtered_cfg).to(device)
    maybe_load_pretrain(fp32_model, model_cfg, device)
    fp32_model.eval()
    return fp32_model


def build_int_qat_config(config: dict[str, Any]) -> IntQATConfig:
    model_cfg = config["model"]
    qat_cfg = config.get("int_qat", {}) or {}
    return IntQATConfig(
        weight_bits=int(qat_cfg.get("weight_bits", model_cfg.get("weight_bits", 12))),
        bias_bits=int(qat_cfg.get("bias_bits", model_cfg.get("bias_bits", 17))),
        downscale_factor=int(model_cfg.get("downscale_factor", qat_cfg.get("downscale_factor", 4))),
        only_train_y=bool(model_cfg.get("only_train_y", qat_cfg.get("only_train_y", True))),
        per_channel_shift=bool(qat_cfg.get("per_channel_shift", model_cfg.get("per_channel_shift", True))),
        min_shift=int(qat_cfg.get("min_shift", model_cfg.get("min_shift", 0))),
        max_shift=int(qat_cfg.get("max_shift", model_cfg.get("max_shift", 30))),
        weight_range_penalty=float(
            qat_cfg.get("weight_range_penalty", model_cfg.get("weight_range_penalty", 1e-6))
        ),
        bias_range_penalty=float(qat_cfg.get("bias_range_penalty", model_cfg.get("bias_range_penalty", 1e-6))),
        bias_l1_weight=float(qat_cfg.get("bias_l1_weight", model_cfg.get("bias_l1_weight", 0.0))),
    )


def build_distillation_loss(config: dict[str, Any]) -> DistillationLoss | None:
    distill_cfg = (config.get("int_qat", {}) or {}).get("distillation", {}) or {}
    if not bool(distill_cfg.get("enabled", True)):
        return None
    return DistillationLoss(
        loss_weight=float(distill_cfg.get("loss_weight", 0.1)),
        only_y=bool(distill_cfg.get("only_y", config["model"].get("only_train_y", True))),
        use_charbonnier=bool(distill_cfg.get("use_charbonnier", False)),
        eps=float(distill_cfg.get("eps", 1e-12)),
    )


def build_optimizer(model: DeployPrefilterIntQAT, train_cfg: dict[str, Any]) -> Adam:
    return Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )


def make_checkpoint(
    *,
    epoch: int,
    current_iter: int,
    best_metric_name: str,
    best_metric_value: float,
    best_psnr: float,
    model: DeployPrefilterIntQAT,
    optimizer: Adam,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "current_iter": current_iter,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_psnr": best_psnr,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "int_export": model.export_int_parameters(),
    }


def append_metrics(metrics_log: Path, payload: dict[str, Any]) -> None:
    with metrics_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    logger_cfg = config.get("logger", {})
    validation_cfg = build_validation_cfg(config)
    qat_cfg = config.get("int_qat", {}) or {}

    set_random_seed(int(train_cfg.get("seed", 123)))
    torch.backends.cudnn.benchmark = True

    device = select_device(train_cfg.get("device", "cuda"))
    data_root = Path(data_cfg["root"]).resolve()
    run_dir = ensure_dir(Path(config["output_root"]) / config["experiment_name"])
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    export_root = ensure_dir(run_dir / "int_exports")

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

    fp32_model = build_fp32_source_model(model_cfg, device)
    int_qat_config = build_int_qat_config(config)
    teacher = build_deploy_fp32_reference(fp32_model).to(device)
    model = DeployPrefilterIntQAT.from_fp32_prefilter(fp32_model, int_qat_config).to(device)
    del fp32_model

    fidelity_loss, perceptual_loss, edge_aux_loss = build_train_loss(train_cfg)
    if fidelity_loss is not None:
        fidelity_loss = fidelity_loss.to(device)
    if perceptual_loss is not None:
        perceptual_loss = perceptual_loss.to(device)
    if edge_aux_loss is not None:
        edge_aux_loss = edge_aux_loss.to(device)
    distillation_loss = build_distillation_loss(config)
    if distillation_loss is not None:
        distillation_loss = distillation_loss.to(device)

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
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))
        current_iter = int(state.get("current_iter", state.get("global_step", 0)))
        primary_metric_name = str(state.get("best_metric_name", primary_metric_name))
        best_metric_value = float(state.get("best_metric_value", state.get("best_psnr", best_metric_value)))
        best_psnr = float(state.get("best_psnr", best_psnr))
        print(f"[INFO] Resumed QAT from {resume_path} @ epoch={start_epoch}, iter={current_iter}")

    config_snapshot = run_dir / "train_int_qat_config.yaml"
    with config_snapshot.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    metrics_log = run_dir / "metrics.jsonl"
    print(
        "[INFO] Int-QAT statistics: "
        f"num_train={len(train_ds)} batch_size={batch_size} "
        f"iters_per_epoch={num_iter_per_epoch} total_epochs={total_epochs} total_iters={total_iters} "
        f"weight_bits={int_qat_config.weight_bits} bias_bits={int_qat_config.bias_bits}"
    )
    print(f"[INFO] Initial quant stats: {model.quantization_stats()}")

    print_freq = int(logger_cfg.get("print_freq", train_cfg.get("print_freq", 1000)))
    save_latest_freq = int(logger_cfg.get("save_latest_freq", train_cfg.get("save_latest_freq", 500)))
    save_checkpoint_freq = int(logger_cfg.get("save_checkpoint_freq", train_cfg.get("save_checkpoint_freq", total_iters)))
    val_freq = int(logger_cfg.get("val_freq", train_cfg.get("val_freq", 1000)))
    warmup_iter = int(train_cfg.get("warmup_iter", 3000))
    export_best = bool(qat_cfg.get("export_best", True))
    export_latest = bool(qat_cfg.get("export_latest", True))

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
                task_loss, loss_logs = compute_train_loss(
                    inputs,
                    preds,
                    targets,
                    roi,
                    only_train_y,
                    fidelity_loss,
                    perceptual_loss,
                    edge_aux_loss,
                )
                loss = task_loss

                if distillation_loss is not None:
                    with torch.no_grad():
                        teacher_preds = teacher(inputs)
                    l_distill = distillation_loss(preds, teacher_preds)
                    loss = loss + l_distill
                    loss_logs["l_distill"] = float(l_distill.detach().item())

                regs = model.auxiliary_regularization()
                for name, value in regs.items():
                    loss = loss + value
                    loss_logs[name] = float(value.detach().item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_logs["l_task"] = loss_logs.pop("loss")
            loss_logs["loss"] = float(loss.detach().item())

            if current_iter % print_freq == 0:
                quant_stats = model.quantization_stats()
                log_msg = (
                    f"[ITER {current_iter:06d}] "
                    f"epoch={epoch:04d} "
                    f"loss={loss_logs['loss']:.6f} "
                    f"task={loss_logs['l_task']:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6e} "
                    f"max_qw={quant_stats['max_abs_q_w']:.0f} "
                    f"max_qb={quant_stats['max_abs_q_b']:.0f} "
                    f"shift={quant_stats['shift_min']:.0f}-{quant_stats['shift_max']:.0f}"
                )
                if "l_distill" in loss_logs:
                    log_msg += f" l_distill={loss_logs['l_distill']:.6f}"
                if "l_fidelity" in loss_logs:
                    log_msg += f" l_fidelity={loss_logs['l_fidelity']:.6f}"
                if "l_perceptual" in loss_logs:
                    log_msg += f" l_perceptual={loss_logs['l_perceptual']:.6f}"
                if "l_edge_aux" in loss_logs:
                    log_msg += f" l_edge_aux={loss_logs['l_edge_aux']:.6f}"
                print(log_msg)

            should_save_iter = current_iter % save_checkpoint_freq == 0
            should_save_latest = current_iter % save_latest_freq == 0
            if should_save_iter or should_save_latest:
                checkpoint = make_checkpoint(
                    epoch=epoch,
                    current_iter=current_iter,
                    best_metric_name=primary_metric_name,
                    best_metric_value=best_metric_value,
                    best_psnr=best_psnr,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                )
            if should_save_iter:
                save_checkpoint(ckpt_dir / f"iter_{current_iter:06d}.pt", checkpoint)
                print(f"[INFO] Saved QAT checkpoint @ iter {current_iter}")

            if should_save_latest:
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
                quant_stats = model.quantization_stats()
                payload = {
                    "epoch": epoch,
                    "current_iter": current_iter,
                    "train_loss": loss_logs["loss"],
                    "train_task_loss": loss_logs["l_task"],
                    "val_loss": val_metrics["loss"],
                    "val_psnr": val_metrics["psnr"],
                    "val_ssim": val_metrics["ssim"],
                    "primary_metric_name": primary_metric_name,
                    "primary_metric_value": val_metrics.get(primary_metric_name),
                    "lr": optimizer.param_groups[0]["lr"],
                    **quant_stats,
                }
                if "l_distill" in loss_logs:
                    payload["train_distill_loss"] = loss_logs["l_distill"]
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
                append_metrics(metrics_log, payload)
                print(f"[VAL {current_iter:06d}] {format_validation_log(val_metrics)}")

                if val_metrics["psnr"] >= best_psnr:
                    best_psnr = val_metrics["psnr"]
                current_primary_value = float(val_metrics.get(primary_metric_name, best_metric_value))
                if is_better_metric(current_primary_value, best_metric_value, primary_higher_is_better):
                    best_metric_value = current_primary_value
                    checkpoint = make_checkpoint(
                        epoch=epoch,
                        current_iter=current_iter,
                        best_metric_name=primary_metric_name,
                        best_metric_value=best_metric_value,
                        best_psnr=best_psnr,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                    )
                    save_checkpoint(ckpt_dir / "best.pt", checkpoint)
                    if export_best:
                        model.save_export(export_root / "best")

    final_checkpoint = make_checkpoint(
        epoch=epoch,
        current_iter=current_iter,
        best_metric_name=primary_metric_name,
        best_metric_value=best_metric_value,
        best_psnr=best_psnr,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
    )
    save_checkpoint(ckpt_dir / "latest.pt", final_checkpoint)
    if export_latest:
        model.save_export(export_root / "latest")
    print(
        f"[DONE] Int-QAT finished. "
        f"Best {primary_metric_name}={best_metric_value:.4f}, "
        f"Best PSNR={best_psnr:.4f}. Checkpoints: {ckpt_dir}. Exports: {export_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
