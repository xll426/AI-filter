#!/usr/bin/env python3
"""用 FP32 teacher 在线监督 W/B 定点 QAT student。

这个脚本是独立训练入口，不改动原来的 train_int_qat.py。

训练策略：
1. 加载 FP32 teacher 权重，例如 weights/iccv_yan_2025_fp32.pth。
2. 用 teacher 的 fused 3x3 deploy 权重初始化 QAT student。
3. 对每个 batch 在线计算 teacher(inputs)，作为训练 target。
4. student(inputs) 只用逐像素 L1 或 L2 拟合 teacher 输出。
5. 保存 best/latest checkpoint，并导出 q_w/q_b/shift。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model import PrefilterNet, load_prefilter_state
from model_int_qat import DeployPrefilterIntQAT, IntQATConfig, build_deploy_fp32_reference
from utils import calculate_psnr, calculate_ssim, ensure_dir, read_csv_rows, set_random_seed, yuvread2tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SourceYuvDataset(torch.utils.data.Dataset):
    """只读取 manifest 里的 input_path，不读取 target/gt。"""

    def __init__(
        self,
        split_dir: str | Path,
        crop_size: int | None = None,
        training: bool = False,
        hflip: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.crop_size = crop_size
        self.training = training
        self.hflip = hflip
        manifest_path = self.split_dir / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.records = read_csv_rows(manifest_path)
        if not self.records:
            raise ValueError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _crop(self, img: torch.Tensor) -> torch.Tensor:
        if not self.crop_size:
            return img
        _, h, w = img.shape
        if self.crop_size >= h or self.crop_size >= w:
            return img
        if self.training:
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
        return img[:, top : top + self.crop_size, left : left + self.crop_size]

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.records[index]
        img_path = self.split_dir / row["input_path"]
        img = yuvread2tensor(
            img_path,
            int(row["width"]),
            int(row["height"]),
            fmt=row["format"],
            bitdepth=int(row["bitdepth"]),
            normalize=True,
        )
        img = self._crop(img)
        if self.training and self.hflip and random.random() < 0.5:
            img = torch.flip(img, dims=[2])
        return {
            "input": img,
            "input_path": str(img_path),
            "source_video": row.get("source_video", ""),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-target W/B QAT training.")
    parser.add_argument("--config", default="configs/teacher_qat_w10_b13.yaml")
    parser.add_argument("--resume", default=None, help="'auto' 或 checkpoint 路径；优先级高于 config.train.resume。")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def resolve_resume_path(run_dir: Path, resume_cfg: str | None) -> Path | None:
    if not resume_cfg or str(resume_cfg).lower() in {"none", "null", ""}:
        return None
    if resume_cfg == "auto":
        candidate = run_dir / "checkpoints" / "latest.pt"
        return candidate if candidate.is_file() else None
    path = resolve_path(resume_cfg)
    return path if path.is_file() else None


def build_scheduler(optimizer: Adam, train_cfg: dict[str, Any], total_iters: int) -> CosineAnnealingLR:
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


def build_fp32_source_model(model_cfg: dict[str, Any], device: torch.device) -> PrefilterNet:
    model_kwargs = {
        key: value
        for key, value in model_cfg.items()
        if key
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
    model = PrefilterNet(**model_kwargs).to(device)
    pretrain_path = model_cfg.get("pretrain_path") or model_cfg.get("pretrain_network_g")
    if not pretrain_path:
        raise ValueError("model.pretrain_path is required for teacher-target QAT.")
    pretrain_path = resolve_path(pretrain_path)
    if not pretrain_path.is_file():
        raise FileNotFoundError(f"Missing teacher checkpoint: {pretrain_path}")

    state = torch.load(pretrain_path, map_location=device)
    strict = bool(model_cfg.get("strict_load", model_cfg.get("strict_load_g", True)))
    missing, unexpected = load_prefilter_state(model, state, strict=strict)
    if missing or unexpected:
        print(f"[WARN] teacher load mismatch: missing={missing}, unexpected={unexpected}")
    print(f"[INFO] loaded FP32 teacher: {pretrain_path}")
    model.eval()
    return model


def build_int_qat_config(config: dict[str, Any]) -> IntQATConfig:
    model_cfg = config["model"]
    qat_cfg = config.get("int_qat", {}) or {}
    return IntQATConfig(
        weight_bits=int(qat_cfg.get("weight_bits", 10)),
        bias_bits=int(qat_cfg.get("bias_bits", 13)),
        downscale_factor=int(model_cfg.get("downscale_factor", qat_cfg.get("downscale_factor", 4))),
        only_train_y=bool(model_cfg.get("only_train_y", qat_cfg.get("only_train_y", True))),
        per_channel_shift=bool(qat_cfg.get("per_channel_shift", True)),
        min_shift=int(qat_cfg.get("min_shift", 0)),
        max_shift=int(qat_cfg.get("max_shift", 30)),
        weight_range_penalty=0.0,
        bias_range_penalty=0.0,
        bias_l1_weight=0.0,
    )


def pixel_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str, only_y: bool) -> torch.Tensor:
    if only_y:
        pred = pred[:, :1]
        target = target[:, :1]
    diff = pred - target
    if loss_type == "l1":
        return diff.abs().mean()
    if loss_type == "l2":
        return (diff * diff).mean()
    raise ValueError(f"Unsupported pixel loss: {loss_type}")


@torch.no_grad()
def validate(
    model: DeployPrefilterIntQAT,
    teacher: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
    only_y: bool,
) -> dict[str, float]:
    model.eval()
    teacher.eval()
    loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        # FP32 teacher 是残差直加结构，输出可能轻微越界；部署写 YUV 前会 clip 到 8-bit 范围。
        teacher_targets = teacher(inputs).clamp(0.0, 1.0)
        preds = model(inputs)
        loss = pixel_loss(preds, teacher_targets, loss_type, only_y)
        loss_sum += float(loss.item())

        for pred, target in zip(preds, teacher_targets):
            pred_y = pred[:1] if only_y else pred
            target_y = target[:1] if only_y else target
            psnr_sum += calculate_psnr(pred_y, target_y)
            ssim_sum += calculate_ssim(pred_y, target_y)
            count += 1

    model.train()
    if count == 0:
        return {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    return {
        "loss": loss_sum / max(len(loader), 1),
        "psnr": psnr_sum / count,
        "ssim": ssim_sum / count,
    }


def make_checkpoint(
    *,
    epoch: int,
    current_iter: int,
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
        "best_metric_name": "psnr",
        "best_metric_value": best_psnr,
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
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    qat_cfg = config.get("int_qat", {}) or {}
    logger_cfg = config.get("logger", {}) or {}

    set_random_seed(int(train_cfg.get("seed", 123)))
    torch.backends.cudnn.benchmark = True
    device = select_device(str(train_cfg.get("device", "cuda")))

    data_root = resolve_path(data_cfg["root"])
    output_root = resolve_path(config.get("output_root", "./runs"))
    run_dir = ensure_dir(output_root / config["experiment_name"])
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    export_root = ensure_dir(run_dir / "int_exports")

    train_ds = SourceYuvDataset(
        data_root / "train",
        crop_size=train_cfg.get("crop_size"),
        training=True,
        hflip=train_cfg.get("hflip", True),
    )
    val_root = data_root / "val"
    val_manifest = val_root / "manifest.csv"
    has_val = val_manifest.is_file() and len(read_csv_rows(val_manifest)) > 0
    val_ds = SourceYuvDataset(val_root, crop_size=None, training=False, hflip=False) if has_val else None

    batch_size = int(train_cfg["batch_size"])
    drop_last = bool(train_cfg.get("drop_last", len(train_ds) >= batch_size))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=drop_last,
    )
    if len(train_loader) == 0:
        raise RuntimeError(
            f"Empty train loader: num_train={len(train_ds)}, batch_size={batch_size}, drop_last={drop_last}"
        )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(train_cfg.get("val_batch_size", 1)),
            shuffle=False,
            num_workers=max(0, int(train_cfg.get("num_workers", 0)) // 2),
        )

    num_iter_per_epoch = len(train_loader)
    total_iters = int(train_cfg.get("total_iter", 0))
    total_epochs = train_cfg.get("epochs")
    if total_iters <= 0:
        if total_epochs is None:
            raise ValueError("train.total_iter or train.epochs must be set")
        total_iters = int(total_epochs) * num_iter_per_epoch
    if total_epochs is None:
        total_epochs = math.ceil(total_iters / num_iter_per_epoch)
    total_epochs = int(total_epochs)

    fp32_source = build_fp32_source_model(model_cfg, device)
    teacher = build_deploy_fp32_reference(fp32_source).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    int_qat_config = build_int_qat_config(config)
    model = DeployPrefilterIntQAT.from_fp32_prefilter(fp32_source, int_qat_config).to(device)
    del fp32_source

    optimizer = Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = build_scheduler(optimizer, train_cfg, total_iters)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(train_cfg.get("mixed_precision", False) and device.type == "cuda"),
    )

    resume_cfg = args.resume if args.resume is not None else train_cfg.get("resume", None)
    resume_path = resolve_resume_path(run_dir, resume_cfg)
    start_epoch = 0
    current_iter = 0
    best_psnr = float("-inf")

    if resume_path is not None:
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))
        current_iter = int(state.get("current_iter", 0))
        best_psnr = float(state.get("best_psnr", state.get("best_metric_value", best_psnr)))
        print(f"[INFO] resumed from {resume_path} @ epoch={start_epoch}, iter={current_iter}")

    with (run_dir / "train_teacher_qat_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    loss_cfg = qat_cfg.get("teacher_target", {}) or {}
    loss_type = str(loss_cfg.get("loss_type", "l1")).lower()
    only_y = bool(loss_cfg.get("only_y", model_cfg.get("only_train_y", True)))
    loss_weight = float(loss_cfg.get("loss_weight", 1.0))
    if loss_type not in {"l1", "l2"}:
        raise ValueError(f"int_qat.teacher_target.loss_type must be l1 or l2, got {loss_type}")

    print(
        "[INFO] Teacher-QAT: "
        f"train={len(train_ds)} val={len(val_ds) if val_ds is not None else 0} "
        f"batch={batch_size} total_iters={total_iters} "
        f"W{int_qat_config.weight_bits}/B{int_qat_config.bias_bits} "
        f"loss={loss_type} only_y={only_y}"
    )
    print(f"[INFO] Initial quant stats: {model.quantization_stats()}")

    metrics_log = run_dir / "metrics.jsonl"
    print_freq = int(logger_cfg.get("print_freq", 100))
    val_freq = int(logger_cfg.get("val_freq", 500))
    save_latest_freq = int(logger_cfg.get("save_latest_freq", 500))
    save_checkpoint_freq = int(logger_cfg.get("save_checkpoint_freq", total_iters))
    warmup_iter = int(train_cfg.get("warmup_iter", 300))
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
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                with torch.no_grad():
                    # 对齐最终部署语义：raw Y 会被 clip 到 [0,255]，归一化后就是 [0,1]。
                    teacher_targets = teacher(inputs).clamp(0.0, 1.0)
                preds = model(inputs)
                loss_pixel = pixel_loss(preds, teacher_targets, loss_type, only_y) * loss_weight
                loss = loss_pixel

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if current_iter % print_freq == 0:
                quant_stats = model.quantization_stats()
                print(
                    f"[ITER {current_iter:06d}] "
                    f"epoch={epoch:04d} "
                    f"loss={float(loss.detach().item()):.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6e} "
                    f"max_qw={quant_stats['max_abs_q_w']:.0f} "
                    f"max_qb={quant_stats['max_abs_q_b']:.0f} "
                    f"shift={quant_stats['shift_min']:.0f}-{quant_stats['shift_max']:.0f}"
                )

            checkpoint = None
            if current_iter % save_checkpoint_freq == 0 or current_iter % save_latest_freq == 0:
                checkpoint = make_checkpoint(
                    epoch=epoch,
                    current_iter=current_iter,
                    best_psnr=best_psnr,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                )
            if checkpoint is not None and current_iter % save_checkpoint_freq == 0:
                save_checkpoint(ckpt_dir / f"iter_{current_iter:06d}.pt", checkpoint)
            if checkpoint is not None and current_iter % save_latest_freq == 0:
                save_checkpoint(ckpt_dir / "latest.pt", checkpoint)

            if val_loader is not None and current_iter % val_freq == 0:
                val_metrics = validate(model, teacher, val_loader, device, loss_type, only_y)
                quant_stats = model.quantization_stats()
                append_metrics(
                    metrics_log,
                    {
                        "epoch": epoch,
                        "current_iter": current_iter,
                        "train_loss": float(loss.detach().item()),
                        "val_loss": val_metrics["loss"],
                        "val_psnr": val_metrics["psnr"],
                        "val_ssim": val_metrics["ssim"],
                        "lr": optimizer.param_groups[0]["lr"],
                        **quant_stats,
                    },
                )
                print(
                    f"[VAL {current_iter:06d}] "
                    f"loss={val_metrics['loss']:.6f} "
                    f"psnr={val_metrics['psnr']:.4f} "
                    f"ssim={val_metrics['ssim']:.4f}"
                )
                if val_metrics["psnr"] >= best_psnr:
                    best_psnr = val_metrics["psnr"]
                    best_checkpoint = make_checkpoint(
                        epoch=epoch,
                        current_iter=current_iter,
                        best_psnr=best_psnr,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                    )
                    save_checkpoint(ckpt_dir / "best.pt", best_checkpoint)
                    if export_best:
                        model.save_export(export_root / "best")

    final_checkpoint = make_checkpoint(
        epoch=epoch,
        current_iter=current_iter,
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
    print(f"[DONE] Teacher-QAT finished. best_psnr={best_psnr:.4f}, checkpoints={ckpt_dir}, exports={export_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
