#!/usr/bin/env python3
"""W10+B13 QAT checkpoint 的 PyTorch 推理脚本。

支持两类输入：
1. 带 `manifest.csv` 的 split 目录；
2. 单个 YUV 文件。

对仅滤波 Y 的模型，脚本读取 Y plane 送入模型，输出时将滤波后的 Y 与
原始 chroma 字节拼回，确保 UV 不被模型修改。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from model_int_qat import DeployPrefilterIntQAT, IntQATConfig
from utils import (
    calculate_psnr,
    calculate_ssim,
    crop_after_pad,
    ensure_dir,
    pad_to_factor,
    read_csv_rows,
    tensor2yuv,
    yuvread2tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 W10+B13 定点 QAT Prefilter 推理。")
    parser.add_argument("--config", default="", help="可选 QAT 配置；不传时使用 checkpoint 内保存的配置。")
    parser.add_argument("--checkpoint", required=True, help="train_int_qat.py 生成的 QAT checkpoint。")
    parser.add_argument("--input", required=True, help="包含 manifest.csv 的 split 目录，或单个 .yuv 文件。")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=None, help="单文件 YUV 推理时必须指定宽度。")
    parser.add_argument("--height", type=int, default=None, help="单文件 YUV 推理时必须指定高度。")
    parser.add_argument("--format", default="yuv420p")
    parser.add_argument("--bitdepth", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_int_qat_config(config: dict[str, Any]) -> IntQATConfig:
    model_cfg = config["model"]
    qat_cfg = config.get("int_qat", {}) or {}
    return IntQATConfig(
        weight_bits=int(qat_cfg.get("weight_bits", model_cfg.get("weight_bits", 10))),
        bias_bits=int(qat_cfg.get("bias_bits", model_cfg.get("bias_bits", 13))),
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


def load_model(config_path: str | Path, checkpoint_path: str | Path, device: torch.device) -> DeployPrefilterIntQAT:
    state = torch.load(checkpoint_path, map_location=device)
    if "model" not in state or not isinstance(state["model"], dict):
        raise RuntimeError(f"Invalid QAT checkpoint, missing model state: {checkpoint_path}")

    config = load_yaml(config_path) if config_path else state.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Missing config. Pass --config or use a checkpoint saved by train_int_qat.py.")

    model_state = state["model"]
    weight_fp = model_state.get("weight_fp")
    bias_fp = model_state.get("bias_fp")
    shift = model_state.get("shift")
    if weight_fp is None or bias_fp is None or shift is None:
        raise RuntimeError("Invalid QAT checkpoint, expected weight_fp, bias_fp, and shift in model state.")

    model = DeployPrefilterIntQAT(
        fused_weight_fp=weight_fp.detach().to(device).float(),
        fused_bias_fp_raw=bias_fp.detach().to(device).float(),
        cfg=build_int_qat_config(config),
        init_shift=shift.detach().to(device).long(),
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    print(f"[INFO] Loaded QAT checkpoint: {checkpoint_path}")
    print(f"[INFO] Quant stats: {model.quantization_stats()}")
    return model


@torch.no_grad()
def infer_one(model: DeployPrefilterIntQAT, img_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = img_tensor.unsqueeze(0).to(device)
    x, pad_hw = pad_to_factor(x, model.downscale_factor)
    pred = model(x)
    pred = crop_after_pad(pred, pad_hw)
    return pred.squeeze(0).cpu()


def yuv_plane_sizes(width: int, height: int, fmt: str, bitdepth: int) -> tuple[int, int, int, int]:
    bytes_per_sample = 1 if bitdepth <= 8 else 2
    y_samples = width * height
    if fmt == "yuv400p":
        u_samples = v_samples = 0
    elif fmt == "yuv420p":
        u_samples = v_samples = (width // 2) * (height // 2)
    elif fmt == "yuv422p":
        u_samples = v_samples = (width // 2) * height
    elif fmt == "yuv444p":
        u_samples = v_samples = width * height
    else:
        raise ValueError(f"Unsupported format for exact Y-only inference: {fmt}")
    return (
        y_samples * bytes_per_sample,
        u_samples * bytes_per_sample,
        v_samples * bytes_per_sample,
        bytes_per_sample,
    )


def read_y_plane_with_chroma(
    path: Path,
    width: int,
    height: int,
    fmt: str,
    bitdepth: int,
) -> tuple[torch.Tensor, bytes]:
    """读取归一化 Y 张量，并返回未修改的 chroma 字节用于透传。"""
    y_bytes, u_bytes, v_bytes, bytes_per_sample = yuv_plane_sizes(width, height, fmt, bitdepth)
    frame_bytes = y_bytes + u_bytes + v_bytes
    with path.open("rb") as f:
        raw = f.read(frame_bytes)
    if len(raw) != frame_bytes:
        raise EOFError(f"Cannot read full frame from {path}: got {len(raw)}, expected {frame_bytes}")

    dtype = np.uint8 if bytes_per_sample == 1 else np.uint16
    max_value = float((1 << bitdepth) - 1)
    y = np.frombuffer(raw[:y_bytes], dtype=dtype).reshape(height, width).astype(np.float32) / max_value
    return torch.from_numpy(y).unsqueeze(0), raw[y_bytes:]


def tensor_y_to_bytes(tensor: torch.Tensor, bitdepth: int) -> bytes:
    if tensor.dim() != 3 or tensor.size(0) != 1:
        raise ValueError(f"Expected Y tensor [1,H,W], got {tuple(tensor.shape)}")
    max_value = (1 << bitdepth) - 1
    dtype = np.uint8 if bitdepth <= 8 else np.uint16
    arr = (tensor.detach().cpu().clamp(0, 1) * float(max_value)).round().to(torch.int64)
    return arr.squeeze(0).numpy().astype(dtype).tobytes()


def write_y_with_original_chroma(tensor_y: torch.Tensor, chroma_payload: bytes, output_path: Path, bitdepth: int) -> None:
    """写出滤波后的 Y，并在其后拼接原始 chroma 字节。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(tensor_y_to_bytes(tensor_y, bitdepth))
        f.write(chroma_payload)


@torch.no_grad()
def infer_y_only(model: DeployPrefilterIntQAT, y_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not model.only_train_y:
        raise ValueError("infer_y_only is only valid when model.only_train_y=True")
    pred = infer_one(model, y_tensor, device)
    return pred[:1]


def run_split_dir(
    model: DeployPrefilterIntQAT,
    split_dir: Path,
    output_dir: Path,
    device: torch.device,
    skip_existing: bool,
) -> None:
    manifest = read_csv_rows(split_dir / "manifest.csv")
    metric_rows = []
    for row in manifest:
        input_path = split_dir / row["input_path"]
        target_path = split_dir / row["target_path"]
        output_path = output_dir / Path(row["input_path"]).relative_to("img")
        if skip_existing and output_path.is_file():
            continue

        width = int(row["width"])
        height = int(row["height"])
        bitdepth = int(row["bitdepth"])
        fmt = row["format"]
        if model.only_train_y:
            img_y, chroma_payload = read_y_plane_with_chroma(input_path, width, height, fmt, bitdepth)
            pred = infer_y_only(model, img_y, device)
            write_y_with_original_chroma(pred, chroma_payload, output_path, bitdepth)
        else:
            img = yuvread2tensor(input_path, width, height, fmt=fmt, bitdepth=bitdepth, normalize=True)
            pred = infer_one(model, img, device)
            tensor2yuv(pred, output_path, fmt=fmt, bitdepth=bitdepth, normalize=True)

        if target_path.is_file():
            if model.only_train_y:
                gt, _ = read_y_plane_with_chroma(target_path, width, height, fmt, bitdepth)
            else:
                gt = yuvread2tensor(target_path, width, height, fmt=fmt, bitdepth=bitdepth, normalize=True)
            metric_rows.append(
                {
                    "sample": Path(row["input_path"]).as_posix(),
                    "psnr": calculate_psnr(pred[:1], gt[:1]),
                    "ssim": calculate_ssim(pred[:1], gt[:1]),
                }
            )

    if metric_rows:
        metrics_path = output_dir / "metrics.json"
        summary = {
            "count": len(metric_rows),
            "avg_psnr": sum(item["psnr"] for item in metric_rows) / len(metric_rows),
            "avg_ssim": sum(item["ssim"] for item in metric_rows) / len(metric_rows),
            "items": metric_rows,
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[DONE] Metrics written to {metrics_path}")


def run_single_file(
    model: DeployPrefilterIntQAT,
    input_path: Path,
    output_dir: Path,
    device: torch.device,
    width: int,
    height: int,
    fmt: str,
    bitdepth: int,
) -> None:
    output_path = output_dir / input_path.name
    if model.only_train_y:
        img_y, chroma_payload = read_y_plane_with_chroma(input_path, width, height, fmt, bitdepth)
        pred = infer_y_only(model, img_y, device)
        write_y_with_original_chroma(pred, chroma_payload, output_path, bitdepth)
    else:
        img = yuvread2tensor(input_path, width, height, fmt=fmt, bitdepth=bitdepth, normalize=True)
        pred = infer_one(model, img, device)
        tensor2yuv(pred, output_path, fmt=fmt, bitdepth=bitdepth, normalize=True)
    print(f"[DONE] {input_path} -> {output_path}")


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    model = load_model(args.config, args.checkpoint, device)

    input_path = Path(args.input).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    if input_path.is_dir() and (input_path / "manifest.csv").is_file():
        run_split_dir(model, input_path, output_dir, device, args.skip_existing)
        return 0

    if not input_path.is_file():
        raise SystemExit(f"Invalid input: {input_path}")
    if args.width is None or args.height is None:
        raise SystemExit("Single-file YUV inference requires --width and --height")
    run_single_file(model, input_path, output_dir, device, args.width, args.height, args.format, args.bitdepth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
