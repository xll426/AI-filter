#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from model import PrefilterNet, load_prefilter_state
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
    parser = argparse.ArgumentParser(description="Run PrefilterNet inference on a split folder or a single YUV file.")
    parser.add_argument("--config", default="configs/train.yaml", help="Used only for model hyper-parameters.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Split dir containing manifest.csv, or a single .yuv file.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=None, help="Required for single-file YUV inference.")
    parser.add_argument("--height", type=int, default=None, help="Required for single-file YUV inference.")
    parser.add_argument("--format", default="yuv420p")
    parser.add_argument("--bitdepth", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def load_model(config_path: str | Path, checkpoint_path: str | Path, device: torch.device) -> PrefilterNet:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    model = PrefilterNet(**{k: v for k, v in model_cfg.items() if k not in {"pretrain_path", "pretrain_network_g", "strict_load", "strict_load_g"}}).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = load_prefilter_state(model, state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


@torch.no_grad()
def infer_one(model: PrefilterNet, img_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = img_tensor.unsqueeze(0).to(device)
    x, pad_hw = pad_to_factor(x, model.downscale_factor)
    pred = model(x)
    pred = crop_after_pad(pred, pad_hw)
    return pred.squeeze(0).cpu()


def run_split_dir(model: PrefilterNet, split_dir: Path, output_dir: Path, device: torch.device, skip_existing: bool) -> None:
    manifest = read_csv_rows(split_dir / "manifest.csv")
    metric_rows = []
    for row in manifest:
        input_path = split_dir / row["input_path"]
        target_path = split_dir / row["target_path"]
        output_path = output_dir / Path(row["input_path"]).relative_to("img")
        if skip_existing and output_path.is_file():
            continue

        img = yuvread2tensor(input_path, int(row["width"]), int(row["height"]), fmt=row["format"], bitdepth=int(row["bitdepth"]), normalize=True)
        pred = infer_one(model, img, device)
        tensor2yuv(pred, output_path, fmt=row["format"], bitdepth=int(row["bitdepth"]), normalize=True)

        if target_path.is_file():
            gt = yuvread2tensor(target_path, int(row["width"]), int(row["height"]), fmt=row["format"], bitdepth=int(row["bitdepth"]), normalize=True)
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


def run_single_file(model: PrefilterNet, input_path: Path, output_dir: Path, device: torch.device, width: int, height: int, fmt: str, bitdepth: int) -> None:
    img = yuvread2tensor(input_path, width, height, fmt=fmt, bitdepth=bitdepth, normalize=True)
    pred = infer_one(model, img, device)
    output_path = output_dir / input_path.name
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
