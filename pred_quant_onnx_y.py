#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pred_int_qat import (
    read_y_plane_with_chroma,
    tensor_y_to_bytes,
    write_y_with_original_chroma,
    yuv_plane_sizes,
)
from utils import calculate_psnr, calculate_ssim, crop_after_pad, ensure_dir, pad_to_factor, read_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exported int-QAT Y-plane ONNX inference.")
    parser.add_argument("--onnx", required=True, help="Y-plane ONNX exported by export_quant_onnx.py.")
    parser.add_argument("--input", required=True, help="Split dir containing manifest.csv, or a single .yuv file.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=None, help="Required for single-file YUV inference.")
    parser.add_argument("--height", type=int, default=None, help="Required for single-file YUV inference.")
    parser.add_argument("--format", default="yuv420p")
    parser.add_argument("--bitdepth", type=int, default=8)
    parser.add_argument("--downscale_factor", type=int, default=4)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def run_onnx_y(session: Any, y_tensor: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    x = y_tensor.unsqueeze(0).to(torch.float32)
    x, pad_hw = pad_to_factor(x, downscale_factor)
    ort_input = x.cpu().numpy().astype(np.float32)
    ort_output = session.run(None, {session.get_inputs()[0].name: ort_input})[0]
    pred = torch.from_numpy(ort_output)
    pred = crop_after_pad(pred, pad_hw)
    if pred.ndim != 4 or pred.size(1) != 1:
        raise RuntimeError(f"Expected ONNX output [N,1,H,W], got {tuple(pred.shape)}")
    return pred.squeeze(0).cpu()


def run_split_dir(
    session: Any,
    split_dir: Path,
    output_dir: Path,
    downscale_factor: int,
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
        y_tensor, chroma_payload = read_y_plane_with_chroma(input_path, width, height, fmt, bitdepth)
        pred_y = run_onnx_y(session, y_tensor, downscale_factor)
        write_y_with_original_chroma(pred_y, chroma_payload, output_path, bitdepth)

        if target_path.is_file():
            gt_y, _ = read_y_plane_with_chroma(target_path, width, height, fmt, bitdepth)
            metric_rows.append(
                {
                    "sample": Path(row["input_path"]).as_posix(),
                    "psnr": calculate_psnr(pred_y[:1], gt_y[:1]),
                    "ssim": calculate_ssim(pred_y[:1], gt_y[:1]),
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
    session: Any,
    input_path: Path,
    output_dir: Path,
    width: int,
    height: int,
    fmt: str,
    bitdepth: int,
    downscale_factor: int,
) -> None:
    output_path = output_dir / input_path.name
    y_tensor, chroma_payload = read_y_plane_with_chroma(input_path, width, height, fmt, bitdepth)
    pred_y = run_onnx_y(session, y_tensor, downscale_factor)
    write_y_with_original_chroma(pred_y, chroma_payload, output_path, bitdepth)
    print(f"[DONE] {input_path} -> {output_path}")


def main() -> int:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required. Install it with `pip install onnxruntime`.") from exc

    args = parse_args()
    session = ort.InferenceSession(str(Path(args.onnx).resolve()), providers=["CPUExecutionProvider"])
    input_path = Path(args.input).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())

    if input_path.is_dir() and (input_path / "manifest.csv").is_file():
        run_split_dir(session, input_path, output_dir, args.downscale_factor, args.skip_existing)
        return 0

    if not input_path.is_file():
        raise SystemExit(f"Invalid input: {input_path}")
    if args.width is None or args.height is None:
        raise SystemExit("Single-file YUV inference requires --width and --height")
    run_single_file(
        session,
        input_path,
        output_dir,
        args.width,
        args.height,
        args.format,
        args.bitdepth,
        args.downscale_factor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
