from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pred_int_qat import infer_one, load_model, select_device
from utils import crop_after_pad, ensure_dir, pad_to_factor, read_csv_rows, tensor2yuv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exported quant ONNX inference with PyTorch int-QAT inference."
    )
    parser.add_argument("--onnx", default="runs/xlx_clean_roi_512_edge_aux_int_qat_w12/onnx/quant_w12_best_y.onnx")
    parser.add_argument("--config", default="configs/int_qat_xlx_clean_roi_512_edge_aux.yaml")
    parser.add_argument(
        "--checkpoint",
        default="runs/xlx_clean_roi_512_edge_aux_int_qat_w12/checkpoints/best.pt",
    )
    parser.add_argument("--split_dir", default="data/xlx_clean_roi_512/test")
    parser.add_argument("--output_dir", default="runs/xlx_clean_roi_512_edge_aux_int_qat_w12/onnx_compare")
    parser.add_argument("--device", default="cpu", help="PyTorch device. ONNXRuntime uses CPUExecutionProvider by default.")
    parser.add_argument("--all", action="store_true", help="Compare every manifest row instead of one sample per test/img folder.")
    parser.add_argument("--save_yuv", action="store_true", help="Save PyTorch and ONNX yuv outputs for inspected samples.")
    parser.add_argument("--atol", type=float, default=0.0, help="Required max normalized abs diff.")
    return parser.parse_args()


def pick_first_sample_per_input_folder(split_dir: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(split_dir / "manifest.csv")
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        input_path = Path(row["input_path"])
        if len(input_path.parts) < 3 or input_path.parts[0] != "img":
            key = row.get("source_video") or input_path.parent.as_posix()
        else:
            key = input_path.parts[1]
        selected.setdefault(key, row)
    return [selected[key] for key in sorted(selected)]


def read_y_plane_tensor(path: Path, width: int, height: int, bitdepth: int) -> torch.Tensor:
    if bitdepth > 8:
        raise ValueError("This Y-only compare currently supports 8-bit YUV files.")
    y_size = width * height
    with path.open("rb") as f:
        raw = f.read(y_size)
    if len(raw) != y_size:
        raise EOFError(f"Cannot read full Y plane from {path}")
    y = np.frombuffer(raw, dtype=np.uint8).reshape(height, width).astype(np.float32) / 255.0
    return torch.from_numpy(y).unsqueeze(0)


def run_onnx(session: Any, img_tensor: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    x = img_tensor.unsqueeze(0).to(torch.float32)
    x, pad_hw = pad_to_factor(x, downscale_factor)
    ort_input = x.cpu().numpy().astype(np.float32)
    ort_output = session.run(None, {session.get_inputs()[0].name: ort_input})[0]
    pred = torch.from_numpy(ort_output)
    pred = crop_after_pad(pred, pad_hw)
    return pred.squeeze(0).cpu()


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    diff = (a - b).abs()
    raw_diff = torch.round(a * 255.0).to(torch.int16) - torch.round(b * 255.0).to(torch.int16)
    return {
        "max_abs_norm": float(diff.max().item()),
        "mean_abs_norm": float(diff.mean().item()),
        "max_abs_luma_norm": float(diff[:1].max().item()),
        "max_abs_raw_lsb": int(raw_diff.abs().max().item()),
        "max_abs_luma_raw_lsb": int(raw_diff[:1].abs().max().item()),
        "nonzero_raw_pixels": int(torch.count_nonzero(raw_diff).item()),
    }


def save_outputs(
    output_dir: Path,
    sample_name: str,
    pytorch_pred: torch.Tensor,
    onnx_pred: torch.Tensor,
    bitdepth: int,
) -> None:
    safe_name = sample_name.replace("/", "__")
    tensor2yuv(pytorch_pred, output_dir / "pytorch_y" / safe_name, fmt="yuv400p", bitdepth=bitdepth, normalize=True)
    tensor2yuv(onnx_pred, output_dir / "onnx_y" / safe_name, fmt="yuv400p", bitdepth=bitdepth, normalize=True)


def main() -> int:
    args = parse_args()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required. Install it with `pip install onnxruntime`.") from exc

    split_dir = Path(args.split_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    rows = read_csv_rows(split_dir / "manifest.csv") if args.all else pick_first_sample_per_input_folder(split_dir)
    if not rows:
        raise SystemExit(f"No samples found in {split_dir}")

    device = select_device(args.device)
    model = load_model(args.config, args.checkpoint, device)
    session = ort.InferenceSession(str(Path(args.onnx).resolve()), providers=["CPUExecutionProvider"])

    results = []
    failed = []
    for row in rows:
        input_rel = row["input_path"]
        input_path = split_dir / input_rel
        width = int(row["width"])
        height = int(row["height"])
        fmt = row["format"]
        bitdepth = int(row["bitdepth"])

        if fmt != "yuv420p":
            raise ValueError(f"Y-only direct raw reader currently expects yuv420p, got {fmt}")
        img_y = read_y_plane_tensor(input_path, width, height, bitdepth)
        pytorch_pred = infer_one(model, img_y, device)
        onnx_pred = run_onnx(session, img_y, model.downscale_factor)
        stats = diff_stats(pytorch_pred, onnx_pred)
        item = {
            "input_path": input_rel,
            "width": width,
            "height": height,
            "format": fmt,
            "bitdepth": bitdepth,
            **stats,
        }
        results.append(item)
        if stats["max_abs_norm"] > args.atol:
            failed.append(item)

        if args.save_yuv:
            save_outputs(output_dir, input_rel, pytorch_pred, onnx_pred, bitdepth)

        print(
            "[COMPARE] "
            f"{input_rel} max_abs_norm={stats['max_abs_norm']:.10f} "
            f"max_abs_raw_lsb={stats['max_abs_raw_lsb']} "
            f"nonzero_raw_pixels={stats['nonzero_raw_pixels']}"
        )

    summary = {
        "onnx": str(Path(args.onnx).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split_dir": str(split_dir),
        "sample_count": len(results),
        "atol": args.atol,
        "passed": len(failed) == 0,
        "max_abs_norm": max(item["max_abs_norm"] for item in results),
        "max_abs_raw_lsb": max(item["max_abs_raw_lsb"] for item in results),
        "items": results,
    }
    summary_path = output_dir / "compare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Summary written: {summary_path}")

    if failed:
        print(f"[FAIL] {len(failed)} sample(s) exceeded --atol {args.atol}")
        return 1
    print("[PASS] ONNX output is identical to PyTorch within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
