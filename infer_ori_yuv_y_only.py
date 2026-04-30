#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ref import generate_reference_tensor
from utils import crop_after_pad, ensure_dir, iter_tile_boxes, pad_to_factor


DEFAULT_ONNX = (
    "runs/xlx_clean_roi_512_edge_aux_int_qat_w10_b13/onnx/"
    "quant_w10_b13_best_y.onnx"
)
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = REPO_ROOT / "data/ori"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/ori_y_filter_results"


@dataclass(frozen=True)
class YuvInfo:
    path: Path
    width: int
    height: int
    fmt: str
    bitdepth: int
    y_bytes: int
    chroma_bytes: int
    frame_bytes: int
    frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run w10_b13 ONNX and/or ref.py on raw YUV files. Only Y is filtered; "
            "the original UV/chroma payload is copied unchanged."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing source .yuv files.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for w10_b13/ref outputs.")
    parser.add_argument("--onnx", default=str(REPO_ROOT / DEFAULT_ONNX), help="w10_b13 Y-plane ONNX model.")
    parser.add_argument("--bitdepth", type=int, default=8, help="Input/output bit depth.")
    parser.add_argument("--downscale-factor", type=int, default=4, help="Model padding factor for dynamic ONNX inputs.")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="ONNX tile size. Defaults to the fixed ONNX input size, or full-frame for dynamic ONNX.",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=None,
        help="ONNX tile stride. Defaults to tile size.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("w10_b13", "ref"),
        default=("w10_b13", "ref"),
        help="Algorithms to run.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Debug option: process only the first N frames.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_info_from_name(path: Path, bitdepth: int) -> YuvInfo:
    match = re.search(r"(?P<w>\d+)x(?P<h>\d+)_(?P<fmt>nv12|yuv420p)(?:_|$)", path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse width/height/format from filename: {path.name}")

    width = int(match.group("w"))
    height = int(match.group("h"))
    fmt = match.group("fmt").lower()
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(f"{fmt} 4:2:0 input requires even width/height: {path.name}")

    bytes_per_sample = 1 if bitdepth <= 8 else 2
    y_bytes = width * height * bytes_per_sample
    if fmt in {"nv12", "yuv420p"}:
        chroma_bytes = width * height // 2 * bytes_per_sample
    else:
        raise ValueError(f"Unsupported YUV format: {fmt}")

    frame_bytes = y_bytes + chroma_bytes
    file_bytes = path.stat().st_size
    if file_bytes % frame_bytes != 0:
        raise ValueError(
            f"File size is not a whole number of frames for {path.name}: "
            f"{file_bytes} bytes, frame={frame_bytes} bytes"
        )

    return YuvInfo(
        path=path,
        width=width,
        height=height,
        fmt=fmt,
        bitdepth=bitdepth,
        y_bytes=y_bytes,
        chroma_bytes=chroma_bytes,
        frame_bytes=frame_bytes,
        frame_count=file_bytes // frame_bytes,
    )


def list_inputs(input_dir: Path, bitdepth: int) -> list[YuvInfo]:
    infos = [parse_info_from_name(path, bitdepth) for path in sorted(input_dir.glob("*.yuv"))]
    if not infos:
        raise FileNotFoundError(f"No .yuv files found in {input_dir}")
    return infos


def iter_frames(info: YuvInfo, max_frames: int | None = None) -> Iterable[tuple[int, np.ndarray, bytes]]:
    dtype = np.uint8 if info.bitdepth <= 8 else np.uint16
    frame_limit = info.frame_count if max_frames is None else min(info.frame_count, max_frames)
    with info.path.open("rb") as f:
        for frame_idx in range(frame_limit):
            y_raw = f.read(info.y_bytes)
            chroma_payload = f.read(info.chroma_bytes)
            if len(y_raw) != info.y_bytes or len(chroma_payload) != info.chroma_bytes:
                raise EOFError(f"Cannot read full frame {frame_idx} from {info.path}")
            y = np.frombuffer(y_raw, dtype=dtype).reshape(info.height, info.width)
            yield frame_idx, y, chroma_payload


def y_to_bytes(y: np.ndarray, bitdepth: int) -> bytes:
    max_value = (1 << bitdepth) - 1
    dtype = np.uint8 if bitdepth <= 8 else np.uint16
    out = np.clip(np.rint(y), 0, max_value).astype(dtype, copy=False)
    return out.tobytes()


def write_frame(out_f, y: np.ndarray, chroma_payload: bytes, bitdepth: int) -> None:
    out_f.write(y_to_bytes(y, bitdepth))
    out_f.write(chroma_payload)


def load_onnx_session(onnx_path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required. Install it with `pip install onnxruntime`.") from exc
    return ort.InferenceSession(str(onnx_path.resolve()), providers=["CPUExecutionProvider"])


def fixed_onnx_hw(session: Any) -> tuple[int | None, int | None]:
    shape = session.get_inputs()[0].shape
    if len(shape) != 4:
        raise RuntimeError(f"Expected ONNX input shape [N,1,H,W], got {shape}")
    h = shape[2] if isinstance(shape[2], int) else None
    w = shape[3] if isinstance(shape[3], int) else None
    return h, w


def run_onnx_tile(session: Any, y: np.ndarray, bitdepth: int, downscale_factor: int) -> np.ndarray:
    max_value = float((1 << bitdepth) - 1)
    x = torch.from_numpy(y.astype(np.float32) / max_value).unsqueeze(0).unsqueeze(0)
    x, pad_hw = pad_to_factor(x, downscale_factor)
    ort_input = x.numpy().astype(np.float32, copy=False)
    ort_output = session.run(None, {session.get_inputs()[0].name: ort_input})[0]
    pred = crop_after_pad(torch.from_numpy(ort_output), pad_hw)
    if pred.ndim != 4 or pred.shape[0] != 1 or pred.shape[1] != 1:
        raise RuntimeError(f"Expected ONNX output [1,1,H,W], got {tuple(pred.shape)}")
    return (pred.squeeze(0).squeeze(0).numpy() * max_value).round()


def run_onnx_y(
    session: Any,
    y: np.ndarray,
    bitdepth: int,
    downscale_factor: int,
    tile_size: int | None,
    tile_stride: int | None,
) -> np.ndarray:
    fixed_h, fixed_w = fixed_onnx_hw(session)
    if fixed_h is None and fixed_w is None:
        return run_onnx_tile(session, y, bitdepth, downscale_factor)
    if fixed_h != fixed_w:
        raise ValueError(f"Only square fixed ONNX inputs are supported, got {fixed_h}x{fixed_w}")

    model_tile = fixed_h
    tile_size = model_tile if tile_size is None else tile_size
    tile_stride = tile_size if tile_stride is None else tile_stride
    if tile_size != model_tile:
        raise ValueError(f"This ONNX expects {model_tile}x{model_tile}, got --tile-size {tile_size}")
    if y.shape[0] < tile_size or y.shape[1] < tile_size:
        raise ValueError(f"Input frame {y.shape[1]}x{y.shape[0]} is smaller than ONNX tile size {tile_size}")

    acc = np.zeros(y.shape, dtype=np.float32)
    weight = np.zeros(y.shape, dtype=np.float32)
    for top, left, tile_h, tile_w in iter_tile_boxes(y.shape[1], y.shape[0], tile_size, tile_stride, "yuv420p"):
        if tile_h != tile_size or tile_w != tile_size:
            raise RuntimeError(f"Unexpected partial tile {tile_w}x{tile_h}; fixed ONNX requires {tile_size}x{tile_size}")
        pred_tile = run_onnx_tile(session, y[top : top + tile_h, left : left + tile_w], bitdepth, downscale_factor)
        acc[top : top + tile_h, left : left + tile_w] += pred_tile.astype(np.float32, copy=False)
        weight[top : top + tile_h, left : left + tile_w] += 1.0
    if np.any(weight == 0):
        raise RuntimeError("Internal tiling error: some pixels were not covered.")
    return acc / weight


def run_ref_y(y: np.ndarray, bitdepth: int) -> np.ndarray:
    if bitdepth != 8:
        raise ValueError("ref.py path currently expects 8-bit Y input.")
    img_tensor = torch.from_numpy(y.astype(np.float32, copy=False)).unsqueeze(0)
    ref_tensor = generate_reference_tensor(img_tensor)
    if ref_tensor.ndim != 3 or ref_tensor.shape[0] < 1:
        raise RuntimeError(f"Expected ref output CHW tensor, got {tuple(ref_tensor.shape)}")
    return ref_tensor[0].detach().cpu().numpy()


def process_one_algorithm(
    info: YuvInfo,
    output_path: Path,
    algorithm: str,
    session: Any | None,
    downscale_factor: int,
    tile_size: int | None,
    tile_stride: int | None,
    max_frames: int | None,
) -> None:
    ensure_dir(output_path.parent)
    frame_limit = info.frame_count if max_frames is None else min(info.frame_count, max_frames)
    with output_path.open("wb") as out_f:
        for frame_idx, y, chroma_payload in iter_frames(info, max_frames=max_frames):
            if algorithm == "w10_b13":
                if session is None:
                    raise RuntimeError("ONNX session is required for w10_b13")
                out_y = run_onnx_y(session, y, info.bitdepth, downscale_factor, tile_size, tile_stride)
            elif algorithm == "ref":
                out_y = run_ref_y(y, info.bitdepth)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            write_frame(out_f, out_y, chroma_payload, info.bitdepth)
            print(f"[{algorithm}] {info.path.name} frame {frame_idx + 1}/{frame_limit}", flush=True)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_root = ensure_dir(Path(args.output_root).resolve())
    infos = list_inputs(input_dir, args.bitdepth)

    session = None
    if "w10_b13" in args.algorithms:
        session = load_onnx_session(Path(args.onnx))

    for info in infos:
        print(
            f"[INFO] {info.path.name}: {info.width}x{info.height} {info.fmt}, "
            f"{info.frame_count} frame(s), bitdepth={info.bitdepth}; "
            f"outputs: {output_root / 'w10_b13'} and {output_root / 'ref'}",
            flush=True,
        )
        for algorithm in args.algorithms:
            output_path = output_root / algorithm / info.path.name
            if args.skip_existing and output_path.is_file():
                print(f"[SKIP] {output_path}", flush=True)
                continue
            process_one_algorithm(
                info=info,
                output_path=output_path,
                algorithm=algorithm,
                session=session,
                downscale_factor=args.downscale_factor,
                tile_size=args.tile_size,
                tile_stride=args.tile_stride,
                max_frames=args.max_frames,
            )
            print(f"[DONE] {algorithm}: {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
