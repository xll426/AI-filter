#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from infer_ori_yuv_y_only import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_ROOT, iter_frames, list_inputs, write_frame
from utils import ensure_dir


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_ONNX = (
    REPO_ROOT
    / "runs/xlx_clean_roi_512_edge_aux_int_qat_w10_b13/onnx/quant_w10_b13_best_y_raw_dynamic.onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw-I/O w10_b13 ONNX on data/ori YUV videos. Y is filtered, UV/chroma is copied unchanged.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "w10_b13"))
    parser.add_argument("--onnx", default=str(DEFAULT_RAW_ONNX))
    parser.add_argument("--bitdepth", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug option: process only the first N frames.")
    return parser.parse_args()


def load_session(onnx_path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required. Install it with `pip install onnxruntime`.") from exc
    return ort.InferenceSession(str(onnx_path.resolve()), providers=["CPUExecutionProvider"])


def run_raw_onnx_y(session: Any, y: np.ndarray) -> np.ndarray:
    x = y.astype(np.float32, copy=False)[None, None, :, :]
    out = session.run(None, {session.get_inputs()[0].name: x})[0]
    if out.ndim != 4 or out.shape[0] != 1 or out.shape[1] != 1:
        raise RuntimeError(f"Expected ONNX output [1,1,H,W], got {tuple(out.shape)}")
    return out[0, 0]


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    session = load_session(Path(args.onnx))
    input_shape = session.get_inputs()[0].shape
    print(f"[INFO] ONNX: {Path(args.onnx).resolve()}", flush=True)
    print(f"[INFO] ONNX input shape: {input_shape}; raw 0..255 float I/O", flush=True)
    print(f"[INFO] Output dir: {output_dir}", flush=True)

    for info in list_inputs(input_dir, args.bitdepth):
        output_path = output_dir / info.path.name
        frame_limit = info.frame_count if args.max_frames is None else min(info.frame_count, args.max_frames)
        print(
            f"[INFO] {info.path.name}: {info.width}x{info.height} {info.fmt}, "
            f"{frame_limit}/{info.frame_count} frame(s)",
            flush=True,
        )
        with output_path.open("wb") as out_f:
            for frame_idx, y, chroma_payload in iter_frames(info, max_frames=args.max_frames):
                out_y = run_raw_onnx_y(session, y)
                write_frame(out_f, out_y, chroma_payload, info.bitdepth)
                print(f"[w10_b13] {info.path.name} frame {frame_idx + 1}/{frame_limit}", flush=True)
        print(f"[DONE] {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
