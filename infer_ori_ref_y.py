#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from infer_ori_yuv_y_only import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_ROOT, iter_frames, list_inputs, write_frame
from ref import generate_reference_tensor
from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ref.py on data/ori YUV videos. Y is filtered, UV/chroma is copied unchanged.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "ref"))
    parser.add_argument("--bitdepth", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug option: process only the first N frames.")
    return parser.parse_args()


def run_ref_y(y: np.ndarray, bitdepth: int) -> np.ndarray:
    if bitdepth != 8:
        raise ValueError("ref.py path currently expects 8-bit Y input.")
    img_tensor = torch.from_numpy(y.astype(np.float32, copy=False)).unsqueeze(0)
    out = generate_reference_tensor(img_tensor)
    if out.ndim != 3 or out.shape[0] < 1:
        raise RuntimeError(f"Expected ref output CHW tensor, got {tuple(out.shape)}")
    return out[0].detach().cpu().numpy()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(Path(args.output_dir).resolve())
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
                out_y = run_ref_y(y, info.bitdepth)
                write_frame(out_f, out_y, chroma_payload, info.bitdepth)
                print(f"[ref] {info.path.name} frame {frame_idx + 1}/{frame_limit}", flush=True)
        print(f"[DONE] {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
