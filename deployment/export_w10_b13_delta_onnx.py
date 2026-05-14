#!/usr/bin/env python3
"""Export W10+B13 as a raw-Y residual ONNX.

Edit the constants in this block when paths or export size need to change.
No command-line arguments are used.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Export settings
# =============================================================================

# 训练交付目录，写死绝对路径。
SOURCE_DELIVER_DIR = "/mnt/d/fudan/prefilter_clean/deliver_w10_b13"

# W10+B13 QAT config and checkpoint.
CONFIG_PATH = "/mnt/d/fudan/prefilter_clean/deliver_w10_b13/configs/w10_b13_qat.yaml"
CHECKPOINT_PATH = "/mnt/d/fudan/prefilter_clean/deliver_w10_b13/models/w10_b13_best.pt"

# 输出 ONNX。该模型输出 residual delta_y，不输出 filtered Y。
# 这是原先带 edge loss 的 W10/B13 交付模型，文件名显式加 edge，避免和后续
# teacher_qat_w10_b12 / fudan_fp16 模型混淆。
OUTPUT_ONNX_PATH = (
    "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/models/"
    "edge_w10_b13_delta_raw_dynamic.onnx"
)

# 用 32x32 dummy 导出，但实际 ONNX 输入 H/W 是动态轴。
# 注意：PixelUnshuffle4 要求实际输入高宽能被 4 整除。
EXPORT_HEIGHT = 32
EXPORT_WIDTH = 32
DYNAMIC_INPUT_SHAPE = True

# CPU is enough for export and avoids CUDA environment differences.
DEVICE = "cpu"
OPSET_VERSION = 17

# Optional validation before writing ONNX.
RUN_TORCH_CHECK = True


CODE_DIR = os.path.join(SOURCE_DELIVER_DIR, "code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from pred_int_qat import load_model, select_device  # noqa: E402


class ExportableQuantPrefilterYDelta(nn.Module):
    """ONNX-friendly Y-only graph that returns the raw residual plane."""

    def __init__(
        self,
        q_w: torch.Tensor,
        q_b: torch.Tensor,
        shift: torch.Tensor,
        downscale_factor: int,
    ) -> None:
        super().__init__()
        if q_w.ndim != 4 or q_w.shape[-2:] != (3, 3):
            raise ValueError(f"Expected q_w as OIHW 3x3, got {tuple(q_w.shape)}")
        if q_b.ndim != 1 or q_b.numel() != q_w.size(0):
            raise ValueError(f"q_b shape mismatch: {tuple(q_b.shape)} vs q_w={tuple(q_w.shape)}")
        if shift.ndim != 1 or shift.numel() != q_w.size(0):
            raise ValueError(f"shift shape mismatch: {tuple(shift.shape)} vs q_w={tuple(q_w.shape)}")

        self.downscale_factor = int(downscale_factor)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale_factor)
        self.pixel_shuffle = nn.PixelShuffle(self.downscale_factor)

        # Stored as float32 for ONNX Runtime Conv, but values are integer q_w/q_b.
        self.register_buffer("q_w_float", q_w.to(torch.float32))
        self.register_buffer("q_b_float", q_b.to(torch.float32))
        inv_shift = torch.pow(torch.tensor(2.0, dtype=torch.float32), -shift.to(torch.float32))
        self.register_buffer("inv_shift", inv_shift.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is raw Y in 0..255 float32. The round+clip keeps uint8 semantics.
        y = torch.clamp(torch.round(x[:, :1]), 0.0, 255.0)
        y_u = self.pixel_unshuffle(y)
        acc = F.conv2d(y_u, self.q_w_float, self.q_b_float, stride=1, padding=1)
        delta_u = torch.round(acc * self.inv_shift)
        return self.pixel_shuffle(delta_u)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    tensor_i64 = tensor.detach().cpu().to(torch.int64)
    return {
        "shape": list(tensor_i64.shape),
        "min": int(tensor_i64.min().item()),
        "max": int(tensor_i64.max().item()),
        "max_abs": int(tensor_i64.abs().max().item()),
    }


def save_sidecar_files(output_path: str, params: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    stem = os.path.splitext(output_path)[0]
    np.savez_compressed(
        f"{stem}.int_params.npz",
        q_w=params["q_w"].numpy().astype(np.int32),
        q_b=params["q_b"].numpy().astype(np.int32),
        shift=params["shift"].numpy().astype(np.int32),
    )
    with open(f"{stem}.export_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def add_onnx_metadata(output_path: str, meta: dict[str, Any]) -> None:
    try:
        import onnx
    except ImportError:
        return

    model_proto = onnx.load(output_path)
    del model_proto.metadata_props[:]
    for key, value in meta.items():
        prop = model_proto.metadata_props.add()
        prop.key = str(key)
        prop.value = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    onnx.save(model_proto, output_path)


def check_export_module(
    export_model: ExportableQuantPrefilterYDelta,
    params: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    raw_y = torch.randint(
        low=0,
        high=256,
        size=(2, 1, EXPORT_HEIGHT, EXPORT_WIDTH),
        generator=generator,
        device=device,
        dtype=torch.int32,
    ).to(torch.float32)

    with torch.no_grad():
        got = export_model(raw_y)
        y_u = export_model.pixel_unshuffle(raw_y)
        inv_shift = torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=device),
            -params["shift"].to(device=device, dtype=torch.float32),
        ).view(1, -1, 1, 1)
        acc = F.conv2d(
            y_u,
            params["q_w"].to(device=device, dtype=torch.float32),
            params["q_b"].to(device=device, dtype=torch.float32),
            stride=1,
            padding=1,
        )
        ref = export_model.pixel_shuffle(torch.round(acc * inv_shift))
        max_abs = torch.max(torch.abs(ref - got)).item()
    if max_abs != 0.0:
        raise RuntimeError(f"Export module mismatch: max_abs={max_abs}")


def main() -> int:
    device = select_device(DEVICE)
    qat_model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)
    params = qat_model.export_int_parameters()

    export_model = ExportableQuantPrefilterYDelta(
        q_w=params["q_w"],
        q_b=params["q_b"],
        shift=params["shift"],
        downscale_factor=qat_model.downscale_factor,
    ).to(device)
    export_model.eval()

    if RUN_TORCH_CHECK:
        check_export_module(export_model, params, device)

    os.makedirs(os.path.dirname(OUTPUT_ONNX_PATH), exist_ok=True)
    dummy = torch.zeros(1, 1, EXPORT_HEIGHT, EXPORT_WIDTH, dtype=torch.float32, device=device)
    dynamic_axes = None
    if DYNAMIC_INPUT_SHAPE:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "delta_y": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        export_model,
        dummy,
        OUTPUT_ONNX_PATH,
        input_names=["input"],
        output_names=["delta_y"],
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    meta = {
        "source_checkpoint": os.path.abspath(CHECKPOINT_PATH),
        "source_config": os.path.abspath(CONFIG_PATH),
        "weight_bits": int(qat_model.cfg.weight_bits),
        "bias_bits": int(qat_model.cfg.bias_bits),
        "downscale_factor": int(qat_model.downscale_factor),
        "dummy_export_height": int(EXPORT_HEIGHT),
        "dummy_export_width": int(EXPORT_WIDTH),
        "dynamic_input_shape": bool(DYNAMIC_INPUT_SHAPE),
        "input": "raw 0..255 Y block, float32 NCHW shape [N,1,H,W], H/W must be divisible by 4",
        "output": "raw residual delta_y block, float32 NCHW shape [N,1,H,W]",
        "onnx_formula": [
            "Y_u = PixelUnshuffle4(round_clip_u8(Y))",
            "acc = Conv2D(Y_u, q_w, q_b)",
            "delta_u = round(acc / 2^shift)",
            "delta_y = PixelShuffle4(delta_u)",
        ],
        "external_formula": "Y_out_block = clip(round(Y_block + rate_block * delta_y_block), 0, 255)",
        "q_w": tensor_stats(params["q_w"]),
        "q_b": tensor_stats(params["q_b"]),
        "shift": tensor_stats(params["shift"]),
        "note": "ONNX Conv constants are float32 but their values are frozen integer q_w/q_b.",
    }
    save_sidecar_files(OUTPUT_ONNX_PATH, params, meta)
    add_onnx_metadata(OUTPUT_ONNX_PATH, meta)

    stem = os.path.splitext(OUTPUT_ONNX_PATH)[0]
    print(f"[DONE] ONNX written: {OUTPUT_ONNX_PATH}")
    print(f"[DONE] Integer sidecar: {stem}.int_params.npz")
    print(f"[DONE] Export meta: {stem}.export_meta.json")
    print(f"[INFO] q_w={meta['q_w']} q_b={meta['q_b']} shift={meta['shift']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
