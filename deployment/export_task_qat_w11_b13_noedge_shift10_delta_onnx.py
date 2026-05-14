#!/usr/bin/env python3
"""导出 task_qat_w11_b13_noedge_shift10 best.pt 为 raw-Y residual ONNX。

ONNX 输入端不做 round/clip，直接使用外部传入的 raw 0..255 Y。
ONNX 只输出原始 Y 域残差 delta_y。外部推理时执行：

    Y_out = clip(round(Y + rate * delta_y), 0, 255)
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from export_w10_b13_delta_onnx import (
    DYNAMIC_INPUT_SHAPE,
    EXPORT_HEIGHT,
    EXPORT_WIDTH,
    OPSET_VERSION,
    RUN_TORCH_CHECK,
    add_onnx_metadata,
    save_sidecar_files,
    tensor_stats,
)


SOURCE_DELIVER_DIR = "/mnt/d/fudan/prefilter_clean/deliver_w10_b13"
CONFIG_PATH = (
    "/mnt/d/fudan/prefilter_clean/deliver_w10_b13/configs/"
    "task_qat_w11_b13_noedge_shift10.yaml"
)
CHECKPOINT_PATH = (
    "/mnt/d/fudan/prefilter_clean/deliver_w10_b13/runs/"
    "task_qat_w11_b13_noedge_shift10/checkpoints/best.pt"
)
OUTPUT_ONNX_PATH = (
    "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/models/"
    "task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx"
)

DEVICE = "cpu"

CODE_DIR = os.path.join(SOURCE_DELIVER_DIR, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from pred_int_qat import load_model, select_device  # noqa: E402


class ExportableQuantPrefilterYDeltaRawInput(nn.Module):
    """ONNX 图：raw 0..255 Y 直接输入，输出 raw delta_y。"""

    def __init__(
        self,
        q_w: torch.Tensor,
        q_b: torch.Tensor,
        shift: torch.Tensor,
        downscale_factor: int,
    ) -> None:
        super().__init__()
        self.downscale_factor = int(downscale_factor)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale_factor)
        self.pixel_shuffle = nn.PixelShuffle(self.downscale_factor)
        self.register_buffer("q_w_float", q_w.to(torch.float32))
        self.register_buffer("q_b_float", q_b.to(torch.float32))
        inv_shift = torch.pow(torch.tensor(2.0, dtype=torch.float32), -shift.to(torch.float32))
        self.register_buffer("inv_shift", inv_shift.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_u = self.pixel_unshuffle(x[:, :1])
        acc = F.conv2d(y_u, self.q_w_float, self.q_b_float, stride=1, padding=1)
        delta_u = torch.round(acc * self.inv_shift)
        return self.pixel_shuffle(delta_u)


def check_export_module(
    export_model: ExportableQuantPrefilterYDeltaRawInput,
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

    export_model = ExportableQuantPrefilterYDeltaRawInput(
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
        "model_name": "task_qat_w11_b13_noedge_shift10",
        "weight_bits": int(qat_model.cfg.weight_bits),
        "bias_bits": int(qat_model.cfg.bias_bits),
        "downscale_factor": int(qat_model.downscale_factor),
        "dummy_export_height": int(EXPORT_HEIGHT),
        "dummy_export_width": int(EXPORT_WIDTH),
        "dynamic_input_shape": bool(DYNAMIC_INPUT_SHAPE),
        "input": (
            "raw 0..255 Y block, float32 NCHW shape [N,1,H,W], "
            "H/W must be divisible by 4; no input round/clip is applied in ONNX"
        ),
        "output": "raw residual delta_y block, float32 NCHW shape [N,1,H,W]",
        "onnx_formula": [
            "Y_u = PixelUnshuffle4(Y)",
            "acc = Conv2D(Y_u, q_w, q_b)",
            "delta_u = round(acc / 2^shift)",
            "delta_y = PixelShuffle4(delta_u)",
        ],
        "external_formula": "Y_out_block = clip(round(Y_block + rate_block * delta_y_block), 0, 255)",
        "q_w": tensor_stats(params["q_w"]),
        "q_b": tensor_stats(params["q_b"]),
        "shift": tensor_stats(params["shift"]),
        "note": (
            "ONNX Conv constants are float32 but their values are frozen integer q_w/q_b. "
            "q_w is stored as signed W11 and q_b is stored as signed B13 sidecar values. "
            "Input round/clip is intentionally omitted because the runtime feeds uint8 Y converted to float32."
        ),
    }
    save_sidecar_files(OUTPUT_ONNX_PATH, params, meta)
    add_onnx_metadata(OUTPUT_ONNX_PATH, meta)

    stem = os.path.splitext(OUTPUT_ONNX_PATH)[0]
    print(f"[DONE] ONNX written: {OUTPUT_ONNX_PATH}")
    print(f"[DONE] Integer sidecar: {stem}.int_params.npz")
    print(f"[DONE] Export meta: {stem}.export_meta.json")
    print(f"[INFO] q_w={json.dumps(meta['q_w'])}")
    print(f"[INFO] q_b={json.dumps(meta['q_b'])}")
    print(f"[INFO] shift={json.dumps(meta['shift'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
