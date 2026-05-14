#!/usr/bin/env python3
"""导出复旦 deploy FP16 权重为 raw-Y residual ONNX。

该权重已经是部署态 fused conv：

    processing.weight: [16, 16, 3, 3]
    processing.bias  : [16]

训练/权重语义在归一化域，因此导出 raw Y residual 时使用：

    delta_raw = Conv2D(Y_raw, weight_fp, bias_fp * 255)

ONNX 只输出 delta_y，不在图里加回原始 Y。
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 配置区：只改这里
# =============================================================================

WEIGHT_PATH = "/mnt/d/fudan/prefilter_clean/weights/iccv_yan_2025_deploy_fp16.pth"
OUTPUT_ONNX_PATH = (
    "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/models/"
    "fudan_iccv_yan_2025_deploy_fp16_delta_raw_dynamic.onnx"
)

EXPORT_HEIGHT = 32
EXPORT_WIDTH = 32
DYNAMIC_INPUT_SHAPE = True

DEVICE = "cpu"
OPSET_VERSION = 17
RUN_TORCH_CHECK = True


class ExportableFpPrefilterYDelta(nn.Module):
    """ONNX 图：raw Y 输入，raw FP delta_y 输出。"""

    def __init__(self, weight_fp: torch.Tensor, bias_fp_norm: torch.Tensor, downscale_factor: int = 4) -> None:
        super().__init__()
        self.downscale_factor = int(downscale_factor)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale_factor)
        self.pixel_shuffle = nn.PixelShuffle(self.downscale_factor)
        self.register_buffer("weight_fp", weight_fp.to(torch.float32))
        self.register_buffer("bias_fp_raw", bias_fp_norm.to(torch.float32) * 255.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(torch.round(x[:, :1]), 0.0, 255.0)
        y_u = self.pixel_unshuffle(y)
        delta_u = F.conv2d(y_u, self.weight_fp, self.bias_fp_raw, stride=1, padding=1)
        return self.pixel_shuffle(delta_u)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().cpu().to(torch.float32)
    return {
        "shape": list(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "max_abs": float(t.abs().max().item()),
    }


def load_deploy_weight(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(state)}")
    if "processing.weight" not in state or "processing.bias" not in state:
        raise RuntimeError("Expected keys: processing.weight, processing.bias")
    weight = state["processing.weight"].detach().to(torch.float32)
    bias = state["processing.bias"].detach().to(torch.float32)
    if tuple(weight.shape) != (16, 16, 3, 3):
        raise RuntimeError(f"Unexpected processing.weight shape: {tuple(weight.shape)}")
    if tuple(bias.shape) != (16,):
        raise RuntimeError(f"Unexpected processing.bias shape: {tuple(bias.shape)}")
    return weight, bias


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


def check_export_module(export_model: ExportableFpPrefilterYDelta, device: torch.device) -> None:
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
        ref = export_model.pixel_shuffle(
            F.conv2d(y_u, export_model.weight_fp, export_model.bias_fp_raw, stride=1, padding=1)
        )
        max_abs = torch.max(torch.abs(ref - got)).item()
    if max_abs != 0.0:
        raise RuntimeError(f"Export module mismatch: max_abs={max_abs}")


def main() -> int:
    device = torch.device(DEVICE)
    weight_fp, bias_fp_norm = load_deploy_weight(WEIGHT_PATH)
    export_model = ExportableFpPrefilterYDelta(weight_fp, bias_fp_norm, downscale_factor=4).to(device)
    export_model.eval()

    if RUN_TORCH_CHECK:
        check_export_module(export_model, device)

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
        "source_checkpoint": os.path.abspath(WEIGHT_PATH),
        "model_name": "fudan_iccv_yan_2025_deploy_fp16",
        "weight_bits": "fp16_source_exported_as_fp32_onnx_constants",
        "bias_bits": "fp16_source_exported_as_fp32_onnx_constants",
        "downscale_factor": 4,
        "dummy_export_height": int(EXPORT_HEIGHT),
        "dummy_export_width": int(EXPORT_WIDTH),
        "dynamic_input_shape": bool(DYNAMIC_INPUT_SHAPE),
        "input": "raw 0..255 Y block, float32 NCHW shape [N,1,H,W], H/W must be divisible by 4",
        "output": "raw residual delta_y block, float32 NCHW shape [N,1,H,W]",
        "onnx_formula": [
            "Y_u = PixelUnshuffle4(round_clip_u8(Y))",
            "delta_u = Conv2D(Y_u, weight_fp, bias_fp_norm * 255)",
            "delta_y = PixelShuffle4(delta_u)",
        ],
        "external_formula": "Y_out_block = clip(round(Y_block + rate_block * delta_y_block), 0, 255)",
        "weight_fp": tensor_stats(weight_fp),
        "bias_fp_norm": tensor_stats(bias_fp_norm),
        "bias_fp_raw": tensor_stats(bias_fp_norm * 255.0),
    }
    stem = os.path.splitext(OUTPUT_ONNX_PATH)[0]
    with open(f"{stem}.export_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    add_onnx_metadata(OUTPUT_ONNX_PATH, meta)

    print(f"[DONE] ONNX written: {OUTPUT_ONNX_PATH}")
    print(f"[DONE] Export meta: {stem}.export_meta.json")
    print(f"[INFO] weight={meta['weight_fp']} bias_norm={meta['bias_fp_norm']} bias_raw={meta['bias_fp_raw']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
