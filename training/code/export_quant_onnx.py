"""将 W10+B13 QAT checkpoint 导出为 Y-plane ONNX。

推荐使用 `--raw_io` 导出最终部署模型：ONNX 输入/输出均为 float32 raw Y，
数值范围 `0..255`。图内部固定保存整数语义的 `q_w/q_b` 和 `1/2^shift`，
UV 不进入 ONNX，由外部推理脚本原样透传。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pred_int_qat import load_model, select_device


class ExportableQuantPrefilterY(nn.Module):
    """ONNX 友好的 Y-only 部署图，使用冻结后的整数参数。"""

    def __init__(
        self,
        q_w: torch.Tensor,
        q_b: torch.Tensor,
        shift: torch.Tensor,
        downscale_factor: int,
        raw_io: bool = False,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if q_w.ndim != 4 or q_w.shape[-2:] != (3, 3):
            raise ValueError(f"Expected q_w as OIHW 3x3, got {tuple(q_w.shape)}")
        if q_b.ndim != 1 or q_b.numel() != q_w.size(0):
            raise ValueError(f"q_b shape mismatch: {tuple(q_b.shape)} vs q_w={tuple(q_w.shape)}")
        if shift.ndim != 1 or shift.numel() != q_w.size(0):
            raise ValueError(f"shift shape mismatch: {tuple(shift.shape)} vs q_w={tuple(q_w.shape)}")

        self.downscale_factor = int(downscale_factor)
        self.raw_io = bool(raw_io)
        self.residual_scale = float(residual_scale)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale_factor)
        self.pixel_shuffle = nn.PixelShuffle(self.downscale_factor)

        self.register_buffer("q_w_float", q_w.to(torch.float32))
        self.register_buffer("q_b_float", q_b.to(torch.float32))
        inv_shift = torch.pow(torch.tensor(2.0, dtype=torch.float32), -shift.to(torch.float32))
        self.register_buffer("inv_shift", inv_shift.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # raw_io=True 是交付 ONNX 接口：输入为 raw Y 0..255。
        # raw_io=False 仅用于需要和归一化 PyTorch 路径做对比的场景。
        if self.raw_io:
            x_raw = torch.clamp(torch.round(x), 0.0, 255.0)
        else:
            x_raw = torch.clamp(torch.round(x * 255.0), 0.0, 255.0)
        y = x_raw[:, :1]
        y_u = self.pixel_unshuffle(y)
        acc = F.conv2d(y_u, self.q_w_float, self.q_b_float, stride=1, padding=1)
        delta = torch.round(acc * self.inv_shift)
        y_u_out = torch.clamp(torch.round(y_u + self.residual_scale * delta), 0.0, 255.0)
        y_out = self.pixel_shuffle(y_u_out)
        if self.raw_io:
            return y_out
        return y_out / 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将定点 QAT Prefilter Y-plane 模型导出为 ONNX。")
    parser.add_argument("--config", default="configs/w10_b13_qat.yaml")
    parser.add_argument(
        "--checkpoint",
        default="models/w10_b13_best.pt",
    )
    parser.add_argument(
        "--output",
        default="models/w10_b13_raw.onnx",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true", help="导出动态 H/W 轴。")
    parser.add_argument("--raw_io", action="store_true", help="导出 raw 0..255 float 输入/输出接口。")
    parser.add_argument("--check", action="store_true", help="保存前执行 PyTorch 原模型与导出模块的一致性检查。")
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=1.0,
        help="整数残差加回 Y 之前使用的缩放系数。",
    )
    return parser.parse_args()


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    tensor_i64 = tensor.detach().cpu().to(torch.int64)
    return {
        "shape": list(tensor_i64.shape),
        "min": int(tensor_i64.min().item()),
        "max": int(tensor_i64.max().item()),
        "max_abs": int(tensor_i64.abs().max().item()),
    }


def save_sidecar_files(output_path: Path, params: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    stem = output_path.with_suffix("")
    np.savez_compressed(
        stem.with_suffix(".int_params.npz"),
        q_w=params["q_w"].numpy().astype(np.int32),
        q_b=params["q_b"].numpy().astype(np.int32),
        shift=params["shift"].numpy().astype(np.int32),
    )
    with stem.with_suffix(".export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def add_onnx_metadata(output_path: Path, meta: dict[str, Any]) -> None:
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


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    qat_model = load_model(args.config, args.checkpoint, device)
    params = qat_model.export_int_parameters()

    export_model = ExportableQuantPrefilterY(
        q_w=params["q_w"],
        q_b=params["q_b"],
        shift=params["shift"],
        downscale_factor=qat_model.downscale_factor,
        raw_io=args.raw_io,
        residual_scale=args.residual_scale,
    ).to(device)
    export_model.eval()

    dummy = torch.zeros(1, 1, args.height, args.width, dtype=torch.float32, device=device)
    if args.check and args.residual_scale == 1.0:
        with torch.no_grad():
            generator = torch.Generator(device=device)
            generator.manual_seed(123)
            raw_y = torch.randint(
                low=0,
                high=256,
                size=dummy.shape,
                generator=generator,
                device=device,
                dtype=torch.int32,
            ).to(torch.float32)
            if args.raw_io:
                ref = qat_model(raw_y / 255.0) * 255.0
                got = export_model(raw_y)
            else:
                norm_y = raw_y / 255.0
                ref = qat_model(norm_y)
                got = export_model(norm_y)
        max_abs = torch.max(torch.abs(ref - got)).item()
        if max_abs != 0.0:
            raise RuntimeError(f"Export module mismatch before ONNX export: max_abs={max_abs}")
    elif args.check:
        print("[WARN] --check skipped because residual_scale != 1.0")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    meta = {
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_config": str(Path(args.config).resolve()),
        "weight_bits": int(qat_model.cfg.weight_bits),
        "bias_bits": int(qat_model.cfg.bias_bits),
        "downscale_factor": int(qat_model.downscale_factor),
        "input": (
            "raw 0..255 Y plane, float32 NCHW shape [N,1,H,W]"
            if args.raw_io
            else "normalized Y plane, NCHW shape [N,1,H,W]"
        ),
        "output": (
            "raw 0..255 filtered Y plane, float32 NCHW shape [N,1,H,W]"
            if args.raw_io
            else "normalized filtered Y plane, NCHW shape [N,1,H,W]"
        ),
        "raw_io": bool(args.raw_io),
        "residual_scale": float(args.residual_scale),
        "only_train_y": True,
        "q_w": tensor_stats(params["q_w"]),
        "q_b": tensor_stats(params["q_b"]),
        "shift": tensor_stats(params["shift"]),
        "integer_formula": "delta = round((conv(pixel_unshuffle(Y_raw), q_w) + q_b) / 2^shift)",
        "residual_formula": "Y_u_out = clip(round(Y_u + residual_scale * delta), 0, 255)",
        "bias_domain": "q_b is quantized from raw-domain bias_fp_raw = bias_norm * 255",
        "rounding_note": "PyTorch/ONNX round semantics must match the target runtime for bit-exact deployment.",
        "note": "ONNX Conv constants are float32 but are frozen from exported int q_w/q_b/shift.",
    }
    save_sidecar_files(output_path, params, meta)
    add_onnx_metadata(output_path, meta)

    print(f"[DONE] ONNX written: {output_path}")
    print(f"[DONE] Integer sidecar: {output_path.with_suffix('').with_suffix('.int_params.npz')}")
    print(f"[DONE] Export meta: {output_path.with_suffix('').with_suffix('.export_meta.json')}")
    print(f"[INFO] q_w={meta['q_w']} q_b={meta['q_b']} shift={meta['shift']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
