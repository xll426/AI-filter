#!/usr/bin/env python3
"""全量视频 ONNX vs PyTorch 一致性验证。

默认验证这个视频：

    /mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/data/kaideo_2560x1440_yuv420p_0.yuv

对三个模型分别执行：

    1. ONNX block batch 推理，写出完整 YUV。
    2. PyTorch block batch 推理，写出完整 YUV。
    3. 比较 ONNX 输出和 PyTorch 输出的 Y 平面逐像素差异。
    4. 检查 UV 字节是否原样透传。

这里的 PyTorch 路径不是训练模型前向，而是与 ONNX 完全同构的 residual 图：

    raw Y -> PixelUnshuffle4 -> Conv -> delta_y -> Y + rate * delta_y
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 配置区：只改这里
# =============================================================================

VIDEO = {
    "input_yuv": (
        "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/data/"
        "kaideo_2560x1440_yuv420p_0.yuv"
    ),
    "width": 2560,
    "height": 1440,
    "format": "yuv420p",
}

MODELS = {
    "edge_w10_b13": {
        "kind": "int",
        "onnx": (
            "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/models/"
            "edge_w10_b13_delta_raw_dynamic.onnx"
        ),
        "int_params": (
            "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/models/"
            "edge_w10_b13_delta_raw_dynamic.int_params.npz"
        ),
    },
    "teacher_qat_w10_b12": {
        "kind": "int",
        "onnx": (
            "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/models/"
            "teacher_qat_w10_b12_delta_raw_dynamic.onnx"
        ),
        "int_params": (
            "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/models/"
            "teacher_qat_w10_b12_delta_raw_dynamic.int_params.npz"
        ),
    },
    "fudan_fp16": {
        "kind": "fp",
        "onnx": (
            "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/models/"
            "fudan_iccv_yan_2025_deploy_fp16_delta_raw_dynamic.onnx"
        ),
        "weight": "/mnt/d/fudan/prefilter_clean/weights/iccv_yan_2025_deploy_fp16.pth",
    },
}

SELECT_MODELS = ["edge_w10_b13", "teacher_qat_w10_b12", "fudan_fp16"]

OUTPUT_ROOT = (
    "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/"
    "outputs/kaideo_full_onnx_pytorch_compare_rate1"
)

SUMMARY_JSON = (
    "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/"
    "outputs/kaideo_full_onnx_pytorch_compare_rate1/summary.json"
)
SUMMARY_MD = (
    "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/"
    "outputs/kaideo_full_onnx_pytorch_compare_rate1/summary.md"
)

BITDEPTH = 8
DEFAULT_RATE = 1.0
BLOCK_SIZE = 32

# 单次送入 ONNX/PyTorch 的 block 数。最后一批不足该数量也会正常推理。
BLOCK_BATCH_SIZE = 1024

# None 表示跑完整视频。这里按用户要求默认跑完整 kaideo 100 帧。
MAX_FRAMES = None

DEVICE = "cpu"
ORT_PROVIDERS = ["CPUExecutionProvider"]


@dataclass(frozen=True)
class VideoLayout:
    """单帧 YUV 的字节布局。

    dtype:
        Y 平面的 numpy 类型。当前只支持 8-bit，所以固定是 uint8。
    y_bytes:
        一帧 Y 平面的字节数，等于 width * height。
    chroma_bytes:
        一帧 UV 部分的总字节数。nv12/yuv420p 是 width * height / 2。
    frame_bytes:
        一整帧的字节数，等于 y_bytes + chroma_bytes。
    """

    dtype: np.dtype
    y_bytes: int
    chroma_bytes: int
    frame_bytes: int


@dataclass(frozen=True)
class BlockInfo:
    """一个 Y block 在整帧里的位置和有效尺寸。"""

    top: int
    left: int
    height: int
    width: int


class QuantDeltaTorch(nn.Module):
    """PyTorch 版整数 residual 图，与整数 ONNX 同构。"""

    def __init__(self, q_w: np.ndarray, q_b: np.ndarray, shift: np.ndarray) -> None:
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.register_buffer("q_w", torch.from_numpy(q_w.astype(np.float32)))
        self.register_buffer("q_b", torch.from_numpy(q_b.astype(np.float32)))
        inv_shift = np.power(2.0, -shift.astype(np.float32)).reshape(1, -1, 1, 1)
        self.register_buffer("inv_shift", torch.from_numpy(inv_shift.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(torch.round(x[:, :1]), 0.0, 255.0)
        y_u = self.pixel_unshuffle(y)
        acc = F.conv2d(y_u, self.q_w, self.q_b, stride=1, padding=1)
        delta_u = torch.round(acc * self.inv_shift)
        return self.pixel_shuffle(delta_u)


class FpDeltaTorch(nn.Module):
    """PyTorch 版 FP residual 图，与复旦 FP16 ONNX 同构。"""

    def __init__(self, weight: torch.Tensor, bias_norm: torch.Tensor) -> None:
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.register_buffer("weight", weight.detach().to(torch.float32))
        self.register_buffer("bias_raw", bias_norm.detach().to(torch.float32) * 255.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(torch.round(x[:, :1]), 0.0, 255.0)
        y_u = self.pixel_unshuffle(y)
        delta_u = F.conv2d(y_u, self.weight, self.bias_raw, stride=1, padding=1)
        return self.pixel_shuffle(delta_u)


def get_yuv_layout(width: int, height: int, yuv_format: str, bitdepth: int) -> VideoLayout:
    if bitdepth != 8:
        raise ValueError("当前脚本只支持 8-bit YUV。")
    y_bytes = width * height
    fmt = yuv_format.lower()
    if fmt == "yuv400p":
        chroma_bytes = 0
    elif fmt in {"nv12", "yuv420p"}:
        chroma_bytes = width * height // 2
    else:
        raise ValueError(f"不支持的 YUV format: {yuv_format}")
    return VideoLayout(np.dtype(np.uint8), y_bytes, chroma_bytes, y_bytes + chroma_bytes)


def get_frame_count(input_yuv: str, frame_bytes: int) -> int:
    file_bytes = os.path.getsize(input_yuv)
    if file_bytes % frame_bytes != 0:
        raise ValueError(f"YUV 文件大小不是完整帧：file_bytes={file_bytes}, frame_bytes={frame_bytes}")
    total = file_bytes // frame_bytes
    return total if MAX_FRAMES is None else min(total, int(MAX_FRAMES))


def iter_block_infos(width: int, height: int) -> list[BlockInfo]:
    infos: list[BlockInfo] = []
    for top in range(0, height, BLOCK_SIZE):
        h = min(BLOCK_SIZE, height - top)
        for left in range(0, width, BLOCK_SIZE):
            w = min(BLOCK_SIZE, width - left)
            infos.append(BlockInfo(top=top, left=left, height=h, width=w))
    return infos


def pad_block(block: np.ndarray) -> np.ndarray:
    h, w = block.shape
    if h == BLOCK_SIZE and w == BLOCK_SIZE:
        return block.astype(np.float32, copy=False)
    return np.pad(block, ((0, BLOCK_SIZE - h), (0, BLOCK_SIZE - w)), mode="edge").astype(np.float32, copy=False)


def make_block_batch(y: np.ndarray, infos: list[BlockInfo]) -> np.ndarray:
    blocks = []
    for info in infos:
        block = y[info.top : info.top + info.height, info.left : info.left + info.width]
        blocks.append(pad_block(block))
    return np.stack(blocks, axis=0).astype(np.float32, copy=False)[:, None, :, :]


def build_torch_model(model_cfg: dict[str, str]) -> nn.Module:
    if model_cfg["kind"] == "int":
        params = np.load(model_cfg["int_params"])
        model = QuantDeltaTorch(params["q_w"], params["q_b"], params["shift"])
    elif model_cfg["kind"] == "fp":
        state = torch.load(model_cfg["weight"], map_location="cpu")
        model = FpDeltaTorch(state["processing.weight"], state["processing.bias"])
    else:
        raise ValueError(f"Unknown model kind: {model_cfg['kind']}")
    model.eval()
    return model.to(torch.device(DEVICE))


def run_onnx_delta(session: ort.InferenceSession, input_name: str, x: np.ndarray) -> np.ndarray:
    delta = session.run(None, {input_name: x})[0]
    if delta.ndim != 4 or delta.shape[1] != 1:
        raise RuntimeError(f"ONNX 输出应为 [N,1,H,W]，当前是 {delta.shape}")
    return delta[:, 0]


@torch.no_grad()
def run_torch_delta(model: nn.Module, x: np.ndarray) -> np.ndarray:
    xt = torch.from_numpy(x).to(torch.device(DEVICE))
    delta = model(xt).cpu().numpy()
    return delta[:, 0]


def assemble_frame(y: np.ndarray, infos: list[BlockInfo], deltas: list[np.ndarray]) -> np.ndarray:
    out_y = y.astype(np.float32, copy=True)
    cursor = 0
    for batch_delta in deltas:
        for delta in batch_delta:
            info = infos[cursor]
            y_block = y[info.top : info.top + info.height, info.left : info.left + info.width]
            delta_valid = delta[: info.height, : info.width]
            out_y[info.top : info.top + info.height, info.left : info.left + info.width] = (
                y_block.astype(np.float32) + float(DEFAULT_RATE) * delta_valid
            )
            cursor += 1
    return np.clip(np.rint(out_y), 0, 255).astype(np.uint8)


def process_frame_pair(
    y: np.ndarray,
    infos: list[BlockInfo],
    session: ort.InferenceSession,
    input_name: str,
    torch_model: nn.Module,
) -> tuple[np.ndarray, np.ndarray]:
    onnx_deltas = []
    torch_deltas = []
    for start in range(0, len(infos), BLOCK_BATCH_SIZE):
        batch_infos = infos[start : start + BLOCK_BATCH_SIZE]
        x = make_block_batch(y, batch_infos)
        onnx_deltas.append(run_onnx_delta(session, input_name, x))
        torch_deltas.append(run_torch_delta(torch_model, x))
    return assemble_frame(y, infos, onnx_deltas), assemble_frame(y, infos, torch_deltas)


def compare_model(model_name: str, model_cfg: dict[str, str]) -> dict[str, object]:
    input_yuv = str(VIDEO["input_yuv"])
    width = int(VIDEO["width"])
    height = int(VIDEO["height"])
    yuv_format = str(VIDEO["format"])
    layout = get_yuv_layout(width, height, yuv_format, BITDEPTH)
    frame_count = get_frame_count(input_yuv, layout.frame_bytes)
    infos = iter_block_infos(width, height)

    session = ort.InferenceSession(model_cfg["onnx"], providers=ORT_PROVIDERS)
    input_name = session.get_inputs()[0].name
    torch_model = build_torch_model(model_cfg)

    model_out_dir = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_yuv))[0]
    onnx_yuv = os.path.join(model_out_dir, f"{stem}_onnx_rate{DEFAULT_RATE:g}.yuv")
    torch_yuv = os.path.join(model_out_dir, f"{stem}_pytorch_rate{DEFAULT_RATE:g}.yuv")

    max_abs_y_diff = 0
    sum_abs_y_diff = 0
    diff_pixel_count = 0
    total_y_count = 0
    uv_mismatch = False

    print(f"model={model_name}")
    print(f"input={input_yuv}")
    print(f"onnx_output={onnx_yuv}")
    print(f"pytorch_output={torch_yuv}")
    print(f"frames={frame_count}, blocks_per_frame={len(infos)}, block_batch={BLOCK_BATCH_SIZE}")

    with open(input_yuv, "rb") as in_f, open(onnx_yuv, "wb") as onnx_f, open(torch_yuv, "wb") as torch_f:
        for frame_idx in range(frame_count):
            y_raw = in_f.read(layout.y_bytes)
            chroma = in_f.read(layout.chroma_bytes)
            if len(y_raw) != layout.y_bytes or len(chroma) != layout.chroma_bytes:
                raise EOFError(f"第 {frame_idx} 帧读取失败。")

            y = np.frombuffer(y_raw, dtype=layout.dtype).reshape(height, width)
            onnx_y, torch_y = process_frame_pair(y, infos, session, input_name, torch_model)

            onnx_f.write(onnx_y.tobytes())
            onnx_f.write(chroma)
            torch_f.write(torch_y.tobytes())
            torch_f.write(chroma)

            diff = onnx_y.astype(np.int16) - torch_y.astype(np.int16)
            abs_diff = np.abs(diff)
            max_abs_y_diff = max(max_abs_y_diff, int(abs_diff.max()))
            sum_abs_y_diff += int(abs_diff.sum())
            diff_pixel_count += int(np.count_nonzero(abs_diff))
            total_y_count += int(abs_diff.size)

            # UV 两边都直接写 chroma，这里保留显式状态，方便 summary 说明。
            uv_mismatch = uv_mismatch or False
            print(f"frame {frame_idx + 1}/{frame_count}")

    expected_size = frame_count * layout.frame_bytes
    onnx_size = os.path.getsize(onnx_yuv)
    torch_size = os.path.getsize(torch_yuv)
    if onnx_size != expected_size or torch_size != expected_size:
        raise RuntimeError(f"输出大小错误：expected={expected_size}, onnx={onnx_size}, pytorch={torch_size}")

    mean_abs_y_diff = sum_abs_y_diff / max(total_y_count, 1)
    result = {
        "model": model_name,
        "input_yuv": input_yuv,
        "onnx": model_cfg["onnx"],
        "onnx_output": onnx_yuv,
        "pytorch_output": torch_yuv,
        "frames": frame_count,
        "width": width,
        "height": height,
        "format": yuv_format,
        "block_size": BLOCK_SIZE,
        "block_batch_size": BLOCK_BATCH_SIZE,
        "rate": DEFAULT_RATE,
        "output_size_expected": expected_size,
        "onnx_output_size": onnx_size,
        "pytorch_output_size": torch_size,
        "max_abs_y_diff": max_abs_y_diff,
        "mean_abs_y_diff": mean_abs_y_diff,
        "diff_y_pixels": diff_pixel_count,
        "total_y_pixels": total_y_count,
        "diff_y_ratio": diff_pixel_count / max(total_y_count, 1),
        "uv_mismatch": uv_mismatch,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def write_summary(results: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(SUMMARY_JSON), exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    lines = [
        "# ONNX vs PyTorch Full Video Compare",
        "",
        f"input: `{VIDEO['input_yuv']}`",
        "",
        "| model | frames | max abs Y diff | mean abs Y diff | diff Y ratio | ONNX output | PyTorch output |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['model']} | {item['frames']} | {item['max_abs_y_diff']} | "
            f"{item['mean_abs_y_diff']:.8g} | {item['diff_y_ratio']:.8g} | "
            f"`{item['onnx_output']}` | `{item['pytorch_output']}` |"
        )
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    results = []
    for model_name in SELECT_MODELS:
        if model_name not in MODELS:
            raise ValueError(f"未知模型：{model_name}")
        results.append(compare_model(model_name, MODELS[model_name]))
    write_summary(results)
    print(f"[DONE] summary json: {SUMMARY_JSON}")
    print(f"[DONE] summary md: {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
