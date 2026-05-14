#!/usr/bin/env python3
"""调试滤波工具 v2：单模型整帧 ONNX residual 推理。

默认使用 task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx。
输入按 8-bit 4:2:0 YUV 解析；脚本只读取 Y，chroma 字节原样透传。
ONNX 输出 delta_y 后在原始 Y 域加回：

    Y_out = clip(round(Y + rate * delta_y), 0, 255)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODEL_NAME = "task_qat_w11_b13_noedge_shift10"
MODEL_FILENAME = "task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx"
BITDEPTH = 8
YUV_FORMAT = "yuv420p"
ORT_PROVIDERS = ["CPUExecutionProvider"]


@dataclass(frozen=True)
class VideoLayout:
    """一帧 8-bit 4:2:0 YUV 的字节布局。

    dtype:
        Y 平面和 UV 平面的 numpy 数据类型。当前只支持 8-bit，所以固定是 uint8。
    y_bytes:
        一帧 Y 平面的字节数，等于 width * height。
    chroma_bytes:
        一帧 chroma 字节数。yuv420p/nv12 都是 width * height / 2。
        当前脚本不解析 chroma 排布，只原样透传。
    frame_bytes:
        一整帧的字节数，等于 y_bytes + chroma_bytes。
    """

    dtype: np.dtype
    y_bytes: int
    chroma_bytes: int
    frame_bytes: int


def script_root_dir() -> Path:
    """返回当前推理脚本所在目录，用来定位默认模型、输入和输出路径。"""
    return Path(__file__).resolve().parent


ROOT_DIR = script_root_dir()
DEFAULT_INPUT_YUV = ROOT_DIR / "data" / "kaideo_2560x1440_yuv420p_0.yuv"
DEFAULT_OUTPUT_YUV = ROOT_DIR / "outputs" / f"{DEFAULT_INPUT_YUV.stem}_w11_b13_shift10_rate1.yuv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / MODEL_FILENAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fudan prefilter v2 ONNX YUV420 推理工具。")
    parser.add_argument("--width", "-w", type=int, default=2560, help="输入 YUV 宽度，默认 2560。")
    parser.add_argument("--height", "-H", type=int, default=1440, help="输入 YUV 高度，默认 1440。")
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT_YUV), help="输入 8-bit 4:2:0 .yuv 路径。")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT_YUV), help="输出 .yuv 文件路径。")
    parser.add_argument("--rate", "-r", type=float, default=1.0, help="残差倍率，默认 1。")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="ONNX 模型路径，默认使用内置模型。")
    return parser.parse_args()


def get_yuv_layout(width: int, height: int) -> VideoLayout:
    if BITDEPTH != 8:
        raise ValueError("当前工具只支持 8-bit 4:2:0 YUV。")
    if width <= 0 or height <= 0:
        raise ValueError(f"宽高必须为正数：width={width}, height={height}")
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("4:2:0 YUV 要求宽高都是偶数。")

    y_bytes = width * height
    chroma_bytes = width * height // 2
    return VideoLayout(np.dtype(np.uint8), y_bytes, chroma_bytes, y_bytes + chroma_bytes)


def get_frame_count(input_yuv: Path, frame_bytes: int) -> int:
    file_bytes = input_yuv.stat().st_size
    if file_bytes % frame_bytes != 0:
        raise ValueError(f"YUV 文件大小不是完整 4:2:0 帧：file_bytes={file_bytes}, frame_bytes={frame_bytes}")
    return file_bytes // frame_bytes


def infer_full_frame_delta(session: ort.InferenceSession, input_name: str, y: np.ndarray) -> np.ndarray:
    x = y.astype(np.float32, copy=False)[None, None, :, :]
    delta = session.run(None, {input_name: x})[0]
    if delta.ndim != 4 or delta.shape != x.shape:
        raise RuntimeError(f"ONNX 输出应为 {x.shape}，当前是 {delta.shape}")
    return delta[0, 0]


def process_one_frame(session: ort.InferenceSession, input_name: str, y: np.ndarray, rate: float) -> np.ndarray:
    delta_y = infer_full_frame_delta(session, input_name, y)
    out_y = y.astype(np.float32) + float(rate) * delta_y
    return np.clip(np.rint(out_y), 0, 255).astype(np.uint8)


def run(
    *,
    model_path: Path,
    input_yuv: Path,
    output_yuv: Path,
    width: int,
    height: int,
    rate: float,
) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX 不存在：{model_path}")
    if not input_yuv.is_file():
        raise FileNotFoundError(f"输入 YUV 不存在：{input_yuv}")

    layout = get_yuv_layout(width, height)
    frame_count = get_frame_count(input_yuv, layout.frame_bytes)
    output_yuv.parent.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(str(model_path), providers=ORT_PROVIDERS)
    input_name = session.get_inputs()[0].name

    print(f"model={MODEL_NAME}")
    print(f"onnx={model_path}")
    print(f"input={input_yuv}")
    print(f"output={output_yuv}")
    print(f"frames={frame_count}, size={width}x{height}, format={YUV_FORMAT}, rate={rate:g}, mode=full-frame")

    with input_yuv.open("rb") as in_f, output_yuv.open("wb") as out_f:
        for frame_idx in range(frame_count):
            y_raw = in_f.read(layout.y_bytes)
            chroma = in_f.read(layout.chroma_bytes)
            if len(y_raw) != layout.y_bytes or len(chroma) != layout.chroma_bytes:
                raise EOFError(f"第 {frame_idx} 帧读取失败。")

            y = np.frombuffer(y_raw, dtype=layout.dtype).reshape(height, width)
            out_y = process_one_frame(session, input_name, y, rate)

            out_f.write(out_y.tobytes())
            out_f.write(chroma)
            print(f"frame {frame_idx + 1}/{frame_count}")

    expected_size = frame_count * layout.frame_bytes
    output_size = output_yuv.stat().st_size
    if output_size != expected_size:
        raise RuntimeError(f"输出大小错误：expected={expected_size}, output={output_size}")
    print("done")


def main() -> int:
    args = parse_args()
    input_yuv = Path(args.input).resolve()
    output_yuv = Path(args.output).resolve()
    if output_yuv.exists() and output_yuv.is_dir():
        raise ValueError(f"--output 必须是明确的 .yuv 文件路径，不能是目录：{output_yuv}")
    run(
        model_path=Path(args.model).resolve(),
        input_yuv=input_yuv,
        output_yuv=output_yuv,
        width=int(args.width),
        height=int(args.height),
        rate=float(args.rate),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
