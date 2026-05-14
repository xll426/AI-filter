#!/usr/bin/env python3
"""生成 infer_block_rate_yuv.py 使用的固定结构 rate map。

所有参数都写在脚本开头，不使用命令行参数。
输出固定为 .npy，shape 固定为 [FRAME_COUNT, block_rows, block_cols]。
"""
from __future__ import annotations

import os

import numpy as np


# =============================================================================
# Rate map 配置
# =============================================================================

# 视频尺寸。你的目标视频使用 WIDTH=3184, HEIGHT=2160。
WIDTH = 3184
HEIGHT = 2160
BLOCK_SIZE = 32

# 视频帧数。这个值必须和推理脚本实际处理的 frame_count 一致。
# 如果 infer_block_rate_yuv.py 里 MAX_FRAMES=None，则这里填整个 yuv 文件的总帧数。
FRAME_COUNT = 1

# 旧版可变 rate map 示例。当前 infer_block_rate_yuv.py 默认 rate 固定为 1.0，
# 不再读取这里生成的 npy；保留该脚本仅用于以后重新启用分块 rate map。
OUTPUT_RATE_MAP = (
    "/mnt/d/fudan/prefilter_clean/deliver_w10_b13_delta_block_rate_onnx/rate_maps/"
    "rate_3184x2160_frames1.npy"
)

# 所有帧、所有 block 的默认倍率。
# rate=0 表示不滤波，rate=1 表示完整使用模型残差。
DEFAULT_RATE = 1.0

# 手动修改某些 block 的 rate，格式固定为：
#     (frame_idx, block_y, block_x, rate)
#
# block_y/block_x 从 0 开始。
# 对 3184x2160：block_y 范围 0..67，block_x 范围 0..99。
MANUAL_RATES = [
    # (0, 0, 0, 0.5),
    # (0, 0, 1, -0.5),
    # (0, 0, 2, 2.0),
]


def main() -> int:
    block_rows = (HEIGHT + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_cols = (WIDTH + BLOCK_SIZE - 1) // BLOCK_SIZE

    rates = np.full((FRAME_COUNT, block_rows, block_cols), DEFAULT_RATE, dtype=np.float32)
    for frame_idx, block_row, block_col, rate in MANUAL_RATES:
        rates[frame_idx, block_row, block_col] = float(rate)

    os.makedirs(os.path.dirname(OUTPUT_RATE_MAP), exist_ok=True)
    np.save(OUTPUT_RATE_MAP, rates)
    print(f"[DONE] wrote: {OUTPUT_RATE_MAP}")
    print(f"[INFO] shape: {rates.shape}")
    print(f"[INFO] block grid: {block_rows}x{block_cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
