# Debug Filter Tool

这是从早期 `调试滤波工具v0` 修正后的精简版本。核心修正是整帧 Y 一次输入 ONNX，替代早期 32x32 block 分块推理，避免 block 边界拼接伪影。

## Original v2 Notes

## 内容

```text
models/   ONNX 模型和整数参数 sidecar
data/     输入 sample YUV
outputs/  推理输出 YUV
infer_block_rate_yuv.py  推理脚本
```

默认模型：

```text
models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx
```

模型语义：

```text
input  = raw 0..255 Y, float32 NCHW [N,1,H,W]
output = raw residual delta_y, float32 NCHW [N,1,H,W]

Y_u     = PixelUnshuffle4(Y)
acc     = Conv2D(Y_u, q_w, q_b)
delta_u = round(acc / 2^10)
delta_y = PixelShuffle4(delta_u)
```

注意：ONNX 输入端不做 `clip/round`。推理脚本从 8-bit YUV 读取的 Y 本身就是 `0..255`，转成 float32 后直接送入模型。

整数参数：

```text
q_w: signed W11, range -1024..1023, actual -466..76
q_b: signed B13, range -4096..4095, actual -1989..-119
shift: 10
```

## 输入 sample

```text
data/kaideo_2560x1440_yuv420p_0.yuv
```

视频参数：

```text
width    = 2560
height   = 1440
format   = yuv420p
frames   = 100
bitdepth = 8
rate     = 1
```

脚本只处理 Y 平面，chroma 字节不解析并原样透传。因此同尺寸 8-bit `yuv420p`/`nv12` 在本工具里只要 Y 平面一致都可处理。

## 安装环境

建议使用 Python 3.10 或更新版本。

```bash
pip install numpy onnxruntime
```

## 推理流程

```text
读取一帧 8-bit 4:2:0 YUV
拆出 Y 平面，chroma 字节保留
整帧 Y 一次输入 ONNX，得到整帧 delta_y
Y_out = clip(round(Y + rate * delta_y), 0, 255)
拼回 Y_out + 原始 chroma
写出完整 YUV
```

## 运行命令

默认 sample：

```bash
cd deployment/debug_filter_tool
python infer_block_rate_yuv.py
```

指定输入输出：

```bash
python infer_block_rate_yuv.py \
  --width 2560 \
  --height 1440 \
  --input data/kaideo_2560x1440_yuv420p_0.yuv \
  --output outputs/kaideo_2560x1440_yuv420p_0_w11_b13_shift10_rate1.yuv
```

调整残差倍率：

```bash
python infer_block_rate_yuv.py --rate 0.5 --output outputs/rate05.yuv
```
