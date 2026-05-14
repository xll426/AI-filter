# 调试滤波工具 v1

当前 deployment 的正式调试工具对应本版本。`调试滤波工具v2` 仅作为备用版本保留在 `../backup_debug_filter_tool_v2/`。

## 内容

```text
models/   ONNX 模型
data/     输入 sample NV12 YUV
outputs/  推理输出 YUV
infer_block_rate_yuv.py  推理脚本
```

输入 sample：

```text
data/kaideo_2560x1440_yuv420p_0.yuv
```

视频参数：

```text
width  = 2560
height = 1440
format = nv12
frames = 100
bitdepth = 8
rate   = 1
```

当前工具只支持 8-bit NV12 输入。

## 安装环境

建议使用 Python 3.10 或更新版本。

需要安装的第三方依赖只有：

```bash
pip install numpy onnxruntime
```

## 推理流程


脚本流程：

```text
读取一帧 NV12
拆出 Y 平面，UV 字节保留
整帧 Y 一次输入 ONNX，得到整帧 delta_y
Y_out = clip(round(Y + rate * delta_y), 0, 255)
拼回 Y_out + 原始 UV
写出完整 NV12 YUV
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
  --output outputs/kaideo_2560x1440_yuv420p_0_rate1.yuv
```

`rate` 默认是 1，需要修改时：

```bash
python infer_block_rate_yuv.py --rate 0.5 --output outputs/rate05.yuv
```
