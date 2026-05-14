# 部署说明

本目录包含当前 AI Filter noedge 版本的部署文件，包括 residual ONNX 模型、整数参数 sidecar、导出脚本、一致性验证摘要和整帧 YUV 推理工具。

当前正式提交的部署修正版对应 `调试滤波工具v1`。`调试滤波工具v2` 仅作为备用版本保留，不作为当前默认 deployment。

## 当前推荐模型

当前正式部署工具默认模型：

```text
debug_filter_tool/models/task_qat_w10_b12_noedge_delta_raw_dynamic.onnx
```

模型配置：

```text
tool version: 调试滤波工具v1
experiment: task_qat_w10_b12_noedge
input format: 8-bit NV12
训练主线: noedge，不启用 EdgeConsistencyLoss
评估主线: noedge 最终选型不使用 selective_score
```

`noedge` 是当前实际部署推荐线。选择原因是实际视频编码后，尤其 QP 较大时，noedge 模型的综合观感和编码后效果更好；未编码前它会相对更接近 reference filtering target。

## 目录结构

```text
models/
  task_qat_w10_b12_noedge_delta_raw_dynamic.onnx
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.int_params.npz
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.export_meta.json
  其他对比模型 ONNX

infer_block_rate_yuv.py
  当前默认推理脚本，来自调试滤波工具v1，固定按 8-bit NV12 解析输入。

debug_filter_tool/
  当前正式调试工具，对应调试滤波工具v1。

backup_debug_filter_tool_v2/
  备用调试工具，对应调试滤波工具v2，不作为当前默认 deployment。

outputs/*/summary.*
  ONNX/PyTorch 一致性验证摘要。

rate_maps/
  block-rate 实验保留的示例 rate map。

EVALUATION_SUMMARY.md
  部署侧模型选型和指标摘要。
```

大体积原始 YUV 和完整输出 YUV 不放在本目录内。需要推理时，可将本地输入放到 `data/`，或通过 `--input` 指定实际路径。

## ONNX 语义

ONNX 输出的是 Y 域残差，不在图内加回原始 Y。

```text
input  = raw 0..255 Y, float32 NCHW [N,1,H,W]
output = raw residual delta_y, float32 NCHW [N,1,H,W]
```

ONNX 内部整数路径：

```text
Y_u     = PixelUnshuffle4(Y)
acc     = Conv2D(Y_u, q_w, q_b)
delta_u = round(acc / 2^10)
delta_y = PixelShuffle4(delta_u)
```

外部推理合成：

```text
Y_out = clip(round(Y + rate * delta_y), 0, 255)
```

当前正式脚本按 8-bit NV12 计算帧布局，只解析 Y 平面，UV 字节不参与模型计算并原样透传。

## 推理命令

安装最小运行依赖：

```bash
pip install numpy onnxruntime
```

整帧推理：

```bash
python infer_block_rate_yuv.py \
  --width 2560 \
  --height 1440 \
  --input data/kaideo_2560x1440_yuv420p_0.yuv \
  --output outputs/kaideo_2560x1440_yuv420p_0_rate1.yuv
```

调整 residual 强度：

```bash
python infer_block_rate_yuv.py \
  --width 2560 \
  --height 1440 \
  --input data/kaideo_2560x1440_yuv420p_0.yuv \
  --output outputs/rate05.yuv \
  --rate 0.5
```

指定其他 ONNX：

```bash
python infer_block_rate_yuv.py \
  --model models/task_qat_w10_b12_noedge_delta_raw_dynamic.onnx \
  --input data/input.yuv \
  --output outputs/output.yuv
```

## 验证记录

`outputs/kaideo_full_onnx_pytorch_compare_rate1/summary.md` 记录了完整视频 ONNX/PyTorch 一致性检查：

| model | frames | max abs Y diff | diff Y pixels |
|---|---:|---:|---:|
| `edge_w10_b13` | 100 | 0 | 0 |
| `teacher_qat_w10_b12` | 100 | 0 | 0 |
| `fudan_fp16` | 100 | 1 | 615 |

当前正式调试工具使用 `task_qat_w10_b12_noedge_delta_raw_dynamic.onnx`。`task_qat_w11_b13_noedge_shift10` 和 `backup_debug_filter_tool_v2/` 作为备用和对照材料保留。训练指标与模型选择细节见 `EVALUATION_SUMMARY.md`。
