# 部署说明

本目录包含当前 AI Filter noedge 版本的部署文件，包括 residual ONNX 模型、整数参数 sidecar、导出脚本、一致性验证摘要和整帧 YUV 推理工具。

## 当前推荐模型

推荐部署模型：

```text
models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx
models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.int_params.npz
models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.export_meta.json
```

模型配置：

```text
experiment: task_qat_w11_b13_noedge_shift10
weight bits: W11
bias bits: B13
shift: 10
训练主线: noedge，不启用 EdgeConsistencyLoss
评估主线: noedge 最终选型不使用 selective_score
```

`noedge` 模型是当前实际部署推荐线。选择原因是实际视频编码后，尤其 QP 较大时，noedge 模型的综合观感和编码后效果更好；未编码前它会相对更接近 reference filtering target。

## 目录结构

```text
models/
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.int_params.npz
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.export_meta.json
  其他对比模型 ONNX

infer_block_rate_yuv.py
  主推理脚本，整帧 Y 输入 ONNX，输出 residual 后在脚本中加回原始 Y。

debug_filter_tool/
  精简调试工具，使用同样的整帧推理逻辑。

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

脚本只解析 Y 平面，chroma 字节不参与模型计算并原样透传。对于 8-bit 4:2:0 输入，`yuv420p` 和 `nv12` 的单帧字节数一致，只要宽高和 Y 平面正确，chroma 会按原字节拼回。

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
  --output outputs/kaideo_2560x1440_yuv420p_0_w11_b13_shift10_rate1.yuv
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

当前选定的 `task_qat_w11_b13_noedge_shift10` 是整数 residual ONNX，并配套保存整数参数和导出元数据。训练指标与模型选择细节见 `EVALUATION_SUMMARY.md`。
