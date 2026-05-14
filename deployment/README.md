# Deployment Package

本目录是上传用部署包，主线模型为 `task_qat_w11_b13_noedge_shift10`。它修正了早期 `调试滤波工具v0` 的 block 推理方式：不再把整帧拆成 32x32 block 分别送入 ONNX，而是整帧 Y 一次推理，避免 block 边界拼接伪影。

## Contents

```text
models/
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.int_params.npz
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.export_meta.json
  other exported comparison models

infer_block_rate_yuv.py
  Main full-frame ONNX residual inference script.

debug_filter_tool/
  Minimal fixed debug tool version, equivalent to the cleaned v2 package.

outputs/*/summary.*
  ONNX vs PyTorch consistency summaries. Large YUV outputs are intentionally not included.

EVALUATION_SUMMARY.md
  Model selection and evaluation notes copied from the training record.
```

Large raw sample videos and full generated YUV outputs are excluded from the upload package. Put local inputs under `data/`, or pass an absolute path with `--input`.

## Model Semantics

The ONNX model consumes raw luma values:

```text
input  = raw 0..255 Y, float32 NCHW [N,1,H,W]
output = raw residual delta_y, float32 NCHW [N,1,H,W]
```

Internal integer path:

```text
Y_u     = PixelUnshuffle4(Y)
acc     = Conv2D(Y_u, q_w, q_b)
delta_u = round(acc / 2^10)
delta_y = PixelShuffle4(delta_u)
```

Runtime applies the residual outside ONNX:

```text
Y_out = clip(round(Y + rate * delta_y), 0, 255)
```

Chroma bytes are not decoded and are copied back unchanged. For 8-bit 4:2:0 files, `yuv420p` and `nv12` both have `width * height / 2` chroma bytes, so the tool can pass either format through as long as the Y plane and frame size are correct.

## Recommended Model

```text
models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx
models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.int_params.npz
models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.export_meta.json
```

Integer parameters:

```text
q_w: signed W11, range -1024..1023, actual -466..76
q_b: signed B13, range -4096..4095, actual -1989..-119
shift: 10
```

## Run Inference

Install minimal dependencies:

```bash
pip install numpy onnxruntime
```

Default command:

```bash
python infer_block_rate_yuv.py \
  --width 2560 \
  --height 1440 \
  --input data/kaideo_2560x1440_yuv420p_0.yuv \
  --output outputs/kaideo_2560x1440_yuv420p_0_w11_b13_shift10_rate1.yuv
```

Use a different residual strength:

```bash
python infer_block_rate_yuv.py \
  --width 2560 \
  --height 1440 \
  --input data/kaideo_2560x1440_yuv420p_0.yuv \
  --output outputs/rate05.yuv \
  --rate 0.5
```

Use a different ONNX:

```bash
python infer_block_rate_yuv.py \
  --model models/task_qat_w10_b12_noedge_delta_raw_dynamic.onnx \
  --input data/input.yuv \
  --output outputs/output.yuv
```

## Validation Record

`outputs/kaideo_full_onnx_pytorch_compare_rate1/summary.md` records the full-video ONNX vs PyTorch check:

| model | frames | max abs Y diff | diff Y pixels |
|---|---:|---:|---:|
| `edge_w10_b13` | 100 | 0 | 0 |
| `teacher_qat_w10_b12` | 100 | 0 | 0 |
| `fudan_fp16` | 100 | 1 | 615 |

The final W11/B13 shift10 model is exported as an integer residual ONNX. Its model selection and training metrics are summarized in `EVALUATION_SUMMARY.md`.
