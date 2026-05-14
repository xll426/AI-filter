# Deployment

This directory contains the deployment package for the current AI Filter noedge release. It includes residual ONNX models, integer sidecar parameters, export scripts, verification summaries, and the full-frame YUV inference tool.

## Current Model

The recommended deployment model is:

```text
models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx
models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.int_params.npz
models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.export_meta.json
```

Model configuration:

```text
experiment: task_qat_w11_b13_noedge_shift10
weight bits: W11
bias bits: B13
shift: 10
training loss line: noedge, no EdgeConsistencyLoss
evaluation line: no selective_score for final noedge selection
```

The noedge model is selected for practical post-encoding quality. It performs better after actual video encoding at larger QP values, while remaining closer to the reference filtering target before encoding.

## Directory Layout

```text
models/
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.int_params.npz
  task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.export_meta.json
  other comparison ONNX exports

infer_block_rate_yuv.py
  Main full-frame residual ONNX inference script.

debug_filter_tool/
  Minimal debug package using the same full-frame inference logic.

outputs/*/summary.*
  ONNX/PyTorch consistency summary files.

rate_maps/
  Example rate-map files retained for block-rate experiments.

EVALUATION_SUMMARY.md
  Deployment-side copy of the model-selection summary.
```

Large raw input videos and full generated YUV outputs are intentionally not stored in this directory. Local input files can be placed under `data/` or passed with `--input`.

## ONNX Semantics

The ONNX model outputs a residual. It does not add the residual back to the source frame.

```text
input  = raw 0..255 Y, float32 NCHW [N,1,H,W]
output = raw residual delta_y, float32 NCHW [N,1,H,W]
```

Integer path inside the ONNX graph:

```text
Y_u     = PixelUnshuffle4(Y)
acc     = Conv2D(Y_u, q_w, q_b)
delta_u = round(acc / 2^10)
delta_y = PixelShuffle4(delta_u)
```

Runtime composition:

```text
Y_out = clip(round(Y + rate * delta_y), 0, 255)
```

Chroma bytes are copied unchanged. The script only parses the Y plane and treats the chroma payload as raw bytes. For 8-bit 4:2:0 inputs, both `yuv420p` and `nv12` have the same frame byte size.

## Inference

Install minimal runtime dependencies:

```bash
pip install numpy onnxruntime
```

Run full-frame inference:

```bash
python infer_block_rate_yuv.py \
  --width 2560 \
  --height 1440 \
  --input data/kaideo_2560x1440_yuv420p_0.yuv \
  --output outputs/kaideo_2560x1440_yuv420p_0_w11_b13_shift10_rate1.yuv
```

Adjust residual strength:

```bash
python infer_block_rate_yuv.py \
  --width 2560 \
  --height 1440 \
  --input data/kaideo_2560x1440_yuv420p_0.yuv \
  --output outputs/rate05.yuv \
  --rate 0.5
```

Use another ONNX model:

```bash
python infer_block_rate_yuv.py \
  --model models/task_qat_w10_b12_noedge_delta_raw_dynamic.onnx \
  --input data/input.yuv \
  --output outputs/output.yuv
```

## Verification

`outputs/kaideo_full_onnx_pytorch_compare_rate1/summary.md` records a full-video ONNX/PyTorch consistency check:

| model | frames | max abs Y diff | diff Y pixels |
|---|---:|---:|---:|
| `edge_w10_b13` | 100 | 0 | 0 |
| `teacher_qat_w10_b12` | 100 | 0 | 0 |
| `fudan_fp16` | 100 | 1 | 615 |

The selected `task_qat_w11_b13_noedge_shift10` model is an integer residual ONNX with separate integer parameter metadata. Training metrics and model-selection details are recorded in `EVALUATION_SUMMARY.md`.
