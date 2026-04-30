# 整数 QAT 结果记录

## 1. 本次评估对象

评估文件：

```text
infer_test_compare/eval_int_qat_w12_vs_fp32.json
```

评估范围：

- `tile_count = 1920`
- `mask_mode = detail_gain`
- `s_thr = 0.25`
- 原图：`data/xlx_clean_roi_512/test/img`
- ref/gt：`data/xlx_clean_roi_512/test/gt`

对比模型：

- `fp32_edge_aux`
  - 旧 FP32 edge auxiliary best
  - 输出目录：`infer_test_compare/xlx_clean_roi_512_edge_aux_best_test`
- `int_qat_w12`
  - 当前 W12 + bias17 整数 QAT best
  - checkpoint：`runs/xlx_clean_roi_512_edge_aux_int_qat_w12/checkpoints/best.pt`
  - 输出目录：`infer_test_compare/xlx_clean_roi_512_int_qat_w12_best_test`

## 2. 当前 QAT 配置

配置文件：

```text
configs/int_qat_xlx_clean_roi_512_edge_aux.yaml
```

核心配置：

```yaml
int_qat:
  weight_bits: 12
  bias_bits: 17
  per_channel_shift: true
  min_shift: 0
  max_shift: 30
  weight_range_penalty: 1.0e-6
  bias_range_penalty: 1.0e-6
  bias_l1_weight: 0.0
  distillation:
    enabled: true
    loss_weight: 0.1
    only_y: true
```

训练配置：

```yaml
train:
  lr: 3.0e-6
  min_lr: 5.0e-7
  total_iter: 8000
  warmup_iter: 300
  batch_size: 16
  crop_size: 224
  fidelity_loss:
    enabled: true
    loss_weight: 1.0
    roi_weight: 20.0
    non_roi_weight: 1.0
  perceptual_loss:
    enabled: true
    loss_weight: 0.16
  edge_aux_loss:
    enabled: true
    loss_weight: 0.05
```

初始化 checkpoint：

```text
runs/xlx_clean_roi_512_edge_aux_finetune/checkpoints/best.pt
```

当前 best 的整数参数统计：

| 项 | 数值 | 说明 |
|---|---:|---|
| `max_abs_q_w` | `1623` | W12 范围是 `[-2048, 2047]` |
| `q_w_usage` | `0.793` | 权重没有顶满 |
| `max_abs_q_b` | `7487` | bias17 范围是 `[-65536, 65535]` |
| `q_b_usage` | `0.114` | bias 余量很大 |
| `shift_min` | `12` | 当前所有通道 shift 一致 |
| `shift_max` | `12` | 当前所有通道 shift 一致 |

## 3. 指标对比

| 指标 | FP32 edge-aux | Int-QAT W12 | 变化 | 方向 |
|---|---:|---:|---:|---|
| `selective_score` | `33.241507` | `34.593271` | `+1.351764` | 更好 |
| `bg_completion` | `0.195257` | `0.213280` | `+0.018023` | 更好 |
| `edge_source_completion` | `0.270574` | `0.285160` | `+0.014586` | 更好 |
| `edge_retention_ratio` | `0.955848` | `0.955472` | `-0.000376` | 基本不变 |
| `edge_oversmooth_vs_src` | `0.298055` | `0.294720` | `-0.003335` | 更好 |
| `bg_hf_error` | `20.113781` | `20.096566` | `-0.017215` | 基本不变略好 |
| `edge_preserve_error` | `91.187806` | `90.760849` | `-0.426957` | 略好 |
| `edge_over_smooth_ratio` | `0.151262` | `0.150618` | `-0.000644` | 略好 |
| `edge_gmsd` | `0.120846` | `0.120727` | `-0.000119` | 基本不变略好 |
| `bg_grad_energy_ratio` | `1.514324` | `1.507908` | `-0.006416` | 略接近 1 |
| `edge_grad_energy_ratio` | `1.096668` | `1.096271` | `-0.000397` | 基本不变 |
| `structure_alignment_error` | `0.031743` | `0.031723` | `-0.000020` | 基本不变略好 |

## 4. 结论

当前 W12 QAT 结果是正向的：

- `selective_score` 从 `33.241507` 提升到 `34.593271`，约 `+4.07%`。
- `bg_completion` 提升，说明背景更接近 ref。
- `edge_source_completion` 提升，说明边缘更接近原图。
- `edge_retention_ratio` 基本不变，说明没有靠牺牲边缘换背景。
- `edge_oversmooth_vs_src` 略降，说明边缘过平滑没有变严重。
- `bg_hf_error`、`edge_preserve_error`、`edge_gmsd` 都没有退化。

所以这版不是“量化后勉强可用”，而是：

> W12 + bias17 + fixed shift 的整数 QAT 路径已经成立，并且在当前评估口径下略优于 FP32 edge-aux best。

结合量化统计：

- W12 权重没有顶满，`q_w_usage = 0.793`
- bias17 远没到瓶颈，`q_b_usage = 0.114`

当前问题不在 bit 上限，也不在 bias 17bit。W12 可以作为第一版硬件部署候选 baseline。

## 5. 下一步计划

### Step 1：做 bit sweep

优先保持当前训练配置不变，只改 `weight_bits`：

1. `W10 + bias17`
2. `W8 + bias17`
3. 视情况补 `W14 + bias17`

判断标准：

- 如果 W10 的 `selective_score` 接近 W12，优先推进 W10。
- 如果 W8 也接近 W12，可以考虑 W8 作为更激进硬件方案。
- 如果 W10 明显掉，保留 W12 作为部署主线。
- 如果 W12 后续在 bit-true 校验中有不可接受误差，再试 W14/W16。

### Step 2：做 bit-true reference

用导出的：

```text
runs/xlx_clean_roi_512_edge_aux_int_qat_w12/int_exports/best/q_w.pt
runs/xlx_clean_roi_512_edge_aux_int_qat_w12/int_exports/best/q_b.pt
runs/xlx_clean_roi_512_edge_aux_int_qat_w12/int_exports/best/shift.pt
```

写一个纯整数 reference 推理，验证：

- 导出参数和 `DeployPrefilterIntQAT.forward()` 输出一致
- round 规则一致
- clip 规则一致
- PixelUnshuffle / PixelShuffle 通道顺序一致

这是进入硬件对齐前必须做的步骤。

### Step 3：主观图对比

从测试集挑几类样本：

- 平坦背景
- 人脸/车牌 ROI
- 强边缘
- 纹理区域
- 暗部和高亮区域

对比：

- 原图
- ref/gt
- FP32 edge-aux
- Int-QAT W12

重点看：

- 背景是否更干净
- 边缘是否发糊
- 是否有量化阶梯感
- 是否出现局部过冲或 clip 饱和

### Step 4：只在需要时调 loss

当前 W12 指标已经优于 FP32，暂时不建议先调 loss。

只有出现下面情况再改：

- W10/W8 掉背景：降低 `distillation.loss_weight` 或继续训练。
- W10/W8 掉边缘：提高 `edge_aux_loss.loss_weight`，例如 `0.05 -> 0.08`。
- W10/W8 权重顶满：提高 bit 或调整 shift 初始化。
- bias 接近 17bit：提高 `bias_range_penalty` 或加轻微 `bias_l1_weight`。

### Step 5：推荐执行顺序

```text
W12 当前结果归档
  -> W10 同配置训练
  -> W10 推理 + 全量评估
  -> W8 同配置训练
  -> W8 推理 + 全量评估
  -> 选 W12/W10/W8 中指标和硬件成本最合适的一版
  -> bit-true reference
  -> 硬件对齐
```

---

## 6. Bit sweep 更新：W10+B13 / W8+B10 / W4+B8

### 6.1 新增实验对象

本轮新增 3 组更低 bit 的整数 QAT：

| 实验 | 配置文件 | checkpoint | 说明 |
|---|---|---|---|
| `W10+B13` | `configs/int_qat_xlx_clean_roi_512_edge_aux_w10_b13.yaml` | `runs/xlx_clean_roi_512_edge_aux_int_qat_w10_b13/checkpoints/best.pt` | 主推小表示候选 |
| `W8+B10` | `configs/int_qat_xlx_clean_roi_512_edge_aux_w8_b10.yaml` | `runs/xlx_clean_roi_512_edge_aux_int_qat_w8_b10/checkpoints/best.pt` | 激进备选 |
| `W4+B8` | `configs/int_qat_xlx_clean_roi_512_edge_aux_w4_b8.yaml` | `runs/xlx_clean_roi_512_edge_aux_int_qat_w4_b8/checkpoints/best.pt` | 探下限实验 |

三组实验均保持原 QAT 训练策略不变：

- 从 `runs/xlx_clean_roi_512_edge_aux_finetune/checkpoints/best.pt` 初始化
- `total_iter = 8000`
- `batch_size = 16`
- `crop_size = 224`
- `lr = 3.0e-6`
- `min_lr = 5.0e-7`
- `distillation.loss_weight = 0.1`
- `edge_aux_loss.loss_weight = 0.05`

### 6.2 训练验证集 best 指标

下面是各实验 `metrics.jsonl` 中按 `primary_metric = selective_score` 选出的 best：

| 实验 | best iter | selective | val PSNR | val SSIM | bg_completion | edge_source_completion | edge_retention_ratio | edge_oversmooth_vs_src |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `W12+B17` | 6500 | 22.081893 | 29.142439 | 0.923051 | 0.343879 | -0.175827 | 0.873106 | 0.583994 |
| `W10+B13` | 4500 | 21.845536 | 29.152281 | 0.923240 | 0.344558 | -0.184612 | 0.872252 | 0.585644 |
| `W8+B10` | 500 | 21.053020 | 29.123300 | 0.922537 | 0.328971 | -0.188872 | 0.872510 | 0.583447 |
| `W4+B8` | 2500 | -2.773623 | 27.479895 | 0.886639 | -0.035610 | -0.429853 | 0.836083 | 0.666849 |

相对 `W12+B17` 的 best selective：

| 实验 | selective 差值 | 判断 |
|---|---:|---|
| `W10+B13` | -0.236357 | 非常接近 W12 |
| `W8+B10` | -1.028873 | 有可见退化，但还可作为激进备选 |
| `W4+B8` | -24.855516 | 明显不可用 |

### 6.3 训练后期走势

各实验最后一次验证，即 `iter = 8000`：

| 实验 | last selective | best selective | 走势 |
|---|---:|---:|---|
| `W12+B17` | 21.977755 | 22.081893 | 稳定，后期轻微回落 |
| `W10+B13` | 21.310123 | 21.845536 | 后期回落，但 best 仍接近 W12 |
| `W8+B10` | 19.245980 | 21.053020 | 训练越久反而下降，说明不是单纯 iter 不够 |
| `W4+B8` | -15.555464 | -2.773623 | 明显恶化，低 bit 容量不足 |

这个趋势很关键：

- `W10+B13` 的 best 出现在 4500 iter，不是没训够，而是后期指标略回落。
- `W8+B10` 的 best 出现在 500 iter，后面持续下降，说明继续堆 iter 不会自然修复，反而可能让模型在低 bit 约束下偏离较好的折中点。
- `W4+B8` 从 best 到 last 都很差，且 PSNR/SSIM/selective 同时明显退化，基本不是训练 iter 不够的问题。

### 6.4 整数量化统计

| 实验 | weight range | max_abs_q_w | q_w_usage | bias range | max_abs_q_b | q_b_usage | shift |
|---|---|---:|---:|---|---:|---:|---:|
| `W12+B17` | [-2048, 2047] | 1623 | 0.793 | [-65536, 65535] | 7487 | 0.114 | 12 |
| `W10+B13` | [-512, 511] | 406 | 0.795 | [-4096, 4095] | 1874 | 0.458 | 10 |
| `W8+B10` | [-128, 127] | 102 | 0.803 | [-512, 511] | 470 | 0.920 | 8 |
| `W4+B8` | [-8, 7] | 6 | 0.857 | [-128, 127] | 29 | 0.228 | 4 |

解释：

- `W10+B13` 的权重使用率约 0.795，bias 使用率约 0.458，整数范围仍有余量，配置合理。
- `W8+B10` 的 bias 使用率已经达到约 0.920，非常接近上限。它虽然没有硬顶满，但 bias 空间已经很紧，继续下降到 `B9/B8` 风险很高。
- `W4+B8` 的数值范围虽然没有显示饱和，但权重只有 4 bit，量化步长太粗，表达能力已经不够。

### 6.5 测试集 PyTorch 推理结果

测试集：

```text
data/xlx_clean_roi_512/test
tile_count = 1920
```

推理脚本：

```text
pred_int_qat.py
```

该脚本在 `only_train_y=True` 时的逻辑已确认：

1. 从输入 YUV 中只读取 Y plane
2. 送入 `DeployPrefilterIntQAT`
3. 模型输出 filtered Y
4. 将 filtered Y 与原始 UV payload 原封不动拼回输出 YUV

测试集 PSNR/SSIM：

| 实验 | 输出目录 | tile count | avg PSNR | avg SSIM |
|---|---|---:|---:|---:|
| `W12+B17` | `infer_test_compare/xlx_clean_roi_512_int_qat_w12_best_test` | 1920 | 36.221612 | 0.958758 |
| `W10+B13` | `infer_test_compare/xlx_clean_roi_512_int_qat_w10_b13_best_test` | 1920 | 36.206724 | 0.958672 |
| `W8+B10` | `infer_test_compare/xlx_clean_roi_512_int_qat_w8_b10_best_test` | 1920 | 36.096575 | 0.957819 |
| `W4+B8` | `infer_test_compare/xlx_clean_roi_512_int_qat_w4_b8_best_test` | 1920 | 32.286338 | 0.909525 |

相对 `W12+B17`：

| 实验 | PSNR 差值 | SSIM 差值 | 判断 |
|---|---:|---:|---|
| `W10+B13` | -0.014888 dB | -0.000087 | 几乎无损 |
| `W8+B10` | -0.125037 dB | -0.000939 | 有小幅退化 |
| `W4+B8` | -3.935275 dB | -0.049234 | 明显不可用 |

### 6.6 W10+B13 ONNX 导出与对齐

已导出 Y-only ONNX：

```text
runs/xlx_clean_roi_512_edge_aux_int_qat_w10_b13/onnx/quant_w10_b13_best_y.onnx
```

同时导出整数参数 sidecar：

```text
runs/xlx_clean_roi_512_edge_aux_int_qat_w10_b13/onnx/quant_w10_b13_best_y.int_params.npz
runs/xlx_clean_roi_512_edge_aux_int_qat_w10_b13/onnx/quant_w10_b13_best_y.export_meta.json
```

ONNX 输入/输出：

- 输入：normalized Y plane，`[N, 1, H, W]`
- 输出：normalized filtered Y plane，`[N, 1, H, W]`
- ONNX 内部 Conv 常量是 float32，但来自冻结后的整数 `q_w/q_b/shift`

导出参数：

| 项 | 数值 |
|---|---:|
| `weight_bits` | 10 |
| `bias_bits` | 13 |
| `q_w.max_abs` | 406 |
| `q_b.max_abs` | 1874 |
| `shift` | 10 |

PyTorch vs ONNX 全量对齐：

```text
runs/xlx_clean_roi_512_edge_aux_int_qat_w10_b13/onnx_compare/compare_summary.json
```

结果：

| 项 | 数值 |
|---|---:|
| sample_count | 1920 |
| passed | true |
| max_abs_norm | 0.0 |
| max_abs_raw_lsb | 0 |

ONNX 推理输出目录：

```text
infer_test_compare/xlx_clean_roi_512_int_qat_w10_b13_best_onnx_test
```

ONNX 测试集结果：

| 实验 | tile count | avg PSNR | avg SSIM |
|---|---:|---:|---:|
| `W10+B13 ONNX` | 1920 | 36.206724 | 0.958672 |

并且 PyTorch 输出 YUV 与 ONNX 输出 YUV 已逐字节比较：

| 项 | 结果 |
|---|---|
| 文件数量 | PyTorch 1920 / ONNX 1920 |
| 文件名集合 | 一致 |
| 文件内容 | byte-identical |

结论：

> W10+B13 的 ONNX 路径已经通过 bit-exact 对齐。后续如果只比较算法指标，ONNX 与 PyTorch 不需要重复算一遍，结果必然一致。

## 7. 最终结论与后续建议

### 7.1 当前推荐版本

推荐主线：

```text
W10+B13
```

理由：

- 相对 `W12+B17`，测试集 PSNR 只低 `0.014888 dB`
- SSIM 只低 `0.000087`
- 验证集 best selective 只低 `0.236357`
- 权重从 12 bit 降到 10 bit
- bias 从 17 bit 降到 13 bit
- ONNX 已经和 PyTorch 全量 bit-exact 对齐

保守 baseline：

```text
W12+B17
```

理由：

- 指标最好
- 整数范围余量最大
- 可作为硬件对齐和回归测试 baseline

激进备选：

```text
W8+B10
```

理由：

- 测试集 PSNR/SSIM 只小幅下降
- 但验证集 selective 已明显低于 W12/W10
- bias 使用率约 0.920，空间很紧
- 可以保留为硬件成本极敏感场景的备选，但不建议作为默认部署版本

不推荐：

```text
W4+B8
```

理由：

- selective 从 W12 的 22.081893 掉到 -2.773623
- 测试集 PSNR 从 36.221612 掉到 32.286338
- SSIM 从 0.958758 掉到 0.909525
- edge_retention_ratio 降低，edge_oversmooth_vs_src 升高
- 这已经不是小幅量化损失，而是表达能力明显不够

### 7.2 更低 bit 是否值得继续

不建议继续系统性往 `W4` 方向压。

原因不是单纯训练 iter 不够：

1. `W8+B10` 的 best 出现在 500 iter，8000 iter 时反而下降。
2. `W4+B8` 的 best 已经很差，8000 iter 更差。
3. 如果只是训练不够，通常会看到 best 出现在末期或仍有上升趋势；这里正相反。
4. `W4` 的权重量化级别只有 16 档，实际 signed 可用范围 `[-8, 7]`，对这个残差滤波器来说步长太粗。
5. `W8+B10` 的 bias 使用率已经约 0.920，继续压 bias 会很容易顶到上限或产生更强量化误差。

所以当前判断：

> `W10+B13` 是质量与硬件成本最平衡的点；`W8+B10` 是还能看的激进点；`W4+B8` 基本拉不回，除非重新设计网络、训练策略或量化方式，而不是简单增加 iter。

### 7.3 如果还要探索更低 bit，建议只做两个方向

如果硬件必须继续压缩，建议不要直接继续 `W4+B8`，而是试：

1. `W8+B11` 或 `W8+B12`
   - 目标是验证 W8 权重下，bias 放宽后是否能追回 selective。
   - 因为当前 `W8+B10` 的 bias 使用率已经约 0.920，bias 可能是瓶颈之一。

2. `W9+B12`
   - 介于 `W10+B13` 和 `W8+B10` 之间。
   - 很可能比 W8 稳，同时比 W10 再省一点。

暂不建议：

```text
W6+B8
W4+B8
W4+B10
```

除非有明确硬件约束必须测试，否则这类配置大概率需要改训练方案或模型结构，不适合在当前 QAT 配置下继续消耗时间。
