# 整数 QAT 推理与对比流程

目标：把 QAT 模型输出成和旧模型一致的 `.yuv` 目录，再用同一套 `evaluate_prefilter_dataset.py` 做横向比较。

## 1. 推理 QAT best

```bash
python pred_int_qat.py \
  --checkpoint runs/xlx_clean_roi_512_edge_aux_int_qat_w12/checkpoints/best.pt \
  --input data/xlx_clean_roi_512/test \
  --output_dir infer_test_compare/xlx_clean_roi_512_int_qat_w12_best_test \
  --device cuda
```

如果只想先快速确认流程，可以加：

```bash
--skip_existing
```

## 2. 推理旧 FP32 best

如果旧 FP32 best 的测试输出目录已经存在，可以直接复用：

```text
infer_test_compare/xlx_clean_roi_512_edge_aux_best_test
```

如果需要重新生成：

```bash
python pred.py \
  --config configs/finetune_xlx_clean_roi_512_edge_aux.yaml \
  --checkpoint runs/xlx_clean_roi_512_edge_aux_finetune/checkpoints/best.pt \
  --input data/xlx_clean_roi_512/test \
  --output_dir infer_test_compare/xlx_clean_roi_512_edge_aux_best_test \
  --device cuda
```

## 3. 多模型统一评估

测试集原图和 ref/gt：

```text
data/xlx_clean_roi_512/test/img
data/xlx_clean_roi_512/test/gt
```

对比 FP32 best 和 QAT best：

```bash
python evaluate_prefilter_dataset.py \
  --ori data/xlx_clean_roi_512/test/img \
  --ref data/xlx_clean_roi_512/test/gt \
  --candidate fp32_edge_aux=infer_test_compare/xlx_clean_roi_512_edge_aux_best_test \
  --candidate int_qat_w12=infer_test_compare/xlx_clean_roi_512_int_qat_w12_best_test \
  --width 512 \
  --height 512 \
  --mask_mode detail_gain \
  --s_thr 0.25 \
  --workers 8 \
  --output_json infer_test_compare/eval_int_qat_w12_vs_fp32.json
```

如果想先抽样：

```bash
--limit 200
```

## 4. 怎么读结果

优先看这些指标：

- `selective_score`：主指标，越大越好。
- `bg_completion`：背景接近 ref 的程度，越大越好。
- `edge_source_completion`：边缘接近原图的程度，越大越好；小于 0 说明伤边明显。
- `edge_retention_ratio`：边缘能量保留，越接近 1 越好，过低是过平滑，过高可能是噪声/锐化过强。
- `edge_oversmooth_vs_src`：相对原图过平滑比例，越小越好。
- `bg_hf_error`：背景高频误差，越小越好。

QAT 优化的判断顺序：

1. 先比 `int_qat_w12` 和 `fp32_edge_aux` 的 `selective_score`。
2. 如果 QAT 低很多，再看是 `bg_completion` 掉了，还是 `edge_source_completion / edge_retention_ratio` 掉了。
3. 再结合训练日志里的 `max_qw/max_qb/shift` 判断是量化表达不够，还是 loss 方向不对。

## 5. 下一步优化决策

### 情况 A：QAT 比 FP32 主要差在背景

表现：

- `bg_completion` 明显低
- `bg_hf_error` 明显高
- `edge_retention_ratio` 接近 FP32

建议：

- 增大 task loss 权重相对 distill 的影响，先把 `distillation.loss_weight` 从 `0.1` 降到 `0.05`。
- 继续训练当前 W12，学习率保持小，例如 `1e-6` 到 `3e-6`。
- 如果 `max_qw` 没到上限，可以继续 W12，不必急着升 bit。

### 情况 B：QAT 比 FP32 主要差在边缘

表现：

- `edge_source_completion` 降低或小于 0
- `edge_retention_ratio` 明显低于 FP32
- `edge_oversmooth_vs_src` 升高

建议：

- 增大 `edge_aux_loss.loss_weight`，例如 `0.05 -> 0.08`。
- 降低 `perceptual_loss.loss_weight`，避免过度贴 ref 导致伤边。
- 保持 distillation，不要立刻关掉，因为 teacher 能约束整体行为。

### 情况 C：QAT 输出整体像 FP32，但指标略低

表现：

- `l_distill` 很小
- `selective_score` 小幅低于 FP32
- 主观图差别不大

建议：

- 先接受 W12，继续做硬件 bit-true reference。
- 同时跑 W10/W8 sweep 看能不能进一步降位宽。

### 情况 D：QAT 明显差，且 `max_qw` 靠近 W12 上限

表现：

- `q_w_usage > 0.95`
- `selective_score` 明显低
- `l_distill` 降不下去

建议：

- 先试 W14 或 W16，确认是不是权重量化精度瓶颈。
- 如果 W14 立刻恢复，说明 W12 太紧。
- 如果 W14 也不恢复，问题更可能在训练 loss 或 raw-domain clip/round 策略。

### 情况 E：bias 接近 17bit 上限

表现：

- `q_b_usage > 0.8`
- `max_qb` 持续增大

建议：

- 增大 `bias_range_penalty`，例如 `1e-6 -> 1e-5`。
- 打开轻微 `bias_l1_weight`，例如 `1e-7` 或 `1e-6`。
- 必要时试无 bias 版本，但这属于第二阶段，不建议第一轮就改结构。

## 6. 当前 W12 训练日志初步判断

当前 QAT best checkpoint 的量化统计：

```text
max_abs_q_w = 1623
max_abs_q_b = 7487
q_w_usage   = 0.793
q_b_usage   = 0.114
shift       = 12
```

含义：

- W12 权重还有余量，没有顶满。
- bias17 余量非常大，不是当前瓶颈。
- 如果测试集指标差，优先怀疑 loss 配比、QAT raw-domain round/clip 带来的边缘/背景取舍，而不是 bit 上限。
