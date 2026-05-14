# 评估结果

更新时间：2026-05-13

## 当前推荐结论

当前按“有符号 W bit，最终右移使用 `W-1`”的硬件口径，已经测试过：

| 模型 | W/B | shift | test PSNR | test SSIM | test loss | val best PSNR | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| task_qat_w10_b12_noedge_shift9 | W10/B12 | 9 | 36.7153 | 0.968990 | 0.015199 | 29.4915 | 位宽最省，但精度最低 |
| task_qat_w11_b13_noedge_shift10 | W11/B13 | 10 | 36.7303 | 0.969035 | 0.015182 | 29.4984 | 推荐平衡方案 |
| task_qat_w12_b14_noedge_shift11 | W12/B14 | 11 | 36.7394 | 0.969061 | 0.015178 | 29.4994 | 当前硬件口径下精度最好 |

推荐：

1. 如果要控制 W/B 位数，优先选 `W11/B13 shift10`。
2. 如果 W12/B14 资源可以接受，选 `W12/B14 shift11`，它是当前固定 shift 规则下 test/val 综合最好的。
3. 不建议继续往下试 `W9/B11 shift8`，从 W10 shift9 的掉点趋势看，大概率继续损失精度。
4. 继续往上只建议最多试一个 `W13/B15 shift12`。预计收益很小，主要是靠近旧 `W12/B14 shift12` 上限，可能只有约 0.002 dB 级别。

## 和旧自动 shift 结果对比

旧配置 `min_shift: 0, max_shift: 30` 会自动找最大可用 shift，因此结果如下：

| 模型 | W/B | shift | test PSNR | test SSIM | q_w usage | q_b usage |
|---|---:|---:|---:|---:|---:|---:|
| task_qat_w10_b12_noedge | W10/B12 | 10 | 36.7303 | 0.969035 | 0.910 | 0.971 |
| task_qat_w11_b13_noedge_shift10 | W11/B13 | 10 | 36.7303 | 0.969035 | 0.455 | 0.486 |
| task_qat_w12_b14_noedge | W12/B14 | 12 | 36.7419 | 0.969079 | 0.910 | 0.971 |
| task_qat_w12_b14_noedge_shift11 | W12/B14 | 11 | 36.7394 | 0.969061 | 0.455 | 0.486 |

解释：

- `W11/B13 shift10` 和旧 `W10/B12 shift10` 的有效 q 值完全对齐，所以 test 指标一致。
- 固定 `shift = W - 1` 后，q 值占用约 45%/49%，说明整数范围不是瓶颈，主要精度差异来自右移小数位。
- `W12/B14 shift11` 比 `W11/B13 shift10` 高约 0.009 dB，比 `W10/B12 shift9` 高约 0.024 dB，提升存在但已经很小。

## Test 汇总

完整 test set：1920 samples，Y channel 指标。

| experiment | W/B | ckpt iter | test PSNR | test SSIM | test loss | val best PSNR | q_w range/use | q_b range/use | shift |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| task_qat_w12_b14_noedge_shift11 | W12/B14 | 1000 | 36.7394 | 0.969061 | 0.015178 | 29.4994 | -932..153 / 0.455 | -3978..-237 / 0.486 | 11..11 |
| task_qat_w12_b14_noedge | W12/B14 | 1000 | 36.7419 | 0.969079 | 0.015178 | 29.4982 | -1864..305 / 0.910 | -7956..-474 / 0.971 | 12..12 |
| task_qat_w11_b13_noedge_shift10 | W11/B13 | 1000 | 36.7303 | 0.969035 | 0.015182 | 29.4984 | -466..76 / 0.455 | -1989..-119 / 0.486 | 10..10 |
| task_qat_w10_b12_noedge_shift9 | W10/B12 | 1000 | 36.7153 | 0.968990 | 0.015199 | 29.4915 | -233..38 / 0.455 | -995..-59 / 0.486 | 9..9 |
| task_qat_w10_b12_noedge | W10/B12 | 1000 | 36.7303 | 0.969035 | 0.015182 | 29.4984 | -466..76 / 0.910 | -1989..-119 / 0.971 | 10..10 |

来源：

- `./runs/task_qat_noedge_test_summary.md`
- `./runs/task_qat_noedge_test_summary.json`

## Baseline 和任务效果

| item | PSNR vs reference | L1 | L2 | 说明 |
|---|---:|---:|---:|---|
| input_vs_reference | 34.0913 | 0.01847592 | 0.00086775712 | 原始输入到 reference |
| fp32_teacher_vs_reference | 36.7181 | 0.01242504 | 0.00038421729 | FP32 部署结构 teacher |
| task_qat_w10_b12_noedge | 36.7303 | 0.01230828 | - | task QAT，略高于 FP32 teacher |
| task_qat_w12_b14_noedge | 36.7419 | 0.01230410 | - | task QAT，旧自动 shift 上限 |

从结果看，task QAT 不是单纯拟合 FP32 teacher，而是在 reference 指标上略微超过 FP32 teacher。当前最有用的主线是 `task_qat_*_noedge`，不是早期 edge 版本。

## Teacher-QAT 实验

这些实验主要用于验证整数量化能否贴近 FP32 teacher，不是最终 reference 指标主线。

| experiment | W/B | loss | test PSNR vs teacher | test SSIM vs teacher | q_w usage | q_b usage | shift |
|---|---:|---|---:|---:|---:|---:|---:|
| teacher_qat_w12_b14 | W12/B14 | l1 | 56.0403 | 0.999060 | 0.910 | 0.972 | 12 |
| teacher_qat_w10_b12 | W10/B12 | l1 | 51.3205 | 0.997320 | 0.912 | 0.972 | 10 |
| teacher_qat_w10_b13 | W10/B13 | l1 | 51.3205 | 0.997320 | 0.912 | 0.486 | 10 |
| teacher_qat_w10_b13_l2 | W10/B13 | l2 | 50.8200 | 0.996857 | 0.912 | 0.484 | 10 |
| teacher_qat_w8_b12 | W8/B12 | l1 | 43.9512 | 0.985299 | 0.913 | 0.243 | 8 |

结论：

- W8 明显不够。
- W10 已经能比较好贴近 teacher。
- W12 贴近 teacher 最好，但对最终 reference 的收益不等同于 56 dB 这个数，因为该表是 vs teacher。

## 训练是否充分

当前这批 task QAT 都是从 FP32 融合权重初始化，训练变量很少，且 shift 固定，所以收敛非常快。

| experiment | best iter | best val PSNR | last iter | last val PSNR | 观察 |
|---|---:|---:|---:|---:|---|
| task_qat_w10_b12_noedge_shift9 | 1000 | 29.4920 左右 | 8000 | 29.4856 | 1000 后没有继续提升 |
| task_qat_w11_b13_noedge_shift10 | 1000 | 29.4984 | 8000 | 29.4865 | 1000 是 best |
| task_qat_w12_b14_noedge_shift11 | 1000 | 29.4994 | 8000 | 29.4875 | 1000 是 best |

判断：

- 对当前配置，8000 iter 已经足够，甚至 best 基本在 1000 iter 定型。
- 后续继续训练没有带来更好 val/test，反而略回落，说明不是训练不充分，更像是小数据/任务 loss 下的轻微过拟合或指标波动。
- 如果后面继续扫 bit，可以先用 2000 iter 快速筛选，只有接近最优的候选再跑 8000 iter 做确认。

## 已导出模型和调试工具

已导出 W11/B13 shift10 residual ONNX：

```text
../deployment/models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.onnx
../deployment/models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.int_params.npz
../deployment/models/task_qat_w11_b13_noedge_shift10_delta_raw_dynamic.export_meta.json
```

ONNX 语义：

```text
input  = raw 0..255 Y, float32 NCHW
output = raw residual delta_y, float32 NCHW
图内无入口 clip/round
图内仅保留 delta_u = round(acc / 2^10)
```

整数参数：

```text
q_w: W11, actual range -466..76
q_b: B13, actual range -1989..-119
shift: 10
```

已新建调试工具：

```text
../deployment/debug_filter_tool
```

已用 v2 推理样例：

```text
input:
../deployment/debug_filter_tool/data/kaideo_2560x1440_yuv420p_0.yuv

output:
../deployment/debug_filter_tool/outputs/kaideo_2560x1440_yuv420p_0_w11_b13_shift10_rate1.yuv
```

校验：

```text
frames: 100
input bytes: 552960000
output bytes: 552960000
same size: true
```

## 有价值实验数量

目前建议作为备份记录保留的有效实验：

1. Baseline/reference：2 条，`input_vs_reference` 和 `fp32_teacher_vs_reference`。
2. Teacher-QAT：5 条，用于量化贴近 teacher 的能力判断。
3. Task-QAT noedge：5 条，是当前最终模型选择主线。
4. 部署导出：W11/B13 shift10 residual ONNX + `deployment debug_filter_tool`。

实际用于最终选型的核心候选是 3 条：

```text
W10/B12 shift9
W11/B13 shift10
W12/B14 shift11
```

当前最终建议：

```text
资源优先：W11/B13 shift10
精度优先：W12/B14 shift11
不建议：W10/B12 shift9 以下继续降 bit
```
