# 整数 QAT 训练方案

本文档对应当前仓库里的实现：

- `model_int_qat.py`：整数仿真 deploy 模型、STE、导出逻辑
- `train_int_qat.py`：独立 QAT 训练入口，不影响原 `train.py`
- `configs/int_qat_xlx_clean_roi_512_edge_aux.yaml`：W12/bias17 的第一版配置

## 1. 目标

当前 FP32 预滤波模型已经能稳定训练。新目标是面向硬件部署重新微调一个整数参数版本：

- 输入：8bit 原始像素，语义为 `0..255`
- 权重：整数，优先从 W12 开始试，最终不超过 17bit
- bias：整数，目标不超过 17bit
- 中间累加器：允许更大位宽
- 输出：8bit 原始像素，语义为 `0..255`
- 训练目标：尽量逼近当前 FP32 baseline 和参考滤波目标

核心不是套通用 int8 QAT，而是训练一个能按硬件路径执行的 raw-domain 定点残差卷积。

## 2. 现有模型部署形态

原 `PrefilterNet` 对 Y 通道做：

$$
Y_u = \operatorname{PixelUnshuffle}(Y)
$$

$$
\Delta_u = \operatorname{MBRConv3}(Y_u)
$$

$$
\hat{Y} = \operatorname{PixelShuffle}(Y_u + \Delta_u)
$$

其中 `MBRConv3.slim()` 能把训练态多分支卷积融合成单个 `3x3` 卷积：

$$
\Delta_u = \operatorname{Conv}_{3\times3}(Y_u, W) + b
$$

这正好适合硬件部署，所以 QAT 从 `slim()` 后的 deploy 结构开始，而不是继续训练多分支结构。

## 3. 为什么需要 shift

残差滤波的卷积核通常是小数系数。例如：

$$
W =
\begin{bmatrix}
0 & 0.05 & 0 \\
0.05 & -0.20 & 0.05 \\
0 & 0.05 & 0
\end{bmatrix}
$$

如果权重只能是裸整数，就无法表示 `0.05`、`-0.20` 这类小系数。定点表达使用：

$$
W_{\mathrm{eff}} = \frac{Q_w}{2^s}
$$

例如 `s=8` 时：

$$
0.05 \times 256 \approx 13,\quad -0.20 \times 256 \approx -51
$$

硬件保存整数 `Q_w`，执行后再右移 `s` 位。因此 shift 的作用不是防止累加器溢出，而是让整数权重能表达小数滤波核。

## 4. 硬件仿真前向

`DeployPrefilterIntQAT.forward()` 的仿真路径是：

$$
x_{\mathrm{raw}} = \operatorname{clip}(\operatorname{round}(255x_{\mathrm{norm}}), 0, 255)
$$

$$
Y_u = \operatorname{PixelUnshuffle}(Y_{\mathrm{raw}})
$$

$$
Q_w = \operatorname{clamp}(\operatorname{round}(W_{\mathrm{fp}}2^s), q_{\min}, q_{\max})
$$

$$
Q_b = \operatorname{clamp}(\operatorname{round}(b_{\mathrm{raw}}2^s), b_{\min}, b_{\max})
$$

$$
acc = \operatorname{Conv}_{3\times3}(Y_u, Q_w) + Q_b
$$

$$
\Delta = \operatorname{round}\left(\frac{acc}{2^s}\right)
$$

$$
Y_u^{out} = \operatorname{clip}(Y_u + \Delta, 0, 255)
$$

$$
\hat{Y} = \operatorname{PixelShuffle}(Y_u^{out})
$$

最后输出再除以 `255` 回到 `[0,1]`，这样可以复用现有 loss 和 validation。

注意：当前 FP32 训练在 `[0,1]` 域，转 raw 域时权重数值不变，bias 需要乘以 `255`：

$$
b_{\mathrm{raw}} = 255 b_{\mathrm{norm}}
$$

## 5. 训练变量和梯度

第一版固定 bit 和 shift，只训练浮点主参数：

- 训练变量：`weight_fp`、`bias_fp`
- 固定变量：`shift`
- 前向：投影成整数参数并模拟硬件
- 反向：对 `round/clamp/shift/clip` 使用 STE

权重梯度近似为：

$$
\frac{\partial L}{\partial W_{\mathrm{fp}}}
\approx
2^s \frac{\partial L}{\partial Q_w}
$$

bias 梯度近似为：

$$
\frac{\partial L}{\partial b_{\mathrm{raw}}}
\approx
2^s \frac{\partial L}{\partial Q_b}
$$

右移处忽略 round 的不可导部分：

$$
\frac{\partial \Delta}{\partial acc} \approx 2^{-s}
$$

## 6. Loss 组成

QAT 总 loss：

$$
L = L_{\mathrm{task}} + L_{\mathrm{distill}} + L_{\mathrm{reg}}
$$

其中：

- `L_task`：复用现有 `compute_train_loss()`，包含 ROI Charbonnier、MS-SSIM、EdgeConsistency
- `L_distill`：student 对齐 deploy-FP32 teacher，默认 Y 通道 L1，权重 `0.1`
- `L_reg`：权重范围正则、bias 范围正则、可选 bias L1

范围正则对量化前的整数 latent 生效：

$$
L_{\mathrm{range}} =
\operatorname{mean}\left(\operatorname{ReLU}(|x|-q_{\max})^2\right)
$$

## 7. 当前实现入口

第一版推荐配置：

```bash
python train_int_qat.py \
  --config configs/int_qat_xlx_clean_roi_512_edge_aux.yaml
```

该配置默认从当前 edge auxiliary FP32 best checkpoint 初始化：

```text
runs/xlx_clean_roi_512_edge_aux_finetune/checkpoints/best.pt
```

训练不会改动原 `train.py`。如果需要恢复 QAT：

```bash
python train_int_qat.py \
  --config configs/int_qat_xlx_clean_roi_512_edge_aux.yaml \
  --resume auto
```

## 8. Checkpoint 和导出

QAT checkpoint：

```text
runs/xlx_clean_roi_512_edge_aux_int_qat_w12/checkpoints/latest.pt
runs/xlx_clean_roi_512_edge_aux_int_qat_w12/checkpoints/best.pt
```

整数参数导出：

```text
runs/xlx_clean_roi_512_edge_aux_int_qat_w12/int_exports/latest/
runs/xlx_clean_roi_512_edge_aux_int_qat_w12/int_exports/best/
```

导出目录包含：

- `int_params.pt`：完整导出 payload
- `q_w.pt`：整数权重，`int32`
- `q_b.pt`：整数 bias，`int32`
- `shift.pt`：每输出通道右移位数，`int32`
- `export_meta.json`：bit 范围和基础元信息

## 9. 训练时重点看什么

日志会输出：

- `loss`：总 loss
- `task`：原任务 loss
- `l_distill`：teacher 对齐 loss
- `max_qw`：当前导出整数权重最大绝对值
- `max_qb`：当前导出整数 bias 最大绝对值
- `shift=min-max`：shift 范围

第一版 W12/bias17 的合法范围：

- W12：`[-2048, 2047]`
- bias17：`[-65536, 65535]`

如果 `max_qw` 或 `max_qb` 长期贴近上限，需要提高对应 range penalty，或者降低 shift / 提高 bit 后再扫。

## 10. 后续 sweep 建议

先跑通 W12：

- `weight_bits=12`
- `bias_bits=17`
- `per_channel_shift=true`
- `distillation.loss_weight=0.1`

再按顺序尝试：

- W10：压缩更强，观察 selective_score 和 edge_retention 是否明显下降
- W8：如果 W10 可接受再试
- W14/W16/W17：如果 W12 损失过大，用更高位宽确认上限效果

第一版不要训练 bit 和整数 shift。只有当固定 shift 的 QAT 已经稳定后，再考虑连续 scale 或 mixed precision。
