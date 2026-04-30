# 部署微调说明：为什么要做整数 QAT

这份文档只保留部署微调的关键推导。可执行训练方案见 `int_qat_training_plan.md`。

## 1. slim 后的部署模型

当前训练态 `PrefilterNet` 是多分支重参数化结构，但部署前可以通过 `MBRConv3.slim()` 融合成单个 `3x3` 卷积。

对 Y 通道而言，部署形态是：

$$
Y_u = \operatorname{PixelUnshuffle}(Y)
$$

$$
\Delta_u = \operatorname{Conv}_{3\times3}(Y_u, W) + b
$$

$$
\hat{Y}_u = Y_u + \Delta_u
$$

$$
\hat{Y} = \operatorname{PixelShuffle}(\hat{Y}_u)
$$

也就是说，模型不是重建整张图，而是在原图上加一个残差修正量。

## 2. 为什么整数权重还需要缩放

假设某个理想 FP32 残差核是：

$$
K =
\begin{bmatrix}
0 & 0.05 & 0 \\
0.05 & -0.20 & 0.05 \\
0 & 0.05 & 0
\end{bmatrix}
$$

对一个局部块：

$$
X =
\begin{bmatrix}
0 & 100 & 0 \\
105 & 120 & 115 \\
0 & 110 & 0
\end{bmatrix}
$$

残差为：

$$
\Delta =
0.05 \times 100
+ 0.05 \times 105
- 0.20 \times 120
+ 0.05 \times 115
+ 0.05 \times 110
= -2.5
$$

输出约为：

$$
120 + (-2.5) = 117.5 \approx 118
$$

如果不允许缩放，只能用裸整数核，例如：

$$
K_{\mathrm{int}} =
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

则：

$$
\Delta_{\mathrm{int}}
= 100 + 105 - 4 \times 120 + 115 + 110
= -50
$$

输出会变成 `70`，强度完全不对。问题不是最终输出能不能 round 成整数，而是整数核本身无法表达小数滤波强度。

## 3. 定点表达

引入每通道 shift：

$$
W_{\mathrm{eff}} = \frac{Q_w}{2^s}
$$

如果 `s=8`，则：

$$
0.05 \times 256 \approx 13,\quad -0.20 \times 256 \approx -51
$$

硬件保存：

$$
Q =
\begin{bmatrix}
0 & 13 & 0 \\
13 & -51 & 13 \\
0 & 13 & 0
\end{bmatrix}
$$

整数 MAC 后右移：

$$
acc = 13 \times 100 + 13 \times 105 - 51 \times 120 + 13 \times 115 + 13 \times 110 = -530
$$

$$
\Delta = \operatorname{round}\left(\frac{-530}{256}\right) \approx -2
$$

输出：

$$
120 - 2 = 118
$$

这就是 shift 的本质：让整数参数表示定点小数核。

## 4. bias 为什么不能替代缩放

卷积残差是：

$$
\Delta = \sum_i w_i x_i + b
$$

其中 `sum_i w_i x_i` 根据局部邻域变化，bias 只是固定偏移。bias 能整体加亮或减暗，但不能表达“边缘处减 2、平坦处不动、不同纹理不同响应”的局部结构规则。

因此 bias 不能替代小数卷积核，也不能替代定点 shift。

## 5. QAT 训练路径

训练仍然吃当前 dataloader 的 `[0,1]` 输入，但模型内部模拟 raw `0..255`：

$$
x_{\mathrm{raw}} = \operatorname{round}(255x_{\mathrm{norm}})
$$

$$
Q_w = \operatorname{clamp}(\operatorname{round}(W_{\mathrm{fp}}2^s), q_{\min}, q_{\max})
$$

$$
Q_b = \operatorname{clamp}(\operatorname{round}(b_{\mathrm{raw}}2^s), b_{\min}, b_{\max})
$$

$$
\Delta = \operatorname{round}\left(\frac{\operatorname{Conv}(Y_u,Q_w)+Q_b}{2^s}\right)
$$

$$
Y_u^{out} = \operatorname{clip}(Y_u+\Delta,0,255)
$$

再除以 `255` 回 `[0,1]` 接现有 loss。

反向不直接训练整数参数，而是训练浮点主参数 `W_fp` 和 `b_raw`，对 `round/clamp/clip` 使用 STE。

## 6. 当前建议

第一版先固定：

- `weight_bits=12`
- `bias_bits=17`
- `per_channel_shift=true`
- `shift` 初始化后不训练
- 加 deploy-FP32 teacher distillation
- 加 weight/bias range penalty

命令：

```bash
python train_int_qat.py \
  --config configs/int_qat_xlx_clean_roi_512_edge_aux.yaml
```

训练完成后使用导出的：

- `q_w.pt`
- `q_b.pt`
- `shift.pt`
- `export_meta.json`

这些文件就是后续硬件 bit-true reference 和 RTL 对齐的输入。
