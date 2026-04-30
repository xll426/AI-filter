# Loss 总结：Charbonnier Loss、MS-SSIM Loss、Edge Loss

这份训练目标可以理解为三层约束：

$$
L_{\mathrm{total}}
= L_{\mathrm{char}}
+ L_{\mathrm{msssim}}
+ L_{\mathrm{edge}}
$$

更贴近当前代码实现时，可以写成：

$$
L_{\mathrm{total}}
=
\lambda_{\mathrm{char}} L_{\mathrm{char}}
+
\lambda_{\mathrm{ms}} L_{\mathrm{msssim}}
+
\lambda_e\left(
\lambda_m L_{\mathrm{match}}
+
\lambda_r L_{\mathrm{retain}}
\right)
$$

其中当前配置大致对应：

```text
lambda_char = 1.0
lambda_ms = 0.16
lambda_e = 0.05
lambda_m = 1.0
lambda_r = 0.25
retain_ratio = 0.90
mask_quantile = 0.90
mask_gamma = 1.5
```

整体目标不是单纯让模型复制参考算法，而是：

```text
整体亮度/背景效果：靠近参考算法 target
局部视觉结构：靠近参考算法 target
关键边缘：尽量保留原始输入 source 的边缘
```

---

# 1. Charbonnier Loss：像素级拟合参考图

Charbonnier Loss 可以看成平滑版 L1 loss。单个像素的形式是：

$$
\ell_{\mathrm{char}}(p,t)
=
\sqrt{(p-t)^2+\epsilon}
$$

其中：

* $p$：模型输出，即 `pred`；
* $t$：参考算法生成的伪标签，即 `target`；
* $\epsilon$：很小的常数，防止 0 附近不可导。

如果忽略 $\epsilon$，它接近：

$$
|p-t|
$$

当前代码里还支持 ROI 加权，因此更准确的批量形式是：

$$
L_{\mathrm{char}}
=
\lambda_{\mathrm{char}}
\cdot
\frac{1}{N}
\sum_i
w_i
\sqrt{(pred_i-target_i)^2+\epsilon}
$$

其中：

$$
w_i =
\begin{cases}
w_{\mathrm{roi}}, & i \in ROI \\
w_{\mathrm{nonroi}}, & i \notin ROI
\end{cases}
$$

在当前训练配置中，ROI 区域权重大，非 ROI 区域权重小。它主要约束：

```text
亮度值是否正确
整体颜色/灰度是否贴近 target
背景滤波结果是否接近 target
```

它的缺点是只看像素差，不真正理解局部结构。因此如果只用它，模型可能学出比较平滑、平均化的结果。

---

# 2. MS-SSIM Loss：局部视觉结构拟合

MS-SSIM Loss 是：

$$
L_{\mathrm{msssim}}
=
\lambda_{\mathrm{ms}}
\left(
1 - MS\text{-}SSIM(pred,target)
\right)
$$

当前代码中实际调用时是：

$$
L_{\mathrm{msssim}}
=
\lambda_{\mathrm{ms}}
\left(
1 - MS\text{-}SSIM(\mathrm{softclip01}(pred),target)
\right)
$$

它不是单纯比较像素差，而是比较局部区域的视觉结构：

```text
一小块图像是否相似，不只看每个像素是否一样，
还要看这一小块整体亮度、对比度、结构变化是否一致。
```

---

# 3. SSIM 的核心公式

对于两个局部图像块 $x$ 和 $y$：

$$
SSIM(x,y)
=
\frac{
(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)
}{
(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)
}
$$

其中：

* $\mu_x,\mu_y$：局部均值；
* $\sigma_x^2,\sigma_y^2$：局部方差；
* $\sigma_{xy}$：局部协方差；
* $C_1,C_2$：稳定常数，防止分母过小。

---

# 4. 局部均值：代表整体亮度水平

普通均值是：

$$
\mu_x
=
\frac{1}{N}
\sum_{i=1}^{N}x_i
$$

如果是高斯加权形式：

$$
\mu_x
=
\sum_i w_i x_i,
\qquad
\sum_i w_i = 1
$$

这里的 $w_i$ 是窗口权重，一般中心像素权重大，周围像素权重小。

例如：

$$
A=
\begin{bmatrix}
98 & 100 & 102 \\
99 & 101 & 103 \\
100 & 102 & 104
\end{bmatrix}
$$

均值大约是：

$$
\mu_A=101
$$

如果另一块是：

$$
B=
\begin{bmatrix}
198 & 200 & 202 \\
199 & 201 & 203 \\
200 & 202 & 204
\end{bmatrix}
$$

均值大约是：

$$
\mu_B=201
$$

所以均值对应人眼看到的“这一片整体偏暗还是偏亮”。

---

# 5. 局部方差：代表对比度和亮度起伏

只看均值是不够的。例如：

$$
A=
\begin{bmatrix}
100 & 100 & 100 \\
100 & 100 & 100 \\
100 & 100 & 100
\end{bmatrix}
$$

和：

$$
B=
\begin{bmatrix}
50 & 100 & 150 \\
50 & 100 & 150 \\
50 & 100 & 150
\end{bmatrix}
$$

它们均值都可能是 100，但视觉上：

```text
A 是平坦区域；
B 有明显亮暗变化，可能有边缘。
```

方差定义为：

$$
\sigma_x^2
=
\frac{1}{N}
\sum_i
(x_i-\mu_x)^2
$$

含义是每个像素距离局部平均亮度有多远：

$$
\sigma_x^2 \text{ 大}
\Rightarrow
\text{局部对比度强}
$$

$$
\sigma_x^2 \text{ 小}
\Rightarrow
\text{局部区域平坦}
$$

---

# 6. 局部协方差：代表结构变化是否同步

协方差是 SSIM 里非常关键的一项：

$$
\sigma_{xy}
=
\frac{1}{N}
\sum_i
(x_i-\mu_x)(y_i-\mu_y)
$$

它比较的是 $x$ 和 $y$ 在同一位置上的亮暗变化是否同步。

例如原图局部：

$$
x=[50,\ 100,\ 150]
$$

预测图局部：

$$
y=[60,\ 110,\ 160]
$$

两者都是左边暗、中间过渡、右边亮。去均值后：

$$
x-\mu_x=[-50,\ 0,\ 50]
$$

$$
y-\mu_y=[-50,\ 0,\ 50]
$$

对应位置相乘后大多为正，所以协方差大，说明两张图局部结构变化方向一致。

如果预测图反过来：

$$
y=[150,\ 100,\ 50]
$$

那么一个是左暗右亮，一个是左亮右暗，协方差会变小甚至为负。

---

# 7. SSIM 为什么能惩罚模糊

假设原图边缘是：

$$
x=[50,\ 50,\ 200,\ 200]
$$

预测图被模糊成：

$$
y=[80,\ 110,\ 140,\ 170]
$$

二者均值可能差不多，但是：

```text
原图亮暗跳变强，方差大；
预测图变化更平缓，方差变小；
原图和预测图的变化不再完全同步，协方差下降。
```

所以 SSIM 会下降。这就是 SSIM 比普通 L1 更能感知模糊的原因。

---

# 8. MS-SSIM：多尺度结构相似性

SSIM 只在一个尺度上看局部窗口。MS-SSIM 会在多个尺度上计算结构相似性：

```text
原图尺度：看细节纹理
下采样后：看中等结构
继续下采样：看大轮廓
```

在这个任务里，MS-SSIM 的作用是让模型输出不仅像素接近 `target`，而且视觉结构、局部对比度、纹理变化也接近 `target`。

---

# 9. Edge Loss：关键边缘保护

Edge Loss 不是让模型继续接近 `target`，而是：

```text
在重要边缘区域，让 pred 的边缘尽量接近 source，
防止模型为了学习参考图 target 而把原图边缘抹掉。
```

如果只用：

$$
L_{\mathrm{char}}(pred,target)
+
L_{\mathrm{msssim}}(pred,target)
$$

模型会倾向于完全学习 `target`。这样背景可能更干净，但边缘也可能被一起学平滑。

所以 Edge Loss 的作用是：

```text
主 loss 学 target 的干净背景；
edge loss 把关键边缘拉回 source。
```

---

# 10. Edge Loss 的计算链路

当前代码里的 Edge Loss 计算过程是：

```text
pred / source / target
   ↓
取 Y 通道
   ↓
用 Sobel 计算 Gx、Gy
   ↓
计算梯度幅值 G
   ↓
用 target 的 G 生成 soft edge mask
   ↓
在 mask 加权区域比较 pred 和 source 的梯度
   ↓
得到 edge loss
```

---

# 11. Sobel 梯度：不是插值，而是差分求导

Sobel 用来计算局部亮度变化。对于一个 $3\times3$ 小块：

$$
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix}
$$

Sobel-x 核是：

$$
K_x=
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$$

计算结果为：

$$
G_x
=
(c+2f+i)
-
(a+2d+g)
$$

它的含义是：

```text
右边一列加权亮度 - 左边一列加权亮度
```

所以 $G_x$ 表示左右方向的亮度变化。

Sobel-y 核是：

$$
K_y=
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$

计算结果为：

$$
G_y
=
(g+2h+i)
-
(a+2b+c)
$$

它表示上下方向的亮度变化。

---

# 12. Sobel 核为什么是三列、中间为 0、还有权重 2

Sobel 的本质是：

```text
差分求导 + 邻域平滑
```

对于 $G_x$：

$$
K_x=
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
1 \\
2 \\
1
\end{bmatrix}
\begin{bmatrix}
-1 & 0 & 1
\end{bmatrix}
$$

其中：

$$
[-1,\ 0,\ 1]
$$

表示左右差分，即“右边 - 左边”。中间为 0，是因为求导只关心两侧变化，不把中心点直接加进去。

而：

$$
[1,\ 2,\ 1]^T
$$

表示在垂直方向做加权平滑：

```text
中间行离当前像素最近，权重大；
上下行离当前像素稍远，权重小。
```

所以 Sobel-x 不是只看一行的左右差，而是：

```text
沿 x 方向做差分；
同时沿 y 方向做平滑抗噪。
```

---

# 13. 梯度幅值 G：边缘强度

得到 $G_x$ 和 $G_y$ 后，计算：

$$
G
=
\sqrt{G_x^2+G_y^2+\epsilon}
$$

其中 $\epsilon$ 是小常数，防止数值问题。$G$ 是梯度向量的长度，表示这个像素附近亮度变化有多强：

$$
G \text{ 大}
\Rightarrow
\text{强边缘}
$$

$$
G \text{ 小}
\Rightarrow
\text{平坦区域}
$$

---

# 14. Edge Mask：边缘重要性权重图

Edge Loss 不是全图平均比较梯度，而是重点关注重要边缘。mask 的构造是：

$$
q
=
\operatorname{Quantile}_{0.9}(G_{\mathrm{target}})
$$

也就是找到 `target` 梯度幅值的第 90 百分位。然后：

$$
M_0
=
\operatorname{clip}
\left(
\frac{G_{\mathrm{target}}}{q+\epsilon},
0,
1
\right)
$$

最后：

$$
M
=
M_0^{1.5}
$$

这个指数会进一步压低弱边缘权重。例如：

$$
0.5^{1.5}\approx0.35,
\qquad
0.8^{1.5}\approx0.72,
\qquad
1^{1.5}=1
$$

所以最终：

```text
强边缘：权重大；
弱纹理：权重变小；
平坦区域：几乎不参与 edge loss。
```

注意这个 mask 不是二值边缘图，而是边缘重要性权重图。当前代码还会对这个 mask 做 `detach()`，避免梯度通过 mask 反向影响 `target` 分支。

---

# 15. 为什么 Edge Mask 来自 target

Edge mask 使用 `target` 的梯度生成，而不是 `pred` 或 `source`。原因是：

```text
pred 在训练早期不稳定，用 pred 生成 mask 会乱；
source 可能包含噪声边缘；
target 是参考算法过滤后的结果，更适合作为“哪些边缘值得关注”的指导。
```

所以 `target` 的梯度用于告诉模型哪些边缘是参考算法认为仍然重要的结构。

---

# 16. Edge Loss 的第一项：梯度对齐项

第一项是：

$$
L_{\mathrm{match}}
=
\frac{
\sum_i
M_i
\sqrt{
(G_{x,i}^{pred}-G_{x,i}^{src})^2
+
(G_{y,i}^{pred}-G_{y,i}^{src})^2
+
\epsilon
}
}{
\max(\sum_i M_i,\epsilon)
}
$$

其中：

* $G_x^{pred},G_y^{pred}$：模型输出的梯度；
* $G_x^{src},G_y^{src}$：原图输入的梯度；
* $M$：边缘权重 mask。

这一项比较的不是像素，而是梯度向量。它同时约束：

```text
边缘强度是否接近；
边缘方向是否接近。
```

如果 `source` 是竖直边缘，`pred` 变成斜边缘，即使梯度幅值类似，$(G_x,G_y)$ 的组合也会不同，所以仍然会被惩罚。

这一项是对称的：`pred` 比 `source` 弱会罚，`pred` 比 `source` 强也会罚。

---

# 17. Edge Loss 的第二项：边缘保持项

第二项是：

$$
L_{\mathrm{retain}}
=
\frac{
\sum_i
M_i
\operatorname{ReLU}
\left(
rG_i^{src}-G_i^{pred}
\right)
}{
\max(\sum_i M_i,\epsilon)
}
$$

其中当前配置：

$$
r=0.9
$$

并且：

$$
G^{src}
=
\sqrt{
(G_x^{src})^2+
(G_y^{src})^2+
\epsilon
}
$$

$$
G^{pred}
=
\sqrt{
(G_x^{pred})^2+
(G_y^{pred})^2+
\epsilon
}
$$

这一项不是对称误差，而是单边约束。它等价于要求：

$$
G^{pred}
\ge
0.9G^{src}
$$

也就是：

```text
pred 的边缘强度至少保留 source 的 90%。
```

---

# 18. ReLU 在 Edge Loss 中的位置

ReLU 不在 $L_{\mathrm{match}}$ 里。Edge Loss 是两个子 loss：

$$
L_{\mathrm{edge}}
=
\lambda_e
\left(
\lambda_m L_{\mathrm{match}}
+
\lambda_r L_{\mathrm{retain}}
\right)
$$

其中：

$$
L_{\mathrm{match}}
=
\frac{
\sum_i M_i \cdot diff_i
}{
\max(\sum_i M_i,\epsilon)
}
$$

而：

$$
L_{\mathrm{retain}}
=
\frac{
\sum_i M_i
\operatorname{ReLU}(0.9G_i^{src}-G_i^{pred})
}{
\max(\sum_i M_i,\epsilon)
}
$$

也就是说：

```text
diff 负责边缘向量对齐；
ReLU 只存在于 retain 项里，负责防止边缘过弱。
```

---

# 19. ReLU 保持项如何影响梯度

对单个像素忽略 mask 后：

$$
\ell_{\mathrm{retain}}
=
\operatorname{ReLU}(0.9G_{src}-G_{pred})
$$

分段看：

$$
\ell_{\mathrm{retain}}
=
\begin{cases}
0.9G_{src}-G_{pred}, & G_{pred}<0.9G_{src} \\
0, & G_{pred}\ge0.9G_{src}
\end{cases}
$$

当：

$$
G_{pred}<0.9G_{src}
$$

loss 对 $G_{pred}$ 的梯度是：

$$
\frac{\partial \ell}{\partial G_{pred}}
=
-1
$$

优化器更新时会使 $G_{pred}$ 增大，也就是说如果 `pred` 边缘太弱，loss 会推动模型把边缘变强。

当：

$$
G_{pred}\ge0.9G_{src}
$$

loss 为 0，梯度为 0。边缘已经足够强，就不再通过 retain 项继续推高。

---

# 20. 为什么是 0.9 而不是 1.0

如果用：

$$
\operatorname{ReLU}(G_{src}-G_{pred})
$$

就要求：

$$
G_{pred}\ge G_{src}
$$

这太严格。参考算法和模型本来就是要做滤波，允许边缘轻微变弱是合理的。所以用：

$$
0.9G_{src}
$$

表示允许最多约 10% 的边缘强度下降。这可以避免训练过程过于刚性，也能减少震荡。

---

# 21. Edge Loss 的最终形式

完整可以写成：

$$
L_{\mathrm{edge}}
=
\lambda_e
\left(
\lambda_m L_{\mathrm{match}}
+
\lambda_r L_{\mathrm{retain}}
\right)
$$

其中：

$$
L_{\mathrm{match}}
=
\frac{
\sum_i
M_i
\left\|
\nabla pred_i-\nabla src_i
\right\|_2
}{
\max(\sum_i M_i,\epsilon)
}
$$

$$
L_{\mathrm{retain}}
=
\frac{
\sum_i
M_i
\operatorname{ReLU}
\left(
0.9
\left\|
\nabla src_i
\right\|_2
-
\left\|
\nabla pred_i
\right\|_2
\right)
}{
\max(\sum_i M_i,\epsilon)
}
$$

这里：

$$
\nabla pred_i=(G_{x,i}^{pred},G_{y,i}^{pred})
$$

$$
\nabla src_i=(G_{x,i}^{src},G_{y,i}^{src})
$$

---

# 22. 三个 Loss 的分工关系

| Loss | 主要比较对象 | 关注内容 | 作用 |
| --- | --- | --- | --- |
| Charbonnier | `pred` vs `target` | 像素值，带 ROI 权重 | 学参考算法输出 |
| MS-SSIM | `pred` vs `target` | 局部亮度、对比度、结构 | 学参考算法的视觉结构 |
| Edge Loss | `pred` vs `source`，mask 来自 `target` | 重要边缘梯度 | 防止边缘被抹掉 |

---

# 23. 总体训练目标的真实含义

虽然表面上 `target` 是训练标签，但整体 loss 并不是让模型完全复制 `target`。

更准确地说：

```text
Charbonnier 和 MS-SSIM 让模型学习 target 的滤波效果；
Edge Loss 让模型在关键边缘处保留 source 的结构。
```

因此训练目标是：

```text
背景区域：向 target 靠近，学习干净和平滑；
边缘区域：不能盲目向 target 靠近，要保留 source 的边缘强度和方向。
```

这和参考算法本身的目标一致：

```text
平坦区域强滤波；
结构区域保细节。
```

---

# 24. 最终核心总结

整个 loss 体系可以浓缩成：

$$
\begin{aligned}
L_{\mathrm{total}}
=&
\underbrace{
L_{\mathrm{char}}(pred,target)
}_{\text{像素拟合参考图}}
+
\underbrace{
L_{\mathrm{msssim}}(pred,target)
}_{\text{局部结构拟合参考图}}
\\
&+
\underbrace{
L_{\mathrm{edge}}(pred,source,target)
}_{\text{关键边缘保留原图}}
\end{aligned}
$$

其中最关键的是：

$$
\begin{aligned}
L_{\mathrm{edge}}
=&
\lambda_e
\Bigg[
\lambda_m
\underbrace{
\frac{
\sum_i M_i
\left\|
\nabla pred_i-\nabla src_i
\right\|_2
}{
\max(\sum_i M_i,\epsilon)
}
}_{\text{边缘方向和强度对齐}}
\\
&+
\lambda_r
\underbrace{
\frac{
\sum_i M_i
\operatorname{ReLU}
\left(
0.9
\left\|
\nabla src_i
\right\|_2
-
\left\|
\nabla pred_i
\right\|_2
\right)
}{
\max(\sum_i M_i,\epsilon)
}
}_{\text{边缘强度下限约束}}
\Bigg]
\end{aligned}
$$

最本质的理解是：

```text
Charbonnier：像素值要像 target
MS-SSIM：局部视觉结构要像 target
Edge Loss：重要边缘不能被 target 和网络一起抹弱
```

最终模型学到的是：

```text
参考算法的背景清洁能力
+
原始输入的关键边缘保持能力
```

---

# 25. 这版文档顺手修正的问题

原文主要问题是公式 Markdown 写法不规范，导致很多地方不能正常渲染：

```text
[ ... ] 应该改成 $$ ... $$
公式里的 ===== 应该改成 LaTeX 的 =
矩阵行尾应该用 \\，不是单个反斜杠
Sobel 公式里被误写成 Markdown 标题的 ## 应该改成减号表达式
下标和范数里的 L*{...}、|...|*2 应该改成 L_{...}、\|\cdot\|_2
```

内容上也补齐了几个和代码实现相关的点：

```text
Charbonnier 实际带 ROI 权重；
MS-SSIM 实际对 pred 先做 softclip01；
Edge loss 实际有 loss_weight、match_weight、retain_weight；
Edge mask 是 soft mask，并且在代码里 detach；
分母实际用 max(sum M, eps) 防止全零 mask。
```
