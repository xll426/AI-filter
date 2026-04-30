下面是按我们前面讨论整理后的 **Markdown 版知识总结**。
我按“任务定义 → 参考算法 → 模型 → 训练 Loss → 验证指标 → 关键易错点”的顺序组织，尽量把概念、公式、代码对应关系和容易混淆的地方都放在一起。你后面可以直接当项目笔记用。关于训练与验证的职责分工、默认主损失、`selective_score` 组成和当前代码落地方式，我也一起合并进去了。 

---

# AI PreFilter 项目核心知识总结

## 1. 任务本质与总体目标

这个任务不是普通的“整图尽量逼近参考图 `ref`”的图像恢复任务，而是一个**带明确区域偏好的前滤波任务**：

* **背景低价值区域**：尽量接近参考算法 `ref.py` 的滤波效果，压制不重要的高频、细碎纹理、杂质。
* **主体结构与关键边缘区域**：尽量保留原图 `ori` 中的重要结构，不能为了追 `ref` 把边缘一并磨掉。

因此，这个项目里天然存在一个张力：

* 越贴近 `ref`，背景通常越干净；
* 但越贴近 `ref`，边缘也更可能被一起削弱。

这也是为什么本项目必须把：

* **训练主损失（loss）**
* **验证主指标（metric）**

分开设计，而不能简单用一个“整图误差”统一代替。

---

## 2. 数据流与三类图像对象

在整个系统里，可以把主要图像对象分成三类：

### 2.1 原图 `ori` / `src`

这是模型输入，也是参考算法的起点。
在训练代码里，`inputs` 就是原始输入图；由于当前配置 `only_train_y=True`，真正参与主要计算的是其 **Y 通道**。 

### 2.2 参考图 `ref` / `target`

这是由参考算法 `generate_reference_tensor(...)` 从原图生成的监督目标。
训练时模型并不是直接学“原图到原图”，而是学：

$$
ori \rightarrow ref
$$

参考图不是简单平滑图，而是“背景被压制、结构细节按规则保留”的结构感知输出。

### 2.3 模型输出 `pred`

这是 `PrefilterNet` 的输出。
当前配置下，模型只真正学习 Y 通道，UV 通道直接透传：

* 输入取 `x[:, :1]` 作为 Y；
* UV 如存在则拼回输出；
* loss 和大多数验证指标只看 Y。 

---

## 3. 参考算法 `ref.py` 的完整结构

参考算法的作用，是从原始亮度图构造一个“更符合任务目标”的监督目标 `ref`。
其核心流程是：

$$
y_{uint8}
\rightarrow
\text{structure\_aware\_median}
\rightarrow
\text{fastNlMeansDenoising}
\rightarrow
\text{compute\_structure\_score}
\rightarrow
\text{smooth\_base\_layer}
\rightarrow
\text{detail\_gain}
\rightarrow
\text{match\_local\_mean\_var}
$$

对应代码在 `generate_reference_tensor(...)` 中是串起来执行的。

---

## 4. 结构分数图 (S) 的定义与意义

### 4.1 (S) 的角色

(S) 是一张与原图同尺寸的**结构重要性分数图**。
对于每个像素 ((i,j))，都有一个分数：

$$
S(i,j)\in[0,1]
$$

其含义是：

* (S(i,j)) 小：更像背景、平坦区、弱结构区；
* (S(i,j)) 大：更像边缘、角点、轮廓、需要保护的强结构区。

它不是最终输出图，而是后续“哪里该滤、哪里该保”的基础控制图。

---

### 4.2 结构张量的构造

对亮度图 (y) 先求 Scharr 梯度：

$$
g_x = \text{Scharr}_x(y), \qquad g_y = \text{Scharr}_y(y)
$$

再构造局部二阶统计量：

$$
g_x^2,\quad g_y^2,\quad g_x g_y
$$

对每个窗口大小 `win` 做局部均值，得到结构张量分量：

$$
j_{11} = \text{boxFilter}(g_x^2),\qquad
j_{22} = \text{boxFilter}(g_y^2),\qquad
j_{12} = \text{boxFilter}(g_x g_y)
$$

从而形成局部结构矩阵：

$$
J =
\begin{bmatrix}
j_{11} & j_{12} \\
j_{12} & j_{22}
\end{bmatrix}
$$

这一步的意义是：不是看单个像素，而是看**当前像素周围一小块区域整体的结构方向统计**。

---

### 4.3 特征值 (\lambda_1,\lambda_2) 的意义

代码中先定义：

$$
tr = j_{11}+j_{22}
$$

$$
\mathrm{det\_term} = \sqrt{(j_{11}-j_{22})^2 + 4j_{12}^2 + \varepsilon}
$$

然后得到两个特征值：

$$
\lambda_1 = \frac{tr + \mathrm{det\_term}}{2},\qquad
\lambda_2 = \frac{tr - \mathrm{det\_term}}{2}
$$

它们不是“横向强度”和“纵向强度”，而是当前局部窗口在**两个主方向**上的结构能量：

* (\lambda_1)：最强主方向强度
* (\lambda_2)：次强正交方向强度

其理论依据是：任意单位方向 (v) 上的局部结构强度可以写成

$$
E(v)=v^T J v,\qquad |v|=1
$$

而这个二次型在所有方向上的最大值和最小值，分别就是 (J) 的最大特征值和最小特征值。
因此：

* 最大特征值 = 最强方向结构强度
* 最小特征值 = 最弱主方向结构强度

---

### 4.4 `coherence` 与 `corner_ratio`

根据特征值构造两个更有图像意义的量：

$$
coherence = \frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\varepsilon}
$$

$$
\mathrm{corner\_ratio} = \frac{\lambda_2}{\lambda_1+\lambda_2+\varepsilon}
$$

含义如下：

* `coherence` 大：更像单方向很清晰的边缘；
* `corner_ratio` 大：第二方向也很强，更像角点、交叉、拐角等复杂结构。

---

### 4.5 `mag`、`percentile` 与 `gate`

代码中进一步定义：

$$
mag = \sqrt{tr+\varepsilon}
$$

它表示局部整体变化强度。
然后在整张 `mag` 图上取两个百分位阈值：

$$
t = percentile(mag, 60),\qquad
t_2 = percentile(mag, 90)
$$

再构造门控：

$$
gate = \operatorname{clip}\left(\frac{mag-t}{t_2-t}, 0, 1\right)
$$

含义是：

* (mag \le t)：局部变化偏弱，`gate = 0`
* (mag \ge t_2)：局部变化很强，`gate = 1`
* 中间区间：线性过渡

这一步的作用是：
即使某个位置“形态上像边/角”，如果整体变化太弱，也不要把它判成重要结构。

---

### 4.6 单尺度分数 (s) 与多尺度融合为 (S)

对于某一个窗口尺度，代码计算：

$$
s = \operatorname{clip}
\left(
gate \cdot
\max\left(coherence^\gamma,\ \alpha\cdot \mathrm{corner\_ratio}^q\right),
0,1\right)
$$

默认参数为：

* (\gamma = 2.0)
* (q = 1.0)
* (\alpha = 1.2) 

这表示：

* 边缘型结构由 (coherence^\gamma) 主导；
* 角点型结构由 (\alpha\cdot corner_ratio^q) 主导；
* 最后再乘上强度门控 `gate`。

对多个尺度 `wins=(5,9)` 分别计算 (s)，再逐像素取最大：

$$
S(i,j)=\max\big(s_{5\times5}(i,j),\ s_{9\times9}(i,j)\big)
$$

于是得到最终的结构分数图 (S)。

---

## 5. 结构感知中值滤波：只修背景坏点，不碰强结构

参考算法第一步真正使用 (S) 的地方是 `structure_aware_median(...)`。
核心公式对应代码中的掩码：

$$
\mathrm{mask} = (S < s_{thr})\ \land\ (|y - med| > \mathrm{outlier\_t})
$$

默认：

$$
s_{thr}=0.25,\qquad \mathrm{outlier\_t}=30
$$

其中：

* (med)：对 (y) 做 `medianBlur(ksize=3)` 得到的中值滤波结果；
* (|y-med|>30)：说明原像素和局部中值差得很大，像坏点；
* (S<0.25)：说明这里是低结构区，可以大胆修。

这一步的意义是：

* 平坦背景区：修离群点、脉冲噪声；
* 强结构区：尽量不动，避免边缘被中值滤波误伤。

---

## 6. 非局部均值去噪：清随机噪声

中值滤波后，参考算法继续对亮度图做：

$$
y_{denoise} = \text{fastNlMeansDenoising}(y_{medfix})
$$

参数为：

* `h = 4`
* `templateWindowSize = 7`
* `searchWindowSize = 15`。

这一步不是修单个坏点，而是进一步压制整片区域里均匀分布的随机噪声，为后续分层提供更干净的起点。

---

## 7. 基础层 `base`、细节层 `detail` 与结构感知细节回灌

### 7.1 平滑基础层 `base`

代码中：

$$
base = \mathrm{smooth\_base\_layer}(y_{denoise})
$$

其内部是：

1. `L0 smooth`
2. `guided filter`

也就是先得到结构骨架，再用引导滤波生成更自然的平滑基础层。
文档与代码对应参数为：

* ( \lambda = 0.005 )
* ( \kappa = 2.0 )
* guided filter radius = 1
* guided filter ( \varepsilon = 50^2 ) 

---

### 7.2 细节层 `detail`

定义为：

$$
detail = y_{denoise} - base
$$

含义是：

> 被 `base` 平滑掉的高频部分，就是细节层。

---

### 7.3 细节增益 `detail_gain`

参考算法再次在 `y_denoise` 上计算结构分数 (S)，然后定义：

$$
\mathrm{detail\_gain} = \operatorname{clip}(S^3,0,1)
$$

这里 (S^3) 的作用是：

* 低结构区：值被进一步压小，几乎不允许细节回来；
* 高结构区：仍保留较明显的回灌能力。

---

### 7.4 结构感知重建公式

最终重建参考亮度图：

$$
y_{ref} = base + \mathrm{detail\_gain} \cdot detail
$$

这句公式的含义是：

* 背景区：(detail_gain \approx 0)，结果接近 `base`
* 结构区：(detail_gain) 较大，重要细节被重新加回去。

所以参考算法不是“简单平滑”，而是“先分离基础层，再按结构重要性选择性回灌细节”。

---

## 8. 局部均值/方差匹配：为什么最后还有

最后一步是：

$$
out = (y_1-\mu_1)\cdot gain + \mu_0
$$

其中：

* (y_0 = y_{orig})
* (y_1 = y_{ref})
* (\mu_0,\mu_1)：原图与重建图的局部均值
* (s_0,s_1)：原图与重建图的局部标准差
* (gain = \operatorname{clip}\left(\dfrac{s_0}{s_1+\varepsilon}, 0.85, 1.18\right)) 

这一步不是把原图加回来，而是做**局部统计风格校正**：

* 主体仍然是 (y_1) 的局部结构起伏；
* 原图 (y_0) 只提供局部均值与局部对比度目标。

从局部仿射变换角度看，这个公式来自于：

$$
out = a y_1 + b
$$

要求满足：

$$
\mu_{out}=\mu_0,\qquad \sigma_{out}\approx \sigma_0
$$

于是可推出：

$$
a = gain,\qquad b = \mu_0 - gain\mu_1
$$

整理后得到：

$$
out = (y_1-\mu_1)\cdot gain + \mu_0
$$

它的作用是把前面已经合理的参考图 `y_ref` 做最后的局部亮度/对比度回正，避免结果发灰、发闷。

---

## 9. 模型结构：`PrefilterNet`

### 9.1 只训练 Y 通道

模型定义里：

$$
\mathrm{only\_train\_y} = True
$$

如果这个开关打开，则：

* 输入只取 (Y = x[:,:1])
* UV 原样透传
* 输出时再把处理后的 Y 和原 UV 拼起来。

---

### 9.2 主干流程

Y 通道经过：

1. `PixelUnshuffle(4)`
2. `MBRConv3`
3. 残差相加
4. `PixelShuffle(4)`

即：

$$
y_{restored} = PixelShuffle\big(PixelUnshuffle(y) + processing(PixelUnshuffle(y))\big)
$$

所以模型本质上是在低分辨率特征域内学习一个残差型 Y 通道前滤波器。

---

## 10. 训练 Loss 设计

当前真实代码中的训练 loss 由 `compute_train_loss(...)` 负责组装。
如果 `only_train_y=True`，则训练时只取：

$$
pred = preds[:, :1],\qquad gt = targets[:, :1]
$$

也就是只在 Y 通道上算 loss。

---

### 10.1 主项：ROI-Charbonnier

代码实现：

$$
L_{char} =
\frac{1}{N}\sum_{i,j}
w(i,j)\cdot \sqrt{(pred_{ij}-target_{ij})^2+\epsilon}
$$

其中：

$$
w(i,j)=
\begin{cases}
w_{roi}, & roi(i,j)=1\
w_{nonroi}, & roi(i,j)=0
\end{cases}
$$

默认：

* `loss_weight = 1.0`
* `eps = 1e-12`
* `roi_weight = 20.0`
* `non_roi_weight = 1.0`。 

这意味着：

* ROI 区域误差比非 ROI 区域重要 20 倍；
* 训练主收敛方向是：**稳定逼近参考图 `ref`，但更重视 ROI 区域**。

---

### 10.2 结构辅助项：MS-SSIM

代码实现是：

$$
L_{ms} = \lambda_{ms}\cdot (1-MS\text{-}SSIM(pred,target))
$$

默认：

* `loss_weight = 0.16`
* `data_range = 1.0`
* `win_size = 11`
* `win_sigma = 1.5`。 

训练时真正调用的是：

$$
L_{perceptual}=MS\text{-}SSIM(softclip01(pred),\ gt)
$$

即先把预测平滑压到更适合 `[0,1]` 范围的数值域，再算多尺度结构相似度。 

它的定位不是主收敛项，而是：

* 防止输出只会做像素平均；
* 给局部结构和多尺度感知一个辅助约束。

---

### 10.3 可选边缘辅助项：`EdgeConsistencyLoss`

如果配置中：

$$
\mathrm{edge\_aux\_loss.enabled} = True
$$

则训练时再加：

$$
L_{\mathrm{edge\_aux}} =
\lambda_e\big(
\lambda_m L_{\mathrm{match}}
+
\lambda_r L_{\mathrm{retain}}
\big)
$$

默认参数：

* `loss_weight = 0.05`
* `match_weight = 1.0`
* `retain_weight = 0.25`
* `retain_ratio = 0.90`
* `mask_quantile = 0.90`
* `mask_gamma = 1.5`。 

#### (1) 梯度匹配项

对 `pred/source/target` 的 Y 通道分别用 Sobel 算梯度：

$$
G_x,\ G_y,\ G=\sqrt{G_x^2+G_y^2+\varepsilon}
$$

再用 `target/ref` 的强边生成 soft edge mask，定义：

$$
L_{\mathrm{match}} =
\frac{
\sum M_{edge}^{soft}\cdot
\sqrt{(G_x^{pred}-G_x^{src})^2+(G_y^{pred}-G_y^{src})^2+\varepsilon}
}{
\sum M_{edge}^{soft}+\varepsilon
}
$$

作用是：让模型在重要边缘处的梯度方向与源图更接近。

#### (2) 保边惩罚项

$$
L_{\mathrm{retain}} =
\frac{
\sum M_{edge}^{soft}\cdot
\max(0,\ \rho G^{src} - G^{pred})
}{
\sum M_{edge}^{soft}+\varepsilon
}
$$

其中 (\rho = 0.90)。
它只惩罚一种情况：

> 预测边缘强度低于原图边缘强度的 90%

这是一种单向的“防磨边”约束。

---

### 10.4 当前总训练损失

默认推荐主损失是：

$$
L_{\mathrm{total}} =
1.00\cdot L_{\mathrm{charbonnier\_roi}}(pred,ref)
+
0.16\cdot L_{\mathrm{ms\_ssim}}(pred,ref)
$$

如果开启 `edge_aux_loss`，则还会再加上一小项 `L_edge_aux`。
代码里三项 loss 就是这样累加的。 

---

## 11. 验证指标体系与 `selective_score`

验证阶段，代码会同时统计：

* `val_loss`
* `psnr`
* `ssim`
* `selective_score`
* `bg_completion`
* `edge_source_completion`
* `edge_retention_ratio`
* `edge_oversmooth_vs_src`
* 以及若干底层分项指标。

而 `best.pt` 默认按：

$$
\mathrm{primary\_metric} = \mathrm{selective\_score}
$$

来选，而不是按 `val_loss` 或 `PSNR`。

---

### 11.1 为什么不能只看 `PSNR/SSIM`

因为这个任务不是“整图越像 `ref` 越好”。
真正的任务规则是：

* 背景区对 `ref`
* 边缘区对 `ori`

所以必须做结构分区。文档已经明确把这种逻辑写成 validation 主指标体系。

---

### 11.2 掩码来源

评估时的背景区 / 边缘区掩码，不是简单梯度阈值，而是复用参考算法的结构判断逻辑，默认：

* `mask_mode = detail_gain`
* `s_thr = 0.25`。 

即：

$$
M_{bg}(i,j)=\mathbf{1}[S(i,j)<0.25],\qquad
M_{edge}(i,j)=\mathbf{1}[S(i,j)\ge 0.25]
$$

---

### 11.3 底层拟合型指标

这些指标回答的是：

> 模型输出和 `ref` 到底有多像？

包括：

* `bg_hf_error`
* `edge_preserve_error`
* `edge_over_smooth_ratio`
* `edge_gmsd`
* `bg_grad_energy_ratio`
* `edge_grad_energy_ratio`
* `structure_alignment_error`。

其中典型公式可写为：

#### 背景高频误差

$$
bg\_hf\_error(pred,ref) =
\frac{\sum M_{bg}\cdot |H(pred)-H(ref)|}
{\sum M_{bg}+\varepsilon}
$$

其中 (H(X)) 是高频响应，例如 Laplacian 幅值。

#### 结构区梯度误差

$$
edge\_preserve\_error(A,B) =
\frac{\sum M_{edge}\cdot |G(A)-G(B)|}
{\sum M_{edge}+\varepsilon}
$$

其中 (G(X)) 是梯度幅值。

#### 结构区过平滑比例

$$
edge\_over\_smooth\_ratio(A,B) =
\frac{
\sum M_{edge}\cdot
\mathbf{1}[G(A)<\rho G(B)]\cdot
\mathbf{1}[G(B)\ge\tau]
}{
\sum M_{edge}\cdot \mathbf{1}[G(B)\ge\tau]+\varepsilon
}
$$

它统计的是：

> 有多少比例的有效边缘被明显磨弱了

这就是 `edge_oversmooth_vs_src` 背后的核心函数。

---

### 11.4 任务型指标

真正用于选 checkpoint 的是下面四个任务量：

#### 背景完成度

$$
bg\_completion =
1-\frac{bg\_hf\_error(pred,ref)}{bg\_hf\_error(ori,ref)}
$$

含义：
相对于原图，背景区域朝 `ref` 前进了多少。

#### 边缘保持完成度

$$
edge\_source\_completion =
1-\frac{edge\_preserve\_error(pred,ori)}{edge\_preserve\_error(ref,ori)}
$$

含义：
相对于参考算法 `ref`，模型是否更接近原图边缘。

#### 边缘保留比

$$
edge\_retention\_ratio = edge\_grad\_energy\_ratio(pred,ori)
$$

含义：
结构区整体边缘能量还剩多少，越接近 1 越好。

#### 边缘过平滑比例

$$
edge\_oversmooth\_vs\_src = edge\_over\_smooth\_ratio(pred,ori)
$$

含义：
结构区里有多少比例的边被明显磨弱了，越小越好。

---

### 11.5 最终主指标 `selective_score`

文档给出的推荐主指标是：

$$
selective\_score =
100\cdot
\Big(
0.45\cdot bg\_completion
+
0.35\cdot \operatorname{clip}(edge\_source\_completion,-1,1)
+
0.10\cdot \frac{\operatorname{clip}(edge\_retention\_ratio,0,1.2)}{1.2}
+
0.10\cdot \max(0,1-edge\_oversmooth\_vs\_src)
\Big)
$$

各项意义如下：

* `0.45`：背景完成度，前滤波本职任务；
* `0.35`：边缘保持完成度，主体边不能坏；
* `0.10`：边缘能量保留，结构安全护栏；
* `0.10`：过平滑惩罚，防止边缘被大面积磨平。 

注意：

* 这个分数不是“参考图 `ref` 自己一定 100 分”的绝对分；
* 因为 `ref` 本身也会牺牲一部分边缘，所以 `ref` 自己在这套体系下也不一定满分。

---

## 12. 训练与验证职责分工

这是整个项目最重要的“系统观”。

### 12.1 训练主损失负责什么

训练 loss 负责：

* 可微
* 稳定
* 快速收敛
* 学到从 `ori` 到 `ref` 的基本映射

默认主损失就是：

$$
L_{\mathrm{total}} = 1.00\cdot L_{\mathrm{charbonnier\_roi}} + 0.16\cdot L_{\mathrm{ms\_ssim}}
$$

必要时再开小权重 `edge_aux_loss`。

### 12.2 验证主指标负责什么

validation 的主指标负责：

> 挑出真正“背景够干净、边又没有坏太多”的模型

所以默认是：

$$
\mathrm{primary\_metric} = \mathrm{selective\_score}
$$

而不是 `val_loss`。

### 12.3 为什么不能把 `selective_score` 直接当主 loss

因为它里面包含：

* percentile gating
* 二值 mask
* `clip/max/min`
* Numpy/CPU 路径

这些非常适合做评估，但不适合直接做主反传损失；否则容易导致：

* 梯度不稳定
* 训练发散或抖动
* 模型投机优化统计量而非整体画质。

---

## 13. `PSNR` 与 `SSIM` 的定位

### 13.1 `PSNR`

来源于均方误差：

$$
MSE=\frac{1}{N}\sum (pred-ref)^2
$$

$$
PSNR = 10\log_{10}\left(\frac{MAX^2}{MSE}\right)
$$

它更偏“像素数值保真”，回答的是：

> 整图平均误差大不大

但它不懂任务分区规则，所以只能做辅助监控。

---

### 13.2 `SSIM`

局部窗口内比较：

* 亮度相似性
* 对比度相似性
* 结构相似性

标准公式是：

$$
SSIM(x,y)=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$

它比 `PSNR` 更接近视觉，但仍然不懂“背景对 ref、边缘对 ori”的任务规则，因此也只是辅助参考，不是主验证指标。

---

## 14. 当前代码的真实落地状态

根据你现在贴出的代码，可以确认：

1. **模型只训练 Y 通道**，UV 透传。
2. **训练 loss** 至少包括：

   * `ROI-Charbonnier`
   * `MS-SSIM`
   * 可选 `EdgeConsistencyLoss`。
3. **validation** 同时统计：

   * `loss`
   * `psnr`
   * `ssim`
   * `selective_score` 及其子指标。
4. **best.pt 默认按 `selective_score` 选**。

---

## 15. 最终结论

这个项目的核心不是“做一个普通去噪网络”，而是：

1. 用参考算法先定义一个**结构感知的监督目标 `ref`**；
2. 用**稳定的训练主损失**让模型学到从 `ori` 到 `ref` 的基本映射；
3. 再用**任务型验证指标 `selective_score`**，从“背景完成度”和“边缘保持度”两个角度选择真正更符合业务目标的模型。

简洁地说：

$$
\text{训练负责稳收敛，验证负责挑对模型}
$$

这也是你当前整套 loss 与指标体系设计最专业、最核心的地方。 

---

如果你想，我可以把这份整理继续扩成两个版本：一个“简版速查笔记”，一个“面试/汇报版讲稿”。
