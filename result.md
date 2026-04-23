# Prefilter 评估结果记录

## 1. 评估对象

本次评估使用以下 4 组数据：

- `ori`
  - `/mnt/d/fudan/AI前滤波_V0302/AI-PreFilter/datasets/xlx_yuv_tiled/ori`
- `ref`
  - `/mnt/d/fudan/AI前滤波_V0302/AI-PreFilter/datasets/xlx_yuv_tiled/ref`
- `fudan_infer`
  - `/mnt/d/fudan/AI前滤波_V0302/AI-PreFilter/datasets/xlx_yuv_tiled/fudan_infer`
- `filt_test_best_loss`
  - `/mnt/d/fudan/AI前滤波_V0302/AI-PreFilter/datasets/xlx_yuv_tiled/filt_test_best_loss`

经过文件名严格求交集后，4 组数据共有：

- `1920` 个可一一配对的 `.yuv` tile

评估使用的正式脚本：

- [evaluate_prefilter_dataset.py](/mnt/d/fudan/prefilter_clean/evaluate_prefilter_dataset.py)

主评估函数：

- [evaluate_selective_prefilter_y](/mnt/d/fudan/prefilter_clean/iqa_metrics_exact_refalgo_y.py:553)

---

## 2. 指标口径

### 2.1 严格拟合 `ref` 的指标

这些指标回答的是：

“模型输出和 `ref` 有多像？”

- `bg_hf_error`
  - 背景区高频误差，越小越好
- `edge_preserve_error`
  - 结构区梯度误差，越小越好
- `edge_over_smooth_ratio`
  - 相对 `ref` 的过平滑比例，越小越好
- `edge_gmsd`
  - 结构区梯度相似性波动，越小越好
- `bg_grad_energy_ratio`
  - 背景区梯度能量比，越接近 `1` 越好
- `edge_grad_energy_ratio`
  - 结构区梯度能量比，越接近 `1` 越好
- `structure_alignment_error`
  - 结构图误差，越小越好

### 2.2 任务型指标

这些指标回答的是：

“背景是否接近 `ref`，同时边缘是否保住了 `ori`？”

- `bg_completion`
  - 背景完成度，越大越好
- `edge_source_completion`
  - 边缘保持完成度，越大越好
  - `> 0` 表示比 `ref` 更接近原图边缘
  - `< 0` 表示比 `ref` 还更伤边
- `edge_retention_ratio`
  - 边缘能量保留比，越接近 `1` 越好
- `edge_oversmooth_vs_src`
  - 相对原图的边缘过平滑比例，越小越好
- `selective_score`
  - 最终任务主指标，越大越好

注意：

- `selective_score` 不是“纯拟合 ref 分数”
- 在这套定义下，`ref` 自己也不会是 `100`
- 因为边缘区故意拿 `ori` 当目标，避免“越像 ref 越好”把边一起磨掉

---

## 3. 全量评估结果

全量汇总文件：

- [eval_prefilter_full.json](/mnt/d/fudan/prefilter_clean/eval_prefilter_full.json)

评估范围：

- `1920 tile`
- `mask_mode = detail_gain`
- `s_thr = 0.25`

### 3.1 核心结论

如果只看“背景像不像 `ref`”：

- `filt_test_best_loss` 更强

如果按真实任务目标“背景要滤干净，主体边缘要保住”：

- `fudan_infer` 更好

也就是说：

- `filt_test_best_loss`
  - 背景滤波更接近 `ref`
  - 但边缘伤得更重
- `fudan_infer`
  - 背景没有 `best_loss` 那么干净
  - 但边缘明显更安全

### 3.2 主指标均值对比

| 模型 | selective_score | bg_completion | edge_source_completion | edge_retention_ratio | edge_oversmooth_vs_src |
|---|---:|---:|---:|---:|---:|
| `fudan_infer` | `32.5852` | `0.2353` | `0.2562` | `0.9102` | `0.4556` |
| `filt_test_best_loss` | `26.5403` | `0.4504` | `-0.1618` | `0.8556` | `0.5204` |

解释：

- `filt_test_best_loss` 的 `bg_completion` 更高，说明背景更像 `ref`
- 但它的 `edge_source_completion < 0`，说明边缘已经比 `ref` 还更伤
- `fudan_infer` 的 `edge_retention_ratio = 0.9102`，显著优于 `best_loss = 0.8556`
- 所以最终 `selective_score` 由 `fudan_infer` 胜出

---

## 4. 全量详细统计

### 4.1 `fudan_infer`

| 指标 | mean | median | p10 | p90 |
|---|---:|---:|---:|---:|
| `selective_score` | `32.585224` | `34.618662` | `23.765056` | `42.521122` |
| `bg_completion` | `0.235315` | `0.376018` | `-0.181496` | `0.452524` |
| `edge_source_completion` | `0.256209` | `0.251621` | `-0.038185` | `0.546191` |
| `edge_retention_ratio` | `0.910220` | `0.893962` | `0.855114` | `1.011026` |
| `edge_oversmooth_vs_src` | `0.455628` | `0.505235` | `0.185230` | `0.606527` |
| `bg_hf_error` | `20.149452` | `14.345866` | `4.140853` | `41.319276` |
| `edge_preserve_error` | `96.470611` | `111.611984` | `19.057744` | `156.564586` |
| `edge_over_smooth_ratio` | `0.199355` | `0.200583` | `0.130389` | `0.263368` |
| `edge_gmsd` | `0.119237` | `0.107852` | `0.090313` | `0.161787` |
| `bg_grad_energy_ratio` | `1.509136` | `1.327647` | `1.164038` | `2.027280` |
| `edge_grad_energy_ratio` | `1.086517` | `1.055050` | `0.993110` | `1.208508` |
| `structure_alignment_error` | `0.037583` | `0.031919` | `0.019390` | `0.063433` |
| `bg_area_ratio` | `0.731984` | `0.722118` | `0.696361` | `0.777294` |
| `edge_area_ratio` | `0.268016` | `0.277882` | `0.222706` | `0.303639` |

### 4.2 `filt_test_best_loss`

| 指标 | mean | median | p10 | p90 |
|---|---:|---:|---:|---:|
| `selective_score` | `26.540304` | `26.662923` | `13.616790` | `39.605080` |
| `bg_completion` | `0.450446` | `0.612927` | `-0.103662` | `0.731073` |
| `edge_source_completion` | `-0.161784` | `-0.197654` | `-0.661222` | `0.389409` |
| `edge_retention_ratio` | `0.855559` | `0.831286` | `0.772359` | `1.010411` |
| `edge_oversmooth_vs_src` | `0.520354` | `0.576890` | `0.222933` | `0.658310` |
| `bg_hf_error` | `11.219421` | `9.028636` | `3.692158` | `20.500827` |
| `edge_preserve_error` | `103.296102` | `95.518719` | `18.017449` | `208.184172` |
| `edge_over_smooth_ratio` | `0.235279` | `0.216226` | `0.119787` | `0.391868` |
| `edge_gmsd` | `0.115527` | `0.108983` | `0.080960` | `0.162137` |
| `bg_grad_energy_ratio` | `1.397143` | `1.240433` | `1.064797` | `1.850472` |
| `edge_grad_energy_ratio` | `1.019819` | `0.980079` | `0.906224` | `1.141305` |
| `structure_alignment_error` | `0.037764` | `0.032840` | `0.016052` | `0.074476` |
| `bg_area_ratio` | `0.731984` | `0.722118` | `0.696361` | `0.777294` |
| `edge_area_ratio` | `0.268016` | `0.277882` | `0.222706` | `0.303639` |

---

## 5. 排序结论

### 5.1 按任务主指标排序

```text
fudan_infer > filt_test_best_loss
```

依据：

- `selective_score`
- `edge_source_completion`
- `edge_retention_ratio`
- `edge_oversmooth_vs_src`

### 5.2 按纯背景接近 `ref` 排序

```text
filt_test_best_loss > fudan_infer
```

依据：

- `bg_completion`
- `bg_hf_error`

### 5.3 最符合主观观察的解释

这和主观看法是一致的：

- `filt_test_best_loss`
  - 背景抹得更狠
  - 但边缘被带坏
- `fudan_infer`
  - 整体更清楚
  - 边更稳
  - 但背景还不够干净

---

## 6. `ref` 作为候选时的参考分数

因为 `selective_score` 的定义是：

- 背景对 `ref`
- 边缘对 `ori`

所以 `ref` 自己不会是 `100`

之前做过一个 `200 tile` 子集估计，结果如下：

- `selective_score` 均值约 `57.0153`
- `edge_retention_ratio` 均值约 `0.8624`
- `edge_oversmooth_vs_src` 均值约 `0.5171`

这个数的意义是：

- `ref` 在背景项上是满分基准
- 但在边缘项上不是满分，因为 `ref` 本身也会牺牲部分边缘

因此：

- 现阶段不要把 `selective_score` 理解成“绝对 100 分制”
- 更适合做相对排名和 checkpoint 选择

---

## 7. 训练/选模建议

### 7.1 训练 loss

当前建议保持：

```text
L_total = 1.00 * ROI-Charbonnier(pred, ref)
        + 0.16 * MS-SSIM(pred, ref)
```

原因：

- 可微
- 稳定
- 收敛行为清晰

### 7.2 验证主指标

当前建议：

```text
primary_metric = selective_score
```

### 7.3 选模护栏

建议同时检查：

- `edge_source_completion >= 0`
- `edge_retention_ratio >= 0.90`

如果某个 checkpoint：

- `selective_score` 高
- 但 `edge_source_completion < 0`

那说明它虽然背景更像 `ref`，但边缘已经开始过度牺牲，不建议作为最终模型。

---

## 8. 当前最终结论

在当前这批结果里：

- 如果你要“更符合整体任务目标”的模型
  - 选 `fudan_infer`
- 如果你只要“背景尽量像 ref 更干净”
  - `filt_test_best_loss` 更接近

但从最终产品目标看：

- `fudan_infer` 更像当前应该继续优化的主线
- 后续重点不是再把背景继续一味压狠
- 而是：
  - 在保持 `edge_retention_ratio` 和 `edge_source_completion` 不掉的前提下
  - 再提升 `bg_completion`

---

## 9. 最新训练结果补充

本次新增评估对象：

- `edge_aux_best`
  - `/mnt/d/fudan/prefilter_clean/infer_test_compare/xlx_clean_roi_512_edge_aux_best_test`

对应评估结果文件：

- [eval_selective.json](/mnt/d/fudan/prefilter_clean/infer_test_compare/xlx_clean_roi_512_edge_aux_best_test/eval_selective.json)

评估范围与前文保持一致：

- `1920 tile`
- `mask_mode = detail_gain`
- `s_thr = 0.25`

### 9.1 相对 `fudan_infer` 的结论

如果按当前任务主指标 `selective_score` 看：

- `edge_aux_best` 已经超过 `fudan_infer`

这次提升的主要来源不是背景进一步逼近 `ref`，而是：

- 边缘保留更好
- 相对原图的过平滑显著减少
- 结构一致性更稳定

同时要注意：

- `bg_completion` 略有下降
- 提升更集中在中高分样本和最好的一段
- 最差那部分样本尾部还需要继续看

### 9.2 主指标均值对比

| 模型 | selective_score | bg_completion | edge_source_completion | edge_retention_ratio | edge_oversmooth_vs_src |
|---|---:|---:|---:|---:|---:|
| `fudan_infer` | `32.5852` | `0.2353` | `0.2562` | `0.9102` | `0.4556` |
| `edge_aux_best` | `33.2415` | `0.1953` | `0.2706` | `0.9558` | `0.2981` |

和 `fudan_infer` 相比：

- `selective_score` 提升 `+0.6563`
- `edge_source_completion` 提升 `+0.0144`
- `edge_retention_ratio` 提升 `+0.0456`
- `edge_oversmooth_vs_src` 降低 `-0.1576`
- `bg_completion` 下降 `-0.0401`

解释：

- 新模型没有把背景再往 `ref` 方向压得更狠
- 但明显更会“保边”
- 所以综合任务分数反而更高

### 9.3 关键误差项对比

| 指标 | fudan_infer | edge_aux_best | 变化 |
|---|---:|---:|---:|
| `edge_preserve_error` | `96.4706` | `91.1878` | `-5.2828` |
| `edge_source_error` | `145.5365` | `93.0968` | `-52.4397` |
| `edge_over_smooth_ratio` | `0.1994` | `0.1513` | `-0.0481` |
| `structure_alignment_error` | `0.0376` | `0.0317` | `-0.0058` |
| `bg_hf_error` | `20.1495` | `20.1138` | `-0.0357` |

这里最明显的是：

- `edge_source_error` 大幅下降
- `edge_over_smooth_ratio` 明显下降
- `bg_hf_error` 基本持平

说明新模型主要是把“边缘副作用”压下来了，而不是靠牺牲背景质量换分。

### 9.4 `selective_score` 分布变化

| 统计项 | fudan_infer | edge_aux_best | 变化 |
|---|---:|---:|---:|
| `mean` | `32.5852` | `33.2415` | `+0.6563` |
| `median` | `34.6187` | `34.5949` | `-0.0238` |
| `p10` | `23.7651` | `19.0145` | `-4.7506` |
| `p90` | `42.5211` | `45.8301` | `+3.3089` |

这个分布很重要：

- `median` 基本持平
- `p90` 提升明显
- `p10` 反而更低

因此更准确的描述不是“所有样本都变好了”，而是：

- 新模型整体均值更强
- 高分样本提升更明显
- 但低分尾部还有退化样本需要继续排查

### 9.5 当前阶段排序

如果按当前任务主指标排序：

```text
edge_aux_best > fudan_infer > filt_test_best_loss
```

如果按“背景尽量贴近 `ref`”排序：

```text
filt_test_best_loss > fudan_infer > edge_aux_best
```

### 9.6 对训练方向的意义

这版 `edge_aux_best` 说明当前这条带 edge auxiliary 的微调方向是有效的：

- 成功超过了原来的 `fudan_infer`
- 改善点集中在更符合主观体验的“保边”和“少过平滑”
- 目前最值得继续做的不是盲目追更高 `bg_completion`
- 而是优先处理 `p10` 那部分差样本，争取把尾部抬起来
