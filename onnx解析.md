# W10+B13 Y-only ONNX 解析

本文记录当前 W10+B13 整数 QAT 模型导出的 ONNX 图结构、每个算子的作用、Y/UV 处理方式，以及 PyTorch/ONNX 对齐结果。

## 1. 文件与基本信息

ONNX 文件：

```text
runs/xlx_clean_roi_512_edge_aux_int_qat_w10_b13/onnx/quant_w10_b13_best_y.onnx
```

导出脚本：

```text
export_quant_onnx.py
```

推理脚本：

```text
pred_quant_onnx_y.py
```

导出 sidecar：

```text
runs/xlx_clean_roi_512_edge_aux_int_qat_w10_b13/onnx/quant_w10_b13_best_y.int_params.npz
runs/xlx_clean_roi_512_edge_aux_int_qat_w10_b13/onnx/quant_w10_b13_best_y.export_meta.json
```

ONNX opset：

```text
opset = 17
```

ONNX 输入/输出：

| 项 | shape | dtype | 含义 |
|---|---|---|---|
| input | `[1, 1, 512, 512]` | float32 | normalized Y plane，范围约 `[0,1]` |
| output | `[1, 1, 512, 512]` | float32 | normalized filtered Y plane，范围约 `[0,1]` |

这个 ONNX 是 **Y-only graph**：

- ONNX 不输入 UV
- ONNX 不输出 UV
- ONNX 只处理 Y plane
- UV 在外部推理脚本中保留并拼回

## 2. 整数量化参数

ONNX 图中有三个核心 initializer：

| initializer | shape | dtype | 含义 |
|---|---|---|---|
| `q_w_float` | `[16, 16, 3, 3]` | float32 | 从整数 `q_w` 导出的卷积权重 |
| `q_b_float` | `[16]` | float32 | 从整数 `q_b` 导出的卷积 bias |
| `inv_shift` | `[1, 16, 1, 1]` | float32 | `1 / 2^shift` |

虽然 ONNX 中 `q_w_float` 和 `q_b_float` 是 float32 tensor，但它们的数值来自导出的整数参数。

实际范围：

| 项 | 数值 |
|---|---:|
| `weight_bits` | 10 |
| `bias_bits` | 13 |
| `q_w.min` | -406 |
| `q_w.max` | 100 |
| `q_w.max_abs` | 406 |
| `q_b.min` | -1874 |
| `q_b.max` | -17 |
| `q_b.max_abs` | 1874 |
| `shift` | 10 |

`q_b` 具体值：

```text
[-965, -819, -790, -630, -17, -1620, -798, -1416,
 -698, -1874, -797, -636, -1403, -754, -1314, -1771]
```

`shift` 具体值：

```text
[10, 10, 10, 10, 10, 10, 10, 10,
 10, 10, 10, 10, 10, 10, 10, 10]
```

因此：

```text
inv_shift = 1 / 2^10 = 1 / 1024
```

## 3. 总体计算公式

整个 ONNX 图等价于下面的整数仿真流程：

```text
x_raw = clip(round(x_norm * 255), 0, 255)

Y_u = PixelUnshuffle4(x_raw)

acc = Conv(Y_u, q_w, q_b)

delta = round(acc / 2^10)

Y_u_out = clip(round(Y_u + delta), 0, 255)

Y_out_raw = PixelShuffle4(Y_u_out)

Y_out_norm = Y_out_raw / 255
```

其中：

- `x_norm` 是输入 normalized Y
- `x_raw` 是 raw 8bit 语义的 Y
- `Y_u` 是 PixelUnshuffle 后的 16 通道低分辨率表示
- `acc` 是整数卷积累加结果
- `delta` 是右移缩放后的残差
- `Y_u_out` 是加残差后的 raw 8bit unshuffle 域输出
- `Y_out_norm` 是最终 ONNX 输出

## 4. 节点级解析

当前 ONNX graph 共 27 个节点。

### 4.1 输入 Y 转 raw 8bit 语义

对应节点：

```text
00 Constant  value = 255.0
01 Mul
02 Round
03 Constant  value = 0.0
04 Constant  value = 255.0
05 Clip
```

公式：

```text
x_raw = clip(round(input * 255), 0, 255)
```

作用：

1. `Mul`：把 normalized Y 从 `[0,1]` 放大到 `[0,255]`
2. `Round`：模拟 8bit 输入像素的整数化
3. `Clip`：限制到合法 luma 范围 `[0,255]`

这一步之后 tensor dtype 仍然是 float32，但数值已经是 8bit raw 整数语义。

### 4.2 Slice 取 Y 通道

对应节点：

```text
06 Constant  axes = [1]
07 Constant  starts = [0]
08 Constant  ends = [1]
09 Constant  steps = [1]
10 Slice
```

公式：

```text
y = x_raw[:, 0:1, :, :]
```

当前 ONNX 输入本身就是 `[N,1,H,W]`，因此这个 Slice 实际上是 no-op。

它存在的原因是导出模型保留了 PyTorch 代码里的：

```python
y = x_raw[:, :1]
```

注意：这里没有 UV。UV 不在 ONNX 图里。

### 4.3 PixelUnshuffle(4)

对应节点：

```text
11 Constant  shape = [-1, 1, 128, 4, 128, 4]
12 Reshape
13 Transpose perm = [0, 1, 3, 5, 2, 4]
14 Constant  shape = [-1, 16, 128, 128]
15 Reshape
```

shape 变化：

```text
[1, 1, 512, 512]
-> [1, 1, 128, 4, 128, 4]
-> [1, 1, 4, 4, 128, 128]
-> [1, 16, 128, 128]
```

作用：

把 512x512 的 Y 平面按 4x4 block 拆成 16 个 phase channel。

也就是：

```text
Y: [1, 1, 512, 512]
变成
Y_u: [1, 16, 128, 128]
```

这一步对应 PyTorch 的：

```python
nn.PixelUnshuffle(4)
```

16 个 channel 分别对应每个 4x4 block 内的 16 个子像素位置。

### 4.4 整数卷积 Conv

对应节点：

```text
16 Conv
```

Conv 属性：

```text
kernel_shape = [3, 3]
pads = [1, 1, 1, 1]
strides = [1, 1]
dilations = [1, 1]
group = 1
```

输入输出：

```text
input : [1, 16, 128, 128]
weight: q_w_float [16, 16, 3, 3]
bias  : q_b_float [16]
output: [1, 16, 128, 128]
```

公式：

```text
acc_k(h,w) =
sum_c sum_i sum_j Y_u_c(h+i,w+j) * q_w[k,c,i,j]
+ q_b[k]
```

作用：

- `q_w_float` 的数值来自整数 `q_w`
- `q_b_float` 的数值来自整数 `q_b`
- ONNX 用 float32 Conv 承载整数卷积累加

因此这个 Conv 的语义是整数卷积：

```text
acc = Conv(Y_u, q_w, q_b)
```

但实际 ONNX Runtime 中 dtype 是 float32。

### 4.5 右移缩放 shift=10

对应节点：

```text
17 Mul
18 Round
```

公式：

```text
delta = round(acc * inv_shift)
```

由于：

```text
inv_shift = 1 / 1024
```

所以等价于：

```text
delta = round(acc / 2^10)
```

作用：

- 模拟硬件中的右移 `shift=10`
- 用 `Round` 保持和 PyTorch QAT 路径一致的 rounding 规则

输出：

```text
delta: [1, 16, 128, 128]
```

### 4.6 残差加回原始 unshuffle Y

对应节点：

```text
19 Add
```

公式：

```text
Y_u_res = Y_u + delta
```

作用：

模型不是直接生成最终 Y，而是在 PixelUnshuffle 后的低分辨率 16 通道空间里预测残差 `delta`，再加回原始 `Y_u`。

这对应 PyTorch 模型里的 residual 结构。

### 4.7 输出再次 round/clip 到 raw 8bit

对应节点：

```text
20 Round
21 Constant value = 0.0
22 Constant value = 255.0
23 Clip
```

公式：

```text
Y_u_out = clip(round(Y_u + delta), 0, 255)
```

作用：

- `Round`：保证 residual add 后仍是整数像素语义
- `Clip`：防止下溢/上溢，限制到 `[0,255]`

输出：

```text
Y_u_out: [1, 16, 128, 128]
```

### 4.8 PixelShuffle(4)

对应节点：

```text
24 DepthToSpace
```

属性：

```text
blocksize = 4
mode = CRD
```

shape 变化：

```text
[1, 16, 128, 128]
-> [1, 1, 512, 512]
```

作用：

这是 PixelUnshuffle(4) 的逆操作，把 16 个 phase channel 重新拼回原始 512x512 的 Y 平面。

它对应 PyTorch 的：

```python
nn.PixelShuffle(4)
```

### 4.9 raw Y 转回 normalized Y

对应节点：

```text
25 Constant value = 255.0
26 Div
```

公式：

```text
output = Y_out_raw / 255
```

作用：

把 raw 8bit 语义的输出重新转回 normalized `[0,1]` Y plane。

最终输出：

```text
output: [1, 1, 512, 512]
```

## 5. UV 如何处理

UV 不在 ONNX 里处理。

ONNX 推理脚本 `pred_quant_onnx_y.py` 的关键逻辑是：

```python
y_tensor, chroma_payload = read_y_plane_with_chroma(input_path, width, height, fmt, bitdepth)
pred_y = run_onnx_y(session, y_tensor, downscale_factor)
write_y_with_original_chroma(pred_y, chroma_payload, output_path, bitdepth)
```

含义：

1. `read_y_plane_with_chroma(...)`
   - 从输入 YUV 文件读取完整 frame
   - 前 `width * height` 字节作为 Y
   - 后面的 U/V 数据保存为 `chroma_payload`

2. `run_onnx_y(...)`
   - 只把 `y_tensor` 送入 ONNX
   - 输入 shape 是 `[1, 1, H, W]`
   - ONNX 输出 filtered Y，shape 也是 `[1, 1, H, W]`

3. `write_y_with_original_chroma(...)`
   - 先写 ONNX 输出的 filtered Y
   - 再写原始 `chroma_payload`

所以输出 YUV 是：

```text
filtered Y + original UV
```

UV 没有经过 ONNX，没有被修改。

## 6. 与 PyTorch 的一致性

PyTorch vs ONNX 全量对齐结果：

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

PyTorch 输出目录：

```text
infer_test_compare/xlx_clean_roi_512_int_qat_w10_b13_best_test
```

两个目录的输出 YUV 已逐字节比较：

| 项 | 结果 |
|---|---|
| PyTorch `.yuv` 数量 | 1920 |
| ONNX `.yuv` 数量 | 1920 |
| 文件名集合 | 一致 |
| 文件内容 | byte-identical |

测试集 PSNR/SSIM 也完全一致：

| 路径 | tile count | avg PSNR | avg SSIM |
|---|---:|---:|---:|
| PyTorch | 1920 | 36.206724 | 0.958672 |
| ONNX | 1920 | 36.206724 | 0.958672 |

结论：

> W10+B13 的 ONNX graph 是当前 PyTorch 整数 QAT deploy 逻辑的 Y-only bit-exact 表达。

## 7. 硬件实现视角

如果后续做硬件对齐，可以把 ONNX 图理解成下面几步：

1. 输入 Y 从 normalized 转为 8bit raw。
2. 做 `PixelUnshuffle(4)`，得到 16 个 phase channel。
3. 使用整数 `q_w/q_b` 做 3x3 卷积。
4. 对卷积累加结果做 `round(acc / 2^10)`。
5. 把 residual 加回原始 unshuffle Y。
6. 对结果做 round + clip 到 `[0,255]`。
7. 做 `PixelShuffle(4)` 还原到原始分辨率。
8. 输出 filtered Y。
9. 外部系统把 filtered Y 与原始 UV 拼接。

最核心的 bit-true 公式是：

```text
delta = round((Conv(PixelUnshuffle(Y_raw), q_w) + q_b) / 2^10)
Y_out = clip(round(PixelUnshuffle(Y_raw) + delta), 0, 255)
Y_out = PixelShuffle(Y_out)
```

其中：

- `q_w` 是 10 bit signed integer
- `q_b` 是 13 bit signed integer
- `shift = 10`
- 输入/输出 Y 都是 8bit raw 语义
- UV 不参与模型计算
