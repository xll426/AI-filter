# Training Package

本目录是上传用训练包，包含训练代码、正式配置、实际运行 config、指标摘要、checkpoint、raw-Y ONNX 和评估记录。

推荐优先阅读：

```text
EVALUATION.md
runs/task_qat_noedge_test_summary.md
configs/task_qat_w11_b13_noedge_shift10.yaml
runs/task_qat_w11_b13_noedge_shift10/train_int_qat_config.yaml
```

当前部署推荐模型来自：

```text
task_qat_w11_b13_noedge_shift10
W11/B13, fixed shift = 10
test PSNR = 36.7303
test SSIM = 0.969035
```

如资源允许并且希望优先追求精度，可参考 `task_qat_w12_b14_noedge_shift11`；如优先压缩位宽，可参考 `task_qat_w10_b12_noedge_shift9`，但它的精度下降更明显。

# W10+B13 Prefilter 训练与部署指南

## 目录结构

本目录包含 W10+B13 版本训练、推理和部署所需的代码、配置、模型与示例数据：

```text
code/
  process_data.py          数据预处理，生成 img/gt/roi/manifest
  train.py                 FP32 PrefilterNet 训练入口
  train_int_qat.py         W10+B13 定点 QAT 训练入口
  train_teacher_qat.py     FP32 teacher 在线监督的定点 QAT 训练入口
  pred_int_qat.py          PyTorch checkpoint 推理入口
  export_quant_onnx.py     导出 raw-Y ONNX
  model.py                 FP32 训练态模型和重参数化 MBRConv3
  model_int_qat.py         定点 QAT 部署模型
  ref.py                   reference algorithm 生成逻辑
  loss.py                  Charbonnier / MS-SSIM / edge loss
  iqa_metrics_exact_refalgo_y.py  selective_score 验证指标
  dataset.py, utils.py     数据读取、YUV 读写、训练工具

configs/
  fp32_edge_aux.yaml        FP32 PrefilterNet 训练配置
  w10_b13_qat.yaml         W10+B13 训练配置
  teacher_qat_*.yaml       FP32 teacher 在线监督 QAT 配置

models/
  fp32_edge_aux_best.pt    W10+B13 QAT 初始化用 FP32 best
  w10_b13_best.pt          W10+B13 best checkpoint
  w10_b13_raw.onnx         raw 0..255 Y 输入/输出 ONNX
  int_params_best/         q_w / q_b / shift

weights/
  iccv_yan_2025_fp32.pth   FP32 训练初始化基座权重

data/xlx_clean_roi_512/   少量真实样本示例，每个 split 2 条，用于说明目录和 manifest 格式
```

## 新策略：FP32 Teacher 直接监督 QAT

如果不再先微调 FP32，也不使用 reference GT，可以直接用
`weights/iccv_yan_2025_fp32.pth` 作为 FP32 teacher。训练时 student 从同一
FP32 权重初始化，teacher 在线生成 target，loss 只保留逐像素 L1 或 L2。

入口脚本：

```bash
python code/train_teacher_qat.py --config configs/teacher_qat_w10_b13.yaml
```

可选配置：

```text
configs/teacher_qat_w8_b12.yaml
configs/teacher_qat_w10_b12.yaml
configs/teacher_qat_w10_b13.yaml
configs/teacher_qat_w10_b13_l2.yaml
configs/teacher_qat_w12_b14.yaml
```

核心配置项：

```yaml
model:
  pretrain_path: ./weights/iccv_yan_2025_fp32.pth

int_qat:
  weight_bits: 10
  bias_bits: 13
  teacher_target:
    loss_type: l1   # 可改 l2
    loss_weight: 1.0
    only_y: true
```

该入口不改动原来的 `train_int_qat.py`。原入口仍保留 reference/任务 loss +
distillation 的旧训练流程。

`data/xlx_clean_roi_512` 仅提供少量真实样本，用于展示数据组织方式和 manifest 格式。正式训练应将 `configs/w10_b13_qat.yaml` 中的 `data.root` 指向完整数据集。

## 项目目标与整体流程

本工程面向视频编码前处理，模型执行 Y 通道选择性滤波。训练监督来自 `ref.py` 中的传统 reference algorithm：

| 区域 | 期望行为 |
| --- | --- |
| 低结构背景区域 | 接近 reference target，压制背景高频、细碎纹理和无效复杂度 |
| 主体结构、文字、边缘区域 | 尽量保留原始输入 source 的结构和梯度，避免过平滑 |
| UV 色度通道 | 不训练、不滤波，推理和部署时原样透传 |

训练和验证中有三个核心图像对象：

| 名称 | 代码变量 | 含义 |
| --- | --- | --- |
| 原始输入 | `input` / `source` | 原始 YUV 读入后的图像，训练中归一化为 `[0,1]` |
| 参考目标 | `target` / `gt` / `ref` | `ref.py` 对原始图像生成的传统滤波监督目标 |
| 模型输出 | `pred` | QAT student 的输出，loss 主要约束 Y 通道 |

主流程如下：

```text
原始视频
  -> process_data.py 抽帧、生成 reference target、切 tile、写 manifest
  -> PairedYuvDataset 按 manifest 读取 input/target/roi
  -> FP32 PrefilterNet 加载预训练权重并 slim 成单个 3x3 部署卷积
  -> DeployPrefilterIntQAT 从融合权重初始化 W10+B13 QAT student
  -> train_int_qat.py 用 task loss + distillation + integer regularization 微调
  -> selective_score 在验证集上选择 best checkpoint
  -> export_quant_onnx.py 导出 raw-Y ONNX 和 q_w/q_b/shift
```

训练接口保持 `[0,1]`；QAT forward 内部恢复到 raw `0..255`，以模拟硬件整数路径。部署 ONNX 使用 raw `0..255` Y 输入/输出。

## 1. 环境

安装依赖：

```bash
pip install -r requirements.txt
```

OpenCV 使用 contrib 版本，以提供 `ximgproc.l0Smooth/guidedFilter`：

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
pip install opencv-contrib-python-headless
```

## 2. 数据预处理

目标目录结构：

```text
data/xlx_clean_roi_512/
  train/manifest.csv
  train/img/
  train/gt/
  train/roi/
  val/manifest.csv
  val/img/
  val/gt/
  val/roi/
  test/manifest.csv
  test/img/
  test/gt/
  test/roi/
```

交付目录内提供示例数据：

```text
data/xlx_clean_roi_512/
  train/  2 个 512x512 样本，含 img/gt/roi
  val/    2 个 512x512 样本，含 img/gt/roi
  test/   2 个 512x512 样本，含 img/gt/roi
```

示例数据用于说明数据读取、训练、推理和导出流程，不代表完整训练分布。正式训练应替换为完整数据集，或将 `configs/w10_b13_qat.yaml` 中的 `data.root` 指向完整数据目录。

### 2.1 manifest.csv

`manifest.csv` 是每个 split 的样本索引表。训练、验证和推理代码均通过它定位样本，不扫描目录、不解析文件名。每一行对应一个 tile 样本。

示例：

```csv
input_path,target_path,width,height,format,bitdepth,source_video,source_frame,tile_id,tile_top,tile_left,roi_path
img/xxx/xxx_tile000_y0000_x0000.yuv,gt/xxx/xxx_tile000_y0000_x0000.yuv,512,512,yuv420p,8,xxx,0,0,0,0,roi/xxx/xxx_tile000_y0000_x0000.npy
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `input_path` | 输入 YUV tile 的相对路径，基于对应 split 目录，例如 `train/img/...` 在 CSV 中写成 `img/...` |
| `target_path` | reference algorithm 2 生成的监督目标 YUV tile，相对对应 split 目录 |
| `width` / `height` | tile 宽高；边界 tile 可能小于 `tile_size` |
| `format` | YUV 存储格式，默认主线为 `yuv420p` |
| `bitdepth` | 位深；W10+B13 按 8-bit raw Y 设计 |
| `source_video` | 样本来源视频名，用于追溯和按视频分组划分 train/val/test |
| `source_frame` | 来源视频中的帧号 |
| `tile_id` | 帧内 tile 编号 |
| `tile_top` / `tile_left` | tile 在原始帧中的左上角坐标 |
| `roi_path` | 可选 ROI mask 路径，`.npy` 格式；为空或文件不存在时训练自动退化为无 ROI 加权 |

`dataset.py` 按 manifest 读取：

```text
img -> sample["input"]
gt  -> sample["target"]
roi -> sample["roi"]，如果存在
```

训练阶段读取后将 YUV 归一化到 `[0,1]`；定点 QAT 模型内部再将 Y 恢复到 raw `0..255` 像素域以模拟硬件路径。

生成命令示例：

```bash
python code/process_data.py \
  --input_dir /path/to/source_videos \
  --output_dir data/xlx_clean_roi_512 \
  --frames_per_video 12 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 123 \
  --pix_fmt yuv420p \
  --yuv_format yuv420p \
  --bitdepth 8 \
  --tile_size 512 \
  --tile_stride 512 \
  --detect_roi \
  --roi_conf 0.3 \
  --roi_iou 0.25 \
  --workers 1 \
  --overwrite
```

关键参数：

| 参数 | 作用 |
| --- | --- |
| `--input_dir` | 原始视频目录，代码按后缀查找视频文件 |
| `--output_dir` | 输出数据集根目录 |
| `--frames_per_video` | 每个视频均匀抽多少帧 |
| `--train_ratio/--val_ratio/--test_ratio` | 按视频分组划分数据集，避免同视频泄漏 |
| `--pix_fmt` | ffmpeg 解码输出像素格式 |
| `--yuv_format` | 落盘 YUV 格式，训练读取时按此解释 |
| `--bitdepth` | 位深；W10+B13 按 8-bit 设计 |
| `--tile_size` | 切块尺寸；`512` 表示切 512x512 tile，`0` 表示保留整帧 |
| `--tile_stride` | 切块步长，通常等于 `tile_size` |
| `--detect_roi` | 解码后做人脸/车牌 ROI 检测并保存 mask |
| `--roi_dir` | 已有 ROI mask 目录；配置后可不启用 `--detect_roi` |
| `--overwrite` | 输出目录非空时允许重建 |

ROI 为可选字段。缺少 ROI 时，训练使用普通区域权重。

启用 `--detect_roi` 时，默认检测权重路径为：

```text
weights/yolov8n-face.pt
weights/yolov8s-plate.pt
```

交付目录不包含上述检测权重；可通过 `--roi_face_model`、`--roi_plate_model` 显式指定，或通过 `--roi_dir` 读取已有 ROI。

数据读取的几个约束：

| 项 | 说明 |
| --- | --- |
| manifest | 训练、验证和 split 目录推理都以 `manifest.csv` 为准，不扫描目录、不从文件名推断尺寸 |
| 数据域 | `dataset.py` 输出 `[0,1]` tensor；QAT 模型内部再做 `round + clip(x * 255, 0, 255)` |
| tile 尺寸 | 边界 tile 可能小于 `tile_size`，宽高以 manifest 为准 |
| 推理 padding | PixelUnshuffle 要求 H/W 可被 4 整除；PyTorch 推理先 pad 到 4 的倍数，再裁回原尺寸 |
| UV | 只训练和导出 Y；UV 不进 ONNX，外部流程原样拼回 |

## 3. 参考算法 2（Reference Algorithm 2）

`ref.py` 负责从原始 Y 生成监督目标 `gt/ref`。该算法执行结构感知的选择性滤波：

1. 反射 padding，避免边界处理影响中心区域。
2. 用 Scharr 梯度和 structure tensor 计算结构强度 `S`。
3. 对非结构区域的异常点做 `structure_aware_median` 修正。
4. 用 `fastNlMeansDenoising` 做轻量去噪。
5. 用 `L0Smooth + guidedFilter` 得到平滑 base layer。
6. 计算 detail：`detail = denoise - base`。
7. 用结构强度控制细节回填：`y_ref = base + S^3 * detail`。
8. 用 `match_local_mean_var` 对局部均值和方差做回调，避免亮度/对比度漂移。

核心目标：

```text
背景区域更接近平滑参考结果；
结构/边缘区域保留原始细节。
```

关键步骤的作用：

| 步骤 | 作用 |
| --- | --- |
| 反射 padding | 减少滤波在图像边界产生伪影 |
| 结构感知中值修正 | 只在低结构区域替换明显异常点 |
| `fastNlMeansDenoising` | 降低噪声和细碎高频 |
| 结构分数 `S` | 区分背景区域和重要结构区域 |
| base layer | 通过 `L0Smooth + guidedFilter` 或 fallback 得到平滑基底 |
| detail 回填 | 用 `S^3` 控制细节回填，背景少回填，边缘多回填 |
| 局部均值方差匹配 | 避免亮度和局部对比度漂移过大 |

## 4. 模型结构

训练态模型在 `model.py`：

```text
Y -> PixelUnshuffle(4)
  -> MBRConv3 多分支卷积
  -> residual add
  -> PixelShuffle(4)
  -> 滤波后的 Y
```

模型配置为仅训练 Y 通道：

```yaml
model.only_train_y: true
```

UV 不参与模型计算，推理时原样复制。

`PixelUnshuffle(4)` 将 1 通道 Y 转换为 16 通道低分辨率特征；W10+B13 部署卷积参数形状为：

```text
q_w: [16, 16, 3, 3]
q_b: [16]
shift: [16]
```

## 5. 重参数化

`MBRConv3` 训练态包含 8 路特征：

```text
3x3 conv
1x1 conv
3x1 conv
1x3 conv
以上 4 路各自的 BN 分支
```

这些分支 concat 后经过 `1x1 conv_out` 压回目标通道。部署前使用 `MBRConv3.slim()` 融合为单分支结构：

1. `1x1` padding 成 `3x3`。
2. `3x1` / `1x3` padding 成 `3x3`。
3. 把 Conv+BN 按 BN running mean/var 融合成等价 Conv。
4. 把所有分支 kernel concat。
5. 用最后的 `conv_out` 权重做线性压缩，得到单个 `3x3` kernel 和 bias。

融合后的部署结构是：

```text
Y_u = PixelUnshuffle(Y)
delta = Conv3x3(Y_u, W, b)
Y_out = PixelShuffle(Y_u + delta)
```

W10+B13 QAT 从这个融合后的部署结构开始训练。

## 6. 定点 QAT

`model_int_qat.py` 的 `DeployPrefilterIntQAT` 在训练时模拟硬件路径：

```text
x_norm [0,1]
  -> round + clip(x_norm * 255, 0, 255)
  -> PixelUnshuffle(4)
  -> Conv2D(q_w, q_b)
  -> round(acc / 2^shift)
  -> residual add
  -> round + clip(0,255)
  -> PixelShuffle(4)
  -> out_norm = out_raw / 255
```

定点表达：

```text
q_w = round(weight_fp * 2^shift)
q_b = round(bias_raw * 2^shift)
effective_weight = q_w / 2^shift
effective_bias = q_b / 2^shift
```

FP32 训练在 `[0,1]` 域。转换到 raw 域时：

```text
weight 数值不变
bias_raw = bias_norm * 255
```

`shift` 用于为整数权重提供小数表达能力。训练时保留浮点 master 参数，前向量化成整数语义，反向对 `round/clamp/clip` 使用 STE。

部署端需与训练/ONNX 保持一致的 round 语义；硬件右移如采用不同半整数取整规则，需要进行 bit-exact 对齐。

从 FP32 融合模型切到 raw 像素域时，weight 数值保持不变，bias 必须乘 255：

```text
y_norm' = y_norm + Conv(y_norm, W) + b_norm
y_raw   = 255 * y_norm
y_raw'  = y_raw + Conv(y_raw, W) + 255 * b_norm
```

因此 QAT 初始化时使用：

```text
fused_weight_fp   = fused_weight_fp
fused_bias_fp_raw = fused_bias_fp * 255
```

训练时优化浮点 master 参数，并在 forward 中投影为整数语义参数：

```text
weight_fp / bias_fp_raw  --forward-->  q_w / q_b  --export-->  int32 q_w / q_b
```

整数参数为离散变量，不适合直接用 Adam 优化。forward 中插入量化以模拟部署误差，backward 使用 STE 近似传梯度，导出阶段固化整数参数。

STE 节点如下：

| 函数 | 前向行为 | 反向近似 |
| --- | --- | --- |
| `ste_round_clamp` | `clamp(round(x), qmin, qmax)` | 近似恒等梯度 |
| `ste_clip_u8` | `clamp(round(x), 0, 255)` | 近似恒等梯度 |
| `ste_round_shift` | `round(acc / 2^shift)` | 保留 `1 / 2^shift` 缩放梯度 |

W10+B13 定点规格：

| 项 | 数值 |
| --- | ---: |
| weight bits | 10 |
| weight range | [-512, 511] |
| bias bits | 13 |
| bias range | [-4096, 4095] |
| shift | 10 |
| q_w max_abs | 406 |
| q_b max_abs | 1874 |

## 7. Loss 设计

W10+B13 训练配置在 `configs/w10_b13_qat.yaml`。

主要 loss：

| loss | 配置项 | 作用 |
| --- | --- | --- |
| ROI Charbonnier | `train.fidelity_loss` | 像素保真，ROI 区域权重更高 |
| MS-SSIM | `train.perceptual_loss` | 保持结构相似性 |
| EdgeConsistencyLoss | `train.edge_aux_loss` | 边缘区域向 source 保留，避免过平滑 |
| DistillationLoss | `int_qat.distillation` | QAT 学生模型对齐融合后的 FP32 教师模型 |
| range penalty | `int_qat.weight_range_penalty/bias_range_penalty` | 防止整数 latent 超出 bit 范围 |

核心训练参数：

| 参数 | 配置值 | 说明 |
| --- | ---: | --- |
| `batch_size` | 16 | 训练 batch |
| `crop_size` | 224 | 从 512 tile 内 random crop |
| `lr` | 3e-6 | QAT 微调学习率 |
| `min_lr` | 5e-7 | cosine 最小学习率 |
| `total_iter` | 8000 | 总训练迭代 |
| `warmup_iter` | 300 | warmup 迭代 |
| `roi_weight` | 20.0 | ROI Charbonnier 权重 |
| `perceptual_loss.weight` | 0.16 | MS-SSIM 权重 |
| `edge_aux_loss.weight` | 0.05 | 边缘辅助约束权重 |
| `distillation.weight` | 0.1 | 教师模型蒸馏对齐权重 |

总损失定义：

```text
L_total = L_task + L_distill + L_reg
L_task  = L_charbonnier + L_ms_ssim + L_edge
```

其中：

| 项 | 说明 |
| --- | --- |
| ROI Charbonnier | 对 `pred` 和 `target/ref` 做像素保真；ROI 区域可使用更高权重 |
| MS-SSIM | 对 `softclip01(pred)` 和 `target/ref` 计算结构相似性损失 |
| EdgeConsistencyLoss | `target/ref` 用于生成 edge mask；在这些位置约束 `pred` 保留 `source` 的梯度方向和强度 |
| DistillationLoss | QAT student 对齐融合后的 FP32 deploy teacher，默认只算 Y 通道 L1 |
| range penalty | 惩罚量化前 latent integer 超出 `[qmin, qmax]`，减少导出时被 clamp 的参数 |

EdgeConsistencyLoss 的计算口径：

```text
target/ref 决定哪里是重要边缘；
source 提供这些位置应尽量保留的原始梯度；
pred 在这些位置保留 source 的边缘强度和方向。
```

该损失不约束 `pred` 的边缘梯度追随 `target/ref`；`target/ref` 仅用于生成边缘区域权重。

## 8. FP32 训练

FP32 阶段训练 `PrefilterNet`，得到 QAT 初始化用 checkpoint。交付目录内已包含训练好的 FP32 checkpoint：

```text
models/fp32_edge_aux_best.pt
```

FP32 训练命令：

```bash
python code/train.py --config configs/fp32_edge_aux.yaml
```

断点恢复：

```bash
python code/train.py --config configs/fp32_edge_aux.yaml --resume auto
```

FP32 训练入口参数：

| 参数 | 必填 | 含义 |
| --- | --- | --- |
| `--config` | 否 | YAML 配置路径；默认 `configs/train.yaml`，交付配置为 `configs/fp32_edge_aux.yaml` |
| `--resume` | 否 | 覆盖 YAML 中的 `train.resume`；可传 `auto` 自动查找 latest checkpoint，也可传具体 checkpoint 路径 |

FP32 训练输出：

```text
runs/fp32_edge_aux/checkpoints/latest.pt
runs/fp32_edge_aux/checkpoints/best.pt
runs/fp32_edge_aux/metrics.jsonl
```

`configs/fp32_edge_aux.yaml` 中的 `model.pretrain_path` 默认指向：

```text
./weights/iccv_yan_2025_fp32.pth
```

该路径用于加载原始 AI-PreFilter FP32 基座权重。若从已有 FP32 checkpoint 继续训练，可将 `model.pretrain_path` 改为对应 checkpoint，例如：

```text
./models/fp32_edge_aux_best.pt
```

W10+B13 QAT 阶段通过 `configs/w10_b13_qat.yaml` 中的 `model.pretrain_path` 加载 FP32 checkpoint，并使用 `MBRConv3.slim()` 融合出部署结构的单个 `3x3` 卷积核。

## 9. W10+B13 QAT 训练

训练命令：

```bash
python code/train_int_qat.py --config configs/w10_b13_qat.yaml
```

断点恢复：

```bash
python code/train_int_qat.py --config configs/w10_b13_qat.yaml --resume auto
```

训练入口参数：

| 参数 | 必填 | 含义 |
| --- | --- | --- |
| `--config` | 否 | YAML 配置路径；默认 `configs/w10_b13_qat.yaml` |
| `--resume` | 否 | 覆盖 YAML 中的 `train.resume`；可传 `auto` 自动查找最新 checkpoint，也可传具体 checkpoint 路径 |

输出：

```text
runs/w10_b13_qat/checkpoints/latest.pt
runs/w10_b13_qat/checkpoints/best.pt
runs/w10_b13_qat/int_exports/best/q_w.pt
runs/w10_b13_qat/int_exports/best/q_b.pt
runs/w10_b13_qat/int_exports/best/shift.pt
```

交付模型路径：

```text
models/w10_b13_best.pt
models/int_params_best/
```

## 10. PyTorch 推理

测试集目录推理：

```bash
python code/pred_int_qat.py \
  --config configs/w10_b13_qat.yaml \
  --checkpoint models/w10_b13_best.pt \
  --input data/xlx_clean_roi_512/test \
  --output_dir outputs/w10_b13_pt_test \
  --device cuda
```

单个 YUV 文件推理：

```bash
python code/pred_int_qat.py \
  --config configs/w10_b13_qat.yaml \
  --checkpoint models/w10_b13_best.pt \
  --input /path/to/input.yuv \
  --output_dir outputs/single \
  --width 512 \
  --height 512 \
  --format yuv420p \
  --bitdepth 8 \
  --device cuda
```

该脚本输出 YUV 文件；Y 由模型滤波，UV 原样复制。split 目录包含 target 时，同时写出包含 PSNR/SSIM 的 `metrics.json`。

PyTorch 推理参数：

| 参数 | 必填 | 含义 |
| --- | --- | --- |
| `--config` | 否 | QAT 配置路径；不传时尝试使用 checkpoint 内保存的配置 |
| `--checkpoint` | 是 | `train_int_qat.py` 保存的 QAT checkpoint，例如 `models/w10_b13_best.pt` |
| `--input` | 是 | 输入路径；支持包含 `manifest.csv` 的 split 目录或单个 `.yuv` 文件 |
| `--output_dir` | 是 | 输出 YUV 和 `metrics.json` 的目录 |
| `--device` | 否 | 推理设备，例如 `cuda`、`cuda:0`、`cpu` |
| `--width` | 单文件必填 | 单个 YUV 文件的宽度；split 目录推理时从 manifest 读取 |
| `--height` | 单文件必填 | 单个 YUV 文件的高度；split 目录推理时从 manifest 读取 |
| `--format` | 单文件必填 | 单个 YUV 文件格式，例如 `yuv420p`、`nv12` |
| `--bitdepth` | 单文件必填 | 单个 YUV 文件位深，W10+B13 主线为 `8` |
| `--skip_existing` | 否 | 输出文件已存在时跳过，适合断点式批量推理 |

## 11. 导出 ONNX

导出 raw `0..255` Y 输入/输出 ONNX：

```bash
python code/export_quant_onnx.py \
  --config configs/w10_b13_qat.yaml \
  --checkpoint models/w10_b13_best.pt \
  --output models/w10_b13_raw.onnx \
  --height 512 \
  --width 512 \
  --opset 17 \
  --device cpu \
  --dynamic \
  --raw_io \
  --check \
  --residual_scale 1.0
```

ONNX 输入输出：

```text
input : float32 [N,1,H,W], raw Y, 0..255
output: float32 [N,1,H,W], 滤波后的 raw Y, 0..255
```

ONNX 内部公式：

```text
Y_u = PixelUnshuffle4(Y)
acc = Conv2D(Y_u, q_w, q_b)
delta = round(acc / 1024)
Y_u_out = clip(round(Y_u + delta), 0, 255)
Y_out = PixelShuffle4(Y_u_out)
```

`q_w/q_b` 在 ONNX initializer 中是 float32 承载，但数值是整数语义。

ONNX metadata 和旁路的 `*.export_meta.json` 用于记录输入域、位宽、公式和参数范围，不参与推理计算。硬件转换工具不兼容 metadata 时，可移除 metadata；图节点和 initializer 不变时，输出不变。

ONNX 导出参数：

| 参数 | 必填 | 含义 |
| --- | --- | --- |
| `--config` | 否 | QAT 配置路径；默认 `configs/w10_b13_qat.yaml` |
| `--checkpoint` | 否 | 要导出的 QAT checkpoint；默认 `models/w10_b13_best.pt` |
| `--output` | 否 | 输出 ONNX 路径；默认 `models/w10_b13_raw.onnx` |
| `--device` | 否 | 导出和检查使用的设备；推荐 `cpu`，便于复现 |
| `--height` | 否 | dummy input 高度；静态导出时决定 ONNX 固定 H |
| `--width` | 否 | dummy input 宽度；静态导出时决定 ONNX 固定 W |
| `--opset` | 否 | ONNX opset 版本；推荐 `17` |
| `--dynamic` | 否 | 导出动态 batch/H/W 轴；部署需要多分辨率时开启 |
| `--raw_io` | 否 | 导出 raw `0..255` Y 输入/输出接口；最终部署推荐开启 |
| `--check` | 否 | 保存前执行 PyTorch QAT 路径和导出模块的一致性检查 |
| `--residual_scale` | 否 | residual 加回前的额外缩放，默认 `1.0`；正式 W10+B13 保持 `1.0` |

## 12. 配置参数

`configs/w10_b13_qat.yaml` 主要字段：

| 字段 | 说明 |
| --- | --- |
| `experiment_name` | 输出到 `runs/<experiment_name>` |
| `data.root` | 数据集根目录 |
| `model.downscale_factor` | PixelUnshuffle/Shuffle 倍率，W10+B13 为 4 |
| `model.only_train_y` | 只训练 Y，UV 旁路 |
| `model.pretrain_path` | QAT 初始化 FP32 checkpoint |
| `int_qat.weight_bits` | 权重整数位宽 |
| `int_qat.bias_bits` | bias 整数位宽 |
| `int_qat.per_channel_shift` | 每个输出通道独立 shift |
| `int_qat.min_shift/max_shift` | shift 搜索边界 |
| `int_qat.export_best/latest` | 训练时是否导出整数参数 |
| `train.resume` | `none` 或 `auto` |
| `logger.val_freq` | 验证频率 |
| `validation.primary_metric` | best 选择指标，W10+B13 使用 `selective_score` |

## 13. 验证指标

训练验证使用 `selective_score` 选择 best checkpoint。该指标按任务目标组合背景拟合、边缘保留和过平滑抑制：

```text
背景接近 ref + 边缘接近 source + 边缘能量保留 + 避免过平滑
```

主要指标含义：

| 指标 | 方向 | 含义 |
| --- | --- | --- |
| `selective_score` | 越大越好 | 综合任务分 |
| `bg_completion` | 越大越好 | 背景接近 ref 的程度 |
| `edge_source_completion` | 越大越好 | 边缘接近 source 的程度 |
| `edge_retention_ratio` | 接近 1 | 边缘梯度能量保留 |
| `edge_oversmooth_vs_src` | 越小越好 | 相对 source 的边缘过平滑比例 |
| `bg_hf_error` | 越小越好 | 背景高频误差 |

更具体地：

```text
bg_completion =
  1 - bg_hf_error(pred, ref) / bg_hf_error(source, ref)

edge_source_completion =
  1 - edge_preserve_error(pred, source) / edge_preserve_error(ref, source)

selective_score = 100 * (
  0.45 * bg_completion
  + 0.35 * clip(edge_source_completion, -1, 1)
  + 0.10 * clip(edge_retention_ratio, 0, 1.2) / 1.2
  + 0.10 * max(0, 1 - edge_oversmooth_vs_src)
)
```

因此 `best.pt` 对应任务指标最优，不保证同时取得最高 PSNR/SSIM。`selective_score` 包含结构 mask、分位数、clip 等非平滑流程，用于验证指标和 best checkpoint 选择，不作为训练主 loss。
