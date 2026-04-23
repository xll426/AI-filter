# Prefilter Clean

`prefilter_clean` 是基于原版 `AI-PreFilter` 重新整理后的可维护工程。

当前目标不是“功能大概能跑”，而是尽量对齐原版训练链路里真正影响结果的部分：

- 模型结构
- 原版权重加载
- `gt/ref` 生成逻辑
- ROI 检测与 ROI 掩码参与训练
- 损失函数和训练调度

## 1. 当前工程包含什么

- [process_data.py](/mnt/d/fudan/prefilter_clean/process_data.py)
  - 从原始 `h265/hevc` 生成 `train/val/test/img|gt|roi`
- [ref.py](/mnt/d/fudan/prefilter_clean/ref.py)
  - 按原版思路生成监督目标 `gt`
- [roi.py](/mnt/d/fudan/prefilter_clean/roi.py)
  - 人脸 + 车牌 ROI 检测，输出二值 mask
- [dataset.py](/mnt/d/fudan/prefilter_clean/dataset.py)
  - 读取 `img/gt/roi`
- [loss.py](/mnt/d/fudan/prefilter_clean/loss.py)
  - ROI 加权 `CharbonnierLoss` + `MsssimLoss`
- [model.py](/mnt/d/fudan/prefilter_clean/model.py)
  - `PrefilterNet`，并兼容原版 `iccv_yan_2025_fp32.pth`
- [train.py](/mnt/d/fudan/prefilter_clean/train.py)
  - 训练入口
- [pred.py](/mnt/d/fudan/prefilter_clean/pred.py)
  - 推理入口
- [finalize_dataset.py](/mnt/d/fudan/prefilter_clean/finalize_dataset.py)
  - 中断后补写 `manifest.csv` 和 `dataset_summary.json`

## 2. OpenCV `ximgproc` 怎么看，是否需要升级版本

结论先说：你现在的问题不是“单纯版本低”，而是 **环境里同时装了多个 OpenCV 包，互相覆盖了功能**。

我当前检查到的环境状态是：

- `opencv-python==4.11.0.86`
- `opencv-contrib-python==4.11.0.86`
- `opencv-python-headless==4.13.0.92`

而 `cv2` 实际表现是：

- `cv2.__version__ == 4.13.0`
- `hasattr(cv2, "ximgproc") == True`
- `hasattr(cv2.ximgproc, "l0Smooth") == False`
- `hasattr(cv2.ximgproc, "guidedFilter") == False`

这说明：

- 不是没有 `ximgproc` 命名空间
- 而是当前被导入的 `cv2` 不是一个带完整 contrib 扩展的构建
- 根因大概率是多个 OpenCV wheel 混装，后装的 `opencv-python-headless` 把真正的 contrib 功能覆盖掉了

### 2.1 这是不是“更新版本”就行

不一定。

关键不是“版本越新越好”，而是：

- 只保留一套 OpenCV wheel
- 这套 wheel 必须是 `contrib`
- 版本最好统一

### 2.2 推荐修法

如果你这个环境主要是跑数据处理和训练，不需要本地图形界面，建议只保留：

```bash
python -m pip uninstall -y \
  opencv-python \
  opencv-python-headless \
  opencv-contrib-python \
  opencv-contrib-python-headless

python -m pip install opencv-contrib-python-headless==4.11.0.86
```

如果你更想用非 headless 版，也可以只保留：

```bash
python -m pip uninstall -y \
  opencv-python \
  opencv-python-headless \
  opencv-contrib-python \
  opencv-contrib-python-headless

python -m pip install opencv-contrib-python==4.11.0.86
```

重点是：

- 只能保留一套
- 必须是 `contrib`

### 2.3 修完后怎么验证

```bash
python - <<'PY'
import cv2
print('cv2_version =', cv2.__version__)
print('has_ximgproc =', hasattr(cv2, 'ximgproc'))
if hasattr(cv2, 'ximgproc'):
    print('has_l0Smooth =', hasattr(cv2.ximgproc, 'l0Smooth'))
    print('has_guidedFilter =', hasattr(cv2.ximgproc, 'guidedFilter'))
PY
```

理想结果是：

- `has_ximgproc = True`
- `has_l0Smooth = True`
- `has_guidedFilter = True`

### 2.4 如果不修，会怎么样

`process_data.py` 仍然能跑。

因为 [ref.py](/mnt/d/fudan/prefilter_clean/ref.py) 做了 fallback：

- 没有原生 `ximgproc.l0Smooth` 时，用纯 NumPy / FFT 版本 `l0_smooth_gray`
- 没有原生 `guidedFilter` 时，用基础 OpenCV 算子实现 `guided_filter_gray`

所以：

- 不修环境：能跑，但不是“最严格原版路径”
- 修好环境：`gt/ref` 生成路径最接近原版

### 2.5 当前环境检查结果

本机当前已经验证通过：

- `cv2.__version__ == 4.11.0`
- `hasattr(cv2, "ximgproc") == True`
- `hasattr(cv2.ximgproc, "l0Smooth") == True`
- `hasattr(cv2.ximgproc, "guidedFilter") == True`

这说明当前 OpenCV 环境已经符合“按原版 `ximgproc` 路径生成 `gt/ref`”的要求。

## 3. `process_data.py` 这条命令到底做了什么

推荐你正式处理完整数据集时使用下面这条命令：

```bash
python process_data.py \
  --input_dir /mnt/d/fudan/xlx_yuv \
  --output_dir /mnt/d/fudan/prefilter_clean/data/xlx_clean_roi \
  --frames_per_video 12 \
  --start_frame 0 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 123 \
  --pix_fmt yuv420p \
  --yuv_format yuv420p \
  --bitdepth 8 \
  --tile_size 0 \
  --tile_stride 0 \
  --detect_roi \
  --roi_conf 0.3 \
  --roi_iou 0.25 \
  --workers 1 \
  --overwrite
```

说明：

- 这一条命令会一次性生成 `train / val / test`
- 不是分别处理三次
- train/val/test 的划分是按视频分组自动完成的

### 3.0 这条命令里最容易混淆的几个参数

下面这几个参数经常会被混在一起理解，但它们属于不同层面：

- `--pix_fmt yuv420p`
  - 这是 `ffmpeg` 从压缩视频里解单帧时使用的输出像素格式
  - 它决定“解出来的原始帧是什么格式”
  - 当前写成 `yuv420p`，表示按 8-bit 4:2:0 planar YUV 解码

- `--yuv_format yuv420p`
  - 这是解码后的帧落盘成 `.yuv` 文件时采用的存储格式
  - 也是后续 [dataset.py](/mnt/d/fudan/prefilter_clean/dataset.py) 读取这些 `.yuv` 时按什么格式解释它们
  - 通常它应当和 `--pix_fmt` 保持一致

- `--bitdepth 8`
  - 这是 YUV 的位深
  - 当前表示每个像素分量按 8-bit 处理
  - 这个值必须和源视频真实位深一致
  - 如果源视频其实是 10-bit，这里就不能写 8

- `--tile_size 0`
  - `0` 表示不切 tile，保留整帧
  - 所以你当前这条命令 **不是切成 `512x512`**
  - 只有当你显式写成 `--tile_size 512` 时，才会把每帧切成 `512x512`

- `--tile_stride 0`
  - 在当前 `tile_size=0` 时，这个参数不会实际参与切块
  - 它主要在 `tile_size>0` 时才有意义，用来控制 tile 滑窗步长
  - 例如：
    - `tile_size=512, tile_stride=512` 表示不重叠切块
    - `tile_size=512, tile_stride=256` 表示重叠切块

- `--overwrite`
  - 如果 `output_dir` 已经存在且里面有内容，允许覆盖
  - 当前实现会先清空旧输出目录，再重新生成整套数据
  - 不加这个参数时，只要输出目录非空，程序就会直接报错退出，避免误覆盖已有结果

所以你当前这条命令的实际含义是：

- 按 `8-bit yuv420p` 解码
- 按 `8-bit yuv420p` 存盘和读取
- 不切成 `512x512`
- 保留整帧样本
- 如果输出目录已存在，就清空后重做

### 3.0.1 预处理切块和训练时 random crop 不是一回事

这一点非常重要：

- `process_data.py` 里的 `tile_size / tile_stride`
  - 属于 **数据预处理阶段**
  - 决定你最终保存到磁盘上的样本，是整帧还是固定 tile

- `train.yaml` 里的 `crop_size`
  - 属于 **训练阶段**
  - 决定 DataLoader 每次取样时，从当前样本里动态裁多大的 patch

也就是说：

- 预处理阶段：你当前是 `tile_size=0`，所以保存的是整帧
- 训练阶段：当前 `train.yaml` 用的是 `crop_size=224`
- 这表示训练时会从整帧里再动态随机裁 `224x224`

并且这个随机裁剪是在训练时实时做的，不是在数据处理时提前裁好的。

### 3.0.2 `tile_size=512` 再 `crop_size=224` 合不合理

合理，这是一个很常见的工程折中方案。

它的含义是：

- 预处理阶段先把 4K 图切成 `512x512`
- 训练阶段再从每个 `512x512` 样本里随机裁 `224x224`

这和“整帧直接 random crop 224”相比，各有特点。

#### 方案 A：`tile_size=0`，整帧保存，再训练时 crop 224

优点：

- 最接近原版“整帧样本 + 训练时裁 patch”的思路
- ROI 裁剪是在完整空间里进行，分布更自然

缺点：

- 单个样本很大，磁盘 IO 和 CPU 解码开销更高
- DataLoader 每次读样本更重

#### 方案 B：`tile_size=512`，训练时 crop 224

优点：

- 更适合 4K 数据，单样本体积明显更小
- 读盘和 DataLoader 压力更小
- 一张 4K 图会拆成很多 tile，样本数会显著增加
- 对局部滤波模型来说，`512 -> 224` 通常是合理的上下文范围

缺点：

- patch 的采样空间先被限制在 `512x512` tile 内
- 和原版“整帧随机 crop”相比，严格一致性会差一点
- 如果大量 tile 都没有 ROI，训练样本会更偏背景

#### 对你当前 4K 数据，我更偏向哪个

如果你的目标是：

- 最大限度贴原版：优先 `tile_size=0`
- 更关注 4K 数据训练效率和样本量：`tile_size=512 + crop_size=224` 是合理的

对你现在这种 4K 场景，我认为：

- `512x512` 上再裁 `224x224` 是一个现实且合理的方案
- 比直接整帧读 4K 更省资源
- 又保留了训练时动态随机裁剪

也就是说：

- 它不是“错误做法”
- 它是“偏工程优化的做法”

#### 旧的 `xlx_clean` 和现在推荐流程的区别

这里要特别区分你之前已经生成好的：

- [xlx_clean](/mnt/d/fudan/prefilter_clean/data/xlx_clean)

和现在推荐重新生成的：

- `xlx_clean_roi_512`

旧的 `xlx_clean` 实际上是早期流程生成的，它的特点是：

- 有 `img`
- 有 `gt`
- 没有 `roi`
- `manifest.csv` 里没有 `roi_path`

所以旧流程更接近：

1. 从 h265 解整帧
2. 在整帧上生成 `gt`
3. 把 `img / gt` 切成 `512x512`
4. 保存 tile
5. 不包含 ROI

它不是“先整图检测 ROI 再 tile”，因为它当时根本没有 ROI。

现在推荐的新流程是：

1. 从 h265 解整帧
2. 在整帧上生成 `gt`
3. 在整张 4K 图上做人脸/车牌检测，得到整图 ROI mask
4. 再把整帧 `img / gt / roi mask` 一起切成 `512x512`
5. 训练时再从 `512x512` tile 中 random crop `224x224`

这个顺序非常重要。

正确顺序是：

- 先整图检测
- 再 tile

不推荐的顺序是：

- 先 tile
- 再在 tile 上做检测

因为如果先切 tile：

- 人脸可能被切到边界
- 车牌可能只剩一半
- 单个 tile 上的检测会更不稳定

所以当前代码里 ROI 检测是在整图上完成的，然后才切 tile。这就是现在推荐的合理流程。

#### 一张 4K 图大概会切出多少个 512 tile

以 `3840x2160`、`tile_size=512`、`tile_stride=512` 为例：

- 宽方向会得到 8 个起点
- 高方向会得到 5 个起点
- 一帧大约得到 `8 x 5 = 40` 个 tile

所以：

- 一个视频抽 12 帧
- 单视频就可能得到约 `12 x 40 = 480` 个样本

样本数会明显增加。

但要注意：

- “样本数增加”不等于“信息量同比例增加”
- 因为这些 tile 之间会有强相关性

#### 边缘不够 512 时怎么处理

不会丢掉边缘，也不会生成小残块。

当前实现会补一个“最后起点”尾块，所以：

- 最后一块仍然是完整 `512x512`
- 但通常会和前一块有重叠

例如 `3840` 宽度：

- 正常步长会取到 `0, 512, 1024, 1536, 2048, 2560, 3072`
- 还会额外补一个 `3328`
- 因为 `3840 - 512 = 3328`

这样最右边区域不会丢失。

它的完整流程如下。

### 3.1 第一步：扫描输入视频

[process_data.py](/mnt/d/fudan/prefilter_clean/process_data.py#L81) 会在 `input_dir` 下找：

- `.h265`
- `.265`
- `.hevc`

每个文件视为一个源视频。

### 3.2 第二步：先按视频划分 train / val / test

不是先按帧划分。

而是：

- 先按 `source_video` 分组
- 再按 `train_ratio / val_ratio / test_ratio` 划分

这样同一个视频的所有帧都只会进入一个 split，不会信息泄漏。

### 3.3 第三步：为每个视频选采样帧

[build_sample_indices](/mnt/d/fudan/prefilter_clean/utils.py) 会在视频总帧数里均匀采样 `frames_per_video` 帧。

你这里是：

- 每个视频抽 12 帧

### 3.4 第四步：把某一帧解码成原始 YUV

[process_data.py](/mnt/d/fudan/prefilter_clean/process_data.py#L115) 会先调用 [decode_frame](/mnt/d/fudan/prefilter_clean/utils.py#L103)：

- 用 `ffmpeg`
- 按 `--pix_fmt yuv420p`
- 把单帧解到临时 `.yuv`

这一步得到的是原始像素输入。

### 3.5 第五步：读取这帧 YUV，作为模型输入 `img`

临时 `.yuv` 会被 [yuvread2tensor](/mnt/d/fudan/prefilter_clean/utils.py) 读成张量。

这里的张量就是后续训练样本里的 `img` 来源。

### 3.6 第六步：从同一帧生成监督目标 `gt`

[process_data.py](/mnt/d/fudan/prefilter_clean/process_data.py#L120) 会调用 [generate_reference_tensor](/mnt/d/fudan/prefilter_clean/ref.py#L177)。

`gt/ref` 生成流程是：

1. 对输入帧做 `reflect pad=16`
2. 做结构感知中值修复
3. 做 `fastNlMeansDenoising`
4. 计算结构分数 `S`
5. 做 `L0 smooth + guided filter`
6. 拆成 base/detail
7. 按 `S^3` 回灌 detail
8. `match_local_mean_var`
9. 裁回原始大小

得到的结果写入 `gt/*.yuv`。

### 3.7 第七步：如果启用 `--detect_roi`，对同一帧做 ROI 检测

这一步不会影响 `img` 和 `gt` 的生成，只是额外生成一个 ROI mask。

[process_data.py](/mnt/d/fudan/prefilter_clean/process_data.py#L129) 会：

1. 把同一帧再解码成 `BGR` 图像，见 [decode_frame_bgr](/mnt/d/fudan/prefilter_clean/utils.py#L128)
2. 调用 [detect_roi_mask](/mnt/d/fudan/prefilter_clean/roi.py#L92)
3. 分别跑：
   - `yolov8n-face.pt`
   - `yolov8s-plate.pt`
4. 把所有人脸框和车牌框区域填成 1
5. 生成一个二维二值 mask，shape 是 `(H, W)`

这个 mask 是：

- 0：非 ROI
- 1：ROI

### 3.8 第八步：切块

你这里用了：

- `--tile_size 0`

这表示：

- 不切 tile
- 保留整帧

这是当前最接近原版训练方式的选项，因为原版训练是在整帧上再做 `224` 随机 crop，而不是先固定切 `512 tile`。

如果 `tile_size > 0`，那么：

- `img`
- `gt`
- `roi`

三者会按完全相同的位置一起切块。

### 3.9 第九步：落盘

最终会写出：

- `img/*.yuv`
- `gt/*.yuv`
- `roi/*.npy`，如果当前帧有 ROI 模式
- `manifest.csv`
- `dataset_summary.json`

## 4. `process_data.py` 每个参数是什么意思

下面按你最常用的完整命令逐个解释。

### 4.1 输入输出相关

- `--input_dir /mnt/d/fudan/xlx_yuv`
  - 原始压缩视频目录
  - 当前支持 `.h265` / `.265` / `.hevc`
  - 目录下每个视频文件会被视为一个源视频

- `--output_dir /mnt/d/fudan/prefilter_clean/data/xlx_clean_roi`
  - 输出数据集根目录
  - 最终会在这里生成：
    - `train/img`
    - `train/gt`
    - `train/roi`
    - `val/img`
    - `val/gt`
    - `val/roi`
    - `test/img`
    - `test/gt`
    - `test/roi`
    - `manifest.csv`
    - `dataset_summary.json`

### 4.2 抽帧相关

- `--frames_per_video 12`
  - 每个视频均匀抽取 12 帧
  - 抽帧越多，训练样本越多，处理时间也越长

- `--start_frame 0`
  - 从第几帧开始参与采样
  - 通常保持 0 即可

- `--end_frame none`
- `--end_frame`
  - 采样终止帧
  - 默认不传，表示到视频最后一帧
  - 如果你只想用视频前半段，可以设置具体帧号，例如 `--end_frame 300`

### 4.3 数据划分相关

- `--train_ratio 0.8`
  - 80% 的视频进入 train

- `--val_ratio 0.1`
  - 10% 的视频进入 val

- `--test_ratio 0.1`
  - 10% 的视频进入 test

- `--seed 123`
  - 控制视频分组划分和部分采样行为
  - 想复现实验时，这个值要固定

### 4.4 解码和像素格式相关

- `--pix_fmt yuv420p`
  - `ffmpeg` 解码单帧时输出的像素格式
  - 这必须和你视频真实位深/采样格式对应
  - 8-bit 4:2:0 常用 `yuv420p`
  - 如果原视频是 10-bit，通常应该改成类似 `yuv420p10le`

- `--yuv_format yuv420p`
  - 落盘到训练集里的裸 YUV 存储格式
  - 当前读取函数支持：
    - `yuv400p`
    - `yuv420p`
    - `yuv422p`
    - `yuv444p`

- `--bitdepth 8`
  - YUV 实际位深
  - 必须和源数据一致
  - 不能把 10-bit 数据按 8-bit 处理

### 4.5 tile 相关

- `--tile_size 0`
  - `0` 表示不切 tile，保留整帧
  - 这是当前最接近原版训练的方式
  - 如果设置成 `512`，就会把每帧切成 `512x512` 样本

- `--tile_stride 0`
  - tile 滑窗步长
  - 当 `tile_size=0` 时，这个值不会实际参与切块
  - 当 `tile_size>0` 时，如果不设，默认等于 `tile_size`

### 4.6 ROI 相关

- `--detect_roi`
  - 直接从视频帧自动检测 ROI
  - 当前 ROI 来源是：
    - 人脸检测
    - 车牌检测

- `--roi_conf 0.3`
  - YOLO 检测置信度阈值
  - 阈值越高，框越少，ROI 更保守

- `--roi_iou 0.25`
  - YOLO NMS 的 IoU 阈值
  - 控制重叠框的合并行为

- `--roi_dir /path/to/roi_npy`
  - 如果你已经有现成 ROI `.npy`，用这个参数替代 `--detect_roi`
  - 不能和 `--detect_roi` 同时使用

- `--roi_face_model`
  - 人脸检测模型权重路径

- `--roi_plate_model`
  - 车牌检测模型权重路径

- `--roi_save_vis`
  - 保存带框的可视化图到 `output_dir/roi_vis`
  - 只用于检查检测结果，不参与训练

### 4.7 执行相关

- `--workers 1`
  - 并行处理视频的 worker 数
  - YOLO + 解码本身就比较吃资源
  - 如果你显存/内存有限，建议先用 1

- `--overwrite`
  - 如果输出目录已存在，允许覆盖
  - 不加这个参数时，目录非空会直接报错退出

## 5. 生成的 ROI 是什么

不是框坐标，不是标签表，不是额外 YUV。

它是一个 **二维二值 mask**：

- 文件格式：`.npy`
- 内容形状：`(H, W)`
- 值域：
  - `0` 表示非 ROI
  - `1` 表示 ROI

当启用切块时，它会变成每个 tile 对应一个 `.npy`。

当 `tile_size=0` 时，就是整帧对应一个 `.npy`。

## 6. 模型训练时到底读什么

这是最重要的点。

### 6.1 模型输入不是把 ROI 当成第四通道

模型本身仍然只吃：

- `img` 里的 YUV

见 [dataset.py](/mnt/d/fudan/prefilter_clean/dataset.py#L93) 和 [model.py](/mnt/d/fudan/prefilter_clean/model.py#L141)。

也就是说：

- 模型输入通道没有变
- 不是 `img + roi` 拼接成 4 通道

### 6.2 `gt` 是监督目标

训练时同时读：

- `input = img`
- `target = gt`

它们都是 YUV 张量。

### 6.3 `roi mask` 的作用有两个

ROI mask 会被数据集和 loss 额外使用，但不会直接进模型。

作用 1：训练裁剪时优先围绕 ROI 取 patch

见 [dataset.py](/mnt/d/fudan/prefilter_clean/dataset.py#L60)。

如果样本有 ROI：

- 训练时 `crop_size=224`
- 会优先围绕 ROI 连通域随机选点
- 然后裁一个 `224x224` patch

如果没有 ROI：

- 就退化成普通随机裁剪

作用 2：ROI 区域 loss 权重大

见 [loss.py](/mnt/d/fudan/prefilter_clean/loss.py#L22)。

当前默认是：

- `roi_weight = 20.0`
- `non_roi_weight = 1.0`

也就是：

- ROI 区域的 Charbonnier loss 放大 20 倍
- 非 ROI 区域保持 1 倍

所以 ROI 的意义不是“模型多读一个输入”，而是：

- 训练采样更关注 ROI
- 损失更关注 ROI

### 6.4 ROI 在训练代码里是怎么流动的

这一段是最核心的“代码级路径”。

#### 第一步：`manifest.csv` 提供 `roi_path`

在数据处理阶段，[process_data.py](/mnt/d/fudan/prefilter_clean/process_data.py#L162) 会把每个样本对应的：

- `input_path`
- `target_path`
- `roi_path`

写入 `manifest.csv`。

#### 第二步：`dataset.py` 把 `roi.npy` 读成张量

[dataset.py](/mnt/d/fudan/prefilter_clean/dataset.py#L95) 会：

1. 从 `row["roi_path"]` 找到 `.npy`
2. 读成 `roi_np`
3. 转成 `torch.Tensor`
4. 变成 shape `(1, H, W)` 的单通道 ROI mask

最终返回给 DataLoader 的 batch 结构里会包含：

- `batch["input"]`
- `batch["target"]`
- `batch["roi"]`

#### 第三步：训练时，ROI 不进模型 forward

[train.py](/mnt/d/fudan/prefilter_clean/train.py#L312) 里：

- `inputs = batch["input"]`
- `targets = batch["target"]`
- `roi = batch.get("roi")`

然后是：

- `preds = model(inputs)`

这里 forward 只用 `inputs`，也就是 `img`。

ROI 没有传进 `model(inputs)`。

#### 第四步：ROI 传给 loss

接着 [train.py](/mnt/d/fudan/prefilter_clean/train.py#L117) 的 `compute_train_loss(...)` 会执行：

- `l_fidelity = fidelity_loss(pred, gt, roi=roi)`

这里 `roi` 被显式传进了 `CharbonnierLoss`。

#### 第五步：ROI 在 `CharbonnierLoss` 里变成权重图

[loss.py](/mnt/d/fudan/prefilter_clean/loss.py#L28) 里的逻辑是：

1. 先计算逐像素误差：

   `sqrt((pred - target)^2 + eps)`

2. 如果 `roi` 存在，就生成二值权重图：

   - ROI 区域：`roi_weight`
   - 非 ROI 区域：`non_roi_weight`

3. 再把逐像素误差乘上这个权重图
4. 最后整体做 `mean`

可以把它理解成：

```text
loss_map = charbonnier(pred, target)
weight_map = roi * roi_weight + (1 - roi) * non_roi_weight
final = mean(loss_map * weight_map)
```

默认配置下：

- `roi_weight = 20`
- `non_roi_weight = 1`

所以：

- ROI 区域一个像素的误差，训练时大约相当于非 ROI 区域 20 个像素的误差权重

#### 第六步：MSSSIM 不使用 ROI 权重

[train.py](/mnt/d/fudan/prefilter_clean/train.py#L131) 里：

- `l_perceptual = perceptual_loss(softclip01(pred), gt)`

这里没有把 `roi` 传进去。

所以当前实现中：

- `CharbonnierLoss` 受 ROI 影响
- `MsssimLoss` 不受 ROI 权重影响

这也符合你现在对原版的对齐目标，因为 ROI 强调主要体现在 fidelity loss 上。

### 6.5 ROI 对 loss 的实际影响可以怎么理解

如果一个 patch 同时包含：

- 人脸/车牌区域
- 大量背景区域

那么没有 ROI 权重时：

- 背景区域像素数量往往远大于人脸/车牌区域
- loss 会更多被大面积背景主导

加入 ROI 权重后：

- 人脸/车牌的小区域误差会被放大
- 优化器更愿意优先修这些区域

所以 ROI 的实际效果是：

- 让训练更偏向保护面部和车牌这种敏感区域
- 即使这些区域面积很小，也不会被背景淹没

### 6.6 ROI 在验证时怎么用

[train.py](/mnt/d/fudan/prefilter_clean/train.py#L154) 到 [train.py](/mnt/d/fudan/prefilter_clean/train.py#L177) 里：

- 验证 loss 也会继续把 `roi` 传给 `compute_train_loss`
- 所以 `val_loss` 也是 ROI 加权后的 fidelity loss + MSSSIM
- 但 `PSNR / SSIM` 还是直接按预测和 GT 算，不带 ROI 权重

也就是说：

- `val_loss` 体现“训练目标”
- `val_psnr / val_ssim` 体现“整体图像指标”

## 7. 数据目录最终长什么样

如果启用了 `--detect_roi`，推荐结构如下：

```text
data/xlx_clean_roi/
  train/
    img/
    gt/
    roi/
    manifest.csv
  val/
    img/
    gt/
    roi/
    manifest.csv
  test/
    img/
    gt/
    roi/
    manifest.csv
  dataset_summary.json
```

其中：

- `img`
  - 输入 YUV
- `gt`
  - 参考目标 YUV
- `roi`
  - 二值 `.npy` 掩码

## 8. `manifest.csv` 里会记录什么

现在的 `manifest.csv` 会包含：

- `input_path`
- `target_path`
- `width`
- `height`
- `format`
- `bitdepth`
- `source_video`
- `source_frame`
- `tile_id`
- `tile_top`
- `tile_left`
- `roi_path`

如果当前样本没有 ROI：

- `roi_path` 为空字符串

如果有 ROI：

- `roi_path` 指向对应的 `.npy`

### 8.1 `roi_path` 和真实文件是否已经验证过

已经验证过。

我实际跑过一个最小样本数据集：

- [roi_smoke](/mnt/d/fudan/prefilter_clean/data/roi_smoke)

并确认：

- [train/manifest.csv](/mnt/d/fudan/prefilter_clean/data/roi_smoke/train/manifest.csv) 已写入 `roi_path`
- 对应的 ROI 文件已经真实落盘，例如：
  - [4mm_human_scene_3840x2160_nv12_frame000000_tile000_y0000_x0000.npy](/mnt/d/fudan/prefilter_clean/data/roi_smoke/train/roi/4mm_human_scene_3840x2160_nv12/4mm_human_scene_3840x2160_nv12_frame000000_tile000_y0000_x0000.npy)

## 9. 你现在这条命令生成出来的数据是否适合训练

适合。

并且相对你之前的 `xlx_clean` 更接近原版，原因是：

- `tile_size=0`
  - 保留整帧
- `detect_roi`
  - 会生成 ROI mask
- 训练时会走：
  - ROI 引导 crop
  - ROI 加权 loss

所以这套数据比之前的：

- 先切 `512 tile`
- 没有 ROI

更接近原版训练分布。

## 10. 推荐的数据处理命令

### 10.1 最接近原版的做法

```bash
cd /mnt/d/fudan/prefilter_clean

python process_data.py \
  --input_dir /mnt/d/fudan/xlx_yuv \
  --output_dir /mnt/d/fudan/prefilter_clean/data/xlx_clean_roi \
  --frames_per_video 12 \
  --pix_fmt yuv420p \
  --yuv_format yuv420p \
  --bitdepth 8 \
  --tile_size 0 \
  --detect_roi \
  --workers 1 \
  --overwrite
```

### 10.2 如果你已经有现成 ROI `.npy`

```bash
python process_data.py \
  --input_dir /mnt/d/fudan/xlx_yuv \
  --output_dir /mnt/d/fudan/prefilter_clean/data/xlx_clean_roi \
  --frames_per_video 12 \
  --pix_fmt yuv420p \
  --yuv_format yuv420p \
  --bitdepth 8 \
  --tile_size 0 \
  --roi_dir /path/to/roi_npy \
  --workers 1 \
  --overwrite
```

### 10.3 如果你想把检测可视化也存下来

```bash
python process_data.py \
  --input_dir /mnt/d/fudan/xlx_yuv \
  --output_dir /mnt/d/fudan/prefilter_clean/data/xlx_clean_roi \
  --frames_per_video 12 \
  --pix_fmt yuv420p \
  --yuv_format yuv420p \
  --bitdepth 8 \
  --tile_size 0 \
  --detect_roi \
  --roi_save_vis \
  --workers 1 \
  --overwrite
```

这会多生成：

- `output_dir/roi_vis/.../*.jpg`

## 11. 这批 `h265` 的实际帧数统计，以及 `frames_per_video` 怎么选

这部分不是拍脑袋估计，是已经对当前目录：

- `/mnt/d/fudan/xlx_yuv`

里的所有 `h265/hevc` 视频做过 `ffprobe` 统计后得到的。

### 11.1 当前视频总体分布

当前视频共：

- `39` 个

帧数统计如下：

- 最少：`196` 帧
- 中位数：`600` 帧
- 平均数：`718.74` 帧
- 最多：`1740` 帧

四分位大致是：

- `P25 = 400`
- `P75 = 1045`

也就是说，这批视频大多数并不是几万帧的长视频，而是大约：

- 几百帧到一千多帧

为主。

### 11.2 这些帧数大概对应多长时间

当前视频帧率基本都是：

- `25 fps`

换算后大致是：

- `196` 帧 ≈ `7.8` 秒
- `400` 帧 ≈ `16` 秒
- `600` 帧 ≈ `24` 秒
- `1045` 帧 ≈ `41.8` 秒
- `1740` 帧 ≈ `69.6` 秒

所以：

- 这批数据主要是 `8 秒 ~ 70 秒` 量级的视频

### 11.3 一些代表样本

短视频代表：

- `ch2_4k15_case01_3840x2160_nv12_fps15.h265`：`196` 帧
- `ch2_4k15_case02_01_3840x2160_nv12_fps15.h265`：`228` 帧
- `CH2_4M_15fps_2Mbps_3840x2160_nv12_fps15.h265`：`252` 帧
- `case_day_avenue3_3840x2160_nv12.h265`：`285` 帧
- `Netflix_BarScene_3840x2160_NV12_20fps.h265`：`400` 帧

中等长度代表：

- `case6_xgm_outdoor_day_car1_3840x2160_nv12.h265`：`600` 帧
- `southgate_garden_scene_3840x2160_nv12.h265`：`793` 帧
- `spring_garden_scene_3840x2160_nv12.h265`：`991` 帧

长视频代表：

- `4mm_human_scene_3840x2160_nv12.h265`：`1031` 帧
- `6mm_human_scene_3840x2160_nv12.h265`：`1125` 帧
- `case_day_avenue_3840x2160_nv12.h265`：`1200` 帧
- `4mm_still_3840x2160_nv12.h265`：`1514` 帧
- `case_night_parking_3840x2160_nv12.h265`：`1740` 帧

### 11.4 `frames_per_video=12` 合不合理

结论：

- 作为第一版正式数据，`12` 是合理的
- 不算偏少
- 但对最长的一批视频来说，时间采样会略稀

原因是你现在不是“每个视频只产出 12 个样本”，而是：

1. 每个视频先抽 `12` 帧
2. 每帧再切 `512x512` tile

也就是说，抽帧数只是第一层采样。

### 11.5 如果 `tile_size=512`，样本量大概有多少

已经估算过整批数据在不同 `frames_per_video` 下的样本量：

- `frames_per_video = 12` 时，约 `18420` 个 tile 样本
- `frames_per_video = 16` 时，约 `24560` 个 tile 样本
- `frames_per_video = 24` 时，约 `36840` 个 tile 样本

所以：

- `12` 帧时，数据量已经不小
- 如果直接上 `24`，数据处理和训练成本会明显增加

### 11.6 为什么 `12` 对当前批次是合理 baseline

因为当前数据有三个放大因素：

1. 视频总数有 `39`
2. 每个视频抽 `12` 帧
3. 每个 4K 帧再切成很多 `512` tile

以 `3840x2160` 为例：

- 一帧大约切成 `40` 个 tile

那么一个视频抽 `12` 帧，大约就是：

- `12 x 40 = 480` 个样本

所以：

- `12` 帧不算“小数据”

### 11.7 `12` 的短板在哪里

短板主要在最长视频上。

例如：

- `1740` 帧视频如果均匀抽 `12` 帧
- 相邻采样点间隔大约 `158` 帧
- 在 `25 fps` 下大约每 `6.3` 秒取一帧

这对长视频来说会偏稀。

但对 `400 ~ 800` 帧这一档视频：

- `12` 帧通常已经足够覆盖主要时序变化

### 11.8 我的建议

推荐顺序如下：

- 第一选择：`frames_per_video = 12`
- 更稳妥但更慢：`frames_per_video = 16`
- 不建议第一版就上：`frames_per_video = 24`

如果你当前是第一次正式生成完整数据并开训，建议先用：

```bash
--frames_per_video 12
```

如果后续你发现：

- 长视频时序变化很大
- ROI 目标出现频率低
- 采样覆盖还不够

再考虑升到：

```bash
--frames_per_video 16
```

## 12. 训练命令

先把 [configs/train.yaml](/mnt/d/fudan/prefilter_clean/configs/train.yaml) 里的：

- `data.root`
- `experiment_name`

改成你自己的数据目录和实验名。

然后运行：

```bash
cd /mnt/d/fudan/prefilter_clean
python train.py --config configs/train.yaml
```

### 12.1 当前推荐的“安全版微调”配置

如果你的目标是：

- 基于已有成熟权重做微调
- 尽量不要比原模型更差
- 先稳住泛化，再追求增益

那么当前推荐优先使用：

- [finetune_xlx_clean_roi_512.yaml](/mnt/d/fudan/prefilter_clean/configs/finetune_xlx_clean_roi_512.yaml)

对应启动命令：

```bash
cd /mnt/d/fudan/prefilter_clean

python train.py \
  --config configs/finetune_xlx_clean_roi_512.yaml
```

这个配置的关键点是：

- 预训练权重：
  - `/mnt/d/fudan/prefilter_clean/weights/iccv_yan_2025_fp32.pth`
- 数据：
  - `./data/xlx_clean_roi_512`
- `crop_size = 224`
- `batch_size = 16`
- `lr = 1e-5`
- `total_iter = 12000`
- `warmup_iter = 500`
- `resume = auto`

### 12.2 为什么把它叫“安全版”

这不是拍脑袋保守，而是结合你当前模型大小、数据量级和目标后定的。

#### 模型很小

当前模型参数量已经实际统计过，大约是：

- `25360`

这是一个非常小的局部滤波模型。

这种模型的特点是：

- 学得快
- 也容易被过大的学习率推偏

所以对“已经训练成熟的预训练权重”做微调时，学习率应该明显小于从头训练时的值。

#### 数据表面很多，但相关性很强

当前新数据集：

- train 样本数：`14580`
- val 样本数：`1920`
- test 样本数：`1920`

如果 `batch_size = 16`，则：

- 每个 epoch 的 iter 数约为 `912`

所以：

- `12000 iter ≈ 13.16 epoch`

这不是只训一两轮，而是大约 `13` 个 epoch。

但这里要注意：

- 这些样本来自 `31` 个 train 视频
- 每视频抽 `12` 帧
- 每帧再切很多 `512 tile`

所以样本之间相关性很强，不能把 `14580` 看成完全独立样本。

这意味着：

- 训练太短可能适配不够
- 训练太长又更容易过拟合这批视频风格

对这种场景，`12000 iter` 属于中等偏保守、但完全合理的第一阶段微调强度。

#### 训练时还有动态 random crop

虽然磁盘上保存的是 `512x512 tile`，但训练时实际喂给模型的是：

- 从 `512x512` 里动态 random crop 的 `224x224`

所以：

- 同一个 tile 在不同 epoch 中看到的 patch 位置并不完全相同
- 有效训练量会大于“死板重复喂同一块图”的情况

这进一步说明：

- `12000 iter` 并不弱

#### 学习率为什么选 `1e-5`

从头训练时，类似这种小模型用：

- `3e-4`

是正常的。

但你现在不是从头训练，而是：

- 加载已经在别的数据上训练饱和过的成熟权重
- 再在你的数据上做微调

这种情况下，经验上通常会把学习率降一个数量级到两个数量级。

所以：

- `1e-5`

是一个典型的“安全起步”值。

它的目标不是最快适应，而是：

- 尽量保留原模型已有能力
- 同时让模型逐步向你的数据分布靠近

### 12.3 为什么不是更短，或者更长

#### 为什么不是更短

如果只训：

- `2000 ~ 4000 iter`

那在你当前数据量级上，往往更像是“轻微调味”，未必足够完成稳定适配。

#### 为什么不是一开始就更长

如果直接上：

- `20000+ iter`

在你这种高相关 tile 数据上，更容易把模型推向当前数据风格，反而增加退化风险。

对于“不要比原模型更差”这个目标，第一阶段更合理的思路是：

- 先用 `12000 iter` 跑一版安全微调
- 看验证集和测试集结果
- 如果还没到平台，再续训

### 12.4 这个安全版怎么续训

当前安全版配置里已经是：

- `resume: auto`

所以继续训练时，直接重复执行同一条命令即可：

```bash
cd /mnt/d/fudan/prefilter_clean

python train.py \
  --config configs/finetune_xlx_clean_roi_512.yaml
```

它会自动从：

- `runs/xlx_clean_roi_512_finetune/checkpoints/latest.pt`

继续训练。

如果后续你观察到：

- `best.pt` 明显优于 baseline
- 但验证曲线还没有完全平台

那么更稳妥的做法不是换大学习率，而是：

1. 保持同一个 `experiment_name`
2. 把 `total_iter` 从 `12000` 提到例如 `18000`
3. 再继续运行同一条命令

这样会比一开始就上长训练更稳。

### 12.5 安全版的核心目标

安全版不是为了追求一上来就最大增益，而是为了：

- 不破坏原模型已经学到的能力
- 用较低风险适配你的新数据
- 让你先得到一个“不更差、而且有机会更稳”的版本

如果后面安全版结果稳定，再去尝试更激进的微调配置，会更合理。

## 13. 已完成的验证

这次改动已经验证过以下内容：

- OpenCV 当前环境已满足：
  - `cv2 4.11.0`
  - `ximgproc.l0Smooth`
  - `ximgproc.guidedFilter`
- `process_data.py / roi.py / utils.py` 语法检查通过
- `ultralytics` 可导入
- 单帧 ROI 检测链路跑通
- 实际样本 `Netflix_ToddlerFountain_3840x2160_NV12_20fps.h265` 的 `frame 10` 检出正 ROI
- `process_data.py --detect_roi` 端到端跑通
- 输出数据集中 `manifest.csv` 已正确写入 `roi_path`
- 对应的 `roi/*.npy` 文件已真实落盘

## 14. 现在最值得优先做的事

顺序建议如下：

1. 现在 OpenCV 已经符合要求，先保持当前环境不动
2. 用 `--tile_size 0 --detect_roi` 重新生成完整数据
3. 确认输出目录里 `train/val/test` 都含有 `manifest.csv` 和 `roi_path`
4. 再开始正式微调

这样你的数据链路会最接近原版。
