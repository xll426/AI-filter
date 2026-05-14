"""FP32 PrefilterNet 与部署重参数化逻辑。

训练态使用过参数化的 `MBRConv3` 多分支结构提升优化稳定性。部署和 QAT 阶段调用
`MBRConv3.slim()`，把多分支结构等价融合为 PixelUnshuffle(4) 后 Y 特征上的
单个 `3x3` 残差卷积。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MBRConv3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4) -> None:
        super().__init__()
        expanded = out_channels * rep_scale
        self.conv = nn.Conv2d(in_channels, expanded, 3, 1, 1)
        self.conv_bn = nn.BatchNorm2d(expanded)
        self.conv1 = nn.Conv2d(in_channels, expanded, 1)
        self.conv1_bn = nn.BatchNorm2d(expanded)
        self.conv_crossh = nn.Conv2d(in_channels, expanded, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.BatchNorm2d(expanded)
        self.conv_crossv = nn.Conv2d(in_channels, expanded, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.BatchNorm2d(expanded)
        self.conv_out = nn.Conv2d(expanded * 8, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv_crossh(x)
        x3 = self.conv_crossv(x)
        x_cat = torch.cat(
            [
                x0,
                x1,
                x2,
                x3,
                self.conv_bn(x0),
                self.conv1_bn(x1),
                self.conv_crossh_bn(x2),
                self.conv_crossv_bn(x3),
            ],
            dim=1,
        )
        return self.conv_out(x_cat)

    def slim(self) -> tuple[torch.Tensor, torch.Tensor]:
        """把所有 MBRConv3 分支融合为一个等价的 `3x3` 卷积核和 bias。

        融合步骤：
        1. 将 `1x1`、`3x1`、`1x3` 分支 padding 到 `3x3` 形状；
        2. 使用 BN 的 running mean / running var 将 Conv+BN 分支折叠成等价 Conv；
        3. 拼接所有分支的 kernel/bias；
        4. 用最后的 `conv_out` 权重做线性压缩，得到最终部署所需的单个卷积核。
        """
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        conv1_weight = torch.nn.functional.pad(self.conv1.weight, (1, 1, 1, 1))
        conv1_bias = self.conv1.bias

        conv_crossh_weight = torch.nn.functional.pad(self.conv_crossh.weight, (1, 1, 0, 0))
        conv_crossh_bias = self.conv_crossh.bias

        conv_crossv_weight = torch.nn.functional.pad(self.conv_crossv.weight, (0, 0, 1, 1))
        conv_crossv_bias = self.conv_crossv.bias

        bn = self.conv_bn
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + (-bn.running_mean * k)
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        bn = self.conv1_bn
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv1_bn_weight = self.conv1.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = torch.nn.functional.pad(conv1_bn_weight, (1, 1, 1, 1))
        conv1_bn_bias = self.conv1.bias * k + (-bn.running_mean * k)
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossh_bn
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = torch.nn.functional.pad(conv_crossh_bn_weight, (1, 1, 0, 0))
        conv_crossh_bn_bias = self.conv_crossh.bias * k + (-bn.running_mean * k)
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossv_bn
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = torch.nn.functional.pad(conv_crossv_bn_weight, (0, 0, 1, 1))
        conv_crossv_bn_bias = self.conv_crossv.bias * k + (-bn.running_mean * k)
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

        weight = torch.cat(
            [
                conv_weight,
                conv1_weight,
                conv_crossh_weight,
                conv_crossv_weight,
                conv_bn_weight,
                conv1_bn_weight,
                conv_crossh_bn_weight,
                conv_crossv_bn_weight,
            ],
            dim=0,
        )
        bias = torch.cat(
            [
                conv_bias,
                conv1_bias,
                conv_crossh_bias,
                conv_crossv_bias,
                conv_bn_bias,
                conv1_bn_bias,
                conv_crossh_bn_bias,
                conv_crossv_bn_bias,
            ],
            dim=0,
        )

        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.view(weight.size(0), -1))
        weight = weight.view(self.conv_out.out_channels, self.conv.in_channels, 3, 3)

        bias = torch.matmul(weight_compress, bias.unsqueeze(-1)).squeeze(-1)
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias


class PrefilterNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        downscale_factor: int = 4,
        rep_scale: int = 4,
        only_train_y: bool = True,
    ) -> None:
        super().__init__()
        self.only_train_y = only_train_y
        self.downscale_factor = downscale_factor
        effective_in = 1 if only_train_y else in_channels
        expanded_channels = effective_in * (downscale_factor ** 2)

        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.processing = MBRConv3(expanded_channels, expanded_channels, rep_scale=rep_scale)
        self.pixel_shuffle = nn.PixelShuffle(downscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 交付模型仅滤波 Y。UV 原样透传，因此部署侧使用只包含 Y 的 ONNX 图，
        # 再把原始 chroma 字节拼回输出 YUV。
        if self.only_train_y:
            y = x[:, :1]
            uv = x[:, 1:] if x.shape[1] > 1 else None
        else:
            y = x
            uv = None

        y_unshuffled = self.pixel_unshuffle(y)
        y_processed = self.processing(y_unshuffled)
        y_restored = self.pixel_shuffle(y_unshuffled + y_processed)

        if self.only_train_y and uv is not None:
            return torch.cat([y_restored, uv], dim=1)
        return y_restored


def _unwrap_state_dict(state: dict) -> dict:
    if "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if "params" in state and isinstance(state["params"], dict):
        return state["params"]
    return state


def convert_state_dict_for_prefilter(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("network.processing.block1.0."):
            new_key = new_key.replace("network.processing.block1.0.", "processing.")
        new_key = new_key.replace("conv_bn.0.", "conv_bn.")
        new_key = new_key.replace("conv1_bn.0.", "conv1_bn.")
        new_key = new_key.replace("conv_crossh_bn.0.", "conv_crossh_bn.")
        new_key = new_key.replace("conv_crossv_bn.0.", "conv_crossv_bn.")
        converted[new_key] = value
    return converted


def load_prefilter_state(model: nn.Module, checkpoint: dict, strict: bool = True) -> tuple[list[str], list[str]]:
    state_dict = convert_state_dict_for_prefilter(_unwrap_state_dict(checkpoint))
    result = model.load_state_dict(state_dict, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
