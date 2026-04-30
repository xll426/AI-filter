from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# STE helpers
# =========================================================

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """
    Forward: round(x)
    Backward: d/dx ~= 1
    """
    x_q = torch.round(x)
    return x + (x_q - x).detach()


def ste_round_clamp(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """
    Forward: clamp(round(x), qmin, qmax)
    Backward: d/dx ~= 1
    """
    x_q = torch.clamp(torch.round(x), float(qmin), float(qmax))
    return x + (x_q - x).detach()


def ste_clip_u8(x: torch.Tensor) -> torch.Tensor:
    """
    Forward: clamp(round(x), 0, 255)
    Backward: d/dx ~= 1
    """
    x_q = torch.clamp(torch.round(x), 0.0, 255.0)
    return x + (x_q - x).detach()


def ste_round_shift(acc: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """
    acc:   [N, C, H, W]
    shift: [C] int-like tensor, per-output-channel fixed shift

    Forward:
        y = round(acc / 2^shift)
    Backward:
        ignore round, so d/dacc ~= 1 / 2^shift
    """
    if acc.ndim != 4:
        raise ValueError(f"acc must be NCHW, got shape={tuple(acc.shape)}")
    if shift.ndim != 1:
        raise ValueError(f"shift must be 1D [C], got shape={tuple(shift.shape)}")
    if acc.size(1) != shift.numel():
        raise ValueError(
            f"shift channels mismatch: acc C={acc.size(1)} vs shift={shift.numel()}"
        )

    scale = torch.pow(
        torch.tensor(2.0, device=acc.device, dtype=acc.dtype),
        -shift.to(device=acc.device, dtype=acc.dtype),
    ).view(1, -1, 1, 1)

    y = acc * scale
    y_q = torch.round(y)
    return y + (y_q - y).detach()


# =========================================================
# Regularization helpers
# =========================================================

def range_penalty(latent_int: torch.Tensor, qmax: int) -> torch.Tensor:
    """
    Penalize latent values that want to exceed integer range before clamp.
    """
    overflow = torch.relu(torch.abs(latent_int) - float(qmax))
    return (overflow ** 2).mean()


def bias_l1_penalty(bias: torch.Tensor) -> torch.Tensor:
    return torch.abs(bias).mean()


# =========================================================
# Configs
# =========================================================

@dataclass
class IntQATConfig:
    # integer parameter ranges
    weight_bits: int = 12
    bias_bits: int = 17

    # model topology
    downscale_factor: int = 4
    only_train_y: bool = True
    per_channel_shift: bool = True

    # fixed shift bounds
    min_shift: int = 0
    max_shift: int = 30

    # regularization weights
    weight_range_penalty: float = 1e-6
    bias_range_penalty: float = 1e-6
    bias_l1_weight: float = 0.0


# =========================================================
# Deploy FP32 teacher model
# =========================================================

class DeployPrefilterFP32Reference(nn.Module):
    """
    Deploy-structure FP32 reference model for teacher distillation.

    It uses the fused 3x3 kernel from fp32_model.processing.slim(),
    but still works in normalized [0,1] domain so it plugs directly
    into the current training/loss pipeline.
    """

    def __init__(
        self,
        fused_weight_fp: torch.Tensor,
        fused_bias_fp: torch.Tensor,
        downscale_factor: int = 4,
        only_train_y: bool = True,
    ) -> None:
        super().__init__()
        self.only_train_y = bool(only_train_y)
        self.downscale_factor = int(downscale_factor)

        out_channels, in_channels, kh, kw = fused_weight_fp.shape
        if kh != 3 or kw != 3:
            raise ValueError(f"Expected 3x3 fused conv, got {tuple(fused_weight_fp.shape)}")
        if fused_bias_fp.ndim != 1 or fused_bias_fp.numel() != out_channels:
            raise ValueError(f"Bias shape mismatch: {tuple(fused_bias_fp.shape)}")

        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale_factor)
        self.pixel_shuffle = nn.PixelShuffle(self.downscale_factor)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        with torch.no_grad():
            self.conv.weight.copy_(fused_weight_fp.float())
            self.conv.bias.copy_(fused_bias_fp.float())

    @staticmethod
    def from_fp32_prefilter(fp32_model: nn.Module) -> "DeployPrefilterFP32Reference":
        if not hasattr(fp32_model, "processing"):
            raise AttributeError("Expected fp32_model to have .processing.slim().")
        fused_weight_fp, fused_bias_fp = fp32_model.processing.slim()
        return DeployPrefilterFP32Reference(
            fused_weight_fp=fused_weight_fp.detach(),
            fused_bias_fp=fused_bias_fp.detach(),
            downscale_factor=int(fp32_model.downscale_factor),
            only_train_y=bool(fp32_model.only_train_y),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW input, got {tuple(x.shape)}")

        if self.only_train_y:
            y = x[:, :1]
            uv = x[:, 1:] if x.size(1) > 1 else None
        else:
            y = x
            uv = None

        y_u = self.pixel_unshuffle(y)
        delta_u = self.conv(y_u)
        y_u_out = y_u + delta_u
        y_out = self.pixel_shuffle(y_u_out)

        if self.only_train_y and uv is not None:
            return torch.cat([y_out, uv], dim=1)
        return y_out


def build_deploy_fp32_reference(fp32_model: nn.Module) -> DeployPrefilterFP32Reference:
    teacher = DeployPrefilterFP32Reference.from_fp32_prefilter(fp32_model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# =========================================================
# Distillation loss
# =========================================================

class DistillationLoss(nn.Module):
    """
    Student and teacher are both expected to be normalized [0,1] tensors.
    Default: L1 on Y channel only.
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        only_y: bool = True,
        use_charbonnier: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.only_y = bool(only_y)
        self.use_charbonnier = bool(use_charbonnier)
        self.eps = float(eps)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if self.only_y:
            student = student[:, :1]
            teacher = teacher[:, :1]

        if self.use_charbonnier:
            diff = torch.sqrt((student - teacher) ** 2 + self.eps)
            return self.loss_weight * diff.mean()

        return self.loss_weight * torch.mean(torch.abs(student - teacher))


# =========================================================
# Integer-simulated deploy student
# =========================================================

class DeployPrefilterIntQAT(nn.Module):
    """
    Integer-aware deploy student.

    Expected external input/output domain:
        normalized [0,1]

    Internal forward simulation:
        x_norm -> round(x_norm * 255) -> raw u8 semantics
               -> take Y
               -> PixelUnshuffle
               -> integer-simulated conv (q_w, q_b)
               -> per-channel right shift
               -> residual add in raw domain
               -> clip to [0,255]
               -> PixelShuffle
               -> concat UV
               -> divide by 255 back to [0,1]
    """

    def __init__(
        self,
        fused_weight_fp: torch.Tensor,
        fused_bias_fp_raw: torch.Tensor,
        cfg: IntQATConfig,
        init_shift: torch.Tensor,
    ) -> None:
        super().__init__()

        if fused_weight_fp.ndim != 4:
            raise ValueError(f"fused_weight_fp must be OIHW, got {tuple(fused_weight_fp.shape)}")
        if fused_bias_fp_raw.ndim != 1:
            raise ValueError(f"fused_bias_fp_raw must be 1D, got {tuple(fused_bias_fp_raw.shape)}")

        out_channels, in_channels, kh, kw = fused_weight_fp.shape
        if kh != 3 or kw != 3:
            raise ValueError("This implementation assumes fused 3x3 conv.")
        if fused_bias_fp_raw.numel() != out_channels:
            raise ValueError("Bias channel count mismatch.")

        self.cfg = cfg
        self.only_train_y = bool(cfg.only_train_y)
        self.downscale_factor = int(cfg.downscale_factor)

        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale_factor)
        self.pixel_shuffle = nn.PixelShuffle(self.downscale_factor)

        # float master params (trainable)
        self.weight_fp = nn.Parameter(fused_weight_fp.clone().float())
        self.bias_fp = nn.Parameter(fused_bias_fp_raw.clone().float())

        # fixed integer shifts
        if init_shift.ndim != 1 or init_shift.numel() != out_channels:
            raise ValueError(
                f"init_shift must be [C_out], got {tuple(init_shift.shape)}, C_out={out_channels}"
            )
        init_shift = torch.clamp(init_shift.long(), min=cfg.min_shift, max=cfg.max_shift)
        self.register_buffer("shift", init_shift)

        # integer ranges
        self.weight_qmin = -(1 << (cfg.weight_bits - 1))
        self.weight_qmax = (1 << (cfg.weight_bits - 1)) - 1
        self.bias_qmin = -(1 << (cfg.bias_bits - 1))
        self.bias_qmax = (1 << (cfg.bias_bits - 1)) - 1

    # -----------------------------------------------------
    # construction helpers
    # -----------------------------------------------------
    @staticmethod
    def from_fp32_prefilter(fp32_model: nn.Module, cfg: IntQATConfig) -> "DeployPrefilterIntQAT":
        """
        Build student from existing PrefilterNet.

        fp32_model is trained in normalized [0,1] domain.
        For raw-domain residual simulation:
            bias_raw = bias_norm * 255
        while the fused weight keeps the same numeric value.
        """
        if not hasattr(fp32_model, "processing"):
            raise AttributeError("Expected fp32_model to have .processing.slim().")

        fused_weight_fp, fused_bias_fp = fp32_model.processing.slim()
        fused_weight_fp = fused_weight_fp.detach().float()
        fused_bias_fp = fused_bias_fp.detach().float()
        fused_bias_fp_raw = fused_bias_fp * 255.0

        init_shift = DeployPrefilterIntQAT._init_shift_from_fp(
            fused_weight_fp=fused_weight_fp,
            fused_bias_fp_raw=fused_bias_fp_raw,
            weight_bits=cfg.weight_bits,
            bias_bits=cfg.bias_bits,
            per_channel=cfg.per_channel_shift,
            min_shift=cfg.min_shift,
            max_shift=cfg.max_shift,
        )

        return DeployPrefilterIntQAT(
            fused_weight_fp=fused_weight_fp,
            fused_bias_fp_raw=fused_bias_fp_raw,
            cfg=cfg,
            init_shift=init_shift,
        )

    @staticmethod
    def _init_shift_from_fp(
        fused_weight_fp: torch.Tensor,
        fused_bias_fp_raw: torch.Tensor,
        weight_bits: int,
        bias_bits: int,
        per_channel: bool = True,
        min_shift: int = 0,
        max_shift: int = 30,
    ) -> torch.Tensor:
        """
        Choose initial integer shift so that both weight and bias try to fit.

            q_w = round(w_fp * 2^s)
            q_b = round(b_fp_raw * 2^s)

        Need both within target integer ranges.
        """
        out_channels = fused_weight_fp.size(0)
        w_qmax = (1 << (weight_bits - 1)) - 1
        b_qmax = (1 << (bias_bits - 1)) - 1

        if per_channel:
            shifts: list[int] = []
            for c in range(out_channels):
                w_abs = fused_weight_fp[c].abs().max().item()
                b_abs = abs(float(fused_bias_fp_raw[c].item()))

                if w_abs == 0.0 and b_abs == 0.0:
                    shifts.append(max_shift)
                    continue

                s_w = max_shift
                s_b = max_shift
                if w_abs > 0.0:
                    s_w = int(math.floor(math.log2(w_qmax / w_abs))) if w_abs < w_qmax else 0
                if b_abs > 0.0:
                    s_b = int(math.floor(math.log2(b_qmax / b_abs))) if b_abs < b_qmax else 0

                s = min(s_w, s_b)
                s = max(min_shift, min(max_shift, s))
                shifts.append(s)
            return torch.tensor(shifts, dtype=torch.long)

        w_abs = fused_weight_fp.abs().max().item()
        b_abs = fused_bias_fp_raw.abs().max().item()

        s_w = max_shift
        s_b = max_shift
        if w_abs > 0.0:
            s_w = int(math.floor(math.log2(w_qmax / w_abs))) if w_abs < w_qmax else 0
        if b_abs > 0.0:
            s_b = int(math.floor(math.log2(b_qmax / b_abs))) if b_abs < b_qmax else 0

        s = min(s_w, s_b)
        s = max(min_shift, min(max_shift, s))
        return torch.full((out_channels,), s, dtype=torch.long)

    # -----------------------------------------------------
    # internal quantization helpers
    # -----------------------------------------------------
    def _shift_scale(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.pow(
            torch.tensor(2.0, device=device, dtype=dtype),
            self.shift.to(device=device, dtype=dtype),
        )

    def _quantize_weight_to_int(self) -> torch.Tensor:
        scale = self._shift_scale(device=self.weight_fp.device, dtype=self.weight_fp.dtype).view(-1, 1, 1, 1)
        latent_int = self.weight_fp * scale
        q_w = ste_round_clamp(latent_int, self.weight_qmin, self.weight_qmax)
        return q_w

    def _quantize_bias_to_int(self) -> torch.Tensor:
        scale = self._shift_scale(device=self.bias_fp.device, dtype=self.bias_fp.dtype)
        latent_int = self.bias_fp * scale
        q_b = ste_round_clamp(latent_int, self.bias_qmin, self.bias_qmax)
        return q_b

    def auxiliary_regularization(self) -> dict[str, torch.Tensor]:
        scale_w = self._shift_scale(device=self.weight_fp.device, dtype=self.weight_fp.dtype).view(-1, 1, 1, 1)
        weight_latent_int = self.weight_fp * scale_w

        scale_b = self._shift_scale(device=self.bias_fp.device, dtype=self.bias_fp.dtype)
        bias_latent_int = self.bias_fp * scale_b

        reg_weight_range = range_penalty(weight_latent_int, self.weight_qmax) * float(self.cfg.weight_range_penalty)
        reg_bias_range = range_penalty(bias_latent_int, self.bias_qmax) * float(self.cfg.bias_range_penalty)
        reg_bias_l1 = bias_l1_penalty(self.bias_fp) * float(self.cfg.bias_l1_weight)

        return {
            "reg_weight_range": reg_weight_range,
            "reg_bias_range": reg_bias_range,
            "reg_bias_l1": reg_bias_l1,
        }

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------
    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        if x_norm.ndim != 4:
            raise ValueError(f"Expected NCHW input, got {tuple(x_norm.shape)}")

        # restore raw pixel domain with integer semantics
        x_raw = ste_clip_u8(x_norm * 255.0)

        if self.only_train_y:
            y = x_raw[:, :1]
            uv = x_raw[:, 1:] if x_raw.size(1) > 1 else None
        else:
            y = x_raw
            uv = None

        y_u = self.pixel_unshuffle(y)

        q_w = self._quantize_weight_to_int()
        q_b = self._quantize_bias_to_int()

        # q_w / q_b are integer-semantics tensors stored in float dtype for autograd
        acc = F.conv2d(y_u, q_w, q_b, stride=1, padding=1)
        delta = ste_round_shift(acc, self.shift)

        y_u_out = ste_clip_u8(y_u + delta)
        y_out = self.pixel_shuffle(y_u_out)

        if self.only_train_y and uv is not None:
            out_raw = torch.cat([y_out, uv], dim=1)
        else:
            out_raw = y_out

        out_norm = out_raw / 255.0
        return out_norm

    # -----------------------------------------------------
    # export
    # -----------------------------------------------------
    @torch.no_grad()
    def export_int_parameters(self) -> dict[str, torch.Tensor]:
        """
        Export true integer deployment params.

        Returns:
            q_w         : int32 [Cout, Cin, 3, 3]
            q_b         : int32 [Cout]
            shift       : int32 [Cout]
            weight_fp   : float32 master weight
            bias_fp_raw : float32 master raw-domain bias
            weight_eff_fp : q_w / 2^shift
            bias_eff_fp   : q_b / 2^shift
        """
        device = self.weight_fp.device
        scale = self._shift_scale(device=device, dtype=torch.float32)

        q_w = torch.clamp(
            torch.round(self.weight_fp.detach().to(torch.float32) * scale.view(-1, 1, 1, 1)),
            float(self.weight_qmin),
            float(self.weight_qmax),
        ).to(torch.int32)

        q_b = torch.clamp(
            torch.round(self.bias_fp.detach().to(torch.float32) * scale),
            float(self.bias_qmin),
            float(self.bias_qmax),
        ).to(torch.int32)

        shift_i32 = self.shift.detach().to(torch.int32)
        denom = torch.pow(torch.tensor(2.0, dtype=torch.float32), shift_i32.to(torch.float32))

        weight_eff_fp = q_w.to(torch.float32) / denom.view(-1, 1, 1, 1)
        bias_eff_fp = q_b.to(torch.float32) / denom

        return {
            "q_w": q_w.cpu(),
            "q_b": q_b.cpu(),
            "shift": shift_i32.cpu(),
            "weight_fp": self.weight_fp.detach().cpu().to(torch.float32),
            "bias_fp_raw": self.bias_fp.detach().cpu().to(torch.float32),
            "weight_eff_fp": weight_eff_fp.cpu(),
            "bias_eff_fp": bias_eff_fp.cpu(),
        }

    @torch.no_grad()
    def quantization_stats(self) -> dict[str, float]:
        """
        Lightweight scalar stats for logging QAT health.
        """
        params = self.export_int_parameters()
        q_w = params["q_w"]
        q_b = params["q_b"]
        shift = params["shift"]
        w_qmax = max(float(self.weight_qmax), 1.0)
        b_qmax = max(float(self.bias_qmax), 1.0)

        max_abs_q_w = float(q_w.abs().max().item())
        max_abs_q_b = float(q_b.abs().max().item())
        return {
            "max_abs_q_w": max_abs_q_w,
            "max_abs_q_b": max_abs_q_b,
            "q_w_usage": max_abs_q_w / w_qmax,
            "q_b_usage": max_abs_q_b / b_qmax,
            "shift_min": float(shift.min().item()),
            "shift_max": float(shift.max().item()),
            "shift_mean": float(shift.float().mean().item()),
        }

    @torch.no_grad()
    def save_export(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = self.export_int_parameters()
        torch.save(payload, output_dir / "int_params.pt")
        torch.save(payload["q_w"], output_dir / "q_w.pt")
        torch.save(payload["q_b"], output_dir / "q_b.pt")
        torch.save(payload["shift"], output_dir / "shift.pt")

        meta: dict[str, Any] = {
            "weight_bits": int(self.cfg.weight_bits),
            "bias_bits": int(self.cfg.bias_bits),
            "downscale_factor": int(self.cfg.downscale_factor),
            "only_train_y": bool(self.cfg.only_train_y),
            "per_channel_shift": bool(self.cfg.per_channel_shift),
            "weight_qmin": int(self.weight_qmin),
            "weight_qmax": int(self.weight_qmax),
            "bias_qmin": int(self.bias_qmin),
            "bias_qmax": int(self.bias_qmax),
            "shift_min": int(self.shift.min().item()),
            "shift_max": int(self.shift.max().item()),
        }
        with (output_dir / "export_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


# =========================================================
# convenience factory
# =========================================================

def build_int_qat_and_teacher_from_fp32(
    fp32_model: nn.Module,
    cfg: IntQATConfig,
) -> tuple[DeployPrefilterIntQAT, DeployPrefilterFP32Reference]:
    student = DeployPrefilterIntQAT.from_fp32_prefilter(fp32_model, cfg)
    teacher = build_deploy_fp32_reference(fp32_model)
    return student, teacher
