from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


class CharbonnierLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        eps: float = 1e-12,
        roi_weight: float = 1.0,
        non_roi_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        self.roi_weight = float(roi_weight)
        self.non_roi_weight = float(non_roi_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = torch.sqrt((pred - target) ** 2 + self.eps)
        if roi is not None:
            roi_mask = (roi > 0).to(dtype=loss.dtype, device=loss.device)
            weight = roi_mask * self.roi_weight + (1.0 - roi_mask) * self.non_roi_weight
            loss = loss * weight
        return self.loss_weight * loss.mean()


class MsssimLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        data_range: float = 1.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.data_range = float(data_range)
        self.win_size = int(win_size)
        self.win_sigma = float(win_sigma)
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        score = ms_ssim(
            pred,
            target,
            data_range=self.data_range,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
            weights=self.weights,
            size_average=True,
        )
        return self.loss_weight * (1.0 - score)


class EdgeConsistencyLoss(nn.Module):
    """
    Small auxiliary loss for preserving important edges from the source image
    while still training towards the filtered target.

    The edge weighting mask is derived from the target/ref image so the model
    focuses on edges that the reference algorithm also considers important.
    """

    def __init__(
        self,
        loss_weight: float = 0.05,
        match_weight: float = 1.0,
        retain_weight: float = 0.25,
        retain_ratio: float = 0.90,
        mask_quantile: float = 0.90,
        mask_gamma: float = 1.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.match_weight = float(match_weight)
        self.retain_weight = float(retain_weight)
        self.retain_ratio = float(retain_ratio)
        self.mask_quantile = float(mask_quantile)
        self.mask_gamma = float(mask_gamma)
        self.eps = float(eps)

        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def _to_y(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW tensor, got {tuple(x.shape)}")
        return x[:, :1]

    def _gradients(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        return gx, gy, mag

    def _edge_mask(self, guide_mag: torch.Tensor) -> torch.Tensor:
        flat = guide_mag.flatten(1)
        q = torch.quantile(flat, self.mask_quantile, dim=1, keepdim=True)
        q = q.view(-1, 1, 1, 1)
        mask = torch.clamp(guide_mag / (q + self.eps), 0.0, 1.0)
        return mask.pow(self.mask_gamma)

    def forward(self, pred: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_y = self._to_y(pred)
        src_y = self._to_y(source)
        tgt_y = self._to_y(target)

        pred_gx, pred_gy, pred_mag = self._gradients(pred_y)
        src_gx, src_gy, src_mag = self._gradients(src_y)
        _, _, tgt_mag = self._gradients(tgt_y)

        edge_mask = self._edge_mask(tgt_mag).detach()
        mask_norm = edge_mask.sum().clamp_min(self.eps)

        grad_vec_diff = torch.sqrt((pred_gx - src_gx) ** 2 + (pred_gy - src_gy) ** 2 + self.eps)
        grad_match = (grad_vec_diff * edge_mask).sum() / mask_norm

        retain_gap = torch.relu(self.retain_ratio * src_mag - pred_mag)
        retain_penalty = (retain_gap * edge_mask).sum() / mask_norm

        return self.loss_weight * (self.match_weight * grad_match + self.retain_weight * retain_penalty)


def softclip01(x: torch.Tensor, k: float = 2.0) -> torch.Tensor:
    if not torch.is_floating_point(x):
        x = x.float()
    k_tensor = torch.as_tensor(k, dtype=x.dtype, device=x.device)
    return 0.5 * (torch.tanh(k_tensor * (x - 0.5)) + 1.0)
