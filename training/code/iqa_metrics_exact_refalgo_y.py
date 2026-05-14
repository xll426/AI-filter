"""W10+B13 训练验证使用的 selective score。

该文件只保留 best 选择所需的指标计算逻辑。背景区域按接近 reference target
评估，边缘区域按接近原始 source 评估，避免模型通过过度平滑边缘来获得更高
传统 PSNR/SSIM。
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Literal

import cv2
import numpy as np

from ref import compute_structure_score, structure_aware_median


ArrayLike = Any


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def to_y_uint8(x: ArrayLike) -> np.ndarray:
    """将常见 tensor/array 形状转换为单通道 uint8 Y plane。

    支持 HxW、1xHxW、3xHxW、1x1xHxW、1x3xHxW 等格式。float 输入可为
    `[0,1]` 或 `[0,255]`，函数会自动归一到 uint8 像素语义。
    """
    arr = np.asarray(_to_numpy(x))

    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        y = arr
    elif arr.ndim == 3 and arr.shape[2] == 1:
        y = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr_f = arr.astype(np.float32)
        if arr_f.max() <= 1.0 + 1e-6:
            arr_f = arr_f * 255.0
        y = 0.299 * arr_f[..., 0] + 0.587 * arr_f[..., 1] + 0.114 * arr_f[..., 2]
    else:
        raise ValueError(f"Unsupported input shape: {arr.shape}")

    y = y.astype(np.float32)
    if y.max() <= 1.0 + 1e-6:
        y = y * 255.0
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def _ensure_same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return a, b


def _safe_mean(x: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(x[mask]))


def _safe_sum(x: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(x[mask]))


def gradient_magnitude(y_u8: np.ndarray) -> np.ndarray:
    y = y_u8.astype(np.float32)
    gx = cv2.Scharr(y, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(y, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy + 1e-6).astype(np.float32)


def high_frequency_energy(y_u8: np.ndarray) -> np.ndarray:
    y = y_u8.astype(np.float32)
    lap = cv2.Laplacian(y, cv2.CV_32F, ksize=3)
    return np.abs(lap).astype(np.float32)


def gms_map(pred_u8: np.ndarray, ref_u8: np.ndarray, c: float = 170.0) -> np.ndarray:
    gp = gradient_magnitude(pred_u8)
    gr = gradient_magnitude(ref_u8)
    return (2.0 * gp * gr + c) / (gp * gp + gr * gr + c)


def edge_band_from_mask(edge_mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return edge_mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    band = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1)
    return band.astype(bool)


def _detail_gain_structure_score(y_source_u8: np.ndarray) -> np.ndarray:
    y_medfix = structure_aware_median(
        y_source_u8,
        s_thr=0.25,
        outlier_t=30,
        use_extreme_gate=False,
    )
    y_denoise = cv2.fastNlMeansDenoising(y_medfix, None, h=4, templateWindowSize=7, searchWindowSize=15)
    return compute_structure_score(y_denoise)["S"].astype(np.float32)


def exact_masks_from_source(
    y_source: ArrayLike,
    mask_mode: Literal["pre_median", "detail_gain"] = "detail_gain",
    s_thr: float = 0.25,
) -> Dict[str, np.ndarray]:
    """从 source Y 生成背景/边缘任务 mask。

    `pre_median` 使用 median 修正前的结构图；
    `detail_gain` 使用 median 修正 + NLM 后的结构图，与 reference 中
    `detail_gain = S^3` 的结构来源一致。
    """
    y_input = to_y_uint8(y_source)
    if mask_mode == "pre_median":
        s = compute_structure_score(y_input)["S"].astype(np.float32)
    elif mask_mode == "detail_gain":
        s = _detail_gain_structure_score(y_input)
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    bg_mask = s < float(s_thr)
    edge_mask = s >= float(s_thr)
    return {
        "S": s,
        "bg_mask": bg_mask.astype(bool),
        "edge_mask": edge_mask.astype(bool),
    }


def bg_hf_error(pred: ArrayLike, ref: ArrayLike, bg_mask: np.ndarray) -> float:
    pred_y, ref_y = _ensure_same_shape(to_y_uint8(pred), to_y_uint8(ref))
    hp = high_frequency_energy(pred_y)
    hr = high_frequency_energy(ref_y)
    return _safe_mean(np.abs(hp - hr), bg_mask)


def edge_preserve_error(pred: ArrayLike, ref: ArrayLike, edge_mask: np.ndarray) -> float:
    pred_y, ref_y = _ensure_same_shape(to_y_uint8(pred), to_y_uint8(ref))
    gp = gradient_magnitude(pred_y)
    gr = gradient_magnitude(ref_y)
    return _safe_mean(np.abs(gp - gr), edge_mask)


def edge_over_smooth_ratio(
    pred: ArrayLike,
    ref: ArrayLike,
    edge_mask: np.ndarray,
    smooth_ratio: float = 0.9,
    min_ref_grad: float = 3.0,
) -> float:
    pred_y, ref_y = _ensure_same_shape(to_y_uint8(pred), to_y_uint8(ref))
    gp = gradient_magnitude(pred_y)
    gr = gradient_magnitude(ref_y)
    valid = edge_mask.astype(bool) & (gr >= float(min_ref_grad))
    if not np.any(valid):
        return float("nan")
    oversmoothed = gp[valid] < (float(smooth_ratio) * gr[valid])
    return float(np.mean(oversmoothed))


def edge_gmsd(pred: ArrayLike, ref: ArrayLike, edge_mask: np.ndarray, band_radius: int = 1) -> float:
    pred_y, ref_y = _ensure_same_shape(to_y_uint8(pred), to_y_uint8(ref))
    band = edge_band_from_mask(edge_mask, radius=band_radius)
    gms = gms_map(pred_y, ref_y)
    if not np.any(band):
        return float("nan")
    return float(np.std(gms[band]))


def bg_grad_energy_ratio(pred: ArrayLike, ref: ArrayLike, bg_mask: np.ndarray) -> float:
    pred_y, ref_y = _ensure_same_shape(to_y_uint8(pred), to_y_uint8(ref))
    gp = gradient_magnitude(pred_y)
    gr = gradient_magnitude(ref_y)
    num = _safe_sum(gp, bg_mask)
    den = _safe_sum(gr, bg_mask)
    if np.isnan(num) or np.isnan(den):
        return float("nan")
    return float(num / max(den, 1e-6))


def edge_grad_energy_ratio(pred: ArrayLike, ref: ArrayLike, edge_mask: np.ndarray) -> float:
    pred_y, ref_y = _ensure_same_shape(to_y_uint8(pred), to_y_uint8(ref))
    gp = gradient_magnitude(pred_y)
    gr = gradient_magnitude(ref_y)
    num = _safe_sum(gp, edge_mask)
    den = _safe_sum(gr, edge_mask)
    if np.isnan(num) or np.isnan(den):
        return float("nan")
    return float(num / max(den, 1e-6))


def structure_alignment_error(pred: ArrayLike, ref: ArrayLike) -> float:
    pred_y, ref_y = _ensure_same_shape(to_y_uint8(pred), to_y_uint8(ref))
    sp = compute_structure_score(pred_y)["S"].astype(np.float32)
    sr = compute_structure_score(ref_y)["S"].astype(np.float32)
    return float(np.mean(np.abs(sp - sr)))


def evaluate_exact_refalgo_y(
    pred: ArrayLike,
    ref: ArrayLike,
    structure_source: ArrayLike | None = None,
    mask_mode: Literal["pre_median", "detail_gain"] = "detail_gain",
    s_thr: float = 0.25,
    edge_band_radius: int = 1,
) -> Dict[str, float]:
    pred_y = to_y_uint8(pred)
    ref_y = to_y_uint8(ref)
    pred_y, ref_y = _ensure_same_shape(pred_y, ref_y)

    if structure_source is None:
        warnings.warn(
            "structure_source is None, so ref is used as the mask-generation proxy.",
            stacklevel=2,
        )
        structure_source = ref_y

    structure_masks = exact_masks_from_source(structure_source, mask_mode=mask_mode, s_thr=s_thr)
    bg_mask = structure_masks["bg_mask"]
    edge_mask = structure_masks["edge_mask"]
    s_map = structure_masks["S"]

    return {
        "bg_hf_error": bg_hf_error(pred_y, ref_y, bg_mask),
        "edge_preserve_error": edge_preserve_error(pred_y, ref_y, edge_mask),
        "edge_over_smooth_ratio": edge_over_smooth_ratio(pred_y, ref_y, edge_mask),
        "edge_gmsd": edge_gmsd(pred_y, ref_y, edge_mask, band_radius=edge_band_radius),
        "bg_grad_energy_ratio": bg_grad_energy_ratio(pred_y, ref_y, bg_mask),
        "edge_grad_energy_ratio": edge_grad_energy_ratio(pred_y, ref_y, edge_mask),
        "structure_alignment_error": structure_alignment_error(pred_y, ref_y),
        "bg_area_ratio": float(np.mean(bg_mask)),
        "edge_area_ratio": float(np.mean(edge_mask)),
        "structure_mean": float(np.mean(s_map)),
    }


def safe_completion_score(pred_error: float, anchor_error: float, eps: float = 1e-6) -> float:
    return float(1.0 - pred_error / max(anchor_error, eps))


def selective_prefilter_score(
    bg_completion: float,
    edge_source_completion: float,
    edge_retention_ratio: float,
    edge_oversmooth_vs_src: float,
) -> float:
    edge_completion_clipped = max(min(edge_source_completion, 1.0), -1.0)
    edge_retention_clipped = max(min(edge_retention_ratio, 1.2), 0.0) / 1.2
    oversmooth_score = max(0.0, 1.0 - edge_oversmooth_vs_src)
    return float(
        100.0
        * (
            0.45 * bg_completion
            + 0.35 * edge_completion_clipped
            + 0.10 * edge_retention_clipped
            + 0.10 * oversmooth_score
        )
    )


def evaluate_selective_prefilter_y(
    pred: ArrayLike,
    ref: ArrayLike,
    source: ArrayLike,
    mask_mode: Literal["pre_median", "detail_gain"] = "detail_gain",
    s_thr: float = 0.25,
    edge_band_radius: int = 1,
) -> Dict[str, float]:
    pred_y = to_y_uint8(pred)
    ref_y = to_y_uint8(ref)
    source_y = to_y_uint8(source)
    pred_y, ref_y = _ensure_same_shape(pred_y, ref_y)
    pred_y, source_y = _ensure_same_shape(pred_y, source_y)

    masks = exact_masks_from_source(source_y, mask_mode=mask_mode, s_thr=s_thr)
    bg_mask = masks["bg_mask"]
    edge_mask = masks["edge_mask"]

    strict_scores = evaluate_exact_refalgo_y(
        pred_y,
        ref_y,
        structure_source=source_y,
        mask_mode=mask_mode,
        s_thr=s_thr,
        edge_band_radius=edge_band_radius,
    )

    bg_anchor = bg_hf_error(source_y, ref_y, bg_mask)
    edge_anchor = edge_preserve_error(ref_y, source_y, edge_mask)
    edge_source_error = edge_preserve_error(pred_y, source_y, edge_mask)

    selective_scores = {
        "bg_completion": safe_completion_score(strict_scores["bg_hf_error"], bg_anchor),
        "edge_source_error": edge_source_error,
        "edge_source_completion": safe_completion_score(edge_source_error, edge_anchor),
        "edge_retention_ratio": edge_grad_energy_ratio(pred_y, source_y, edge_mask),
        "edge_oversmooth_vs_src": edge_over_smooth_ratio(pred_y, source_y, edge_mask),
    }
    selective_scores["selective_score"] = selective_prefilter_score(
        bg_completion=selective_scores["bg_completion"],
        edge_source_completion=selective_scores["edge_source_completion"],
        edge_retention_ratio=selective_scores["edge_retention_ratio"],
        edge_oversmooth_vs_src=selective_scores["edge_oversmooth_vs_src"],
    )
    return {**strict_scores, **selective_scores}
