from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import cv2
import numpy as np


ArrayLike = Any


# =========================
# Exact reference-algorithm building blocks
# =========================

def native_ximgproc_available() -> bool:
    return bool(
        hasattr(cv2, "ximgproc")
        and hasattr(cv2.ximgproc, "l0Smooth")
        and hasattr(cv2.ximgproc, "guidedFilter")
    )


def psf2otf(psf: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    psf = np.asarray(psf, dtype=np.float32)
    pad = np.zeros(out_size, dtype=np.float32)
    pad[: psf.shape[0], : psf.shape[1]] = psf
    for axis, size in enumerate(psf.shape):
        pad = np.roll(pad, -int(size / 2), axis=axis)
    return np.fft.fft2(pad)


def l0_smooth_gray(image_u8: np.ndarray, lambda_: float = 0.005, kappa: float = 2.0, beta_max: float = 1e5) -> np.ndarray:
    s = image_u8.astype(np.float32) / 255.0
    h, w = s.shape

    otf_fx = psf2otf(np.array([[1, -1]], dtype=np.float32), (h, w))
    otf_fy = psf2otf(np.array([[1], [-1]], dtype=np.float32), (h, w))
    denormin2 = np.abs(otf_fx) ** 2 + np.abs(otf_fy) ** 2
    normin1 = np.fft.fft2(s)

    beta = 2.0 * lambda_
    while beta < beta_max:
        h_grad = np.concatenate([np.diff(s, axis=1), s[:, :1] - s[:, -1:]], axis=1)
        v_grad = np.concatenate([np.diff(s, axis=0), s[:1, :] - s[-1:, :]], axis=0)

        mask = (h_grad ** 2 + v_grad ** 2) < (lambda_ / beta)
        h_grad[mask] = 0
        v_grad[mask] = 0

        normin2 = np.zeros_like(s)
        normin2[:, :1] = h_grad[:, -1:] - h_grad[:, :1]
        normin2[:, 1:] += -np.diff(h_grad, axis=1)
        normin2[:1, :] += v_grad[-1:, :] - v_grad[:1, :]
        normin2[1:, :] += -np.diff(v_grad, axis=0)

        fs = (normin1 + beta * np.fft.fft2(normin2)) / (1.0 + beta * denormin2)
        s = np.real(np.fft.ifft2(fs)).astype(np.float32)
        beta *= kappa

    return np.clip(s * 255.0, 0, 255).astype(np.uint8)


def guided_filter_gray(guide_u8: np.ndarray, src_u8: np.ndarray, radius: int = 1, eps: float = 50.0 * 50.0) -> np.ndarray:
    guide = guide_u8.astype(np.float32)
    src = src_u8.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_i = cv2.boxFilter(guide, -1, ksize, borderType=cv2.BORDER_REFLECT, normalize=True)
    mean_p = cv2.boxFilter(src, -1, ksize, borderType=cv2.BORDER_REFLECT, normalize=True)
    corr_i = cv2.boxFilter(guide * guide, -1, ksize, borderType=cv2.BORDER_REFLECT, normalize=True)
    corr_ip = cv2.boxFilter(guide * src, -1, ksize, borderType=cv2.BORDER_REFLECT, normalize=True)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, -1, ksize, borderType=cv2.BORDER_REFLECT, normalize=True)
    mean_b = cv2.boxFilter(b, -1, ksize, borderType=cv2.BORDER_REFLECT, normalize=True)
    q = mean_a * guide + mean_b
    return np.clip(q, 0, 255).astype(np.float32)


def smooth_base_layer(y_denoise_u8: np.ndarray) -> np.ndarray:
    if native_ximgproc_available():
        filtered = cv2.ximgproc.l0Smooth(y_denoise_u8, lambda_=0.005, kappa=2.0)
        return cv2.ximgproc.guidedFilter(
            guide=filtered,
            src=y_denoise_u8.astype(np.float32),
            radius=1,
            eps=50.0 * 50.0,
        ).astype(np.float32)

    filtered = l0_smooth_gray(y_denoise_u8, lambda_=0.005, kappa=2.0)
    return guided_filter_gray(filtered, y_denoise_u8, radius=1, eps=50.0 * 50.0)


def match_local_mean_var(
    y_orig_u8: np.ndarray,
    y_ref_u8: np.ndarray,
    ks: int = 31,
    gain_clip: Tuple[float, float] = (0.85, 1.18),
) -> np.ndarray:
    y0 = y_orig_u8.astype(np.float32)
    y1 = y_ref_u8.astype(np.float32)
    if ks % 2 == 0:
        ks += 1

    mu0 = cv2.GaussianBlur(y0, (ks, ks), 0)
    mu1 = cv2.GaussianBlur(y1, (ks, ks), 0)
    v0 = cv2.GaussianBlur(y0 * y0, (ks, ks), 0) - mu0 * mu0
    v1 = cv2.GaussianBlur(y1 * y1, (ks, ks), 0) - mu1 * mu1
    s0 = np.sqrt(np.maximum(v0, 1e-6))
    s1 = np.sqrt(np.maximum(v1, 1e-6))
    gain = np.clip(s0 / (s1 + 1e-6), gain_clip[0], gain_clip[1])
    out = (y1 - mu1) * gain + mu0
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_structure_score(
    y_uint8: np.ndarray,
    wins: Tuple[int, int] = (5, 9),
    eps: float = 1e-6,
    gamma: float = 2.0,
    q: float = 1.0,
    alpha: float = 1.2,
    gate_p1: float = 60.0,
    gate_p2: float = 90.0,
) -> Dict[str, np.ndarray]:
    """
    Exact structure-score implementation adapted from the provided reference algorithm.
    Returns:
      - S: final multi-scale structure score in [0,1]
      - mag: gradient-magnitude proxy selected at the winning scale
    """
    y = y_uint8.astype(np.float32)
    gx = cv2.Scharr(y, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(y, cv2.CV_32F, 0, 1)
    gx2, gy2, gxy = gx * gx, gy * gy, gx * gy

    s_all = None
    mag_all = None
    for win in wins:
        j11 = cv2.boxFilter(gx2, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)
        j22 = cv2.boxFilter(gy2, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)
        j12 = cv2.boxFilter(gxy, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)

        tr = j11 + j22
        det_term = np.sqrt((j11 - j22) ** 2 + 4.0 * (j12 ** 2) + eps)
        lam1 = 0.5 * (tr + det_term)
        lam2 = 0.5 * (tr - det_term)

        coherence = np.clip((lam1 - lam2) / (lam1 + lam2 + eps), 0.0, 1.0)
        corner_ratio = np.clip(lam2 / (lam1 + lam2 + eps), 0.0, 1.0)
        mag = np.sqrt(tr + eps)

        t = np.percentile(mag, gate_p1)
        t2 = np.percentile(mag, gate_p2)
        gate = np.clip((mag - t) / max(t2 - t, 1e-6), 0.0, 1.0)

        s = np.clip(gate * np.maximum(coherence ** gamma, alpha * (corner_ratio ** q)), 0.0, 1.0).astype(np.float32)
        if s_all is None:
            s_all = s
            mag_all = mag.astype(np.float32)
        else:
            mask = s > s_all
            s_all = np.maximum(s_all, s)
            mag_all = np.where(mask, mag, mag_all).astype(np.float32)

    return {"S": s_all, "mag": mag_all}


def structure_aware_median(
    y_uint8: np.ndarray,
    ksize: int = 3,
    s_thr: float = 0.25,
    outlier_t: int = 30,
    use_extreme_gate: bool = False,
    extreme_low: int = 8,
    extreme_high: int = 247,
) -> np.ndarray:
    out = compute_structure_score(y_uint8)
    s = out["S"]
    med = cv2.medianBlur(y_uint8, ksize)
    diff = cv2.absdiff(y_uint8, med)
    mask = (s < s_thr) & (diff.astype(np.int16) > outlier_t)
    if use_extreme_gate:
        extreme = (y_uint8 <= extreme_low) | (y_uint8 >= extreme_high)
        mask &= extreme
    result = y_uint8.copy()
    result[mask] = med[mask]
    return result


# =========================
# IO / conversion helpers
# =========================

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
    """
    Strict Y-only conversion.
    Accepted shapes:
      HxW, HxWx1, HxWx3(BGR/RGB treated as gray via cvtColor),
      1xHxW, 3xHxW, 1x1xHxW, 1x3xHxW.
    Assumes uint8 in [0,255] or float in [0,255]/[0,1].
    """
    arr = _to_numpy(x)
    arr = np.asarray(arr)

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
        # RGB/BGR ambiguity exists for arbitrary arrays; for Y-only metrics the exact
        # color conversion is usually secondary. We use standard RGB weights.
        y = 0.299 * arr_f[..., 0] + 0.587 * arr_f[..., 1] + 0.114 * arr_f[..., 2]
    else:
        raise ValueError(f"Unsupported input shape: {arr.shape}")

    y = y.astype(np.float32)
    if y.max() <= 1.0 + 1e-6:
        y = y * 255.0
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


# =========================
# Exact reference-pipeline helpers for masks / structure source
# =========================

@dataclass
class ReferenceProxy:
    y_input: np.ndarray
    y_medfix: np.ndarray
    y_denoise: np.ndarray
    structure_S: np.ndarray
    structure_mag: np.ndarray
    base: np.ndarray
    y_ref_before_match: np.ndarray
    y_ref_after_match: np.ndarray


def build_reference_proxy_from_source(y_source: ArrayLike) -> ReferenceProxy:
    """
    Replays the provided reference algorithm on a Y image.
    This is the exact sequence from the shared code, except without torch padding.
    """
    y_input = to_y_uint8(y_source)
    y_medfix = structure_aware_median(
        y_input,
        s_thr=0.25,
        outlier_t=30,
        use_extreme_gate=False,
    )
    y_denoise = cv2.fastNlMeansDenoising(y_medfix, None, h=4, templateWindowSize=7, searchWindowSize=15)

    structure = compute_structure_score(y_denoise)
    s = structure["S"].astype(np.float32)
    base = smooth_base_layer(y_denoise)

    detail = y_denoise.astype(np.float32) - base
    detail_gain = np.clip(np.power(s, 3.0), 0.0, 1.0)
    y_ref = base + detail_gain * detail
    y_ref_u8 = np.clip(y_ref, 0, 255).astype(np.uint8)
    y_out = match_local_mean_var(y_input, y_ref_u8)

    return ReferenceProxy(
        y_input=y_input,
        y_medfix=y_medfix,
        y_denoise=y_denoise,
        structure_S=structure["S"].astype(np.float32),
        structure_mag=structure["mag"].astype(np.float32),
        base=base.astype(np.float32),
        y_ref_before_match=y_ref_u8,
        y_ref_after_match=y_out,
    )


# =========================
# Metric helpers
# =========================

def _ensure_same_shape(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def exact_masks_from_source(
    y_source: ArrayLike,
    mask_mode: Literal["pre_median", "detail_gain"] = "detail_gain",
    s_thr: float = 0.25,
) -> Dict[str, np.ndarray]:
    """
    Exact mask generation aligned with the shared reference code.

    mask_mode='pre_median':
        Uses compute_structure_score(y_input)['S'] and threshold 0.25,
        which is exactly what structure_aware_median uses.

    mask_mode='detail_gain':
        Replays the exact preprocessing y_input -> structure_aware_median -> NLM,
        then computes S on y_denoise, which is exactly what the reference code uses
        for detail_gain = S^3 in the detail reinjection stage.
    """
    y_input = to_y_uint8(y_source)
    if mask_mode == "pre_median":
        s = compute_structure_score(y_input)["S"].astype(np.float32)
    elif mask_mode == "detail_gain":
        proxy = build_reference_proxy_from_source(y_input)
        s = proxy.structure_S.astype(np.float32)
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    bg_mask = s < float(s_thr)
    edge_mask = s >= float(s_thr)
    return {
        "S": s,
        "bg_mask": bg_mask.astype(bool),
        "edge_mask": edge_mask.astype(bool),
    }


# =========================
# Exact Y-only task metrics
# =========================

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
    structure_source: Optional[ArrayLike] = None,
    mask_mode: Literal["pre_median", "detail_gain"] = "detail_gain",
    s_thr: float = 0.25,
    edge_band_radius: int = 1,
) -> Dict[str, float]:
    """
    Evaluate a model output against the reference output using masks derived from the
    actual shared reference algorithm.

    Args:
        pred: model output Y / grayscale image.
        ref: reference output Y / grayscale image.
        structure_source:
            Preferred: original input Y before filtering.
            If None, this function uses `ref` as a proxy source so the interface can still be
            called with just (pred, ref). That is convenient, but not as exact as providing the
            original input Y.
        mask_mode:
            'pre_median'  -> masks consistent with structure_aware_median stage.
            'detail_gain' -> masks consistent with detail_gain = S^3 stage.
    """
    pred_y = to_y_uint8(pred)
    ref_y = to_y_uint8(ref)
    pred_y, ref_y = _ensure_same_shape(pred_y, ref_y)

    if structure_source is None:
        warnings.warn(
            "structure_source is None, so ref is used as the mask-generation proxy. "
            "For the most exact masks, pass the original input Y as structure_source.",
            stacklevel=2,
        )
        structure_source = ref_y

    structure_masks = exact_masks_from_source(structure_source, mask_mode=mask_mode, s_thr=s_thr)
    bg_mask = structure_masks["bg_mask"]
    edge_mask = structure_masks["edge_mask"]
    s_map = structure_masks["S"]

    scores: Dict[str, float] = {
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
    return scores


def safe_completion_score(pred_error: float, anchor_error: float, eps: float = 1e-6) -> float:
    return float(1.0 - pred_error / max(anchor_error, eps))


def selective_prefilter_score(
    bg_completion: float,
    edge_source_completion: float,
    edge_retention_ratio: float,
    edge_oversmooth_vs_src: float,
) -> float:
    """
    Task-oriented aggregate score.
    Larger is better.

    Design intent:
    - background should approach ref output
    - important edges should stay close to original input
    """
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
    """
    Evaluate according to the actual task objective instead of pure ref imitation.

    Background-like regions:
      compare pred against ref
    Edge / structure regions:
      compare pred against original source
    """
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


def composite_prefilter_score(scores: Dict[str, float]) -> float:
    """
    Heuristic aggregate score.
    Larger is better.
    This is not part of the original reference algorithm; it is a convenience wrapper.
    """
    # Prefer lower errors, prefer energy ratios near 1.0
    bg_ratio_pen = abs(scores.get("bg_grad_energy_ratio", np.nan) - 1.0)
    edge_ratio_pen = abs(scores.get("edge_grad_energy_ratio", np.nan) - 1.0)

    return float(
        -1.25 * scores.get("bg_hf_error", 0.0)
        -1.25 * scores.get("edge_preserve_error", 0.0)
        -0.80 * scores.get("edge_over_smooth_ratio", 0.0)
        -0.80 * scores.get("edge_gmsd", 0.0)
        -1.00 * scores.get("structure_alignment_error", 0.0)
        -0.50 * bg_ratio_pen
        -0.50 * edge_ratio_pen
    )


__all__ = [
    "compute_structure_score",
    "structure_aware_median",
    "smooth_base_layer",
    "match_local_mean_var",
    "build_reference_proxy_from_source",
    "exact_masks_from_source",
    "bg_hf_error",
    "edge_preserve_error",
    "edge_over_smooth_ratio",
    "edge_gmsd",
    "bg_grad_energy_ratio",
    "edge_grad_energy_ratio",
    "structure_alignment_error",
    "evaluate_exact_refalgo_y",
    "safe_completion_score",
    "selective_prefilter_score",
    "evaluate_selective_prefilter_y",
    "composite_prefilter_score",
    "to_y_uint8",
]
