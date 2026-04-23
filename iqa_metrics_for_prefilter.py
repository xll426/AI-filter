from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import cv2

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("This module requires PyTorch. Please install torch first.") from e

ArrayLike = Union[np.ndarray, torch.Tensor]


# =========================
# Basic conversions
# =========================

def _np_from_tensor(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _squeeze_batch(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported for numpy input, got shape {x.shape}")
        x = x[0]
    return x


def _guess_channel_first(x: np.ndarray) -> bool:
    if x.ndim != 3:
        return False
    return x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3)


def _to_numpy_image(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = _np_from_tensor(x)
    elif not isinstance(x, np.ndarray):
        raise TypeError(f"Unsupported input type: {type(x)}")

    x = _squeeze_batch(x)

    if x.ndim == 2:
        return x

    if x.ndim != 3:
        raise ValueError(f"Expected 2D/3D image or 4D batch, got shape {x.shape}")

    if _guess_channel_first(x):
        x = np.transpose(x, (1, 2, 0))

    if x.shape[2] not in (1, 3):
        raise ValueError(f"Expected channel dim 1 or 3, got shape {x.shape}")

    return x


def _normalize_to_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.max() > 1.0 or x.min() < 0.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def to_rgb01(x: ArrayLike) -> np.ndarray:
    """
    Convert input to HxWx3 RGB float32 in [0, 1].

    Supported input shapes:
    - HxW
    - HxWx1 / HxWx3
    - 1xHxW / 3xHxW
    - 1x1xHxW / 1x3xHxW
    """
    arr = _to_numpy_image(x)
    arr = _normalize_to_01(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr.astype(np.float32)


def to_gray01(x: ArrayLike) -> np.ndarray:
    """
    Convert input to HxW grayscale float32 in [0, 1].
    RGB is converted with BT.601 luma weights.
    """
    arr = _to_numpy_image(x)
    arr = _normalize_to_01(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.shape[2] == 1:
        return arr[..., 0].astype(np.float32)
    # RGB -> Y
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return y.astype(np.float32)


def to_torch_nchw01(x: ArrayLike, device: Optional[torch.device] = None) -> torch.Tensor:
    arr = to_rgb01(x)
    ten = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).float()
    if device is not None:
        ten = ten.to(device)
    return ten


# =========================
# Gradient / structure tools
# =========================

def _grad_mag(gray01: np.ndarray, method: str = "scharr") -> np.ndarray:
    if method.lower() == "scharr":
        gx = cv2.Scharr(gray01, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray01, cv2.CV_32F, 0, 1)
    elif method.lower() == "sobel":
        gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    else:
        raise ValueError("method must be 'scharr' or 'sobel'")
    mag = np.sqrt(gx * gx + gy * gy)
    return mag.astype(np.float32)


def compute_structure_map(
    ref: ArrayLike,
    grad_method: str = "scharr",
    sigma: float = 1.2,
) -> np.ndarray:
    """
    Lightweight structure map in [0, 1].
    This is not your exact traditional algorithm implementation,
    but it is a stable mask generator for evaluation.
    """
    y = to_gray01(ref)
    g = _grad_mag(y, method=grad_method)
    s = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
    p99 = float(np.percentile(s, 99.0))
    if p99 < 1e-8:
        return np.zeros_like(s, dtype=np.float32)
    s = np.clip(s / p99, 0.0, 1.0)
    return s.astype(np.float32)


def build_region_masks(
    ref: ArrayLike,
    structure_map: Optional[np.ndarray] = None,
    low_thr: Optional[float] = None,
    high_thr: Optional[float] = None,
    low_quantile: float = 0.35,
    high_quantile: float = 0.85,
    edge_dilate: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
    - structure_map: HxW float32 in [0,1]
    - bg_mask: HxW bool, low-structure background-like region
    - edge_mask: HxW bool, high-structure edge/foreground-like region
    """
    s = compute_structure_map(ref) if structure_map is None else structure_map.astype(np.float32)
    if low_thr is None:
        low_thr = float(np.quantile(s, low_quantile))
    if high_thr is None:
        high_thr = float(np.quantile(s, high_quantile))

    bg_mask = s <= low_thr
    edge_mask = s >= high_thr

    if edge_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_dilate + 1, 2 * edge_dilate + 1))
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), k, iterations=1).astype(bool)

    # avoid overlap
    bg_mask = np.logical_and(bg_mask, np.logical_not(edge_mask))
    return s, bg_mask, edge_mask


# =========================
# Task-specific metrics
# =========================

def _masked_mean(x: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    mask_f = mask.astype(np.float32)
    denom = float(mask_f.sum())
    if denom < eps:
        return float("nan")
    return float((x * mask_f).sum() / denom)


def _masked_std(x: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    mask_f = mask.astype(np.float32)
    denom = float(mask_f.sum())
    if denom < eps:
        return float("nan")
    mean = float((x * mask_f).sum() / denom)
    var = float((((x - mean) ** 2) * mask_f).sum() / denom)
    return float(np.sqrt(max(var, 0.0)))


def _hf_map(gray01: np.ndarray, method: str = "laplacian") -> np.ndarray:
    if method.lower() == "laplacian":
        hf = np.abs(cv2.Laplacian(gray01, cv2.CV_32F, ksize=3))
    elif method.lower() in ("scharr", "sobel"):
        hf = _grad_mag(gray01, method=method.lower())
    else:
        raise ValueError("hf method must be 'laplacian', 'scharr', or 'sobel'")
    return hf.astype(np.float32)


def calc_bg_hf_error(
    pred: ArrayLike,
    ref: ArrayLike,
    bg_mask: Optional[np.ndarray] = None,
    structure_map: Optional[np.ndarray] = None,
    hf_method: str = "laplacian",
) -> float:
    """
    Background High-Frequency Error.
    Lower is better.
    Measures whether low-structure regions are filtered enough.
    """
    pred_y = to_gray01(pred)
    ref_y = to_gray01(ref)
    if bg_mask is None:
        _, bg_mask, _ = build_region_masks(ref_y, structure_map=structure_map)
    hf_pred = _hf_map(pred_y, method=hf_method)
    hf_ref = _hf_map(ref_y, method=hf_method)
    err = np.abs(hf_pred - hf_ref)
    return _masked_mean(err, bg_mask)


def calc_edge_preserve_error(
    pred: ArrayLike,
    ref: ArrayLike,
    edge_mask: Optional[np.ndarray] = None,
    structure_map: Optional[np.ndarray] = None,
    grad_method: str = "scharr",
) -> float:
    """
    Edge Preserve Error.
    Lower is better.
    Measures whether important edges are preserved.
    """
    pred_y = to_gray01(pred)
    ref_y = to_gray01(ref)
    if edge_mask is None:
        _, _, edge_mask = build_region_masks(ref_y, structure_map=structure_map)
    g_pred = _grad_mag(pred_y, method=grad_method)
    g_ref = _grad_mag(ref_y, method=grad_method)
    err = np.abs(g_pred - g_ref)
    return _masked_mean(err, edge_mask)


def calc_edge_over_smooth_ratio(
    pred: ArrayLike,
    ref: ArrayLike,
    edge_mask: Optional[np.ndarray] = None,
    structure_map: Optional[np.ndarray] = None,
    grad_method: str = "scharr",
    eps: float = 1e-8,
) -> float:
    """
    Edge Over-Smooth Ratio.
    Lower is better.
    >0 means edge energy of pred is weaker than ref.
    """
    pred_y = to_gray01(pred)
    ref_y = to_gray01(ref)
    if edge_mask is None:
        _, _, edge_mask = build_region_masks(ref_y, structure_map=structure_map)
    g_pred = _grad_mag(pred_y, method=grad_method)
    g_ref = _grad_mag(ref_y, method=grad_method)
    mean_pred = _masked_mean(g_pred, edge_mask)
    mean_ref = _masked_mean(g_ref, edge_mask)
    if np.isnan(mean_pred) or np.isnan(mean_ref):
        return float("nan")
    return float(max(mean_ref - mean_pred, 0.0) / (mean_ref + eps))


def calc_edge_gmsd(
    pred: ArrayLike,
    ref: ArrayLike,
    edge_mask: Optional[np.ndarray] = None,
    structure_map: Optional[np.ndarray] = None,
    grad_method: str = "scharr",
    c: float = 0.0026,
) -> float:
    """
    Edge-region GMSD-like metric.
    Lower is better.

    Uses gradient magnitude similarity map restricted to edge_mask,
    then computes the standard deviation of similarity values.
    """
    pred_y = to_gray01(pred)
    ref_y = to_gray01(ref)
    if edge_mask is None:
        _, _, edge_mask = build_region_masks(ref_y, structure_map=structure_map)
    g_pred = _grad_mag(pred_y, method=grad_method)
    g_ref = _grad_mag(ref_y, method=grad_method)
    gms = (2.0 * g_pred * g_ref + c) / (g_pred * g_pred + g_ref * g_ref + c)
    # Original GMSD uses std over the GMS map; here we restrict to edge region.
    return _masked_std(gms, edge_mask)


def calc_region_energy_ratio(
    pred: ArrayLike,
    ref: ArrayLike,
    bg_mask: Optional[np.ndarray] = None,
    edge_mask: Optional[np.ndarray] = None,
    structure_map: Optional[np.ndarray] = None,
    grad_method: str = "scharr",
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Returns background and edge gradient energy ratios.
    Closer to 1 is better.
    """
    pred_y = to_gray01(pred)
    ref_y = to_gray01(ref)
    if bg_mask is None or edge_mask is None:
        _, bg_mask2, edge_mask2 = build_region_masks(ref_y, structure_map=structure_map)
        bg_mask = bg_mask if bg_mask is not None else bg_mask2
        edge_mask = edge_mask if edge_mask is not None else edge_mask2

    gp = _grad_mag(pred_y, method=grad_method)
    gr = _grad_mag(ref_y, method=grad_method)

    bg_ratio = (_masked_mean(gp, bg_mask) + eps) / (_masked_mean(gr, bg_mask) + eps)
    edge_ratio = (_masked_mean(gp, edge_mask) + eps) / (_masked_mean(gr, edge_mask) + eps)
    return {
        "bg_grad_energy_ratio": float(bg_ratio),
        "edge_grad_energy_ratio": float(edge_ratio),
    }


# =========================
# Library-backed FR-IQA wrappers
# =========================

def calc_pyiqa_metric(
    pred: ArrayLike,
    ref: ArrayLike,
    metric_name: str,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    Generic wrapper for pyiqa full-reference metrics.

    Examples:
        calc_pyiqa_metric(pred, ref, 'dists')
        calc_pyiqa_metric(pred, ref, 'lpips')
        calc_pyiqa_metric(pred, ref, 'fsim')
        calc_pyiqa_metric(pred, ref, 'gmsd')
        calc_pyiqa_metric(pred, ref, 'vif')
    """
    try:
        import pyiqa
    except ImportError as e:  # pragma: no cover
        raise ImportError("Please install pyiqa first: pip install pyiqa") from e

    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = to_torch_nchw01(pred, device=device)
    y = to_torch_nchw01(ref, device=device)

    metric = pyiqa.create_metric(metric_name, device=device)
    with torch.no_grad():
        score = metric(x, y)
    return float(score.reshape(-1)[0].detach().cpu().item())


def calc_lpips_official(
    pred: ArrayLike,
    ref: ArrayLike,
    net: str = "alex",
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    Official LPIPS wrapper using the `lpips` package.

    net: 'alex', 'vgg', or 'squeeze'
    Lower is better.
    """
    try:
        import lpips
    except ImportError as e:  # pragma: no cover
        raise ImportError("Please install lpips first: pip install lpips") from e

    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = to_torch_nchw01(pred, device=device) * 2.0 - 1.0
    y = to_torch_nchw01(ref, device=device) * 2.0 - 1.0

    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()
    with torch.no_grad():
        score = loss_fn(x, y)
    return float(score.reshape(-1)[0].detach().cpu().item())


def calc_piqa_metric(
    pred: ArrayLike,
    ref: ArrayLike,
    metric_name: str,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    Wrapper for piqa metrics.

    Supported here:
        'haarpsi', 'gmsd', 'fsim', 'vsi', 'ssim', 'ms_ssim', 'lpips'

    Note:
        - For metrics whose objective is max (e.g. HaarPSI/FSIM), larger is better.
        - For metrics whose objective is min (e.g. GMSD/LPIPS), smaller is better.
    """
    try:
        import piqa
    except ImportError as e:  # pragma: no cover
        raise ImportError("Please install piqa first: pip install piqa") from e

    name = metric_name.lower()
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = to_torch_nchw01(pred, device=device)
    y = to_torch_nchw01(ref, device=device)

    metric_map = {
        "haarpsi": piqa.HaarPSI,
        "gmsd": piqa.GMSD,
        "fsim": piqa.FSIM,
        "vsi": piqa.VSI,
        "ssim": piqa.SSIM,
        "ms_ssim": piqa.MS_SSIM,
        "lpips": piqa.LPIPS,
    }
    if name not in metric_map:
        raise ValueError(f"Unsupported piqa metric '{metric_name}'. Supported: {list(metric_map.keys())}")

    metric = metric_map[name]().to(device)
    metric.eval()
    with torch.no_grad():
        score = metric(x, y)
    return float(score.reshape(-1)[0].detach().cpu().item())


# =========================
# All-in-one evaluator
# =========================

@dataclass
class EvalConfig:
    use_pyiqa_dists: bool = True
    use_pyiqa_fsim: bool = True
    use_pyiqa_vif: bool = True
    use_pyiqa_gmsd: bool = True
    use_piqa_haarpsi: bool = True
    use_official_lpips: bool = True
    lpips_net: str = "alex"
    device: Optional[Union[str, torch.device]] = None
    hf_method: str = "laplacian"
    grad_method: str = "scharr"


def evaluate_prefilter_metrics(
    pred: ArrayLike,
    ref: ArrayLike,
    structure_map: Optional[np.ndarray] = None,
    bg_mask: Optional[np.ndarray] = None,
    edge_mask: Optional[np.ndarray] = None,
    config: Optional[EvalConfig] = None,
) -> Dict[str, float]:
    cfg = config or EvalConfig()
    scores: Dict[str, float] = {}

    # region masks
    if bg_mask is None or edge_mask is None:
        structure_map, bg_mask2, edge_mask2 = build_region_masks(ref, structure_map=structure_map)
        bg_mask = bg_mask if bg_mask is not None else bg_mask2
        edge_mask = edge_mask if edge_mask is not None else edge_mask2

    # task-specific metrics
    scores["bg_hf_error"] = calc_bg_hf_error(pred, ref, bg_mask=bg_mask, structure_map=structure_map, hf_method=cfg.hf_method)
    scores["edge_preserve_error"] = calc_edge_preserve_error(pred, ref, edge_mask=edge_mask, structure_map=structure_map, grad_method=cfg.grad_method)
    scores["edge_over_smooth_ratio"] = calc_edge_over_smooth_ratio(pred, ref, edge_mask=edge_mask, structure_map=structure_map, grad_method=cfg.grad_method)
    scores["edge_gmsd"] = calc_edge_gmsd(pred, ref, edge_mask=edge_mask, structure_map=structure_map, grad_method=cfg.grad_method)
    scores.update(calc_region_energy_ratio(pred, ref, bg_mask=bg_mask, edge_mask=edge_mask, structure_map=structure_map, grad_method=cfg.grad_method))

    # library-backed metrics; if dependency is missing, skip gracefully
    if cfg.use_pyiqa_dists:
        try:
            scores["dists"] = calc_pyiqa_metric(pred, ref, "dists", device=cfg.device)
        except Exception:
            pass
    if cfg.use_pyiqa_fsim:
        try:
            scores["fsim"] = calc_pyiqa_metric(pred, ref, "fsim", device=cfg.device)
        except Exception:
            pass
    if cfg.use_pyiqa_vif:
        try:
            scores["vif"] = calc_pyiqa_metric(pred, ref, "vif", device=cfg.device)
        except Exception:
            pass
    if cfg.use_pyiqa_gmsd:
        try:
            scores["gmsd"] = calc_pyiqa_metric(pred, ref, "gmsd", device=cfg.device)
        except Exception:
            pass
    if cfg.use_piqa_haarpsi:
        try:
            scores["haarpsi"] = calc_piqa_metric(pred, ref, "haarpsi", device=cfg.device)
        except Exception:
            pass
    if cfg.use_official_lpips:
        try:
            scores["lpips_official"] = calc_lpips_official(pred, ref, net=cfg.lpips_net, device=cfg.device)
        except Exception:
            pass

    return scores


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # Example with uint8 HxWx3 numpy arrays
    pred = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    ref = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    scores = evaluate_prefilter_metrics(pred, ref)
    for k, v in scores.items():
        print(f"{k}: {v:.6f}")
