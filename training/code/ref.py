"""参考算法 2：用于生成训练监督目标 `gt`。

该算法执行结构感知滤波：平坦背景区域向平滑结果靠近，边缘和细节区域
按结构强度回填细节分量，以保留真实边界。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


def native_ximgproc_available() -> bool:
    return bool(
        hasattr(cv2, "ximgproc")
        and hasattr(cv2.ximgproc, "l0Smooth")
        and hasattr(cv2.ximgproc, "guidedFilter")
    )


def psf2otf(psf: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
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


def match_local_mean_var(y_orig_u8: np.ndarray, y_ref_u8: np.ndarray, ks: int = 31, gain_clip: tuple[float, float] = (0.85, 1.18)) -> np.ndarray:
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
    wins: tuple[int, int] = (5, 9),
    eps: float = 1e-6,
    gamma: float = 2.0,
    q: float = 1.0,
    alpha: float = 1.2,
    gate_p1: float = 60.0,
    gate_p2: float = 90.0,
) -> dict[str, np.ndarray]:
    """基于 Scharr 梯度和局部结构张量计算结构强度图 `S`。

    `S` 接近 0 表示平坦/背景类区域；`S` 接近 1 表示方向一致的边缘或角点结构。
    后续 reference 生成和 selective score 都使用同一类结构概念区分背景与边缘。
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


def generate_reference_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
    """从一个 raw 像素域 CHW 输入张量生成对应的参考监督目标。

    模型只训练 Y，因此此处仅修改 Y；UV 分量原样透传。函数内部先做 reflect
    padding，以降低边界滤波伪影，输出前再裁回原始尺寸。
    """
    if img_tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(img_tensor.shape)}")

    pad_size = 16
    padded = torch.nn.functional.pad(
        img_tensor.detach().cpu(),
        (pad_size, pad_size, pad_size, pad_size),
        mode="reflect",
    )
    img_np = padded.numpy().astype(np.float32)
    y_uint8 = np.clip(img_np[0], 0, 255).astype(np.uint8)

    y_medfix = structure_aware_median(
        y_uint8,
        s_thr=0.25,
        outlier_t=30,
        use_extreme_gate=False,
    )
    y_denoise = cv2.fastNlMeansDenoising(y_medfix, None, h=4, templateWindowSize=7, searchWindowSize=15)
    structure = compute_structure_score(y_denoise)
    s = structure["S"].astype(np.float32)
    y_denoise_f = y_denoise.astype(np.float32)

    base = smooth_base_layer(y_denoise)
    detail = y_denoise_f - base
    detail_gain = np.clip(np.power(s, 3.0), 0.0, 1.0)
    y_ref = base + detail_gain * detail
    y_ref_u8 = np.clip(y_ref, 0, 255).astype(np.uint8)
    y_out = match_local_mean_var(y_uint8, y_ref_u8)

    out_np = img_np.copy()
    out_np[0] = y_out.astype(np.float32)
    out = torch.from_numpy(out_np)
    return out[:, pad_size:-pad_size, pad_size:-pad_size]
