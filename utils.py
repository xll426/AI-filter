from __future__ import annotations

import csv
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

VIDEO_SUFFIXES = (".h265", ".265", ".hevc")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_command(cmd: Sequence[str]) -> str:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def probe_video(video_path: str | Path) -> Dict[str, int]:
    video_path = str(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_packets",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_read_packets",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        video_path,
    ]
    output = run_command(cmd)
    info: Dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()

    return {
        "width": int(info["width"]),
        "height": int(info["height"]),
        "total_frames": int(info["nb_read_packets"]),
    }


def build_sample_indices(total_frames: int, start_frame: int, end_frame: int | None, frames_per_video: int) -> List[int]:
    if total_frames <= 0:
        return []

    start = max(0, start_frame)
    end = total_frames - 1 if end_frame is None else min(total_frames - 1, end_frame)
    if start > end:
        return []

    available = end - start + 1
    if frames_per_video >= available:
        return list(range(start, end + 1))
    if frames_per_video <= 1:
        return [start]

    step = (available - 1) / float(frames_per_video - 1)
    indices = []
    seen = set()
    for i in range(frames_per_video):
        idx = start + int(round(i * step))
        idx = min(end, max(start, idx))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)

    current = start
    while len(indices) < frames_per_video and current <= end:
        if current not in seen:
            indices.append(current)
            seen.add(current)
        current += 1

    indices.sort()
    return indices


def decode_frame(video_path: str | Path, frame_idx: int, output_path: str | Path, pix_fmt: str = "yuv420p") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_idx})",
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-pix_fmt",
        pix_fmt,
        "-f",
        "rawvideo",
        "-y",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def decode_frame_bgr(video_path: str | Path, frame_idx: int, width: int, height: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_idx})",
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    expected = width * height * 3
    if len(result.stdout) < expected:
        raise EOFError(f"Cannot decode frame {frame_idx} from {video_path} to BGR")
    arr = np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape((height, width, 3))
    return arr.copy()


def write_csv(path: str | Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def allocate_split_counts(sample_count: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, int]:
    raw_counts = {
        "train": sample_count * train_ratio,
        "val": sample_count * val_ratio,
        "test": sample_count * test_ratio,
    }
    counts = {key: int(math.floor(value)) for key, value in raw_counts.items()}
    remaining = sample_count - sum(counts.values())
    if remaining <= 0:
        return counts

    order = sorted(
        raw_counts.keys(),
        key=lambda key: (
            1 if raw_counts[key] > 0 else 0,
            raw_counts[key] - counts[key],
            raw_counts[key],
        ),
        reverse=True,
    )
    positive_keys = [key for key in order if raw_counts[key] > 0]
    if not positive_keys:
        raise ValueError("train/val/test ratio cannot all be zero")

    for idx in range(remaining):
        counts[positive_keys[idx % len(positive_keys)]] += 1
    return counts


def assign_split_by_group(groups: Sequence[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, str]:
    groups = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(groups)
    counts = allocate_split_counts(len(groups), train_ratio, val_ratio, test_ratio)
    split_map: Dict[str, str] = {}
    cursor = 0
    for phase in ("train", "val", "test"):
        for group in groups[cursor : cursor + counts[phase]]:
            split_map[group] = phase
        cursor += counts[phase]
    return split_map


def alignment_for_format(fmt: str) -> int:
    return 2 if fmt in {"yuv420p", "yuv422p"} else 1


def compute_starts(length: int, tile_size: int, stride: int, align: int) -> List[int]:
    if tile_size <= 0 or tile_size >= length:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final_start = length - tile_size
    if final_start % align != 0:
        final_start -= final_start % align
    final_start = max(0, final_start)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return sorted(set(starts))


def iter_tile_boxes(width: int, height: int, tile_size: int, stride: int, fmt: str) -> Iterable[Tuple[int, int, int, int]]:
    if tile_size <= 0:
        yield 0, 0, height, width
        return
    align = alignment_for_format(fmt)
    tile_w = min(tile_size, width)
    tile_h = min(tile_size, height)
    for top in compute_starts(height, tile_h, stride, align):
        for left in compute_starts(width, tile_w, stride, align):
            yield top, left, tile_h, tile_w


def yuvread2tensor(
    path: str | Path,
    width: int,
    height: int,
    fmt: str = "yuv420p",
    bitdepth: int = 8,
    frame_idx: int = 0,
    normalize: bool = False,
) -> torch.Tensor:
    bps = 1 if bitdepth <= 8 else 2
    y_sz = width * height
    if fmt == "yuv400p":
        u_sz = v_sz = 0
    elif fmt == "yuv420p":
        u_sz = v_sz = (width // 2) * (height // 2)
    elif fmt == "yuv422p":
        u_sz = v_sz = (width // 2) * height
    elif fmt == "yuv444p":
        u_sz = v_sz = width * height
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    frame_bytes = (y_sz + u_sz + v_sz) * bps
    with open(path, "rb") as f:
        f.seek(frame_idx * frame_bytes, os.SEEK_SET)
        raw = f.read(frame_bytes)
        if len(raw) < frame_bytes:
            raise EOFError(f"Cannot read frame {frame_idx} from {path}")

    off = 0
    y_raw = raw[off : off + y_sz * bps]
    off += y_sz * bps
    u_raw = raw[off : off + u_sz * bps]
    off += u_sz * bps
    v_raw = raw[off : off + v_sz * bps]

    dtype = np.uint8 if bps == 1 else np.uint16
    y = np.frombuffer(y_raw, dtype=dtype).reshape((height, width))
    if fmt == "yuv400p":
        arr = y[np.newaxis, :, :].astype(np.float32)
    else:
        if fmt == "yuv420p":
            h2, w2 = height // 2, width // 2
        elif fmt == "yuv422p":
            h2, w2 = height, width // 2
        else:
            h2, w2 = height, width
        u = np.frombuffer(u_raw, dtype=dtype).reshape((h2, w2))
        v = np.frombuffer(v_raw, dtype=dtype).reshape((h2, w2))
        if fmt == "yuv420p":
            u = u.repeat(2, axis=0).repeat(2, axis=1)
            v = v.repeat(2, axis=0).repeat(2, axis=1)
        elif fmt == "yuv422p":
            u = u.repeat(2, axis=1)
            v = v.repeat(2, axis=1)
        arr = np.stack([y, u, v], axis=0).astype(np.float32)

    if normalize:
        arr /= float((1 << bitdepth) - 1)
    return torch.from_numpy(arr)


def tensor2yuv(
    tensor: torch.Tensor,
    output_path: str | Path,
    fmt: str = "yuv420p",
    bitdepth: int = 8,
    normalize: bool = False,
) -> None:
    if tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(tensor.shape)}")
    c, _, _ = tensor.shape
    if fmt == "yuv400p" and c != 1:
        raise ValueError("yuv400p expects 1 channel")
    if fmt != "yuv400p" and c != 3:
        raise ValueError(f"{fmt} expects 3 channels")

    max_val = (1 << bitdepth) - 1
    arr = tensor.detach().cpu().clone()
    if normalize:
        arr = arr * float(max_val)
    arr = arr.clamp(0, max_val).round().to(torch.int64)
    dtype = np.uint8 if bitdepth <= 8 else np.uint16

    y = arr[0].numpy().astype(dtype)
    if fmt == "yuv400p":
        payload = y.tobytes()
    else:
        u_full = arr[1].numpy().astype(dtype)
        v_full = arr[2].numpy().astype(dtype)
        if fmt == "yuv420p":
            u = u_full[::2, ::2]
            v = v_full[::2, ::2]
        elif fmt == "yuv422p":
            u = u_full[:, ::2]
            v = v_full[:, ::2]
        elif fmt == "yuv444p":
            u = u_full
            v = v_full
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        payload = y.tobytes() + u.tobytes() + v.tobytes()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(payload)


def pad_to_factor(x: torch.Tensor, factor: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = (-h) % factor
    pad_w = (-w) % factor
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (pad_h, pad_w)


def crop_after_pad(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return x
    return x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: torch.device | None = None) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    return kernel_2d.unsqueeze(0).unsqueeze(0)


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    pred_y = pred[:, :1]
    target_y = target[:, :1]
    device = pred_y.device
    kernel = _gaussian_kernel(device=device)
    padding = kernel.shape[-1] // 2

    mu_x = F.conv2d(pred_y, kernel, padding=padding)
    mu_y = F.conv2d(target_y, kernel, padding=padding)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred_y * pred_y, kernel, padding=padding) - mu_x2
    sigma_y2 = F.conv2d(target_y * target_y, kernel, padding=padding) - mu_y2
    sigma_xy = F.conv2d(pred_y * target_y, kernel, padding=padding) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return float(ssim_map.mean().item())
