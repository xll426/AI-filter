"""基于 manifest.csv 的成对 YUV 数据读取器。

`manifest.csv` 是样本路径、尺寸、YUV 格式和位深的唯一来源。Dataset 会把
YUV 读成 `[0,1]` 归一化张量；W10+B13 QAT 模型在前向内部再把 Y 恢复到
raw `0..255` 像素域，以模拟最终部署的整数计算路径。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import read_csv_rows, yuvread2tensor


class PairedYuvDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        crop_size: int | None = None,
        training: bool = False,
        hflip: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.crop_size = crop_size
        self.training = training
        self.hflip = hflip

        manifest_path = self.split_dir / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.records: List[Dict[str, str]] = read_csv_rows(manifest_path)
        if not self.records:
            raise ValueError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _crop_pair(
        self,
        img: torch.Tensor,
        gt: torch.Tensor,
        roi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.crop_size:
            return img, gt, roi
        _, h, w = img.shape
        if self.crop_size >= h or self.crop_size >= w:
            return img, gt, roi
        if self.training:
            top, left = self._sample_train_crop(h, w, roi)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
        img = img[:, top : top + self.crop_size, left : left + self.crop_size]
        gt = gt[:, top : top + self.crop_size, left : left + self.crop_size]
        if roi is not None:
            roi = roi[:, top : top + self.crop_size, left : left + self.crop_size]
        return img, gt, roi

    def _sample_train_crop(self, h: int, w: int, roi: torch.Tensor | None) -> tuple[int, int]:
        if roi is None or torch.count_nonzero(roi) == 0:
            return random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)

        roi_np = roi.squeeze(0).numpy()
        roi_uint8 = (roi_np > 0).astype(np.uint8)
        contours, _ = cv2.findContours(roi_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            selected = random.choice(contours)
            x, y, w_box, h_box = cv2.boundingRect(selected)
            center_x = random.randint(x, x + w_box - 1)
            center_y = random.randint(y, y + h_box - 1)
        else:
            ys, xs = np.where(roi_np > 0)
            if len(xs) == 0:
                return random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            idx = random.randint(0, len(xs) - 1)
            center_y = int(ys[idx])
            center_x = int(xs[idx])

        top = max(0, min(center_y - self.crop_size // 2, h - self.crop_size))
        left = max(0, min(center_x - self.crop_size // 2, w - self.crop_size))
        return top, left

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.records[index]
        # 路径相对于 split 目录，例如 `train/img/...` 在 manifest 中写为
        # `img/...`。示例数据和完整数据集共用同一套读取逻辑。
        img_path = self.split_dir / row["input_path"]
        gt_path = self.split_dir / row["target_path"]
        width = int(row["width"])
        height = int(row["height"])
        fmt = row["format"]
        bitdepth = int(row["bitdepth"])

        img = yuvread2tensor(img_path, width, height, fmt=fmt, bitdepth=bitdepth, normalize=True)
        gt = yuvread2tensor(gt_path, width, height, fmt=fmt, bitdepth=bitdepth, normalize=True)
        roi = None
        roi_rel = row.get("roi_path", "").strip()
        if roi_rel:
            roi_path = self.split_dir / roi_rel
            if roi_path.is_file():
                roi_np = np.load(roi_path)
                if roi_np.ndim == 3:
                    roi_np = roi_np[0]
                roi = torch.from_numpy(roi_np.astype(np.float32)).unsqueeze(0)

        img, gt, roi = self._crop_pair(img, gt, roi)

        if self.training and self.hflip and random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            gt = torch.flip(gt, dims=[2])
            if roi is not None:
                roi = torch.flip(roi, dims=[2])

        sample = {
            "input": img,
            "target": gt,
            "input_path": str(img_path),
            "target_path": str(gt_path),
            "source_video": row.get("source_video", ""),
        }
        if roi is not None:
            sample["roi"] = roi
        return sample
