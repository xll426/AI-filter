"""ROI mask 检测与缓存工具。

数据预处理阶段调用人脸/车牌检测器生成 ROI mask。训练损失对 ROI
区域提高像素保真权重，从而减少人脸、车牌等敏感区域被过度平滑的风险。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DEFAULT_FACE_MODEL = "./weights/yolov8n-face.pt"
DEFAULT_PLATE_MODEL = "./weights/yolov8s-plate.pt"

_DETECTOR_CACHE: dict[tuple[str, str, float, float], "YoloRoiDetector"] = {}


class YoloRoiDetector:
    """基于两套 YOLO 权重的人脸/车牌 ROI 检测器。

    返回的 mask 为 `uint8` 二值图，1 表示需要提高保真约束的 ROI 区域。
    检测器初始化开销较大，因此通过 `get_roi_detector()` 做进程内缓存。
    """

    def __init__(
        self,
        face_model_path: str | Path = DEFAULT_FACE_MODEL,
        plate_model_path: str | Path = DEFAULT_PLATE_MODEL,
        conf: float = 0.3,
        iou: float = 0.25,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ROI detection requires the `ultralytics` package. "
                "Install it in the prefilter_clean environment first."
            ) from exc

        self.face_model_path = str(Path(face_model_path).resolve())
        self.plate_model_path = str(Path(plate_model_path).resolve())
        self.conf = float(conf)
        self.iou = float(iou)

        if not Path(self.face_model_path).is_file():
            raise FileNotFoundError(f"Face detector weight not found: {self.face_model_path}")
        if not Path(self.plate_model_path).is_file():
            raise FileNotFoundError(f"Plate detector weight not found: {self.plate_model_path}")

        self.face_model = YOLO(self.face_model_path)
        self.plate_model = YOLO(self.plate_model_path)

    def detect(self, image_bgr: np.ndarray, return_vis: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected HWC BGR image, got {image_bgr.shape}")

        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        vis = image_bgr.copy() if return_vis else None

        detections = [
            (self.face_model, (0, 255, 0), "face"),
            (self.plate_model, (255, 0, 0), "plate"),
        ]
        for model, color, label in detections:
            result = model.predict(image_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int).tolist()
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                mask[y1:y2, x1:x2] = 1
                if vis is not None:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return mask, vis


def get_roi_detector(
    face_model_path: str | Path = DEFAULT_FACE_MODEL,
    plate_model_path: str | Path = DEFAULT_PLATE_MODEL,
    conf: float = 0.3,
    iou: float = 0.25,
) -> YoloRoiDetector:
    """按模型路径和阈值缓存 YOLO 检测器，避免每帧重复加载权重。"""
    key = (str(Path(face_model_path).resolve()), str(Path(plate_model_path).resolve()), float(conf), float(iou))
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = YoloRoiDetector(
            face_model_path=face_model_path,
            plate_model_path=plate_model_path,
            conf=conf,
            iou=iou,
        )
    return _DETECTOR_CACHE[key]


def detect_roi_mask(
    image_bgr: np.ndarray,
    face_model_path: str | Path = DEFAULT_FACE_MODEL,
    plate_model_path: str | Path = DEFAULT_PLATE_MODEL,
    conf: float = 0.3,
    iou: float = 0.25,
    return_vis: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """检测单帧 BGR 图像中的人脸/车牌区域并返回 ROI mask。"""
    detector = get_roi_detector(
        face_model_path=face_model_path,
        plate_model_path=plate_model_path,
        conf=conf,
        iou=iou,
    )
    return detector.detect(image_bgr, return_vis=return_vis)
