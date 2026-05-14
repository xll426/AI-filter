#!/usr/bin/env python3
"""构建 W10+B13 QAT 训练使用的成对 YUV 数据集。

每个 split 下的 `manifest.csv` 是训练样本索引表：一行对应一个 tile 样本，
并记录原始输入 `img`、reference 监督目标 `gt`、可选 ROI mask `roi` 的相对路径。
这些路径都相对于 `train/`、`val/`、`test/` 目录，因此数据集支持整体复制到交付环境，
不需要改写绝对路径。
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from roi import DEFAULT_FACE_MODEL, DEFAULT_PLATE_MODEL, detect_roi_mask
from ref import generate_reference_tensor, native_ximgproc_available
from utils import (
    VIDEO_SUFFIXES,
    assign_split_by_group,
    build_sample_indices,
    decode_frame,
    decode_frame_bgr,
    ensure_dir,
    iter_tile_boxes,
    probe_video,
    tensor2yuv,
    write_csv,
    yuvread2tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="解码源视频，生成 reference 监督目标，并写出 train/val/test 数据集。")
    parser.add_argument("--input_dir", required=True, help="原始 .h265/.265/.hevc 视频所在目录。")
    parser.add_argument("--output_dir", required=True, help="输出数据集根目录。")
    parser.add_argument("--roi_dir", default=None, help="可选：已有逐帧 ROI npy 文件目录。")
    parser.add_argument("--detect_roi", action="store_true", help="直接从解码帧检测人脸/车牌 ROI。")
    parser.add_argument("--roi_face_model", default=DEFAULT_FACE_MODEL, help="YOLO 人脸检测权重路径。")
    parser.add_argument("--roi_plate_model", default=DEFAULT_PLATE_MODEL, help="YOLO 车牌检测权重路径。")
    parser.add_argument("--roi_conf", type=float, default=0.3, help="YOLO 置信度阈值。")
    parser.add_argument("--roi_iou", type=float, default=0.25, help="YOLO NMS IoU 阈值。")
    parser.add_argument("--roi_save_vis", action="store_true", help="将 ROI 可视化 JPG 保存到 output_dir/roi_vis。")
    parser.add_argument("--frames_per_video", type=int, default=12, help="每个视频均匀抽取的帧数。")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tile_size", type=int, default=512, help="tile 尺寸；0 表示保留整帧。")
    parser.add_argument("--tile_stride", type=int, default=None, help="tile 步长；默认等于 tile_size。")
    parser.add_argument("--pix_fmt", default="yuv420p", help="ffmpeg 解码输出像素格式。")
    parser.add_argument("--yuv_format", default="yuv420p", help="落盘保存的 planar YUV 格式。")
    parser.add_argument("--bitdepth", type=int, default=8)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="按源视频并行处理的进程数。")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_roi_mode(args: argparse.Namespace) -> str:
    if args.roi_dir and args.detect_roi:
        raise SystemExit("Use either --roi_dir or --detect_roi, not both.")
    if args.detect_roi:
        return "detect"
    if args.roi_dir:
        return "dir"
    return "none"


def resolve_roi_path(roi_dir: Path | None, video_stem: str, frame_idx: int) -> Path | None:
    if roi_dir is None:
        return None
    candidates = [
        roi_dir / video_stem / f"{video_stem}_frame{frame_idx:06d}.npy",
        roi_dir / f"{video_stem}_frame{frame_idx:06d}.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def find_videos(input_dir: Path) -> list[Path]:
    return [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]


def process_one_video(task: dict) -> dict:
    """处理单个视频：解码抽帧、生成 reference 目标、切 tile，并返回 manifest 行。"""
    video_path = Path(task["video_path"])
    output_dir = Path(task["output_dir"])
    tmp_dir = Path(task["tmp_dir"])
    phase = task["phase"]
    frames_per_video = int(task["frames_per_video"])
    start_frame = int(task["start_frame"])
    end_frame = task["end_frame"]
    tile_size = int(task["tile_size"])
    tile_stride = int(task["tile_stride"])
    pix_fmt = task["pix_fmt"]
    yuv_format = task["yuv_format"]
    bitdepth = int(task["bitdepth"])
    roi_mode = task.get("roi_mode", "none")
    roi_dir = Path(task["roi_dir"]) if task.get("roi_dir") else None
    roi_face_model = task.get("roi_face_model", DEFAULT_FACE_MODEL)
    roi_plate_model = task.get("roi_plate_model", DEFAULT_PLATE_MODEL)
    roi_conf = float(task.get("roi_conf", 0.3))
    roi_iou = float(task.get("roi_iou", 0.25))
    roi_save_vis = bool(task.get("roi_save_vis", False))
    roi_vis_root = Path(task["roi_vis_root"]) if task.get("roi_vis_root") else None

    probe = probe_video(video_path)
    width = probe["width"]
    height = probe["height"]
    frame_indices = build_sample_indices(probe["total_frames"], start_frame, end_frame, frames_per_video)

    rows: list[dict[str, object]] = []
    sample_count = 0

    for frame_idx in frame_indices:
        frame_name = f"{video_path.stem}_frame{frame_idx:06d}.yuv"
        tmp_frame_path = tmp_dir / frame_name
        decode_frame(video_path, frame_idx, tmp_frame_path, pix_fmt=pix_fmt)

        img_tensor = yuvread2tensor(tmp_frame_path, width, height, fmt=yuv_format, bitdepth=bitdepth, normalize=False)
        # 参考算法 2 在 raw 像素域执行，输出与输入 tile 配对的监督目标 `gt`。
        # 这里先对整帧生成参考结果，再跟输入一起切 tile，避免 tile 边界影响参考滤波。
        gt_tensor = generate_reference_tensor(img_tensor)
        frame_roi = None
        if roi_mode == "dir":
            roi_source = resolve_roi_path(roi_dir, video_path.stem, frame_idx)
            if roi_source is not None:
                frame_roi = np.load(roi_source)
                if frame_roi.ndim == 3:
                    frame_roi = frame_roi[0]
        elif roi_mode == "detect":
            frame_bgr = decode_frame_bgr(video_path, frame_idx, width, height)
            frame_roi, roi_vis = detect_roi_mask(
                frame_bgr,
                face_model_path=roi_face_model,
                plate_model_path=roi_plate_model,
                conf=roi_conf,
                iou=roi_iou,
                return_vis=roi_save_vis,
            )
            if roi_save_vis and roi_vis_root is not None and roi_vis is not None:
                vis_path = roi_vis_root / video_path.stem / f"{video_path.stem}_frame{frame_idx:06d}.jpg"
                vis_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_path), roi_vis)

        for tile_id, (top, left, tile_h, tile_w) in enumerate(iter_tile_boxes(width, height, tile_size, tile_stride, yuv_format)):
            # 这些相对路径会写入 manifest.csv。训练时 `PairedYuvDataset` 不再扫描目录，
            # 而是严格按 manifest 中的路径和元数据读取样本。
            tile_stem = f"{video_path.stem}_frame{frame_idx:06d}_tile{tile_id:03d}_y{top:04d}_x{left:04d}"
            rel_img = Path("img") / video_path.stem / f"{tile_stem}.yuv"
            rel_gt = Path("gt") / video_path.stem / f"{tile_stem}.yuv"
            rel_roi = None
            out_img_path = output_dir / phase / rel_img
            out_gt_path = output_dir / phase / rel_gt

            img_tile = img_tensor[:, top : top + tile_h, left : left + tile_w]
            gt_tile = gt_tensor[:, top : top + tile_h, left : left + tile_w]
            tensor2yuv(img_tile, out_img_path, fmt=yuv_format, bitdepth=bitdepth, normalize=False)
            tensor2yuv(gt_tile, out_gt_path, fmt=yuv_format, bitdepth=bitdepth, normalize=False)
            if frame_roi is not None:
                rel_roi = Path("roi") / video_path.stem / f"{tile_stem}.npy"
                out_roi_path = output_dir / phase / rel_roi
                out_roi_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_roi_path, frame_roi[top : top + tile_h, left : left + tile_w])

            rows.append(
                {
                    "input_path": rel_img.as_posix(),
                    "target_path": rel_gt.as_posix(),
                    "width": tile_w,
                    "height": tile_h,
                    "format": yuv_format,
                    "bitdepth": bitdepth,
                    "source_video": video_path.stem,
                    "source_frame": frame_idx,
                    "tile_id": tile_id,
                    "tile_top": top,
                    "tile_left": left,
                    "roi_path": "" if rel_roi is None else rel_roi.as_posix(),
                }
            )
            sample_count += 1

        tmp_frame_path.unlink(missing_ok=True)

    return {
        "video_name": video_path.name,
        "phase": phase,
        "width": width,
        "height": height,
        "total_frames": probe["total_frames"],
        "sampled_frames": len(frame_indices),
        "rows": rows,
        "sample_count": sample_count,
    }


def main() -> int:
    args = parse_args()
    roi_mode = resolve_roi_mode(args)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory already exists and is not empty: {output_dir}. Use --overwrite to reuse it.")

    input_dir = Path(args.input_dir).resolve()
    videos = find_videos(input_dir)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    if not videos:
        raise SystemExit(f"No H.265 files found in: {input_dir}")

    split_map = assign_split_by_group(
        [video.stem for video in videos],
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    tile_stride = args.tile_stride if args.tile_stride is not None else args.tile_size

    if args.overwrite and output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()

    tmp_dir = ensure_dir(output_dir / ".tmp")
    roi_vis_root = output_dir / "roi_vis" if args.roi_save_vis else None
    manifests: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    split_counts: dict[str, dict[str, int]] = {phase: {"videos": 0, "samples": 0} for phase in manifests}
    for phase in manifests:
        ensure_dir(output_dir / phase / "img")
        ensure_dir(output_dir / phase / "gt")

    print(f"[INFO] Native ximgproc available: {native_ximgproc_available()}")
    print(f"[INFO] ROI mode: {roi_mode}")
    print(f"[INFO] Processing {len(videos)} source videos into {output_dir}")

    tasks = []
    for video_path in videos:
        phase = split_map[video_path.stem]
        split_counts[phase]["videos"] += 1
        tasks.append(
            {
                "video_path": str(video_path),
                "output_dir": str(output_dir),
                "tmp_dir": str(tmp_dir),
                "phase": phase,
                "frames_per_video": args.frames_per_video,
                "start_frame": args.start_frame,
                "end_frame": args.end_frame,
                "tile_size": args.tile_size,
                "tile_stride": tile_stride,
                "pix_fmt": args.pix_fmt,
                "yuv_format": args.yuv_format,
                "bitdepth": args.bitdepth,
                "roi_mode": roi_mode,
                "roi_dir": args.roi_dir,
                "roi_face_model": args.roi_face_model,
                "roi_plate_model": args.roi_plate_model,
                "roi_conf": args.roi_conf,
                "roi_iou": args.roi_iou,
                "roi_save_vis": args.roi_save_vis,
                "roi_vis_root": None if roi_vis_root is None else str(roi_vis_root),
            }
        )

    if args.workers <= 1:
        results = [process_one_video(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(process_one_video, tasks))

    for result in results:
        print(
            f"[INFO] {result['video_name']}: "
            f"{result['width']}x{result['height']}, "
            f"sampled {result['sampled_frames']}/{result['total_frames']} frames -> {result['phase']}"
        )
        manifests[result["phase"]].extend(result["rows"])
        split_counts[result["phase"]]["samples"] += int(result["sample_count"])

    manifest_fields = [
        "input_path",
        "target_path",
        "width",
        "height",
        "format",
        "bitdepth",
        "source_video",
        "source_frame",
        "tile_id",
        "tile_top",
        "tile_left",
        "roi_path",
    ]
    for phase, rows in manifests.items():
        write_csv(output_dir / phase / "manifest.csv", rows, manifest_fields)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config": {
            "frames_per_video": args.frames_per_video,
            "tile_size": args.tile_size,
            "tile_stride": tile_stride,
            "pix_fmt": args.pix_fmt,
            "yuv_format": args.yuv_format,
            "bitdepth": args.bitdepth,
            "roi_mode": roi_mode,
            "roi_dir": args.roi_dir,
            "roi_face_model": args.roi_face_model if roi_mode == "detect" else None,
            "roi_plate_model": args.roi_plate_model if roi_mode == "detect" else None,
            "roi_conf": args.roi_conf if roi_mode == "detect" else None,
            "roi_iou": args.roi_iou if roi_mode == "detect" else None,
            "roi_save_vis": args.roi_save_vis if roi_mode == "detect" else False,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        },
        "split_counts": split_counts,
        "native_ximgproc_available": native_ximgproc_available(),
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[DONE] Dataset prepared at {output_dir}")
    print(json.dumps(summary["split_counts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
