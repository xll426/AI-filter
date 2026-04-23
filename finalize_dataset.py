#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild manifest.csv and dataset_summary.json from an existing train/val/test/img|gt dataset tree.")
    parser.add_argument("--data_root", required=True, help="Dataset root created by process_data.py")
    parser.add_argument("--input_dir", default="", help="Optional original raw input dir to record in summary")
    parser.add_argument("--frames_per_video", type=int, default=None)
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--tile_stride", type=int, default=None)
    parser.add_argument("--pix_fmt", default="yuv420p")
    parser.add_argument("--yuv_format", default="yuv420p")
    parser.add_argument("--bitdepth", type=int, default=8)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


PATTERN = re.compile(
    r"^(?P<video>.+)_frame(?P<frame>\d+)_tile(?P<tile>\d+)_y(?P<top>\d+)_x(?P<left>\d+)$"
)


def scan_phase(phase_root: Path, bitdepth: int, yuv_format: str) -> tuple[list[dict[str, object]], set[str]]:
    img_root = phase_root / "img"
    gt_root = phase_root / "gt"
    rows: list[dict[str, object]] = []
    source_videos: set[str] = set()

    for img_path in sorted(img_root.rglob("*.yuv")):
        rel_img = img_path.relative_to(phase_root).as_posix()
        rel_gt = (Path("gt") / img_path.relative_to(img_root)).as_posix()
        rel_roi_path = Path("roi") / img_path.relative_to(img_root)
        rel_roi = rel_roi_path.with_suffix(".npy").as_posix() if (phase_root / rel_roi_path.with_suffix(".npy")).is_file() else ""
        gt_path = phase_root / rel_gt
        if not gt_path.is_file():
            continue

        stem = img_path.stem
        match = PATTERN.match(stem)
        if match is None:
            continue

        video = match.group("video")
        frame = int(match.group("frame"))
        tile_id = int(match.group("tile"))
        top = int(match.group("top"))
        left = int(match.group("left"))
        size_bytes = img_path.stat().st_size
        if bitdepth <= 8 and yuv_format == "yuv420p":
            pixels = int(size_bytes / 1.5)
        else:
            raise ValueError("finalize_dataset.py currently assumes 8-bit yuv420p outputs.")
        side = int(round(pixels ** 0.5))
        width = side
        height = side

        rows.append(
            {
                "input_path": rel_img,
                "target_path": rel_gt,
                "width": width,
                "height": height,
                "format": yuv_format,
                "bitdepth": bitdepth,
                "source_video": video,
                "source_frame": frame,
                "tile_id": tile_id,
                "tile_top": top,
                "tile_left": left,
                "roi_path": rel_roi,
            }
        )
        source_videos.add(video)

    return rows, source_videos


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    split_counts = {}

    for phase in ["train", "val", "test"]:
        phase_root = data_root / phase
        rows, videos = scan_phase(phase_root, args.bitdepth, args.yuv_format)
        write_manifest(phase_root / "manifest.csv", rows)
        split_counts[phase] = {"videos": len(videos), "samples": len(rows)}

    summary = {
        "input_dir": args.input_dir,
        "output_dir": str(data_root),
        "config": {
            "frames_per_video": args.frames_per_video,
            "tile_size": args.tile_size,
            "tile_stride": args.tile_stride,
            "pix_fmt": args.pix_fmt,
            "yuv_format": args.yuv_format,
            "bitdepth": args.bitdepth,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        },
        "split_counts": split_counts,
        "native_ximgproc_available": False,
    }
    with (data_root / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
