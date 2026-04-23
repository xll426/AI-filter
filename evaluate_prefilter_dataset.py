from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from iqa_metrics_exact_refalgo_y import (
    evaluate_selective_prefilter_y,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation for prefilter outputs with exact ref-algorithm masks."
    )
    parser.add_argument("--ori", required=True, help="Original tiled YUV directory.")
    parser.add_argument("--ref", required=True, help="Reference-algorithm tiled YUV directory.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate in the form name=/path/to/dir. Can be passed multiple times.",
    )
    parser.add_argument("--width", type=int, default=512, help="Tile width.")
    parser.add_argument("--height", type=int, default=512, help="Tile height.")
    parser.add_argument(
        "--mask_mode",
        choices=["pre_median", "detail_gain"],
        default="detail_gain",
        help="Exact mask mode aligned with ref.py.",
    )
    parser.add_argument("--s_thr", type=float, default=0.25, help="Structure threshold for bg/edge masks.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of common tiles.")
    parser.add_argument("--workers", type=int, default=0, help="Process workers; 0 means auto.")
    parser.add_argument("--output_json", default="", help="Optional JSON output path.")
    return parser.parse_args()


def parse_candidate_specs(specs: Iterable[str]) -> Dict[str, Path]:
    candidates: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --candidate '{spec}', expected name=/path/to/dir")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name:
            raise ValueError(f"Invalid --candidate '{spec}', name is empty")
        if not raw_path:
            raise ValueError(f"Invalid --candidate '{spec}', path is empty")
        candidates[name] = Path(raw_path)
    if not candidates:
        raise ValueError("At least one --candidate name=/path is required")
    return candidates


def build_file_index(root: Path) -> Dict[str, str]:
    return {path.name: str(path) for path in root.rglob("*.yuv")}


def read_y_plane(path: str, width: int, height: int) -> np.ndarray:
    y_size = width * height
    with open(path, "rb") as f:
        raw = f.read(y_size)
    if len(raw) != y_size:
        raise ValueError(f"Cannot read full Y plane from {path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)


def evaluate_one_tile(
    name: str,
    ori_path: str,
    ref_path: str,
    candidate_paths: Dict[str, str],
    width: int,
    height: int,
    mask_mode: str,
    s_thr: float,
) -> Dict[str, Dict[str, float]]:
    src = read_y_plane(ori_path, width=width, height=height)
    ref = read_y_plane(ref_path, width=width, height=height)

    out: Dict[str, Dict[str, float]] = {}
    for model_name, pred_path in candidate_paths.items():
        pred = read_y_plane(pred_path, width=width, height=height)
        metrics = evaluate_selective_prefilter_y(
            pred,
            ref,
            src,
            mask_mode=mask_mode,
            s_thr=s_thr,
        )
        out[model_name] = metrics
    return out


def evaluate_one_tile_star(args: Tuple[str, str, str, Dict[str, str], int, int, str, float]) -> Dict[str, Dict[str, float]]:
    return evaluate_one_tile(*args)


def finite_values(rows: List[Dict[str, float]], key: str) -> List[float]:
    vals = [row[key] for row in rows if key in row]
    return [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]


def summarize_model(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float] | int]:
    summary: Dict[str, Dict[str, float] | int] = {"tile_count": len(rows)}
    if not rows:
        return summary

    for key in rows[0].keys():
        vals = finite_values(rows, key)
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }
    return summary


def print_summary(model_name: str, summary: Dict[str, Dict[str, float] | int]) -> None:
    print(f"\n== {model_name} ==")
    print(f"tile_count: {summary['tile_count']}")
    ordered_keys = [
        "selective_score",
        "bg_completion",
        "edge_source_completion",
        "edge_retention_ratio",
        "edge_oversmooth_vs_src",
        "bg_hf_error",
        "edge_preserve_error",
        "edge_over_smooth_ratio",
        "edge_gmsd",
        "bg_grad_energy_ratio",
        "edge_grad_energy_ratio",
        "structure_alignment_error",
    ]
    for key in ordered_keys:
        value = summary.get(key)
        if not isinstance(value, dict):
            continue
        print(
            f"{key}: "
            f"mean={value['mean']:.6f} "
            f"median={value['median']:.6f} "
            f"p10={value['p10']:.6f} "
            f"p90={value['p90']:.6f}"
        )


def main() -> None:
    args = parse_args()
    candidate_roots = parse_candidate_specs(args.candidate)

    ori_index = build_file_index(Path(args.ori))
    ref_index = build_file_index(Path(args.ref))
    candidate_indices = {name: build_file_index(path) for name, path in candidate_roots.items()}

    common = set(ori_index) & set(ref_index)
    for index in candidate_indices.values():
        common &= set(index)
    common_names = sorted(common)
    if args.limit > 0:
        common_names = common_names[: args.limit]

    if not common_names:
        raise RuntimeError("No common .yuv tile names found across ori/ref/candidates")

    workers = args.workers if args.workers > 0 else min(8, max(1, (os.cpu_count() or 4) - 1))
    per_model_rows: Dict[str, List[Dict[str, float]]] = {name: [] for name in candidate_roots}

    tasks: List[Tuple[str, str, str, Dict[str, str], int, int, str, float]] = []
    for name in common_names:
        candidate_paths = {model_name: index[name] for model_name, index in candidate_indices.items()}
        tasks.append(
            (
                name,
                ori_index[name],
                ref_index[name],
                candidate_paths,
                args.width,
                args.height,
                args.mask_mode,
                args.s_thr,
            )
        )

    if workers == 1:
        results = [evaluate_one_tile(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(evaluate_one_tile_star, tasks, chunksize=8))

    for tile_result in results:
        for model_name, metrics in tile_result.items():
            per_model_rows[model_name].append(metrics)

    final_summary = {
        "tile_count": len(common_names),
        "mask_mode": args.mask_mode,
        "s_thr": args.s_thr,
        "models": {model_name: summarize_model(rows) for model_name, rows in per_model_rows.items()},
    }

    print(f"common_tile_count: {len(common_names)}")
    print(f"mask_mode: {args.mask_mode}")
    print(f"s_thr: {args.s_thr}")
    for model_name, summary in final_summary["models"].items():
        print_summary(model_name, summary)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\njson_saved: {output_path}")


if __name__ == "__main__":
    main()
