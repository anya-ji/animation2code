"""Compare each video inside its own animated region (the paper's appearance metric with DreamSim).

Steps:
  1. Find the tightest square around the changing pixels in the render and in the gold.
  2. Crop each video to its own square.
  3. Downscale the larger crop to match the smaller.
  4. Compare frames with SSIM, CLIP or DreamSim, aligned with DTW by default.

Using separate crops stops position shifts between the videos from dominating the score.
A static render uses the full frame as its crop and is marked is_static=True.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from methods._utils import (
    build_clip_distance_matrix,
    build_dreamsim_distance_matrix,
    build_ssim_distance_matrix,
    clip_embeddings,
    crop_frames,
    find_changing_bbox,
    is_static_video,
    load_frames,
    mean_dreamsim_over_frames,
    mean_ssim_over_frames,
)


def _resize_frames(frames: list[np.ndarray], target_h: int, target_w: int) -> list[np.ndarray]:
    return [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA) for f in frames]


def run(
    render_path: str,
    gold_path: str,
    *,
    with_dtw: bool = True,
    sim: str = "ssim",
    device: str = "cpu",
) -> dict:
    """Compute the bbox score between a render and a gold video.

    Args:
        render_path: Path to the rendered video.
        gold_path: Path to the gold video.
        with_dtw: Align frames with DTW; otherwise compare matching indices.
        sim: Similarity metric: "ssim", "clip" or "dreamsim".
        device: Torch device.

    Returns a dict with:
        score: Mean similarity over DTW-aligned pairs.
        per_frame_dtw: Similarity of each aligned pair.
        dtw_cost: Total DTW distance.
        dtw_path: List of [render_idx, gold_idx] pairs.
        n_render_frames, n_gold_frames: Frames in each video.
        n_paired_frames: Pairs in the DTW path.
        render_bbox, gold_bbox: [x1, y1, x2, y2] crop of each video.
        target_side: Side length of the compared region.
        sim: Similarity metric used.
        is_static: True if the render has no changing pixels.

    Without DTW, per_frame and n_frames replace the DTW and frame-count fields.
    """
    render_frames = load_frames(render_path)
    gold_frames = load_frames(gold_path)

    is_static = is_static_video(render_frames)

    # If the render is static, find_changing_bbox falls back to the full frame
    # (largest possible bbox), which is the correct behaviour for a static render.
    render_bbox = find_changing_bbox(render_frames, square=True)
    gold_bbox = find_changing_bbox(gold_frames, square=True)

    render_crops = crop_frames(render_frames, render_bbox)
    gold_crops = crop_frames(gold_frames, gold_bbox)

    render_side = render_bbox[2] - render_bbox[0]
    gold_side = gold_bbox[2] - gold_bbox[0]
    target_side = min(render_side, gold_side)
    if render_side > target_side:
        render_crops = _resize_frames(render_crops, target_side, target_side)
    elif gold_side > target_side:
        gold_crops = _resize_frames(gold_crops, target_side, target_side)

    if with_dtw:
        from methods.dtw import dtw as _dtw

        if sim == "clip":
            D = build_clip_distance_matrix(render_crops, gold_crops, bbox=None, device=device)
        elif sim == "dreamsim":
            D = build_dreamsim_distance_matrix(render_crops, gold_crops, bbox=None, device=device)
        else:
            D = build_ssim_distance_matrix(render_crops, gold_crops, bbox=None, device=device)

        dtw_cost, path, _ = _dtw(D)
        pair_scores = [float(1.0 - D[p[0], p[1]]) for p in path]
        avg_score = float(np.mean(pair_scores)) if pair_scores else 0.0

        return {
            "score": avg_score,
            "per_frame_dtw": pair_scores,
            "dtw_cost": dtw_cost,
            "dtw_path": path,
            "render_bbox": list(render_bbox),
            "gold_bbox": list(gold_bbox),
            "target_side": target_side,
            "n_render_frames": len(render_frames),
            "n_gold_frames": len(gold_frames),
            "n_paired_frames": len(path),
            "sim": sim,
            "is_static": is_static,
        }

    else:
        n = min(len(render_crops), len(gold_crops))

        if sim == "clip":
            emb_r = clip_embeddings(render_crops[:n], bbox=None, device=device)
            emb_g = clip_embeddings(gold_crops[:n], bbox=None, device=device)
            per_frame = [float(emb_r[i] @ emb_g[i]) for i in range(n)]
            mean_score = float(np.mean(per_frame)) if per_frame else 0.0
        elif sim == "dreamsim":
            mean_score, per_frame = mean_dreamsim_over_frames(
                render_crops[:n], gold_crops[:n], bbox=None, device=device
            )
        else:
            mean_score, per_frame = mean_ssim_over_frames(
                render_crops[:n], gold_crops[:n], bbox=None
            )

        return {
            "score": mean_score,
            "per_frame": per_frame,
            "render_bbox": list(render_bbox),
            "gold_bbox": list(gold_bbox),
            "target_side": target_side,
            "n_frames": n,
            "sim": sim,
            "is_static": is_static,
        }


def main():
    parser = argparse.ArgumentParser(
        description="BBox metric: similarity in each video's own animated bounding box"
    )
    parser.add_argument("render_video", help="Path to the render (candidate) video")
    parser.add_argument("gold_video", help="Path to the gold (reference) video")
    parser.add_argument(
        "--without-dtw",
        action="store_true",
        help="Disable DTW alignment; compare frames at matching indices instead",
    )
    parser.add_argument(
        "--sim",
        choices=["ssim", "clip", "dreamsim"],
        default="ssim",
        help="Similarity metric: 'ssim' (default), 'clip' (CLIP ViT-B/32), or 'dreamsim'",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for CLIP embeddings (default: cpu)",
    )
    args = parser.parse_args()

    result = run(
        args.render_video,
        args.gold_video,
        with_dtw=not args.without_dtw,
        sim=args.sim,
        device=args.device,
    )

    if len(result.get("dtw_path", [])) > 100:
        result = dict(result)
        result["dtw_path"] = result["dtw_path"][:10] + ["... (truncated)"]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
