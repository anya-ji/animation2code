"""
METHOD1: Precision

Measures how well the rendered animation matches the gold animation
within the region where the render is active.

Algorithm:
  1. Find the bounding box of changing pixels in the RENDER video
     (tightest square bbox — union of changed pixels across all frames).
  2. Crop that same region from both the render and gold videos.
  3. Compute per-frame similarity on the cropped regions (SSIM or CLIP).
  4. Return the mean similarity as the precision score.

Rationale: using the render's bounding box tests whether what the
render actually produced matches the gold in that same area.

With --with-dtw, a full n×m similarity matrix is built and DTW is used
to find the optimal frame alignment before scoring.  Output then contains:
  - score: mean similarity over DTW-aligned pairs
  - per_frame_dtw: similarity for each pair in dtw_path
  - dtw_cost: total accumulated distance along the optimal DTW path
  - dtw_path: list of [render_idx, gold_idx] frame index pairs

With --sim clip, CLIP ViT-B/32 cosine similarity replaces SSIM.

Usage (standalone):
    uv run python evaluation/methods/precision.py render.mp4 gold.mp4
    uv run python evaluation/methods/precision.py render.mp4 gold.mp4 --with-dtw
    uv run python evaluation/methods/precision.py render.mp4 gold.mp4 --sim clip --with-dtw
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from methods._utils import (
    build_clip_distance_matrix,
    build_ssim_distance_matrix,
    clip_embeddings,
    find_changing_bbox,
    is_static_video,
    load_frames,
    mean_ssim_over_frames,
)


def run(
    render_path: str,
    gold_path: str,
    *,
    with_dtw: bool = True,
    sim: str = "ssim",
    device: str = "cpu",
) -> dict:
    """Compute precision score between render and gold videos.

    Args:
        render_path: path to the candidate/render video.
        gold_path:   path to the gold/reference video.
        with_dtw:    if True, use DTW alignment (the chosen similarity is the
                     matching criterion).
        sim:         similarity metric — 'ssim' (default) or 'clip'.
        device:      torch device for CLIP embeddings (only used when sim='clip').

    Returns (without DTW):
        score        – mean similarity over aligned frames
        per_frame    – per-frame similarity values
        bbox         – [x1, y1, x2, y2] bounding box used (from render)
        n_frames     – number of frames compared
        sim          – similarity metric used

    Returns (with DTW):
        score            – mean similarity over DTW-aligned pairs
        per_frame_dtw    – similarity for each pair in dtw_path
        dtw_cost         – total accumulated DTW distance
        dtw_path         – list of [render_idx, gold_idx] frame index pairs
        bbox             – [x1, y1, x2, y2] bounding box used (from render)
        n_render_frames  – frames extracted from render video
        n_gold_frames    – frames extracted from gold video
        n_paired_frames  – number of pairs in the DTW path
        sim              – similarity metric used
    """
    render_frames = load_frames(render_path)
    gold_frames = load_frames(gold_path)

    is_static = is_static_video(render_frames)
    # Bounding box from the render (precision perspective)
    bbox = find_changing_bbox(render_frames, square=True)

    if with_dtw:
        from methods.dtw import dtw as _dtw

        if sim == "clip":
            D = build_clip_distance_matrix(render_frames, gold_frames, bbox, device)
        else:
            D = build_ssim_distance_matrix(render_frames, gold_frames, bbox, device)

        dtw_cost, path, _ = _dtw(D)
        pair_scores = [float(1.0 - D[p[0], p[1]]) for p in path]
        avg_score = float(np.mean(pair_scores)) if pair_scores else 0.0

        return {
            "score": avg_score,
            "per_frame_dtw": pair_scores,
            "dtw_cost": dtw_cost,
            "dtw_path": path,
            "bbox": list(bbox),
            "n_render_frames": len(render_frames),
            "n_gold_frames": len(gold_frames),
            "n_paired_frames": len(path),
            "sim": sim,
            "is_static": is_static,
        }

    else:
        n = min(len(render_frames), len(gold_frames))

        if sim == "clip":
            emb_r = clip_embeddings(render_frames[:n], bbox, device)
            emb_g = clip_embeddings(gold_frames[:n], bbox, device)
            per_frame = [float(emb_r[i] @ emb_g[i]) for i in range(n)]
            mean_score = float(np.mean(per_frame)) if per_frame else 0.0
        else:
            mean_score, per_frame = mean_ssim_over_frames(
                render_frames[:n], gold_frames[:n], bbox
            )

        return {
            "score": mean_score,
            "per_frame": per_frame,
            "bbox": list(bbox),
            "n_frames": n,
            "sim": sim,
            "is_static": is_static,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Precision metric: similarity in the render's bounding box"
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
        choices=["ssim", "clip"],
        default="ssim",
        help="Similarity metric: 'ssim' (default) or 'clip' (CLIP ViT-B/32)",
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

    # dtw_path can be large — truncate for stdout readability
    if len(result.get("dtw_path", [])) > 100:
        result = dict(result)
        result["dtw_path"] = result["dtw_path"][:10] + ["... (truncated)"]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
