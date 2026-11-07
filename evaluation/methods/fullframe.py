"""Compare a render and a gold video over the entire frame, with no crop.

Uses SSIM, CLIP or DreamSim. Frames are aligned with DTW by default; with --without-dtw they are
compared at matching indices, truncated to the shorter video.
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
    """Compute the full-frame similarity between a render and a gold video.

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
        sim: Similarity metric used.
    is_static: True if the render has no changing pixels.

    Without DTW, per_frame and n_frames replace the DTW and frame-count fields.
    """
    render_frames = load_frames(render_path)
    gold_frames = load_frames(gold_path)

    is_static = is_static_video(render_frames)

    if with_dtw:
        from methods.dtw import dtw as _dtw

        if sim == "clip":
            D = build_clip_distance_matrix(render_frames, gold_frames, bbox=None, device=device)
        else:
            D = build_ssim_distance_matrix(render_frames, gold_frames, bbox=None, device=device)

        dtw_cost, path, _ = _dtw(D)
        pair_scores = [float(1.0 - D[p[0], p[1]]) for p in path]
        avg_score = float(np.mean(pair_scores)) if pair_scores else 0.0

        return {
            "score": avg_score,
            "per_frame_dtw": pair_scores,
            "dtw_cost": dtw_cost,
            "dtw_path": path,
            "n_render_frames": len(render_frames),
            "n_gold_frames": len(gold_frames),
            "n_paired_frames": len(path),
            "sim": sim,
            "is_static": is_static,
        }

    else:
        n = min(len(render_frames), len(gold_frames))

        if sim == "clip":
            emb_r = clip_embeddings(render_frames[:n], bbox=None, device=device)
            emb_g = clip_embeddings(gold_frames[:n], bbox=None, device=device)
            per_frame = [float(emb_r[i] @ emb_g[i]) for i in range(n)]
            mean_score = float(np.mean(per_frame)) if per_frame else 0.0
        else:
            mean_score, per_frame = mean_ssim_over_frames(
                render_frames[:n], gold_frames[:n], bbox=None
            )

        return {
            "score": mean_score,
            "per_frame": per_frame,
            "n_frames": n,
            "sim": sim,
            "is_static": is_static,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Full-frame similarity metric (no bounding box)"
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

    if len(result.get("dtw_path", [])) > 100:
        result = dict(result)
        result["dtw_path"] = result["dtw_path"][:10] + ["... (truncated)"]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
