"""
Temporal fidelity evaluation using CoTracker3.

Extends motion_fidelity by combining direction cosine similarity with
speed similarity (min/max displacement magnitude ratio). This captures
both the direction and pace of motion, improving agreement with human
temporal quality judgments (ROC-AUC 0.731 vs 0.637 for direction-only).

Score = 0.5 * direction_cosine + 0.5 * speed_similarity
Speed similarity per timestep: min(|d1|, |d2|) / max(|d1|, |d2|) ∈ [0, 1]
Score range: [-0.5, 1.0] (1 = identical motion, 0 = orthogonal/speed mismatch)
"""
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _utils import load_frames, find_changing_bbox, crop_frames
from _motion_fidelity import (
    frames_to_tensor, resize_tensor, upscale_tensor, match_video_sizes,
    subsample_frames, load_video, adapt_grid_size,
    compute_first_appearance_queries, get_tracklets, motion_fidelity_score,
)


def get_similarity_matrix(
    tracklets1: torch.Tensor, tracklets2: torch.Tensor,
    qt1: np.ndarray, qt2: np.ndarray,
) -> torch.Tensor:
    """0.5 * direction_cosine + 0.5 * speed_similarity (min/max ratio).

    Direction cosine: cosine similarity of frame-to-frame displacement vectors.
    Speed similarity: min(|d1|, |d2|) / max(|d1|, |d2|) per timestep ∈ [0, 1].
    Both are averaged over the union active window (same denominator as MF baseline).
    """
    d1 = tracklets1[:, 1:] - tracklets1[:, :-1]  # (N1, T_steps, 2)
    d2 = tracklets2[:, 1:] - tracklets2[:, :-1]  # (N2, T_steps, 2)
    T_steps = d1.shape[1]
    device = d1.device

    t_idx = torch.arange(T_steps, device=device)
    qt1_t = torch.from_numpy(qt1).to(device)              # (N1,)
    qt2_t = torch.from_numpy(qt2).to(device)              # (N2,)
    d1 = d1 * (t_idx[None, :] >= qt1_t[:, None]).unsqueeze(-1).float()
    d2 = d2 * (t_idx[None, :] >= qt2_t[:, None]).unsqueeze(-1).float()

    mag1 = d1.norm(dim=-1)                                    # (N1, T_steps)
    mag2 = d2.norm(dim=-1)                                    # (N2, T_steps)
    d1_dir = d1 / mag1.clamp(min=1e-6).unsqueeze(-1)
    d2_dir = d2 / mag2.clamp(min=1e-6).unsqueeze(-1)

    dots = torch.einsum("itc,jtc->ijt", d1_dir, d2_dir)      # (N1, N2, T_steps)

    min_mag = torch.minimum(mag1.unsqueeze(1), mag2.unsqueeze(0))  # (N1, N2, T_steps)
    max_mag = torch.maximum(mag1.unsqueeze(1), mag2.unsqueeze(0))
    speed_sim = min_mag / max_mag.clamp(min=1e-6)             # ∈ [0, 1]

    union_start = torch.minimum(qt1_t[:, None], qt2_t[None, :])   # (N1, N2)
    count = (T_steps - union_start).clamp(min=1).float()

    dir_matrix = dots.sum(-1) / count
    spd_matrix = speed_sim.sum(-1) / count
    return 0.5 * dir_matrix + 0.5 * spd_matrix               # (N1, N2)


def run(
    render_path: str,
    gold_path: str,
    *,
    device: str = "cpu",
    motion_threshold: int = 8,
    grid_size: int = 30,
    min_track_side: int = 256,
    max_frames: int | None = None,
    backward: bool = False,
    use_bbox: bool = True,
    model=None,
) -> dict:
    """
    Compute temporal fidelity between a rendered video and a gold reference.

    Combines direction cosine similarity with speed similarity (min/max ratio)
    at each timestep: score = 0.5 * direction + 0.5 * speed.

    Args:
        render_path:      Path to the predicted/rendered video.
        gold_path:        Path to the ground-truth video.
        device:           Torch device ("cpu" or "cuda").
        motion_threshold: Pixel-value diff threshold for first-appearance queries.
        grid_size:        Base NxN grid density (scaled by crop resolution).
        min_track_side:   Upscale crops to at least this many px before tracking.
        max_frames:       Uniformly subsample both videos to at most this many frames.
        backward:         Enable CoTracker backward tracking pass.
        model:            Pre-loaded CoTracker model; loaded from torch.hub if None.

    Returns dict with keys: score, min_score, max_score, std_score, gt_n_tracks,
    pred_n_tracks, gt_frames, pred_frames, shared_track_shape, motion_threshold,
    grid_size, max_frames.
    """
    if model is None:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model = model.to(device).eval()

    _, gt_video, gt_frames, _ = load_video(Path(gold_path), device, use_bbox=use_bbox,
                                            min_track_side=min_track_side, max_frames=max_frames)
    _, pred_video, pred_frames, _ = load_video(Path(render_path), device, use_bbox=use_bbox,
                                               min_track_side=min_track_side, max_frames=max_frames)

    gt_video, gt_frames, pred_video, pred_frames = match_video_sizes(
        gt_video, gt_frames, pred_video, pred_frames
    )
    _, _, _, H, W = gt_video.shape
    shared_grid = adapt_grid_size(H, W, grid_size)

    def _skip(reason, n_gt=0, n_pred=0):
        return {
            "score": 0.0, "min_score": 0.0, "max_score": 0.0, "std_score": 0.0,
            "gt_n_tracks": n_gt, "pred_n_tracks": n_pred,
            "gt_frames": int(gt_video.shape[1]), "pred_frames": int(pred_video.shape[1]),
            "shared_track_shape": [int(H), int(W)],
            "motion_threshold": motion_threshold, "grid_size": int(shared_grid),
            "max_frames": max_frames, "skip_reason": reason,
            "mask_type": "motion" if use_bbox else "none",
            "query_mode": "first_appearance",
        }

    gt_queries, gt_query_frames = compute_first_appearance_queries(
        gt_frames, motion_threshold, shared_grid, device)
    pred_queries, pred_query_frames = compute_first_appearance_queries(
        pred_frames, motion_threshold, shared_grid, device)

    if gt_queries is None or pred_queries is None:
        missing = "gt" if gt_queries is None else "pred"
        warnings.warn(f"No animated pixels in {missing} video — returning score=0")
        return _skip(f"no_animated_pixels_{missing}")

    try:
        gt_tracks, _   = get_tracklets(model, gt_video,   gt_queries,   backward)
        pred_tracks, _ = get_tracklets(model, pred_video, pred_queries, backward)
        gt_flat   = rearrange(gt_tracks,   "b t n c -> (b n) t c")
        pred_flat = rearrange(pred_tracks, "b t n c -> (b n) t c")

        min_t = min(gt_flat.shape[1], pred_flat.shape[1])
        gt_flat, pred_flat = gt_flat[:, :min_t], pred_flat[:, :min_t]

        T_steps = min_t - 1
        valid_gt   = gt_query_frames   < T_steps
        valid_pred = pred_query_frames < T_steps
        gt_flat,   gt_query_frames   = gt_flat[valid_gt],     gt_query_frames[valid_gt]
        pred_flat, pred_query_frames = pred_flat[valid_pred], pred_query_frames[valid_pred]

        if valid_gt.sum() == 0 or valid_pred.sum() == 0:
            empty = "gt" if valid_gt.sum() == 0 else "pred"
            return _skip(f"no_valid_tracks_{empty}",
                         n_gt=int(gt_flat.shape[0]), n_pred=int(pred_flat.shape[0]))

        sim_matrix = get_similarity_matrix(gt_flat, pred_flat, gt_query_frames, pred_query_frames)
        scores = motion_fidelity_score(sim_matrix)
    except torch.cuda.OutOfMemoryError as e:
        warnings.warn(f"CUDA OOM for {render_path}: {e}")
        result = _skip("oom")
        result["score"] = None
        return result

    return {
        "score":              scores["average_score"],
        "min_score":          scores["min_score"],
        "max_score":          scores["max_score"],
        "std_score":          scores["std_score"],
        "gt_n_tracks":        int(gt_flat.shape[0]),
        "pred_n_tracks":      int(pred_flat.shape[0]),
        "gt_frames":          int(gt_video.shape[1]),
        "pred_frames":        int(pred_video.shape[1]),
        "shared_track_shape": [int(H), int(W)],
        "motion_threshold":   motion_threshold,
        "grid_size":          int(shared_grid),
        "max_frames":         max_frames,
        "mask_type":          "motion" if use_bbox else "none",
        "query_mode":         "first_appearance",
    }
