"""Measure how similar two animations move, independent of how they look (the paper's temporal metric).

Steps:
  1. Crop each video to its own animated square and downscale the larger crop to match.
  2. In each cell of a grid, pick the pixel that changes first and start a track there, at that frame.
  3. Track the points with CoTracker3.
  4. Score each pair of tracks by comparing their frame-to-frame displacements.
  5. Match tracks in both directions (Chamfer) and average.

Pair score = 0.5 * direction + 0.5 * speed, averaged from the earlier track's start, where:
  direction = cosine similarity of the displacements
  speed = min(|d1|, |d2|) / max(|d1|, |d2|)

Scores range from -0.5 to 1, where 1 is identical motion.
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


def frames_to_tensor(frames: list[np.ndarray], device: str) -> torch.Tensor:
    arr = np.stack(frames)  # (T, H, W, 3)
    return torch.from_numpy(arr).permute(0, 3, 1, 2)[None].float().to(device)


def resize_tensor(video: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
    _, _, _, H, W = video.shape
    if H == new_h and W == new_w:
        return video
    flat = F.interpolate(video[0], size=(new_h, new_w), mode="bilinear", align_corners=False)
    return flat[None]


def upscale_tensor(video: torch.Tensor, min_side: int) -> torch.Tensor:
    _, _, _, H, W = video.shape
    short = min(H, W)
    if short >= min_side:
        return video
    scale = min_side / short
    new_h, new_w = round(H * scale), round(W * scale)
    return resize_tensor(video, new_h, new_w)


def match_video_sizes(
    v1: torch.Tensor, f1: list[np.ndarray],
    v2: torch.Tensor, f2: list[np.ndarray],
) -> tuple[torch.Tensor, list[np.ndarray], torch.Tensor, list[np.ndarray]]:
    """Downscale the larger of two (1,T,C,H,W) videos so both share the same H,W."""
    _, _, _, H1, W1 = v1.shape
    _, _, _, H2, W2 = v2.shape
    if H1 == H2 and W1 == W2:
        return v1, f1, v2, f2
    tH, tW = min(H1, H2), min(W1, W2)
    if H1 != tH or W1 != tW:
        v1 = resize_tensor(v1, tH, tW)
        f1 = [np.array(Image.fromarray(f).resize((tW, tH), Image.BILINEAR)) for f in f1]
    if H2 != tH or W2 != tW:
        v2 = resize_tensor(v2, tH, tW)
        f2 = [np.array(Image.fromarray(f).resize((tW, tH), Image.BILINEAR)) for f in f2]
    return v1, f1, v2, f2


def subsample_frames(frames: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
    """Uniformly subsample to at most max_frames frames."""
    n = len(frames)
    if n <= max_frames:
        return frames
    indices = [round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frames[i] for i in indices]


def load_video(
    video_path: Path, device: str, use_bbox: bool, min_track_side: int,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray], tuple | None]:
    frames = load_frames(str(video_path))
    if max_frames is not None and len(frames) > max_frames:
        frames = subsample_frames(frames, max_frames)
    original = frames_to_tensor(frames, device)
    if use_bbox:
        bbox = find_changing_bbox(frames, square=True)
        track_frames = crop_frames(frames, bbox)
    else:
        bbox = None
        track_frames = frames
    cropped = frames_to_tensor(track_frames, device)
    track_vid = upscale_tensor(cropped, min_track_side)
    _, _, _, H, W = track_vid.shape
    if (track_frames[0].shape[0], track_frames[0].shape[1]) != (H, W):
        track_frames = [
            np.array(Image.fromarray(f).resize((W, H), Image.BILINEAR))
            for f in track_frames
        ]
    return original, track_vid, track_frames, bbox


def adapt_grid_size(H: int, W: int, max_grid_size: int) -> int:
    return max(3, round(max_grid_size * min(H, W) / 512))


def compute_first_appearance_queries(
    frames: list[np.ndarray], threshold: int, grid_size: int, device: str
) -> tuple[torch.Tensor | None, np.ndarray | None]:
    base = frames[0].astype(np.float32)
    H, W = base.shape[:2]
    T = len(frames)
    first_frame = np.full((H, W), T, dtype=np.int32)
    for t in range(1, T):
        diff = np.abs(frames[t].astype(np.float32) - base).max(axis=-1)
        newly = (diff > threshold) & (first_frame == T)
        first_frame[newly] = t
    cell_h = H / grid_size
    cell_w = W / grid_size
    queries_list = []
    for row in range(grid_size):
        for col in range(grid_size):
            y0, y1 = int(row * cell_h), int((row + 1) * cell_h)
            x0, x1 = int(col * cell_w), int((col + 1) * cell_w)
            cell = first_frame[y0:y1, x0:x1]
            if cell.min() >= T:
                continue
            ly, lx = np.unravel_index(np.argmin(cell), cell.shape)
            y, x = y0 + ly, x0 + lx
            queries_list.append([float(first_frame[y, x]), float(x), float(y)])
    if not queries_list:
        return None, None
    q_arr = np.array(queries_list, dtype=np.float32)
    queries = torch.from_numpy(q_arr)[None].to(device)  # (1, N, 3)
    query_frames = q_arr[:, 0].astype(int)              # (N,)
    return queries, query_frames


def get_tracklets(model, video: torch.Tensor, queries: torch.Tensor, backward: bool = False):
    return model(video, queries=queries, backward_tracking=backward)


def motion_fidelity_score(sim: torch.Tensor) -> dict:
    """Chamfer-style bidirectional score from a (N_gt, N_pred) similarity matrix."""
    gt_to_pred, _ = sim.max(dim=1)    # (N_gt,)
    pred_to_gt, _ = sim.max(dim=0)    # (N_pred,)
    average = (gt_to_pred.mean() + pred_to_gt.mean()) / 2
    return {
        "average_score": float(average.item()),
        "min_score":     float(gt_to_pred.min().item()),
        "max_score":     float(gt_to_pred.max().item()),
        "std_score":     float(gt_to_pred.std().item()),
    }


def get_similarity_matrix(
    tracklets1: torch.Tensor, tracklets2: torch.Tensor,
    qt1: np.ndarray, qt2: np.ndarray,
) -> torch.Tensor:
    """Score every pair of tracks as 0.5 * direction + 0.5 * speed.

    Displacements before a track's first appearance count as zero, so timing mismatches lower the score.

    Args:
        tracklets1: (N1, T, 2) gold tracks.
        tracklets2: (N2, T, 2) predicted tracks.
        qt1: (N1,) first-appearance frame of each gold track.
        qt2: (N2,) first-appearance frame of each predicted track.

    Returns:
        (N1, N2) similarity matrix.
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
    """Compute temporal fidelity between a render and a gold video.

    Args:
        render_path: Path to the rendered video.
        gold_path: Path to the gold video.
        device: Torch device.
        motion_threshold: Pixel change that counts as a first appearance.
        grid_size: Grid density at 512 px, scaled with crop size.
        min_track_side: Upscale crops to at least this many pixels before tracking.
        max_frames: Subsample each video to at most this many frames; None uses all frames.
        backward: Also track backward in time.
        use_bbox: Crop each video to its animated square.
        model: Loaded CoTracker model; loaded from torch.hub if None.

    Returns a dict with:
        score: Temporal similarity. None if the GPU ran out of memory.
        min_score, max_score, std_score: Per-track statistics in the gold-to-render direction.
        gt_n_tracks, pred_n_tracks: Tracks in each video.
        gt_frames, pred_frames: Frames in each video.
        shared_track_shape: [height, width] used for tracking.
        motion_threshold, grid_size, max_frames, mask_type, query_mode: Settings used.
        skip_reason: Present when scoring was skipped, e.g. no_animated_pixels_pred (score is 0).
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


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render")
    parser.add_argument("gold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.render, args.gold, device=args.device), indent=2))
