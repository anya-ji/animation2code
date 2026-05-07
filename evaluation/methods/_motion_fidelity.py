"""
Motion fidelity evaluation using CoTracker3.

Tracks points in GT and predicted videos using first-appearance queries and
measures how well motion directions in the prediction match those in the GT.

Per grid cell, the pixel with the earliest first-appearance time is queried,
handling stroke/draw animations where elements appear progressively.

Similarity: cosine similarity of frame-to-frame displacement directions with
zero-padding before each track's appearance and a union-window denominator,
giving a Chamfer-style bidirectional score.
Score range: [-1, 1] (1 = identical motion, 0 = orthogonal, -1 = opposite).
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


def get_similarity_matrix(
    tracklets1: torch.Tensor, tracklets2: torch.Tensor,
    qt1: np.ndarray, qt2: np.ndarray,
) -> torch.Tensor:
    """Cosine similarity of displacement directions with zero-padding and union-window denominator.

    Displacements before each track's first appearance are zeroed. The
    denominator per pair is T_steps - min(qt1[i], qt2[j]) — the union of both
    active windows — so timing mismatches reduce the score proportionally.
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

    d1 = d1 / d1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    d2 = d2 / d2.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    dots = torch.einsum("itc,jtc->ijt", d1, d2)           # (N1, N2, T_steps)

    union_start = torch.minimum(qt1_t[:, None], qt2_t[None, :])  # (N1, N2)
    count = (T_steps - union_start).clamp(min=1).float()

    return dots.sum(-1) / count                            # (N1, N2)


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
    Compute motion fidelity between a rendered video and a gold reference.

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

    gt_tracks, _   = get_tracklets(model, gt_video,   gt_queries,   backward)
    pred_tracks, _ = get_tracklets(model, pred_video, pred_queries, backward)
    gt_flat   = rearrange(gt_tracks,   "b t n c -> (b n) t c")
    pred_flat = rearrange(pred_tracks, "b t n c -> (b n) t c")

    min_t = min(gt_flat.shape[1], pred_flat.shape[1])
    gt_flat,   pred_flat   = gt_flat[:, :min_t],   pred_flat[:, :min_t]

    # Drop tracks whose first appearance is at or after the last displacement step
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

    return {
        "score":               scores["average_score"],
        "min_score":           scores["min_score"],
        "max_score":           scores["max_score"],
        "std_score":           scores["std_score"],
        "gt_n_tracks":         int(gt_flat.shape[0]),
        "pred_n_tracks":       int(pred_flat.shape[0]),
        "gt_frames":           int(gt_video.shape[1]),
        "pred_frames":         int(pred_video.shape[1]),
        "shared_track_shape":  [int(H), int(W)],
        "motion_threshold":    motion_threshold,
        "grid_size":           int(shared_grid),
        "max_frames":          max_frames,
        "mask_type":           "motion" if use_bbox else "none",
        "query_mode":          "first_appearance",
    }
