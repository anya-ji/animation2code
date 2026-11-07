"""Align two videos in time with dynamic time warping (DTW).

Finds the lowest total cost of matching every frame in one video to a frame in the other,
allowing the time axis to stretch. Frame cost is the mean absolute grayscale difference, in [0, 1].

    DTW[i, j] = cost(i, j) + min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from methods._utils import load_frames, to_grayscale


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean absolute pixel difference between two grayscale frames, in [0, 1].

    Frames are resized to the smaller of the two sizes, so videos of different resolutions can be compared.

    Args:
        a, b: Grayscale float32 frames.
    """
    if a.shape != b.shape:
        th = min(a.shape[0], b.shape[0])
        tw = min(a.shape[1], b.shape[1])
        a = a[:th, :tw]
        b = b[:th, :tw]
    return float(np.mean(np.abs(a - b)))


def build_cost_matrix(render_frames: list[np.ndarray], gold_frames: list[np.ndarray]) -> np.ndarray:
    """Build the pairwise distance matrix, where D[i, j] = dist(render[i], gold[j]).

    Args:
        render_frames: Grayscale render frames.
        gold_frames: Grayscale gold frames.

    Returns:
        n×m float32 array.
    """
    n = len(render_frames)
    m = len(gold_frames)

    render_gray = [to_grayscale(f) for f in render_frames]
    gold_gray = [to_grayscale(f) for f in gold_frames]

    D = np.empty((n, m), dtype=np.float32)
    for i, rf in enumerate(render_gray):
        for j, gf in enumerate(gold_gray):
            D[i, j] = frame_distance(rf, gf)
    return D


def dtw(D: np.ndarray) -> tuple[float, list[list[int]], list[list[float]]]:
    """Run DTW on a precomputed n×m distance matrix.

    Args:
        D: Pairwise distance matrix.

    Returns:
        alignment_cost: Total DTW distance.
        path: 0-indexed [i, j] pairs from [0, 0] to [n-1, m-1].
        cost_matrix: Accumulated cost matrix as a nested list.
    """
    n, m = D.shape

    C = np.full((n, m), np.inf, dtype=np.float64)
    C[0, 0] = D[0, 0]

    for j in range(1, m):
        C[0, j] = D[0, j] + C[0, j - 1]

    for i in range(1, n):
        C[i, 0] = D[i, 0] + C[i - 1, 0]

    for i in range(1, n):
        for j in range(1, m):
            C[i, j] = D[i, j] + min(
                C[i - 1, j],      # insertion
                C[i, j - 1],      # deletion
                C[i - 1, j - 1],  # match
            )

    alignment_cost = float(C[n - 1, m - 1])

    # Traceback from (n-1, m-1) to (0, 0) to recover the optimal path
    path = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        path.append([i, j])
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Move to the predecessor with the lowest accumulated cost
            best = min((C[i - 1, j - 1], i - 1, j - 1),
                       (C[i - 1, j],     i - 1, j),
                       (C[i, j - 1],     i,     j - 1))
            i, j = best[1], best[2]
    path.append([0, 0])
    path.reverse()

    return alignment_cost, path, C.tolist()


def run(render_path: str, gold_path: str) -> dict:
    """Compute the DTW alignment between a render and a gold video.

    Args:
        render_path: Path to the rendered video.
        gold_path: Path to the gold video.

    Returns a dict with:
        alignment_cost: Total DTW cost.
        path: Matched [i, j] frame index pairs.
        cost_matrix: n×m accumulated cost matrix.
        n_render_frames, n_gold_frames: Frames in each video.
    """
    render_frames = load_frames(render_path)
    gold_frames = load_frames(gold_path)

    D = build_cost_matrix(render_frames, gold_frames)
    alignment_cost, path, cost_matrix = dtw(D)

    return {
        "alignment_cost": alignment_cost,
        "path": path,
        "cost_matrix": cost_matrix,
        "n_render_frames": len(render_frames),
        "n_gold_frames": len(gold_frames),
    }
