"""
METHOD3: Dynamic Time Warping (DTW)

Aligns two video sequences in time using DTW, measuring the minimum
accumulated cost of matching every frame in one video to a frame in
the other, allowing non-linear warping of the time axis.

Frame distance: mean absolute difference of grayscale pixel values,
normalised to [0, 1].

Algorithm (follows the standard DTW recurrence):
    DTW[0, 0] = 0
    DTW[i, j] = cost(i, j) + min(DTW[i-1, j],    # insertion
                                  DTW[i, j-1],    # deletion
                                  DTW[i-1, j-1])  # match

Outputs:
    alignment_cost  – scalar total alignment cost (DTW[n, m])
    path            – list of [i, j] frame index pairs tracing the
                      optimal alignment from start to end
    cost_matrix     – full n×m accumulated cost matrix (2-D list)

Usage (standalone):
    uv run python evaluation/methods/dtw.py render.mp4 gold.mp4
"""

import sys
from pathlib import Path

import numpy as np

# Ensure evaluation/ is on sys.path so 'methods' is importable when run standalone
sys.path.insert(0, str(Path(__file__).parent.parent))
from methods._utils import load_frames, to_grayscale


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference between two grayscale float32 frames.

    Both inputs are resized to the smaller of the two spatial dimensions
    so that videos recorded at different resolutions can still be compared.

    Returns a value in [0, 1].
    """
    # Resize b to match a's shape if they differ
    if a.shape != b.shape:
        # Simple nearest-neighbour downscale to the smaller of the two
        th = min(a.shape[0], b.shape[0])
        tw = min(a.shape[1], b.shape[1])
        a = a[:th, :tw]
        b = b[:th, :tw]
    return float(np.mean(np.abs(a - b)))


def build_cost_matrix(render_frames: list[np.ndarray], gold_frames: list[np.ndarray]) -> np.ndarray:
    """Build the pairwise frame-distance matrix D where D[i,j] = dist(render[i], gold[j]).

    Returns an n×m float32 array.
    """
    n = len(render_frames)
    m = len(gold_frames)

    # Pre-convert all frames to grayscale once to avoid redundant work
    render_gray = [to_grayscale(f) for f in render_frames]
    gold_gray = [to_grayscale(f) for f in gold_frames]

    D = np.empty((n, m), dtype=np.float32)
    for i, rf in enumerate(render_gray):
        for j, gf in enumerate(gold_gray):
            D[i, j] = frame_distance(rf, gf)
    return D


def dtw(D: np.ndarray) -> tuple[float, list[list[int]], list[list[float]]]:
    """Run DTW on a precomputed pairwise distance matrix D (n×m).

    Returns:
        alignment_cost  – scalar DTW distance
        path            – list of [i, j] pairs (0-indexed) from [0,0] to [n-1, m-1]
        cost_matrix     – accumulated cost matrix as a 2-D Python list
    """
    n, m = D.shape

    # Accumulated cost matrix; use float64 for numerical stability
    C = np.full((n, m), np.inf, dtype=np.float64)
    C[0, 0] = D[0, 0]

    # Fill first row (only deletion moves available)
    for j in range(1, m):
        C[0, j] = D[0, j] + C[0, j - 1]

    # Fill first column (only insertion moves available)
    for i in range(1, n):
        C[i, 0] = D[i, 0] + C[i - 1, 0]

    # Fill the rest of the matrix
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
    """Compute DTW alignment between render and gold videos.

    Args:
        render_path: path to the candidate/render video.
        gold_path:   path to the gold/reference video.

    Returns:
        dict with keys:
          alignment_cost  – scalar total DTW cost
          path            – list of [i, j] matched frame index pairs
          cost_matrix     – 2-D accumulated cost matrix (n×m)
          n_render_frames – number of frames in the render video
          n_gold_frames   – number of frames in the gold video
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


