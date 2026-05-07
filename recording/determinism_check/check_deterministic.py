#!/usr/bin/env python3
"""
Compare determinism runs across output/run1/ and output/run2/.

Checks:
  1. Byte-level SHA256 hash (fast)
  2. Frame-by-frame pixel comparison via ffmpeg (if hashes differ)

Usage:
    uv run python check_deterministic.py --runs ../output/run1 ../output/run2 --out comparison.csv
"""
import hashlib
import json
import subprocess
import csv
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent

# Videos with PSNR >= this threshold are considered perceptually identical
# (sub-pixel rendering noise). Empirically: CSS/GSAP sub-pixel noise ≥ 61 dB,
# meaningful visual differences ≤ 56 dB.
PSNR_THRESHOLD = 40.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_frames(video: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), str(out_dir / "frame_%05d.png")],
        check=True, capture_output=True,
    )
    return sorted(out_dir.glob("frame_*.png"))


def compare_frames(frames1, frames2):
    if len(frames1) != len(frames2):
        return {"frame_count_mismatch": (len(frames1), len(frames2))}

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("  numpy/PIL not available; using file-hash comparison per frame")
        diffs = sum(1 for a, b in zip(frames1, frames2) if sha256(a) != sha256(b))
        return {"different_frames": diffs, "total_frames": len(frames1), "method": "hash"}

    max_diff = 0.0
    mean_diff = 0.0
    total_mse = 0.0
    different_frames = 0
    for a, b in zip(frames1, frames2):
        ia = np.array(Image.open(a)).astype(float)
        ib = np.array(Image.open(b)).astype(float)
        d = np.abs(ia - ib)
        frame_max = d.max()
        frame_mean = d.mean()
        if frame_max > 0:
            different_frames += 1
        max_diff = max(max_diff, frame_max)
        mean_diff += frame_mean
        total_mse += np.mean(d ** 2)

    mean_diff /= len(frames1)
    mse = total_mse / len(frames1)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
    return {
        "total_frames": len(frames1),
        "different_frames": different_frames,
        "max_pixel_diff": max_diff,
        "mean_pixel_diff": mean_diff,
        "psnr": psnr,
        "method": "pixel",
    }

def compare_two(v1: Path, v2: Path) -> Tuple[bool, str, dict]:
    h1 = sha256(v1)
    h2 = sha256(v2)
    if h1 == h2:
        return True, "sha256", {}

    tmp = Path(tempfile.mkdtemp())
    try:
        d1, d2 = tmp / "v1", tmp / "v2"
        frames1 = extract_frames(v1, d1)
        frames2 = extract_frames(v2, d2)
        stats = compare_frames(frames1, frames2)
        if stats.get("different_frames", 1) == 0:
            return True, "pixel", {}
        if stats.get("psnr", 0) >= PSNR_THRESHOLD:
            return True, "psnr", {"hash1": h1, "hash2": h2, **stats}
        return False, "pixel", {"hash1": h1, "hash2": h2, **stats}
    finally:
        shutil.rmtree(tmp)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Compare determinism runs.")
    parser.add_argument(
        "--runs",
        nargs=2,
        metavar=("RUN1", "RUN2"),
        default=[REPO_ROOT / "output" / "run1", REPO_ROOT / "output" / "run2"],
        help="Paths to run directories (each contains */animation.mp4).",
    )
    parser.add_argument(
        "--out",
        default=SCRIPT_DIR / "determinism_comparison.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--out-json",
        default=SCRIPT_DIR / "nondeterministic.json",
        help="JSON output path for non-identical example names.",
    )
    return parser.parse_args()


def list_examples(run_dir: Path):
    if not run_dir.exists():
        return set()
    return {p.parent.name for p in run_dir.glob("*/animation.mp4")}


def resolve_video(run_dir: Path, example: str) -> Optional[Path]:
    path = run_dir / example / "animation.mp4"
    return path if path.exists() else None


def main():
    args = parse_args()
    run1, run2 = [Path(p) for p in args.runs]
    out_path = Path(args.out)
    out_json = Path(args.out_json)

    for run_dir in (run1, run2):
        if not run_dir.exists():
            print(f"ERROR: {run_dir} not found.")
            sys.exit(1)

    examples = sorted(list_examples(run1) | list_examples(run2))
    if not examples:
        print("No examples found. Expecting */animation.mp4 in each run.")
        sys.exit(1)

    rows = []
    for example in tqdm(examples, desc="Comparing videos", unit="example"):
        v1 = resolve_video(run1, example)
        v2 = resolve_video(run2, example)
        if not v1 or not v2:
            rows.append(
                {
                    "example": example,
                    "identical": "false",
                    "method": "missing",
                    "run1": str(v1) if v1 else "",
                    "run2": str(v2) if v2 else "",
                    "hash1": "", "hash2": "",
                    "total_frames": "", "different_frames": "",
                    "max_pixel_diff": "", "mean_pixel_diff": "",
                }
            )
            continue

        identical, method, diff_stats = compare_two(v1, v2)
        row = {
            "example": example,
            "identical": "true" if identical else "false",
            "method": method,
            "run1": str(v1),
            "run2": str(v2),
            "hash1": diff_stats.get("hash1", ""),
            "hash2": diff_stats.get("hash2", ""),
            "total_frames": diff_stats.get("total_frames", ""),
            "different_frames": diff_stats.get("different_frames", ""),
            "max_pixel_diff": diff_stats.get("max_pixel_diff", ""),
            "mean_pixel_diff": diff_stats.get("mean_pixel_diff", ""),
            "psnr": diff_stats.get("psnr", ""),
        }
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "example", "identical", "method", "run1", "run2",
        "hash1", "hash2", "total_frames", "different_frames",
        "max_pixel_diff", "mean_pixel_diff", "psnr",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

    non_identical = sorted(r["example"] for r in rows if r["identical"] == "false")
    with open(out_json, "w") as f:
        json.dump(non_identical, f, indent=2)
    print(f"Wrote {len(non_identical)} non-identical examples to {out_json}")


if __name__ == "__main__":
    main()
