"""
Batch evaluation invoke script.

Discovers matching pairs across result and gold directories, runs one or more
evaluation methods on each pair, and writes results to a file.

Video methods (precision, recall, fullframe, bbox, temporal_fidelity):
  Pair discovery looks for video files in result_dir subdirectories and matches
  them against gold_dir.  DTW alignment is ON by default; use --without-dtw
  to compare frames at matching indices instead.

Code methods (tsed, tsed_tripart):
  Pair discovery looks for full.html in result_dir subdirectories and matches
  against {gold-html-dir}/{name}.html.  Use extract_gold_html.py to build the
  gold HTML directory from metadata.jsonl.  No DTW.

Output formats: csv (default) or json.

Usage:
    # Video metrics
    uv run python evaluation/run_eval.py \\
        --result-dir <path> --gold-dir <path> \\
        --method precision,recall,fullframe,bbox \\
        --outpath results.json --format json

    # Code metric
    uv run python evaluation/run_eval.py \\
        --result-dir <path> --gold-html-dir <path> \\
        --method tsed --outpath results.json --format json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Make the evaluation package importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

import methods.bbox as bbox_method
import methods.fullframe as fullframe_method
import methods.temporal_fidelity as temporal_fidelity_method
import methods.precision as precision_method
import methods.recall as recall_method
import methods.tsed as tsed_method
import methods.tsed_tripart as tsed_tripart_method

AVAILABLE_METHODS = {
    "precision": precision_method.run,
    "recall": recall_method.run,
    "fullframe": fullframe_method.run,
    "bbox": bbox_method.run,
    "temporal_fidelity": temporal_fidelity_method.run,
    "tsed": tsed_method.run,
    "tsed_tripart": tsed_tripart_method.run,
}

# Methods that operate on HTML files rather than videos
HTML_METHODS = {"tsed", "tsed_tripart"}

# CSV columns written for each row
CSV_FIELDNAMES = ["name", "method", "score", "extra_json", "render_path", "gold_path"]


def discover_pairs(
    result_dir: str,
    gold_dir: str,
    gold_filename: str | None = None,
    render_filename: str | None = None,
) -> list[dict]:
    """Find matching video pairs between result_dir and gold_dir.

    Supports two layouts:
      - Nested: result_dir/{name}/animation.mp4  ↔  gold_dir/{name}/animation.mp4
      - Flat:   result_dir/{name}.mp4            ↔  gold_dir/{name}.mp4

    gold_filename: when provided, use this as the sole gold filename candidate
    instead of the default list ("animation.mp4", "original.mp4").
    render_filename: when provided, use this as the sole render filename candidate
    instead of the default list ("animation.mp4", "generated.mp4", "output.mp4").

    Returns a sorted list of dicts with keys: name, render_path, gold_path.
    """
    result_root = Path(result_dir)
    gold_root = Path(gold_dir)
    gold_candidates = (gold_filename,) if gold_filename else ("animation.mp4", "original.mp4")
    render_candidates = (render_filename,) if render_filename else ("animation.mp4", "generated.mp4", "output.mp4")
    pairs = []

    for entry in sorted(result_root.iterdir()):
        if entry.is_dir():
            # Nested layout — try multiple naming conventions
            render_mp4 = None
            for rname in render_candidates:
                candidate = entry / rname
                if candidate.exists():
                    render_mp4 = candidate
                    break
            if render_mp4 is None:
                continue
            gold_mp4 = None
            for gname in gold_candidates:
                candidate = gold_root / entry.name / gname
                if candidate.exists():
                    gold_mp4 = candidate
                    break
            if gold_mp4 is None:
                # Try flat layout in gold dir
                flat = gold_root / f"{entry.name}.mp4"
                if flat.exists():
                    gold_mp4 = flat
            if gold_mp4 is None:
                continue
            name = entry.name
        elif entry.suffix == ".mp4":
            # Flat layout: file is already the video
            render_mp4 = entry
            gold_mp4 = gold_root / entry.name
            name = entry.stem
        else:
            continue

        if render_mp4.exists() and gold_mp4.exists():
            pairs.append(
                {
                    "name": name,
                    "render_path": str(render_mp4),
                    "gold_path": str(gold_mp4),
                }
            )

    return pairs


def discover_html_pairs(result_dir: str, gold_html_dir: str) -> list[dict]:
    """Find matching HTML pairs for code-similarity methods (e.g. tsed).

    Render:  result_dir/{name}/full.html
    Gold:    gold_html_dir/{name}.html

    Returns a sorted list of dicts with keys: name, render_path, gold_path.
    """
    result_root = Path(result_dir)
    gold_root = Path(gold_html_dir)
    pairs = []
    for entry in sorted(result_root.iterdir()):
        if not entry.is_dir():
            continue
        render_html = entry / "full.html"
        if not render_html.exists():
            continue
        gold_html = gold_root / f"{entry.name}.html"
        if not gold_html.exists():
            continue
        pairs.append({
            "name": entry.name,
            "render_path": str(render_html),
            "gold_path": str(gold_html),
        })
    return pairs


def build_record(name: str, method: str, render_path: str, gold_path: str, result: dict) -> dict:
    """Build a unified result record for a single (example, method) pair."""
    return {
        "name": name,
        "method": method,
        "render_path": render_path,
        "gold_path": gold_path,
        **result,
    }


def record_to_csv_row(record: dict) -> dict:
    """Flatten a result record into a single CSV row.

    The primary scalar is 'score' for all methods.
    Everything else goes into extra_json.
    """
    method = record["method"]

    if method == "tsed":
        score = record.get("score")
        extra = {
            "n_render_nodes": record.get("n_render_nodes"),
            "n_gold_nodes": record.get("n_gold_nodes"),
        }
    elif method == "bbox":
        score = record.get("score")
        extra = {
            "per_frame": record.get("per_frame") or record.get("per_frame_dtw"),
            "render_bbox": record.get("render_bbox"),
            "gold_bbox": record.get("gold_bbox"),
            "target_side": record.get("target_side"),
            "n_frames": record.get("n_frames") or record.get("n_paired_frames"),
            "sim": record.get("sim"),
            "is_static": record.get("is_static"),
        }
        if "dtw_cost" in record:
            extra["dtw_cost"] = record["dtw_cost"]
            extra["dtw_path"] = record.get("dtw_path")
            extra["n_render_frames"] = record.get("n_render_frames")
            extra["n_gold_frames"] = record.get("n_gold_frames")
    else:  # precision, recall, fullframe
        score = record.get("score")
        extra = {
            "per_frame": record.get("per_frame") or record.get("per_frame_dtw"),
            "bbox": record.get("bbox"),
            "n_frames": record.get("n_frames") or record.get("n_paired_frames"),
            "sim": record.get("sim"),
            "is_static": record.get("is_static"),
        }
        if "dtw_cost" in record:
            extra["dtw_cost"] = record["dtw_cost"]
            extra["dtw_path"] = record.get("dtw_path")
            extra["n_render_frames"] = record.get("n_render_frames")
            extra["n_gold_frames"] = record.get("n_gold_frames")

    return {
        "name": record["name"],
        "method": method,
        "score": score,
        "extra_json": json.dumps(extra),
        "render_path": record["render_path"],
        "gold_path": record["gold_path"],
    }


def write_csv(outpath: str, records: list[dict], append: bool = False) -> None:
    """Write records to a CSV file."""
    mode = "a" if append and Path(outpath).exists() else "w"
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        for record in records:
            writer.writerow(record_to_csv_row(record))


def write_json(outpath: str, records: list[dict]) -> None:
    """Write records to a JSON file."""
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(records, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation: run methods over result/gold video pairs"
    )
    parser.add_argument("--result-dir", required=True, help="Directory of render outputs")
    parser.add_argument("--gold-dir", default=None, help="Directory of gold videos (video methods)")
    parser.add_argument("--gold-html-dir", default=None,
                        help="Directory of gold HTML files named {name}.html (tsed method). "
                             "Build with: uv run python evaluation/extract_gold_html.py")
    parser.add_argument(
        "--method",
        default="precision,recall,fullframe",
        help="Comma-separated methods: precision, recall, fullframe, bbox, temporal_fidelity, "
             "tsed, tsed_tripart (default: precision,recall,fullframe)",
    )
    parser.add_argument("--outpath", required=True, help="Output file path")
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="(CSV only) Append to existing file instead of overwriting",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate all pairs even if outpath already has results (default: skip already-done)",
    )
    parser.add_argument(
        "--without-dtw",
        action="store_true",
        help=(
            "Disable DTW alignment (enabled by default). "
            "When disabled, frames are compared at matching indices."
        ),
    )
    parser.add_argument(
        "--sim",
        choices=["ssim", "clip", "dreamsim"],
        default="ssim",
        help="Similarity metric for bbox/precision/recall/fullframe (default: ssim)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for CLIP (default: cuda)",
    )
    parser.add_argument("--gold-filename", default=None,
                        help="Override gold video filename in pair discovery "
                             "(e.g. animation_15fps.mp4). Default tries animation.mp4, original.mp4.")
    parser.add_argument("--render-filename", default=None,
                        help="Override render video filename in pair discovery "
                             "(e.g. animation_30fps.mp4). Default tries animation.mp4, generated.mp4, output.mp4.")
    parser.add_argument("--mf-motion-threshold", type=int, default=8,
                        help="Pixel-value threshold for temporal_fidelity motion mask (default: 8)")
    parser.add_argument("--mf-grid-size", type=int, default=30,
                        help="Base NxN grid density for temporal_fidelity (default: 30)")
    parser.add_argument("--mf-max-frames", type=int, default=None,
                        help="Subsample videos to at most N frames for temporal_fidelity.")
    parser.add_argument("--mf-min-track-side", type=int, default=256,
                        help="Min crop side length for temporal_fidelity tracking (default: 256).")
    parser.add_argument("--mf-mask-type", default="motion",
                        choices=["none", "motion"],
                        help="temporal_fidelity mask type (default: motion)")
    args = parser.parse_args()

    methods = [m.strip().lower() for m in args.method.split(",") if m.strip()]
    unknown = [m for m in methods if m not in AVAILABLE_METHODS]
    if unknown:
        print(
            f"Unknown method(s): {unknown}. Choose from: {list(AVAILABLE_METHODS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Separate HTML vs video methods (can't mix in one invocation)
    html_methods = [m for m in methods if m in HTML_METHODS]
    video_methods = [m for m in methods if m not in HTML_METHODS]
    if html_methods and video_methods:
        print("Cannot mix HTML methods (tsed) with video methods in a single run.", file=sys.stderr)
        sys.exit(1)

    if html_methods:
        if not args.gold_html_dir:
            print("--gold-html-dir is required for tsed. "
                  "Run: uv run python evaluation/extract_gold_html.py --help", file=sys.stderr)
            sys.exit(1)
        pairs = discover_html_pairs(args.result_dir, args.gold_html_dir)
        if not pairs:
            print("No matching HTML pairs found.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(pairs)} example(s). Running: {methods}")
    else:
        if not args.gold_dir:
            print("--gold-dir is required for video methods.", file=sys.stderr)
            sys.exit(1)
        pairs = discover_pairs(args.result_dir, args.gold_dir, gold_filename=args.gold_filename,
                               render_filename=args.render_filename)
        if not pairs:
            print("No matching video pairs found.", file=sys.stderr)
            sys.exit(1)
        with_dtw = not args.without_dtw
        print(f"Found {len(pairs)} example(s). Running: {methods}")
        print(f"  DTW: {'disabled' if not with_dtw else f'enabled (criterion: {args.sim})'}")

    # FPS validation — log errors but continue processing
    if not html_methods:
        from methods._utils import check_video_fps
        for pair in pairs:
            for key in ("render_path", "gold_path"):
                if not check_video_fps(pair[key]):
                    import cv2 as _cv2
                    cap = _cv2.VideoCapture(pair[key])
                    actual = cap.get(_cv2.CAP_PROP_FPS)
                    cap.release()
                    print(f"ERROR: expected 30 fps but got {actual:.1f} fps: {pair[key]}",
                          file=sys.stderr)

    # Pre-load CoTracker model once if needed (temporal_fidelity)
    _mf_model = None
    if "temporal_fidelity" in methods:
        import torch
        print("Loading CoTracker3 (offline)...")
        _mf_model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        _mf_model = _mf_model.to(args.device).eval()

    # Pre-load DreamSim once if needed (warm the module-level cache)
    if args.sim == "dreamsim" and not html_methods:
        from methods._utils import _get_dreamsim
        print("Loading DreamSim...")
        _get_dreamsim(args.device)

    from tqdm import tqdm

    # Load existing results to support resume (skip already-done pairs)
    records = []
    done_set: set[tuple[str, str]] = set()
    if not args.overwrite and Path(args.outpath).exists() and args.format == "json":
        with open(args.outpath) as _f:
            records = json.load(_f)
        for r in records:
            done_set.add((r["name"], r["method"]))
        if done_set:
            print(f"Resuming: {len(done_set)} result(s) already done, skipping.")

    for pair in tqdm(pairs, desc=f"[{','.join(methods)}]", unit="pair"):
        name = pair["name"]
        pair_had_new = False
        for method in methods:
            if (name, method) in done_set:
                continue

            tqdm.write(f"  [{method}] {name} ...", end=" ")

            if method == "tsed":
                result = tsed_method.run(pair["render_path"], pair["gold_path"])
            elif method == "temporal_fidelity":
                result = temporal_fidelity_method.run(
                    pair["render_path"],
                    pair["gold_path"],
                    device=args.device,
                    motion_threshold=args.mf_motion_threshold,
                    grid_size=args.mf_grid_size,
                    max_frames=args.mf_max_frames,
                    min_track_side=args.mf_min_track_side,
                    use_bbox=(args.mf_mask_type != "none"),
                    model=_mf_model,
                )
            else:
                result = AVAILABLE_METHODS[method](
                    pair["render_path"],
                    pair["gold_path"],
                    with_dtw=with_dtw,
                    sim=args.sim,
                    device=args.device,
                )

            record = build_record(name, method, pair["render_path"], pair["gold_path"], result)
            records.append(record)
            done_set.add((name, method))
            pair_had_new = True

            scalar = result.get("score")
            skip = result.get("skip_reason")
            suffix = f" [{skip}]" if skip else ""
            tqdm.write(f"{scalar:.4f}{suffix}" if scalar is not None else f"done{suffix}")

        # Write incrementally after each pair that produced new results
        if pair_had_new:
            if args.format == "json":
                write_json(args.outpath, records)
            else:
                write_csv(args.outpath, records, append=False)

    print(f"\nResults written to {args.outpath}")


if __name__ == "__main__":
    main()
