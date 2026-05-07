#!/usr/bin/env python3
"""
Baseline pipeline: inference → record.

Runs model inference (run.py) then records animation.mp4 for each output (record.py).
Both steps use the same model config; recording runs in a screen session by default.

Usage:
    python pipeline.py --config config/gemini.yaml
    python pipeline.py --config config/gemini.yaml --recording-fps 12
    python pipeline.py --config config/gemini.yaml --recording-fps 12 --record-sessions 2
    python pipeline.py --config config/gemini.yaml --output-dir gemini/output/20260318_123456
    python pipeline.py --config config/gemini.yaml --record-only --output-dir gemini/output/20260318_123456
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINES_DIR.parent
DEFAULT_RECORDER = REPO_ROOT / "data_processing" / "video_processing" / "v2" / "record.js"


def run_inference(config: str, output_dir: str | None, extra_args: list[str]) -> str:
    """Run inference and return the output directory path."""
    cmd = [sys.executable, str(BASELINES_DIR / "run.py"), "--config", config]
    if output_dir:
        cmd += ["--output-dir", output_dir]
    cmd += extra_args
    print(f"\n=== Inference ===")
    print(f"  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True, capture_output=False)
    return output_dir


def run_record(
    input_dir: str,
    recorder: str,
    fps: float | None,
    sessions: int,
    concurrency: int,
    rewrite: bool,
    output_name: str | None = None,
) -> None:
    """Launch recording via record.py."""
    cmd = [
        sys.executable, str(BASELINES_DIR / "record.py"),
        "--input", input_dir,
        "--recorder", recorder,
        "--sessions", str(sessions),
        "--concurrency", str(concurrency),
    ]
    if fps is not None:
        cmd += ["--fps", str(fps)]
    if rewrite:
        cmd.append("--rewrite")
    if output_name is not None:
        cmd += ["--output-name", output_name]
    print(f"\n=== Recording ===")
    print(f"  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Baseline pipeline: inference + record")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument(
        "--output-dir",
        help="Use an existing run dir (skips creating a new timestamped one). "
             "Required for --record-only.",
    )
    parser.add_argument(
        "--record-only", action="store_true",
        help="Skip inference, only run recording on an existing --output-dir",
    )
    parser.add_argument(
        "--recording-fps", type=float, default=None,
        help="Recording frame rate passed to record.js (default: record.js default = 30)",
    )
    parser.add_argument(
        "--recorder", default=str(DEFAULT_RECORDER),
        help=f"Path to record.js (default: {DEFAULT_RECORDER})",
    )
    parser.add_argument(
        "--record-sessions", type=int, default=1,
        help="Parallel screen sessions for recording (default: 1 = foreground)",
    )
    parser.add_argument(
        "--record-concurrency", type=int, default=1,
        help="Browser workers per recording session (default: 1)",
    )
    parser.add_argument(
        "--rewrite", action="store_true",
        help="Re-record even if animation.mp4 already exists",
    )
    parser.add_argument(
        "--output-name", default=None,
        help="Output filename per example (default: animation.mp4)",
    )
    # Pass-through args for run.py (inference)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()

    if args.record_only and not args.output_dir:
        parser.error("--record-only requires --output-dir")

    config = str(Path(args.config).resolve())

    import yaml
    with open(config, encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    _base = BASELINES_DIR / _cfg["output"]["base_dir"]

    # ── Pre-generate output dir to avoid concurrent-run race conditions ───
    # Pass it explicitly to run.py so both inference and recording use the
    # exact same dir regardless of what other pipelines create concurrently.
    if not args.record_only and not args.output_dir:
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _suffix = str(_cfg.get("output", {}).get("suffix", "")).strip()
        _run_name = f"{_ts}_{_suffix}" if _suffix else _ts
        args.output_dir = str(_base / _run_name)

    # ── Inference ─────────────────────────────────────────────────────────
    if not args.record_only:
        extra = []
        if args.start is not None:
            extra += ["--start", str(args.start)]
        if args.end is not None:
            extra += ["--end", str(args.end)]
        if args.ids:
            extra += ["--ids"] + args.ids
        run_inference(config, args.output_dir, extra)

    # ── Determine output dir for recording ────────────────────────────────
    output_dir = str(Path(args.output_dir).resolve())
    print(f"Recording from run: {output_dir}")

    # ── Recording ─────────────────────────────────────────────────────────
    run_record(
        input_dir=output_dir,
        recorder=args.recorder,
        fps=args.recording_fps,
        sessions=args.record_sessions,
        concurrency=args.record_concurrency,
        rewrite=args.rewrite,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
