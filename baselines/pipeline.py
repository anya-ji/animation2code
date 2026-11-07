#!/usr/bin/env python3
"""Run inference (run.py), then record each output to animation.mp4 (record.py), with one config."""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINES_DIR.parent
DEFAULT_RECORDER = REPO_ROOT / "recording" / "record.js"


def run_inference(config: str, output_dir: str | None, extra_args: list[str]) -> None:
    """Run run.py for a config, writing to output_dir."""
    cmd = [sys.executable, str(BASELINES_DIR / "run.py"), "--config", config]
    if output_dir:
        cmd += ["--output-dir", output_dir]
    cmd += extra_args
    print(f"\n=== Inference ===")
    print(f"  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def run_record(
    input_dir: str,
    recorder: str,
    fps: float | None,
    sessions: int,
    concurrency: int,
    rewrite: bool,
    output_name: str | None = None,
) -> None:
    """Run record.py on a finished run.

    Args:
        input_dir: Run directory with <id>/full.html.
        recorder: Path to record.js.
        fps: Recording frame rate; None uses the recorder's default of 30.
        sessions: Number of parallel screen sessions.
        concurrency: Browser workers per session.
        rewrite: Re-record finished examples.
        output_name: Output filename per example.
    """
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

    # Fix the run dir up front so concurrent pipelines cannot mix up their outputs.
    if not args.record_only and not args.output_dir:
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _suffix = str(_cfg.get("output", {}).get("suffix", "")).strip()
        _run_name = f"{_ts}_{_suffix}" if _suffix else _ts
        args.output_dir = str(_base / _run_name)

    if not args.record_only:
        extra = []
        if args.start is not None:
            extra += ["--start", str(args.start)]
        if args.end is not None:
            extra += ["--end", str(args.end)]
        if args.ids:
            extra += ["--ids"] + args.ids
        run_inference(config, args.output_dir, extra)

    output_dir = str(Path(args.output_dir).resolve())
    print(f"Recording from run: {output_dir}")

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
