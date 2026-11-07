#!/usr/bin/env python3
"""Record animation.mp4 for each full.html in a run directory.

Wraps recording/record.js. With --sessions > 1 it starts parallel screen sessions;
otherwise it runs in the foreground.
"""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Record animation.mp4 for model outputs")
    parser.add_argument(
        "--input", required=True,
        help="Model run dir containing per-example subdirs with full.html",
    )
    parser.add_argument(
        "--recorder", default=str(Path(__file__).resolve().parent.parent / "recording" / "record.js"),
        help="Path to record.js",
    )
    parser.add_argument(
        "--sessions", type=int, default=1,
        help="Parallel screen sessions (default: 1 = run in foreground)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Browser workers per session (default: 1)",
    )
    parser.add_argument(
        "--rewrite", action="store_true",
        help="Re-record examples that already have animation.mp4",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Recording frame rate (default: record.js CONFIG.fps = 30)",
    )
    parser.add_argument(
        "--output-name", default=None,
        help="Output filename per example (default: animation.mp4)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    recorder = Path(args.recorder).resolve()

    if not input_dir.exists():
        print(f"Error: input dir not found: {input_dir}")
        raise SystemExit(1)
    if not recorder.exists():
        print(f"Error: recorder not found: {recorder}")
        raise SystemExit(1)

    examples = sorted(
        d.name for d in input_dir.iterdir()
        if d.is_dir() and (d / "full.html").exists()
    )
    total = len(examples)
    if total == 0:
        print("No examples with full.html found in input dir.")
        raise SystemExit(0)

    n = min(args.sessions, total)
    chunk = (total + n - 1) // n  # ceiling division
    node_modules = recorder.parent / "node_modules"

    print(f"Input:       {input_dir}")
    print(f"Recorder:    {recorder}")
    print(f"Examples:    {total}")
    print(f"Sessions:    {n}  (~{chunk} each)")
    print(f"Concurrency: {args.concurrency} per session")
    print()

    for i in range(n):
        start = i * chunk
        end = min(start + chunk, total)
        if start >= total:
            break

        cmd_parts = [
            f"NODE_PATH={node_modules}",
            f"node {recorder}",
            f"--html-dir {input_dir}",
            f"--start {start}",
            f"--end {end}",
            f"--session {i}",
            f"--concurrency {args.concurrency}",
        ]
        cmd_parts.append("--js-inject")
        if args.rewrite:
            cmd_parts.append("--rewrite")
        if args.fps is not None:
            cmd_parts.append(f"--fps {args.fps}")
        if args.output_name is not None:
            cmd_parts.append(f"--output-name {args.output_name}")
        cmd = " ".join(cmd_parts)

        if n == 1:
            subprocess.run(cmd, shell=True, check=True)
        else:
            session_name = f"record_{i}"
            screen_cmd = f"screen -dmS {session_name} bash -c '{cmd}'"
            subprocess.run(screen_cmd, shell=True, check=True)
            print(f"  [{i}] examples [{start}:{end}]  →  screen '{session_name}'")

    if n > 1:
        print("All sessions launched.")
        print("  Monitor:  screen -ls")
        print("  Attach:   screen -r record_0")


if __name__ == "__main__":
    main()
