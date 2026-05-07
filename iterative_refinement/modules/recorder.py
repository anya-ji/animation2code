"""
Python wrapper around recorder.js.

Spawns `node recorder.js` as a subprocess to record a given HTML file to MP4.
recorder.js must be in the parent directory (experiments/iterative_refinement/).

On first use, run.py ensures that a node_modules symlink exists pointing to
data_processing/video_processing/v3/node_modules.
"""

import subprocess
from pathlib import Path

# recorder.js sits one level up from this file (in iterative_refinement/)
RECORDER_JS = Path(__file__).parent.parent / "recorder.js"


class Recorder:
    def __init__(self, cfg: dict):
        """
        Args:
            cfg: The recorder block from config.yaml with keys:
                 fps, duration, width, height
        """
        self.fps = int(cfg.get("fps", 12))
        self.duration = float(cfg.get("duration", 10))
        self.width = int(cfg.get("width", 1024))
        self.height = int(cfg.get("height", 768))
        self.skip = bool(cfg.get("skip", False))

    def record(self, html_path: Path, output_path: Path, timeout: int = 300) -> None:
        """
        Record html_path to output_path as an MP4.

        Args:
            html_path:   Absolute path to the HTML file to record.
            output_path: Absolute path where the MP4 will be written.
            timeout:     Subprocess timeout in seconds.

        Raises:
            RuntimeError: If recorder.js exits with a non-zero code.
        """
        if self.skip:
            return  # Skip recording (will be done locally later)

        if not RECORDER_JS.exists():
            raise FileNotFoundError(f"recorder.js not found at {RECORDER_JS}")

        cmd = [
            "node", str(RECORDER_JS),
            "--html",     str(html_path),
            "--output",   str(output_path),
            "--fps",      str(self.fps),
            "--duration", str(self.duration),
            "--width",    str(self.width),
            "--height",   str(self.height),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            raise RuntimeError(
                f"recorder.js failed (exit {result.returncode})\n"
                f"stdout: {stdout}\nstderr: {stderr}"
            )
