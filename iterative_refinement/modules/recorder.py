"""Python wrapper that runs recorder.js as a subprocess to record an HTML file to MP4."""

import subprocess
from pathlib import Path

RECORDER_JS = Path(__file__).parent.parent / "recorder.js"


class Recorder:
    def __init__(self, cfg: dict):
        """Read recording settings.

        Args:
            cfg: The config's recorder block, with fps, duration, width, height and optional skip.
        """
        self.fps = int(cfg.get("fps", 12))
        self.duration = float(cfg.get("duration", 10))
        self.width = int(cfg.get("width", 1024))
        self.height = int(cfg.get("height", 768))
        self.skip = bool(cfg.get("skip", False))

    def record(self, html_path: Path, output_path: Path, timeout: int = 300) -> None:
        """Record an HTML file to MP4.

        Args:
            html_path: Absolute path to the HTML file.
            output_path: Absolute path for the MP4.
            timeout: Subprocess timeout in seconds.

        Raises:
            RuntimeError: If recorder.js fails.
        """
        if self.skip:
            return

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
