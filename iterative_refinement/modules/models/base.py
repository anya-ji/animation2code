"""
Abstract base class for all models used in the iterative refinement pipeline.

Each stage of the pipeline calls a different method:
  - generate_code   : initial code generation from a ground-truth video
  - critique        : compare two videos and produce actionable feedback
  - edit_code       : apply feedback (and optional target video) to update code

Concrete implementations live in gemini.py and qwen3vl.py.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseModel(ABC):
    """Common interface for vision-language models in the pipeline."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def generate_code(
        self,
        video_path: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Watch a single video and generate HTML/CSS/JS code.

        Args:
            video_path:    Path to the ground-truth animation MP4.
            system_prompt: Role/persona instruction for the model.
            user_prompt:   Task instruction appended after the video.

        Returns:
            Raw model response text (may include markdown fences).
        """
        ...

    @abstractmethod
    def critique(
        self,
        target_video: Path,
        generated_video: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Compare a target animation to the current generated attempt.

        Args:
            target_video:    Ground-truth animation MP4.
            generated_video: Current iteration's recorded animation MP4.
            system_prompt:   Role/persona instruction.
            user_prompt:     Task instruction appended after both videos.

        Returns:
            Raw model response text describing visual differences and fixes.
        """
        ...

    @abstractmethod
    def edit_code(
        self,
        current_code: str,
        feedback: str,
        system_prompt: str,
        user_prompt: str,
        target_video: Path | None = None,
    ) -> str:
        """
        Revise code based on critic feedback.

        Args:
            current_code:  The full HTML document from the previous iteration.
            feedback:      Parsed critic text describing what to fix.
            system_prompt: Role/persona instruction.
            user_prompt:   Task instruction (will contain code + feedback).
            target_video:  Optional ground-truth video for reference context.

        Returns:
            Raw model response text containing the revised HTML document.
        """
        ...
