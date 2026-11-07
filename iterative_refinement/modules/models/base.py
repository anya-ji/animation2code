"""Interface for the models used by the pipeline, implemented in gemini.py and qwen3vl.py.

Each stage calls one method:
  generate_code: Generate code from a gold video.
  critique: Compare two videos and produce feedback.
  edit_code: Apply feedback to the code.
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
        """Watch a video and generate HTML/CSS/JS code.

        Args:
            video_path: Path to the gold video.
            system_prompt: Role instruction for the model.
            user_prompt: Task instruction, placed after the video.

        Returns:
            Raw response text, which may include markdown fences.
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
        """Compare the target animation with the current attempt.

        Args:
            target_video: Path to the gold video.
            generated_video: Path to the current iteration's video.
            system_prompt: Role instruction for the model.
            user_prompt: Task instruction, placed after both videos.

        Returns:
            Raw response text describing the differences and fixes.
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
        """Revise code based on critic feedback.

        Args:
            current_code: HTML document from the previous iteration.
            feedback: Parsed critic text describing what to fix.
            system_prompt: Role instruction for the model.
            user_prompt: Task instruction, containing the code and feedback.
            target_video: Optional gold video for reference.

        Returns:
            Raw response text containing the revised HTML document.
        """
        ...
