"""
Gemini model implementation using the google-genai SDK.

Videos are passed as raw bytes with explicit VideoMetadata to control the
sampling fps (number of frames Gemini extracts from the video). Lower fps
means fewer tokens and faster/cheaper calls; higher fps captures more motion.

For multi-video calls (critique stage) both clips are included in the same
contents list: [target_part, label, generated_part, label, user_prompt]

Config keys (from config.yaml generator/critic/editor.gemini block):
  model_id          : e.g. "gemini-3-flash-preview"
  fps               : frames per second Gemini samples from each video (e.g. 5)
  temperature       : float
  max_output_tokens : int
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from modules.models.base import BaseModel

load_dotenv()


class GeminiModel(BaseModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model_id = cfg["model_id"]
        self.fps = float(cfg.get("fps", 5))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_output_tokens = int(cfg.get("max_output_tokens", 65536))
        # Client is cheap to create; API key is read from GOOGLE_API_KEY env var.
        self.client = genai.Client()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _gen_cfg(self, system_prompt: str) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_prompt,
            max_output_tokens=self.max_output_tokens,
        )

    def _video_part(self, video_path: Path) -> types.Part:
        """Build a video Part with explicit fps sampling via VideoMetadata."""
        return types.Part(
            inline_data=types.Blob(
                data=video_path.read_bytes(),
                mime_type="video/mp4",
            ),
            video_metadata=types.VideoMetadata(fps=self.fps),
        )

    def _call(self, contents: list, system_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=self._gen_cfg(system_prompt),
        )
        return response.text

    # ── BaseModel interface ───────────────────────────────────────────────

    def generate_code(
        self,
        video_path: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        contents = [self._video_part(video_path), user_prompt]
        return self._call(contents, system_prompt)

    def critique(
        self,
        target_video: Path,
        generated_video: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        contents = [
            self._video_part(target_video),
            "This is the TARGET animation.",
            self._video_part(generated_video),
            "This is the CURRENT attempt.",
            user_prompt,
        ]
        return self._call(contents, system_prompt)

    def edit_code(
        self,
        current_code: str,
        feedback: str,
        system_prompt: str,
        user_prompt: str,
        target_video: Path | None = None,
    ) -> str:
        # Optionally prepend the target video for additional visual context.
        contents = []
        if target_video is not None:
            contents.append(self._video_part(target_video))
            contents.append("This is the TARGET animation you must match.")
        contents.append(user_prompt)
        return self._call(contents, system_prompt)
