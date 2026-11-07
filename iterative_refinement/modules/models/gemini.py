"""Gemini model using the google-genai SDK.

Videos are sent as bytes with an explicit sampling rate: lower fps uses fewer tokens, and higher fps
captures more motion. Critique calls send [target, label, generated, label, prompt].

Config keys:
  model_id: e.g. "gemini-3-flash-preview"
  fps: Frames per second sampled from each video
  temperature: Sampling temperature
  max_output_tokens: Maximum tokens to generate
"""

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
        self.client = genai.Client()

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
        contents = []
        if target_video is not None:
            contents.append(self._video_part(target_video))
            contents.append("This is the TARGET animation you must match.")
        contents.append(user_prompt)
        return self._call(contents, system_prompt)
