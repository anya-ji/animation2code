import io
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from shared.parsing import parse_response
from shared.video_sampler import sample_frames

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

SYSTEM_PROMPT = """You are an expert web developer specializing in CSS animations and JavaScript.
Your task is to analyze an animation video and generate the HTML/CSS/JavaScript code that recreates it.
Focus on accuracy - match the timing, easing, colors, and movement as closely as possible."""

USER_PROMPT = """Watch this animation video carefully and generate the complete code to recreate it.

Provide your response as a single, complete HTML document only.
Include all CSS inside <style> tags and any JavaScript inside <script> tags.
Keep the code concise — avoid generating hundreds of repetitive CSS rules.

Be precise about:
- Animation duration and timing functions
- Colors and gradients
- Transform properties (rotate, scale, translate)
- Keyframe percentages
- Any JavaScript timing or interactions"""

USER_PROMPT_IMAGES = """Watch this animation carefully (shown as a sequence of frames) and generate the complete code to recreate it.

Provide your response as a single, complete HTML document only.
Include all CSS inside <style> tags and any JavaScript inside <script> tags.
Keep the code concise — avoid generating hundreds of repetitive CSS rules.

Be precise about:
- Animation duration and timing functions
- Colors and gradients
- Transform properties (rotate, scale, translate)
- Keyframe percentages
- Any JavaScript timing or interactions"""


class GeminiModel:
    def __init__(self, config: dict, logger):
        self.logger = logger
        m = config["model"]
        self.model_id = m["model_id"]
        self.max_output_tokens = m.get("max_output_tokens", 8192)
        media_res_str = m.get("media_resolution", "MEDIA_RESOLUTION_MEDIUM")
        self.media_resolution = getattr(types.MediaResolution, media_res_str)
        self.client = genai.Client()

        sampling = m.get("image_sampling")
        if sampling:
            self.image_mode = True
            self.img_fps = sampling.get("fps", 2)
            self.img_fmt = sampling.get("format", "JPEG").upper()
            self.img_quality = sampling.get("quality", 90)
        else:
            self.image_mode = False
            self.fps = m.get("fps", 5)

    def run(self, examples: list[str], get_video_path, run_dir: Path):
        ok, err = 0, 0
        if self.image_mode:
            self.logger.info(
                "Gemini inference (image mode): model=%s  img_fps=%s  max_tokens=%s",
                self.model_id, self.img_fps, self.max_output_tokens,
            )
        else:
            self.logger.info(
                "Gemini inference: model=%s  fps=%s  max_tokens=%s",
                self.model_id, self.fps, self.max_output_tokens,
            )

        pbar = tqdm(examples, desc="inference", unit="ex")
        for name in pbar:
            video_path = get_video_path(name)
            out_dir = run_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                if self.image_mode:
                    frames = sample_frames(video_path, self.img_fps)
                    parts = []
                    for frame in frames:
                        buf = io.BytesIO()
                        frame.save(buf, format=self.img_fmt, quality=self.img_quality)
                        parts.append(types.Part(
                            inline_data=types.Blob(data=buf.getvalue(), mime_type="image/jpeg")
                        ))
                    parts.append(USER_PROMPT_IMAGES)
                    frame_info = f"  frames={len(frames)}"
                else:
                    video_bytes = Path(video_path).read_bytes()
                    parts = [
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                            video_metadata=types.VideoMetadata(fps=self.fps),
                            media_resolution=self.media_resolution,
                        ),
                        USER_PROMPT,
                    ]
                    frame_info = ""

                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )

                finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                raw_text = response.text
                (out_dir / "_raw_output.txt").write_text(raw_text, encoding="utf-8")

                full_html, error = parse_response(raw_text)
                if full_html is not None:
                    (out_dir / "full.html").write_text(full_html, encoding="utf-8")
                    ok += 1
                    self.logger.info("OK    %s  finish_reason=%s%s", name, finish_reason, frame_info)
                else:
                    (out_dir / "_error.txt").write_text(error, encoding="utf-8")
                    err += 1
                    self.logger.warning("ERROR %s: %s  finish_reason=%s%s", name, error, finish_reason, frame_info)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                (out_dir / "_error.txt").write_text(error_msg, encoding="utf-8")
                err += 1
                self.logger.error("EXCEPTION %s: %s", name, error_msg)

            pbar.set_postfix(ok=ok, err=err)

        self.logger.info("Complete. OK: %d  Errors: %d", ok, err)
        print(f"\nDone.  OK: {ok}  Errors: {err}")
