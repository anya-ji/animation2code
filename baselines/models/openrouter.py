import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from shared.parsing import parse_response
from shared.video_sampler import frames_to_base64, sample_frames

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

SYSTEM_PROMPT = """You are an expert web developer specializing in CSS animations and JavaScript.
Your task is to analyze an animation video and generate the HTML/CSS/JavaScript code that recreates it.
Focus on accuracy - match the timing, easing, colors, and movement as closely as possible."""

USER_PROMPT = """Watch this animation carefully (shown as a sequence of frames) and generate the complete code to recreate it.

Provide your response as a single, complete HTML document only.
Include all CSS inside <style> tags and any JavaScript inside <script> tags.
Keep the code concise — avoid generating hundreds of repetitive CSS rules.

Be precise about:
- Animation duration and timing functions
- Colors and gradients
- Transform properties (rotate, scale, translate)
- Keyframe percentages
- Any JavaScript timing or interactions"""

MIME = {"JPEG": "image/jpeg", "PNG": "image/png"}


class OpenRouterModel:
    def __init__(self, config: dict, logger):
        self.logger = logger
        m = config["model"]
        self.model_id = m["model_id"]
        self.max_completion_tokens = m.get("max_completion_tokens", 16384)
        self.temperature = m.get("temperature", 0)

        sampling = m.get("image_sampling", {})
        self.fps = sampling.get("fps", 1)
        self.img_fmt = sampling.get("format", "JPEG").upper()
        self.img_quality = sampling.get("quality", 90)

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def run(self, examples: list[str], get_video_path, run_dir: Path):
        ok, err = 0, 0
        self.logger.info(
            "OpenRouter inference: model=%s  image_fps=%s  max_completion_tokens=%s  temperature=%s",
            self.model_id, self.fps, self.max_completion_tokens, self.temperature,
        )

        pbar = tqdm(examples, desc="inference", unit="ex")
        for name in pbar:
            video_path = get_video_path(name)
            out_dir = run_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                frames = sample_frames(video_path, self.fps)
                b64_frames = frames_to_base64(frames, fmt=self.img_fmt, quality=self.img_quality)
                mime = MIME[self.img_fmt]

                content = [{"type": "text", "text": USER_PROMPT}]
                for b64 in b64_frames:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })

                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    max_completion_tokens=self.max_completion_tokens,
                    temperature=self.temperature,
                )

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                raw_text = choice.message.content or ""

                if finish_reason == "length":
                    self.logger.warning(
                        "CONTEXT_LENGTH %s: output truncated (finish_reason=length)  frames=%d",
                        name, len(frames),
                    )

                (out_dir / "_raw_output.txt").write_text(raw_text, encoding="utf-8")

                full_html, error = parse_response(raw_text)
                if full_html is not None:
                    (out_dir / "full.html").write_text(full_html, encoding="utf-8")
                    ok += 1
                    self.logger.info(
                        "OK    %s  frames=%d  finish_reason=%s",
                        name, len(frames), finish_reason,
                    )
                else:
                    (out_dir / "_error.txt").write_text(error, encoding="utf-8")
                    err += 1
                    self.logger.warning(
                        "ERROR %s: %s  frames=%d  finish_reason=%s",
                        name, error, len(frames), finish_reason,
                    )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                (out_dir / "_error.txt").write_text(error_msg, encoding="utf-8")
                err += 1
                self.logger.error("EXCEPTION %s: %s", name, error_msg)

            pbar.set_postfix(ok=ok, err=err)

        self.logger.info("Complete. OK: %d  Errors: %d", ok, err)
        print(f"\nDone.  OK: {ok}  Errors: {err}")
