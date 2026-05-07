import gc
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from shared.parsing import parse_response

# USER_PROMPT = """Watch this animation video and write a single complete HTML document that recreates it exactly.

# The HTML document must:
# - Include all CSS inside <style> tags
# - Include any JavaScript inside <script> tags
# - Start with <!DOCTYPE html> and contain a full <html>...</html> structure

# Match the animation as closely as possible: colors, timing, easing, movement, and layout.
# Output only the HTML document, nothing else."""
USER_PROMPT = """Watch this animation video and write a single complete HTML document that recreates it exactly."""


class MolmoModel:
    def __init__(self, config: dict, logger):
        self.logger = logger
        m = config["model"]
        self.model_id = m["model_id"]
        self.fps = m.get("fps", 2.0)
        self.max_new_tokens = m.get("max_new_tokens", 16384)
        self._model = None
        self._processor = None

    def _load(self):
        self.logger.info("Loading model: %s", self.model_id)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        )
        self.logger.info("Model loaded. GPUs: %d", torch.cuda.device_count())

    def _generate(self, video_path: Path) -> tuple[str, int]:
        # Molmo's apply_chat_template (transformers 4.57.1) iterates over all
        # message content looking for images/videos, and breaks on plain-string
        # system content. Prepend the system prompt to the user text instead.
        messages = [
            {
                "role": "user",
                "content": [
                    dict(type="text", text=USER_PROMPT),
                    dict(type="video", video=str(video_path), max_fps=self.fps),
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_tokens = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_tokens = generated_ids[0, input_tokens:]
        response = self._processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        del inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return response, input_tokens

    def run(self, examples: list[str], get_video_path, run_dir: Path):
        self._load()
        self.logger.info(
            "Molmo inference: model=%s  fps=%s  max_new_tokens=%s",
            self.model_id, self.fps, self.max_new_tokens,
        )

        ok, err = 0, 0
        pbar = tqdm(examples, desc="inference", unit="ex")
        for i, name in enumerate(pbar):
            video_path = get_video_path(name)
            out_dir = run_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                t0 = time.time()
                response, input_tokens = self._generate(video_path)
                elapsed = time.time() - t0

                (out_dir / "_raw_output.txt").write_text(response, encoding="utf-8")

                full_html, error = parse_response(response)
                if full_html is not None:
                    (out_dir / "full.html").write_text(full_html, encoding="utf-8")
                    ok += 1
                    self.logger.info(
                        "[%d/%d] OK    %s  tokens=%d  chars=%d  %.1fs",
                        i + 1, len(examples), name, input_tokens, len(response), elapsed,
                    )
                else:
                    (out_dir / "_error.txt").write_text(error, encoding="utf-8")
                    err += 1
                    self.logger.warning(
                        "[%d/%d] ERROR %s: %s  %.1fs",
                        i + 1, len(examples), name, error, elapsed,
                    )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                (out_dir / "_error.txt").write_text(error_msg, encoding="utf-8")
                err += 1
                self.logger.exception("[%d/%d] EXCEPTION %s: %s", i + 1, len(examples), name, e)
                gc.collect()
                torch.cuda.empty_cache()

            pbar.set_postfix(ok=ok, err=err)

        self.logger.info("Complete. OK: %d  Errors: %d", ok, err)
        print(f"\nDone.  OK: {ok}  Errors: {err}")
