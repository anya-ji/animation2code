import gc
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from shared.parsing import parse_response
from shared.prompts import SYSTEM_PROMPT, USER_PROMPT, USER_PROMPT_IMAGES
from shared.video_sampler import sample_frames


class Qwen3VLModel:
    def __init__(self, config: dict, logger):
        self.logger = logger
        m = config["model"]
        self.model_id = m["model_id"]
        self.max_pixels = m.get("max_pixels", 512 * 384)
        self.max_new_tokens = m.get("max_new_tokens", 8192)
        # qwen_vl_utils clamps to 4 frames by default, which overrides low fps such as 0.5.
        self.min_frames = m.get("min_frames")
        self._model = None
        self._processor = None

        sampling = m.get("image_sampling")
        if sampling:
            self.image_mode = True
            self.img_fps = sampling.get("fps", 2)
            self.img_fmt = sampling.get("format", "JPEG").upper()
            self.img_quality = sampling.get("quality", 90)
        else:
            self.image_mode = False
            self.fps = m.get("fps", 30.0)

    def _load(self):
        self.logger.info("Loading model: %s", self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype="auto", device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self.logger.info("Model loaded. GPUs: %d", torch.cuda.device_count())

    def _generate(self, video_path: Path) -> tuple[str, int]:
        video_ele = {
            "type": "video",
            "video": str(video_path),
            "max_pixels": self.max_pixels,
            "fps": self.fps,
        }
        if self.min_frames is not None:
            video_ele["min_frames"] = self.min_frames
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    video_ele,
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = self._processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            padding=True,
            do_resize=False,
            **video_kwargs,
        ).to(self._model.device)

        input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, input_tokens:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del inputs, output_ids, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return response, input_tokens

    def _generate_images(self, frames) -> tuple[str, int]:
        content = [
            {"type": "image", "image": img, "max_pixels": self.max_pixels}
            for img in frames
        ]
        content.append({"type": "text", "text": USER_PROMPT_IMAGES})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )
        inputs = self._processor(
            text=[text],
            images=images,
            videos=None,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        ).to(self._model.device)

        input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, input_tokens:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del inputs, output_ids, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return response, input_tokens

    def run(self, examples: list[str], get_video_path, run_dir: Path):
        self._load()
        if self.image_mode:
            self.logger.info(
                "Qwen3-VL inference (image mode): img_fps=%s  max_pixels=%s  max_new_tokens=%s",
                self.img_fps, self.max_pixels, self.max_new_tokens,
            )
        else:
            self.logger.info(
                "Qwen3-VL inference: fps=%s  max_pixels=%s  max_new_tokens=%s",
                self.fps, self.max_pixels, self.max_new_tokens,
            )

        ok, err = 0, 0

        pbar = tqdm(examples, desc="inference", unit="ex")
        for i, name in enumerate(pbar):
            video_path = get_video_path(name)
            out_dir = run_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                t0 = time.time()
                if self.image_mode:
                    frames = sample_frames(video_path, self.img_fps)
                    response, input_tokens = self._generate_images(frames)
                else:
                    frames = None
                    response, input_tokens = self._generate(video_path)
                elapsed = time.time() - t0

                (out_dir / "_raw_output.txt").write_text(response, encoding="utf-8")

                full_html, error = parse_response(response)
                frame_info = f"  frames={len(frames)}" if frames is not None else ""
                if full_html is not None:
                    (out_dir / "full.html").write_text(full_html, encoding="utf-8")
                    ok += 1
                    self.logger.info(
                        "[%d/%d] OK    %s  tokens=%d  chars=%d  %.1fs%s",
                        i + 1, len(examples), name, input_tokens, len(response), elapsed, frame_info,
                    )
                else:
                    (out_dir / "_error.txt").write_text(error, encoding="utf-8")
                    err += 1
                    self.logger.warning(
                        "[%d/%d] ERROR %s: %s  %.1fs%s",
                        i + 1, len(examples), name, error, elapsed, frame_info,
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
