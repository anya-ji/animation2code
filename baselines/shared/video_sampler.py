"""Shared utility for sampling image frames from videos.

Used by image-only models (GPT, etc.) that accept images instead of video.
"""
import base64
import io
from pathlib import Path

import av
from PIL import Image


def sample_frames(video_path: str | Path, fps: float) -> list[Image.Image]:
    """Sample frames from a video at the given rate.

    Args:
        video_path: Path to video file.
        fps: Frames per second to sample (e.g. 1, 2, 4, 8).

    Returns:
        List of PIL Images, in temporal order.
    """
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        video_fps = float(stream.average_rate)
        if video_fps <= 0:
            video_fps = 30.0
        step = max(1.0, video_fps / fps)  # native frames between each sample

        next_sample = 0.0
        for i, frame in enumerate(container.decode(video=0)):
            if i >= next_sample:
                frames.append(frame.to_image())
                next_sample += step

    # Always return at least one frame
    if not frames:
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_image())
                break

    return frames


def frames_to_base64(
    frames: list[Image.Image],
    fmt: str = "JPEG",
    quality: int = 90,
) -> list[str]:
    """Encode PIL images as base64 strings.

    Args:
        frames: List of PIL Images.
        fmt: Image format ("JPEG" or "PNG").
        quality: JPEG quality (ignored for PNG).

    Returns:
        List of base64-encoded strings (no data-URI prefix).
    """
    result = []
    for img in frames:
        buf = io.BytesIO()
        save_kwargs = {"format": fmt}
        if fmt.upper() == "JPEG":
            save_kwargs["quality"] = quality
            if img.mode != "RGB":
                img = img.convert("RGB")
        buf.seek(0)
        img.save(buf, **save_kwargs)
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result
