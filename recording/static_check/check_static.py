"""Find static videos in a recording run and list them in static_videos.json.

A video is static only if every consecutive frame pair has a mean absolute pixel difference below
a very small threshold, meaning the frames differ by codec noise only.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

DATASET_DIR = os.environ.get("RUN_DIR", os.path.join(os.path.dirname(__file__), "..", "output", "run1"))
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "static_videos.json")

WORKERS = 8
# Mean absolute pixel difference (0–255 scale) between consecutive frames.
# Truly frozen H.264 video has codec noise of ~0.01; any real animation
# pushes this above 0.05. We use a small buffer above the noise floor.
THRESHOLD = 0.01


def is_static(video_path: str) -> bool:
    """Return True only if every consecutive frame pair is near-identical."""
    cap = cv2.VideoCapture(video_path)

    ok, prev = cap.read()
    if not ok:
        cap.release()
        return False

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.int16)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diff = np.mean(np.abs(gray - prev_gray))

        if diff > THRESHOLD:
            cap.release()
            return False

        prev_gray = gray

    cap.release()
    return True


def check_example(name: str):
    video_path = os.path.join(DATASET_DIR, name, "animation.mp4")

    if not os.path.exists(video_path):
        return name, False

    return name, is_static(video_path)


def main():
    examples = sorted(
        d
        for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
        and os.path.exists(os.path.join(DATASET_DIR, d, "animation.mp4"))
    )

    print(f"Checking {len(examples)} videos with {WORKERS} workers...")

    static = []
    done = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(check_example, name): name for name in examples}

        for future in as_completed(futures):
            name, result = future.result()

            done += 1

            if result:
                static.append(name)
                print(f"  [{done}/{len(examples)}] STATIC: {name}")
            elif done % 100 == 0:
                print(f"  [{done}/{len(examples)}] ...")

    static.sort()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(static, f, indent=2)
        f.write("\n")

    print(f"\nDone. Found {len(static)} static videos -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
