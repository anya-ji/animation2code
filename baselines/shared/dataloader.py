import json
from pathlib import Path


def load_examples(config: dict) -> tuple[list[str], object]:
    """Return (example_names_sorted, get_video_path_fn).

    Config keys:
        local_dataset  — path to local dir (auto-detects format)
        hf_dataset     — HuggingFace repo id (fallback)
        split          — "test" or "train" (default: "test")
    """
    if config.get("local_dataset"):
        p = Path(config["local_dataset"]).expanduser()
        if not p.is_absolute():
            # resolve relative to baselines/ (parent of this shared/ dir)
            p = (Path(__file__).resolve().parent.parent / p).resolve()
        dataset_dir = p
        return _load_local(dataset_dir)
    return _load_hf(config["hf_dataset"], config.get("split", "test"))


def _load_local(dataset_dir: Path) -> tuple[list[str], object]:
    """Auto-detect format:
    - HF staging: contains metadata.jsonl + videos/<id>.mp4
    - Dataset:    contains <id>/animation.mp4 subdirs
    """
    metadata_path = dataset_dir / "metadata.jsonl"
    if metadata_path.exists():
        examples = []
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line)["id"])
        video_dir = dataset_dir / "videos"

        def get_video_path(name, _vd=video_dir):
            return _vd / f"{name}.mp4"

        return sorted(examples), get_video_path

    examples = sorted(
        d.name
        for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "animation.mp4").exists()
    )

    def get_video_path(name, _dd=dataset_dir):
        return _dd / name / "animation.mp4"

    return examples, get_video_path


def _load_hf(repo_id: str, split: str = "test") -> tuple[list[str], object]:
    """Download from HuggingFace and return examples."""
    from huggingface_hub import snapshot_download

    baselines_dir = Path(__file__).resolve().parent.parent
    cache_dir = baselines_dir / "hf_data" / split
    video_dir = cache_dir / "videos"
    metadata_path = cache_dir / "metadata.jsonl"

    if video_dir.exists() and metadata_path.exists() and any(video_dir.iterdir()):
        n = sum(1 for f in video_dir.iterdir() if f.suffix == ".mp4")
        print(f"Using cached {split} set: {n} videos in {video_dir}")
    else:
        print(f"Downloading {split} split from {repo_id}...")
        dl_dir = baselines_dir / "hf_data"
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dl_dir),
            allow_patterns=[f"data/{split}/*"],
        )
        # snapshot_download preserves repo structure: dl_dir/data/<split>/
        src = dl_dir / "data" / split
        video_dir = src / "videos"
        metadata_path = src / "metadata.jsonl"

    examples = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line)["id"])

    def get_video_path(name, _vd=video_dir):
        return _vd / f"{name}.mp4"

    return sorted(examples), get_video_path
