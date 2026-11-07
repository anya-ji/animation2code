from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parent.parent


def load_examples(config: dict) -> tuple[list[str], object]:
    """List the examples in a dataset split.

    Args:
        config: The config's data block. local_dataset is the split directory, relative to
            baselines/, containing <id>/animation.mp4.

    Returns:
        (sorted example names, function mapping a name to its video path)
    """
    dataset_dir = (BASELINES_DIR / config["local_dataset"]).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    examples = sorted(
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "animation.mp4").exists()
    )
    return examples, lambda name: dataset_dir / name / "animation.mp4"
