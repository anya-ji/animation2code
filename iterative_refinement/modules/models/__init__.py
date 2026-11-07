"""Model factory: return the model class for a stage's config block."""

from modules.models.base import BaseModel
from modules.models.gemini import GeminiModel
from modules.models.qwen3vl import Qwen3VLModel


def get_model(model_name: str, model_cfg: dict) -> BaseModel:
    """Create a model from its name and config.

    Args:
        model_name: "gemini" or "qwen3vl".
        model_cfg: That model's config block.
    """
    if model_name == "gemini":
        return GeminiModel(model_cfg)
    elif model_name == "qwen3vl":
        return Qwen3VLModel(model_cfg)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Supported: gemini, qwen3vl")


__all__ = ["get_model", "BaseModel", "GeminiModel", "Qwen3VLModel"]
