"""
Model factory — returns the right BaseModel subclass for a given role config block.

Usage:
    from modules.models import get_model
    gen = get_model("gemini", config["generator"]["gemini"])
    crt = get_model("qwen3vl", config["critic"]["qwen3vl"])
"""

from modules.models.base import BaseModel
from modules.models.gemini import GeminiModel
from modules.models.qwen3vl import Qwen3VLModel


def get_model(model_name: str, model_cfg: dict) -> BaseModel:
    """Instantiate the model class for the given model_name."""
    if model_name == "gemini":
        return GeminiModel(model_cfg)
    elif model_name == "qwen3vl":
        return Qwen3VLModel(model_cfg)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Supported: gemini, qwen3vl")


__all__ = ["get_model", "BaseModel", "GeminiModel", "Qwen3VLModel"]
