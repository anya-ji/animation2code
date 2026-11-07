import os

from openai import OpenAI

from models.gpt import GptModel


class OpenRouterModel(GptModel):
    label = "OpenRouter"

    def _make_client(self) -> OpenAI:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
