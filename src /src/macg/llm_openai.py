from __future__ import annotations

from openai import OpenAI
from macg.llm_base import LLMClient


class OpenAIResponsesLLM(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5",
        base_url: str = "https://api.openai.com/v1",
        temperature: float | None = 0.2,
        max_output_tokens: int = 900,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI api_key is required.")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system: str, prompt: str) -> str:
        params = {
            "model": self.model,
            "instructions": system,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
        }

        # Some models reject temperature; include it only if set
        if self.temperature is not None:
            params["temperature"] = self.temperature

        try:
            resp = self.client.responses.create(**params)
            return resp.output_text
        except Exception as e:
            msg = str(e)
            # If the model doesn't support temperature, retry without it
            if "Unsupported parameter" in msg and "temperature" in msg:
                params.pop("temperature", None)
                resp = self.client.responses.create(**params)
                return resp.output_text
            raise
