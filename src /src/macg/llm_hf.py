from __future__ import annotations
import os
import time
import requests
from macg.llm_base import LLMClient

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if "```" not in t:
        return t
    for fence in ("```python", "```py", "```"):
        if fence in t:
            start = t.find(fence) + len(fence)
            end = t.find("```", start)
            if end != -1:
                return t[start:end].strip()
    return t.replace("```", "").strip()

class HuggingFaceInferenceLLM(LLMClient):
    def __init__(
        self,
        model: str,
        token: str | None = None,
        max_new_tokens: int = 900,
        temperature: float = 0.2,
        retries: int = 4,
        timeout_s: int = 90,
    ) -> None:
        self.model = model
        # ✅ allow either HF_TOKEN or HUGGINGFACEHUB_API_TOKEN, and allow missing token
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.retries = retries
        self.timeout_s = timeout_s

    def complete(self, system: str, prompt: str) -> str:
        url = f"https://api-inference.huggingface.co/models/{self.model}"

        # ✅ if token exists, use it; otherwise call unauthenticated
        headers = {}
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}

        payload = {
            "inputs": f"{system}\n\n{prompt}".strip(),
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }

        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if r.status_code in (503, 504):
                    time.sleep(min(2 * attempt, 8))
                    continue
                r.raise_for_status()
                data = r.json()

                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return _strip_code_fences(data[0]["generated_text"])
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(f"HF error: {data['error']}")
                raise RuntimeError(f"Unexpected HF response: {data}")
            except Exception as e:
                last_err = e
                time.sleep(min(2 * attempt, 8))

        raise RuntimeError(f"HF inference failed. Last error: {last_err}")

