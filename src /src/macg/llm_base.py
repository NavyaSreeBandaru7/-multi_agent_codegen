from __future__ import annotations

class LLMClient:
    def complete(self, system: str, prompt: str) -> str:
        raise NotImplementedError
