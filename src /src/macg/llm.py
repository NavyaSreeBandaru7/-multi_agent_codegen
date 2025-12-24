from __future__ import annotations

class LLMClient:
    """
    Start small: a stub interface. Replace with OpenAI / Ollama / etc later.
    For now, you can manually paste outputs OR implement one adapter.
    """
    def complete(self, system: str, prompt: str) -> str:
        raise NotImplementedError("Plug in an LLM adapter (OpenAI/Ollama) or start with manual mode.")


class ManualLLM(LLMClient):
    """
    Manual mode: prints prompt and asks you to paste the response.
    Great for learning + debugging the agent loop.
    """
    def complete(self, system: str, prompt: str) -> str:
        print("\n" + "="*80)
        print("SYSTEM:\n", system)
        print("-"*80)
        print("PROMPT:\n", prompt)
        print("="*80)
        print("Paste model output below. End with a line containing only: END\n")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        return "\n".join(lines)
