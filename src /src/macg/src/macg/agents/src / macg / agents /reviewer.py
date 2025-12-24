from __future__ import annotations
import ast
from macg.llm import LLMClient
from macg.protocol import Artifact

REVIEWER_SYSTEM = """You are ReviewerAgent.
Goal: improve correctness, safety, style, and testability.
Rules:
- Be specific and actionable.
- If code likely fails tests, explain why.
- Output only bullet points (no code).
"""

class ReviewerAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm  # optional

    def _static_checks(self, code: str) -> list[str]:
        notes = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            notes.append(f"- SyntaxError: {e}")
            return notes

        if "eval(" in code or "exec(" in code:
            notes.append("- Avoid eval/exec unless absolutely necessary.")
        if "import *" in code:
            notes.append("- Avoid wildcard imports.")
        return notes

    def run(self, art: Artifact) -> Artifact:
        notes = self._static_checks(art.code)

        if self.llm:
            prompt = f"""
Task:
{art.task}
Code:
{art.code}
Existing notes:
{chr(10).join(notes) if notes else "(none)"}
Return improved review notes as bullet points.
""".strip()
            llm_notes = self.llm.complete(REVIEWER_SYSTEM, prompt).strip()
            art.review_notes = (("\n".join(notes) + "\n") if notes else "") + llm_notes
        else:
            art.review_notes = "\n".join(notes) if notes else "- Looks OK (static checks only)."

        return art
