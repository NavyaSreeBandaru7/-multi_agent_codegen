from __future__ import annotations
from macg.llm import LLMClient
from macg.protocol import Artifact

CODER_SYSTEM = """You are CoderAgent.
Write correct, minimal Python code for the requested task.
Rules:
- Output ONLY python code.
- Include type hints.
- Include a docstring for the main function/class.
- No extra text.
"""

class CoderAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, art: Artifact) -> Artifact:
        prompt = f"""
Task:
{art.task}
Write a Python module named {art.module_name}.py implementing the solution.
If you need to revise code based on review notes, use them:
Review notes (if any):
{art.review_notes}
Tests (if any):
{art.tests}
""".strip()

        art.code = self.llm.complete(CODER_SYSTEM, prompt).strip()
        return art
