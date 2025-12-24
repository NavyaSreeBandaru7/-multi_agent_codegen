from __future__ import annotations
from macg.llm import LLMClient
from macg.protocol import Artifact
from macg.tools.python_runner import run_pytest

TESTER_SYSTEM = """You are TesterAgent.
Write pytest tests that validate correctness and edge cases.
Rules:
- Output ONLY python test code.
- Use pytest.
- Do not import external libs besides pytest.
"""

class TesterAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, art: Artifact) -> Artifact:
        test_prompt = f"""
Task:
{art.task}
Code under test (module {art.module_name}.py):
{art.code}
Write pytest tests for the expected behavior.
""".strip()

        art.tests = self.llm.complete(TESTER_SYSTEM, test_prompt).strip()

        result = run_pytest(module_name=art.module_name, code=art.code, tests=art.tests, timeout_s=15)
        art.passed = result.ok
        art.test_report = f"returncode={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return art
