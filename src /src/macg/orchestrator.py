from __future__ import annotations
from dataclasses import asdict
from macg.protocol import Artifact
from macg.agents.coder import CoderAgent
from macg.agents.reviewer import ReviewerAgent
from macg.agents.tester import TesterAgent

class Orchestrator:
    def __init__(self, coder: CoderAgent, reviewer: ReviewerAgent, tester: TesterAgent) -> None:
        self.coder = coder
        self.reviewer = reviewer
        self.tester = tester

    def run(self, task: str, max_iters: int = 3) -> Artifact:
        art = Artifact(task=task, iteration=0)

        for i in range(1, max_iters + 1):
            art.iteration = i

            # 1) code
            art = self.coder.run(art)

            # 2) review
            art = self.reviewer.run(art)

            # 3) test
            art = self.tester.run(art)

            if art.passed:
                return art

            # loop: reviewer notes will guide next coder revision
            art.review_notes = art.review_notes + "\n- Tests failed; revise code to satisfy failing cases.\n"
        return art
