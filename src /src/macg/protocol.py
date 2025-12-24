from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import time
import uuid

Role = Literal["user", "orchestrator", "coder", "reviewer", "tester", "tool"]

@dataclass
class Message:
    role: Role
    content: str
    meta: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=lambda: time.time())
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

@dataclass
class Artifact:
    """A shared bundle agents pass around: code, tests, notes, scores, etc."""
    task: str
    module_name: str = "solution"
    code: str = ""
    tests: str = ""
    review_notes: str = ""
    test_report: str = ""
    passed: bool = False
    iteration: int = 0
