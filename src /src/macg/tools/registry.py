from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]

class ToolRegistry:
    """
    Think of this as a tiny MCP-style “tool server”.
    Agents don’t call random Python—they call named tools with structured inputs.
    """
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name].fn(**kwargs)
