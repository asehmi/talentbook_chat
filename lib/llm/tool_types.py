"""Shared tool-calling types for the three LLM provider clients.

Trimmed from the parent Talentbook monorepo's `lib/llm/tools.py`, which also
defines a local `TOOL_DEFINITIONS` catalog wrapping that app's own database
services — not needed here. This chat client's tools come entirely from the
connected MCP server instead (see `mcp_chat.py`); only the provider-neutral
`ToolCall` shape and the loop guard are shared infrastructure.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    """One tool invocation requested by a model — provider-agnostic, built
    fresh by each provider client from whatever shape that provider's API
    returns (Anthropic's `tool_use` block, Gemini's `function_call` part,
    OpenRouter's streamed OpenAI-style tool call)."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolCallLoopGuard:
    """Detects a stuck agentic tool-call loop and raises before it can run
    away — shared across all three provider clients so this safety net
    can't silently drift out of sync between them.

    Two independent triggers:
    - **Too many rounds total** (`max_rounds`) — a blunt backstop.
    - **The same tool call (name + arguments) repeated back to back**
      (`max_consecutive_repeats`) — a much sharper signal that something's
      actually stuck, not just that a legitimately multi-step task is
      taking a few rounds (e.g. calling one tool then another is two
      different rounds and never trips this).
    """

    def __init__(self, max_rounds: int = 8, max_consecutive_repeats: int = 2):
        self.max_rounds = max_rounds
        self.max_consecutive_repeats = max_consecutive_repeats
        self._rounds = 0
        self._last_signature: str | None = None
        self._repeat_count = 0

    def check(self, calls: list[ToolCall]) -> None:
        """Call once per round, with that round's requested tool calls,
        before executing any of them. Raises `RuntimeError` if the loop
        looks stuck; otherwise returns normally."""
        self._rounds += 1
        if self._rounds > self.max_rounds:
            raise RuntimeError(
                f"Tool-call loop exceeded {self.max_rounds} rounds without "
                "converging"
            )

        signature = "|".join(
            sorted(
                f"{c.name}:{json.dumps(c.arguments, sort_keys=True)}"
                for c in calls
            )
        )
        if signature == self._last_signature:
            self._repeat_count += 1
        else:
            self._last_signature = signature
            self._repeat_count = 1

        if self._repeat_count > self.max_consecutive_repeats:
            raise RuntimeError(
                f"Same tool call(s) repeated {self._repeat_count} times in "
                f"a row, treating as a stuck loop: {signature}"
            )
