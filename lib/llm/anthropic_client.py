"""Anthropic client, with agentic tool-calling support.

`generate_response()`/`generate_response_stream()` are the two methods a
caller needs — same shape every provider client in this package shares, so
switching provider is a one-line change in `get_llm_client()`.

Pass `tools` (built from whatever MCP server is connected — see
`mcp_chat.py`) and `tool_handler` and the call runs an agentic loop —
Anthropic's own `tools=[...]` param and `tool_use`/`tool_result` content
blocks, no manual response parsing needed for this provider. Uses a manual
`tenacity.Retrying()` loop per call rather than the `@retry` decorator, so
retry bookkeeping is always scoped to one call, not shared across concurrent
callers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

from tenacity import Retrying, stop_after_attempt, wait_exponential

from ..core.env import ANTHROPIC_API_KEY
from ..utils.io import get_logger
from .tool_types import ToolCall, ToolCallLoopGuard

logger = get_logger("llm")

DEFAULT_MODEL = "claude-sonnet-5"
# Baseline output budget — a chat reply, not a long report. Grows per retry
# attempt (below) in case a truncated response, not a transient error,
# caused the previous failure.
BASE_MAX_TOKENS = 20000
MAX_TOKENS_GROWTH_FACTOR = 1.25


def _parse_json_response(text: str) -> dict[str, Any]:
    """Strip optional markdown fences and parse JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned)


def _to_anthropic_tools(definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provider-neutral tool definitions use a `parameters` key (so
    Gemini/OpenRouter can reshape it their own way); Anthropic's Messages
    API requires the schema under `input_schema` specifically — confirmed
    live: passing `parameters` as-is is silently accepted by the SDK's
    typing but rejected by the API itself with `tools.0.custom.input_schema:
    Field required`."""
    return [
        {"name": d["name"], "description": d["description"], "input_schema": d["parameters"]}
        for d in definitions
    ]


def _max_tokens_for_attempt(attempt_number: int) -> int:
    """Token budget for a given (1-indexed) retry attempt."""
    return round(BASE_MAX_TOKENS * (MAX_TOKENS_GROWTH_FACTOR ** (attempt_number - 1)))


def _log_retry_attempt(retry_state) -> None:
    logger.debug(
        f"[LLM] attempt {retry_state.attempt_number} failed, retrying..."
    )


class AnthropicClient:
    """Client for Anthropic's Messages API."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        import anthropic as _anthropic

        api_key = api_key or ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY in "
                ".streamlit/secrets.toml."
            )
        self.api_key = api_key
        self.model_name = model
        self._anthropic = _anthropic

    def generate_response(
        self,
        system_instruction: str,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_handler: Callable[[ToolCall], str] | None = None,
    ) -> str | dict[str, Any]:
        """Generate a response, with retry + growing token budget on failure.

        Anthropic has no native JSON-schema-constrained output, so when
        `response_schema` is given it's appended to the prompt as an
        instruction and the response is parsed as JSON; otherwise the raw
        text is returned.

        When `tools` and `tool_handler` are both given, runs the standard
        Anthropic agentic loop: call, and while `stop_reason == "tool_use"`,
        execute each requested tool via `tool_handler`, feed the results
        back as a `tool_result` user turn, and call again — until Claude
        stops asking for tools.
        """
        client = self._anthropic.Anthropic(api_key=self.api_key)

        final_prompt = prompt
        if response_schema is not None:
            schema_str = json.dumps(response_schema, indent=2)
            final_prompt = (
                f"{prompt}\n\nRespond ONLY with a valid JSON object matching "
                f"this schema exactly. No markdown fences, no explanation:\n"
                f"{schema_str}"
            )

        anthropic_tools = _to_anthropic_tools(tools) if tools else None

        retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            before_sleep=_log_retry_attempt,
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                max_tokens = _max_tokens_for_attempt(attempt.retry_state.attempt_number)
                logger.info(
                    f"[ANTHROPIC] calling {self.model_name}, max_tokens={max_tokens}"
                )
                messages: list[dict[str, Any]] = [
                    {"role": "user", "content": final_prompt}
                ]
                loop_guard = ToolCallLoopGuard()
                response = None
                while True:
                    create_kwargs: dict[str, Any] = {
                        "model": self.model_name,
                        "max_tokens": max_tokens,
                        "thinking": {"type": "adaptive"},
                        "system": system_instruction,
                        "messages": messages,
                    }
                    if anthropic_tools:
                        create_kwargs["tools"] = anthropic_tools
                    # Streaming, not .create(): at this max_tokens the SDK's
                    # own timeout heuristic requires it ("Streaming is
                    # required for operations that may take longer than 10
                    # minutes"). get_final_message() collapses the stream
                    # back into the same Message shape .create() returns, so
                    # nothing below this needs to change.
                    with client.messages.stream(**create_kwargs) as stream:
                        response = stream.get_final_message()

                    if response.stop_reason != "tool_use" or not tool_handler:
                        break

                    calls = [
                        ToolCall(id=b.id, name=b.name, arguments=b.input)
                        for b in response.content if b.type == "tool_use"
                    ]
                    loop_guard.check(calls)

                    messages.append({"role": "assistant", "content": response.content})
                    tool_results = []
                    for call in calls:
                        logger.info(f"[ANTHROPIC] tool call: {call.name}({call.arguments})")
                        result_text = tool_handler(call)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": result_text,
                        })
                    messages.append({"role": "user", "content": tool_results})

                text_content = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                if not text_content:
                    raise ValueError("Anthropic returned an empty text response")

                if response_schema is None:
                    logger.info("[ANTHROPIC] response received")
                    return text_content

                parsed = _parse_json_response(text_content)
                logger.info("[ANTHROPIC] response received and parsed as JSON")
                return parsed

        raise RuntimeError("unreachable: Retrying always returns or raises")  # pragma: no cover

    def generate_response_stream(
        self,
        system_instruction: str,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
        tool_handler: Callable[[ToolCall], str] | None = None,
    ) -> Iterator[str]:
        """Like `generate_response`, but yields text as it arrives instead
        of waiting for the full response — for a chat-style UI that wants
        to show a reply being written live.

        No `response_schema` param, deliberately: partial JSON isn't useful
        to a caller the way partial prose is.

        No retry/growing-token-budget wrapper either, unlike
        `generate_response`: once text has been yielded, a caller may
        already be showing it, so silently retrying the whole call on a
        transient failure would mean either duplicating what's on screen
        or discarding it without the caller knowing. A mid-stream failure
        raises; it's the caller's job to decide what "try again" means for
        their own UI.

        Tool-calling still works, same agentic loop as `generate_response`:
        each round of `tool_use` is resolved (necessarily un-streamed — a
        tool call has to finish before its result can be fed back) before
        the *next* round's text streams to the caller.
        """
        client = self._anthropic.Anthropic(api_key=self.api_key)
        anthropic_tools = _to_anthropic_tools(tools) if tools else None
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        loop_guard = ToolCallLoopGuard()

        while True:
            create_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": BASE_MAX_TOKENS,
                "thinking": {"type": "adaptive"},
                "system": system_instruction,
                "messages": messages,
            }
            if anthropic_tools:
                create_kwargs["tools"] = anthropic_tools

            logger.info(f"[ANTHROPIC] streaming call to {self.model_name}")
            with client.messages.stream(**create_kwargs) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield event.delta.text
                response = stream.get_final_message()

            if response.stop_reason != "tool_use" or not tool_handler:
                return

            calls = [
                ToolCall(id=b.id, name=b.name, arguments=b.input)
                for b in response.content if b.type == "tool_use"
            ]
            loop_guard.check(calls)

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for call in calls:
                logger.info(f"[ANTHROPIC] tool call: {call.name}({call.arguments})")
                result_text = tool_handler(call)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})
