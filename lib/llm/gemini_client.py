"""Gemini client, with agentic tool-calling support.

Same `google-genai` config-kwargs-dict convention as Google's own samples,
same cheap/fast default model. Synchronous (`client.models.generate_content`,
not `client.aio.models.generate_content`), matching `AnthropicClient` and
this app's own synchronous Streamlit rerun model.

The tool-calling loop below (parsing `function_call` parts, building the
follow-up `function_response` turn) follows the standard `google-genai`
function-calling shape.

`ToolCallLoopGuard` (`lib/llm/tool_types.py`) caps rounds and detects the
same tool call repeating back to back.

No `cache_control`-equivalent call here, unlike `AnthropicClient` — Gemini's
2.5+/3.x models (this one included) apply prompt caching implicitly and
automatically server-side: no SDK parameter to set, it just takes effect
once a request's prefix is long enough and byte-identical to a recent call.
The call shape below already satisfies that for free: `system_instruction`
and `tools` are both passed via `config`, which `google-genai` always places
ahead of `contents` on the wire, and both are the same fixed
`SYSTEM_INSTRUCTION`/tool catalog on every round of the agentic loop and
every turn of a session (see `mcp_st_bridge.py`). Google also offers
*explicit* caching (`client.caches.create()`, a `CachedContent` object you
manage with your own TTL and pay hourly storage for) — deliberately not
used here; it's real added complexity for a model (`flash-lite`/cheap tier)
where the gap over implicit caching isn't worth it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

from tenacity import Retrying, stop_after_attempt, wait_exponential

from ..core.env import GEMINI_API_KEY
from ..utils.io import get_logger
from .tool_types import ToolCall, ToolCallLoopGuard

logger = get_logger("llm")

DEFAULT_MODEL = "gemini-3.1-flash-lite"


def _log_retry_attempt(retry_state) -> None:
    logger.debug(f"[LLM] attempt {retry_state.attempt_number} failed, retrying...")


# Keys the MCP server's (and FastMCP/pydantic's) JSON Schema output carries
# that Gemini's stricter Schema proto rejects outright with "Unknown name
# ... Cannot find field" — unlike Anthropic/OpenRouter, which both just
# ignore extra JSON-Schema keywords they don't need. Stripped recursively
# since any nested object/array param carries its own copy.
_GEMINI_UNSUPPORTED_SCHEMA_KEYS = {"additionalProperties", "$schema", "$id", "$defs", "$ref"}


def _sanitize_schema_for_gemini(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            k: _sanitize_schema_for_gemini(v)
            for k, v in schema.items()
            if k not in _GEMINI_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_sanitize_schema_for_gemini(v) for v in schema]
    return schema


def _to_gemini_tool(definitions: list[dict[str, Any]]):
    from google.genai import types

    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=d["name"],
                description=d["description"],
                parameters=_sanitize_schema_for_gemini(d["parameters"]),
            )
            for d in definitions
        ]
    )


class GeminiClient:
    """Client for Google's Gemini API."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        from google import genai

        api_key = api_key or GEMINI_API_KEY
        if not api_key:
            raise ValueError(
                "Gemini API key not configured. Set GOOGLE_API_KEY or "
                "GEMINI_API_KEY in .streamlit/secrets.toml."
            )
        self.api_key = api_key
        self.model_name = model
        self._client = genai.Client(api_key=api_key)

    def generate_response(
        self,
        system_instruction: str,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_handler: Callable[[ToolCall], str] | None = None,
    ) -> str | dict[str, Any]:
        """Generate a response. See `AnthropicClient.generate_response` for
        the shared contract (tools/tool_handler run an agentic loop; a
        given `response_schema` still applies to the final text)."""
        from google.genai import types

        config_kwargs: dict[str, Any] = {"system_instruction": system_instruction}
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
        if tools:
            config_kwargs["tools"] = [_to_gemini_tool(tools)]

        retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            before_sleep=_log_retry_attempt,
            reraise=True,
        )
        text = ""
        for attempt in retrying:
            with attempt:
                text = self._run(prompt, config_kwargs, tools, tool_handler)

        if response_schema is None:
            logger.info("[GEMINI] response received")
            return text

        parsed = json.loads(text)
        logger.info("[GEMINI] response received and parsed as JSON")
        return parsed

    def _run(
        self,
        prompt: str,
        config_kwargs: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        tool_handler: Callable[[ToolCall], str] | None,
    ) -> str:
        """One full attempt: the agentic loop, ending when the model stops
        requesting tools. Retried as a whole by `generate_response`."""
        from google.genai import types

        contents: list[Any] = [prompt]
        loop_guard = ToolCallLoopGuard()
        response = None
        while True:
            logger.info(f"[GEMINI] calling {self.model_name}")
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            function_calls = [
                part.function_call
                for part in response.candidates[0].content.parts
                if getattr(part, "function_call", None)
            ]
            if not function_calls or not tool_handler:
                break

            calls = [
                ToolCall(id=fc.name, name=fc.name, arguments=dict(fc.args or {}))
                for fc in function_calls
            ]
            loop_guard.check(calls)

            contents.append(response.candidates[0].content)
            response_parts = []
            for call in calls:
                logger.info(f"[GEMINI] tool call: {call.name}({call.arguments})")
                result_text = tool_handler(call)
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=call.name, response={"result": result_text}
                        )
                    )
                )
            contents.append(types.Content(role="user", parts=response_parts))

        if not response.text:
            raise ValueError("Gemini returned an empty response")
        return response.text

    def generate_response_stream(
        self,
        system_instruction: str,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
        tool_handler: Callable[[ToolCall], str] | None = None,
    ) -> Iterator[str]:
        """Like `generate_response`, but yields text as it arrives instead
        of waiting for the full response. See
        `AnthropicClient.generate_response_stream` for the shared contract
        and reasoning (no `response_schema` param, no retry wrapper — a
        mid-stream failure raises rather than silently retrying over
        content a caller may already be showing).

        Each streamed chunk's parts are collected into `accumulated_parts`
        as they're seen — text parts already yielded to the caller, plus
        any `function_call` parts — so that if the model does call a tool,
        the full turn (not just whichever chunk happened to carry the
        call) is what gets appended to `contents` for the follow-up round.
        A single chunk carrying the whole turn is the common case, but
        nothing here assumes it.
        """
        from google.genai import types

        config_kwargs: dict[str, Any] = {"system_instruction": system_instruction}
        if tools:
            config_kwargs["tools"] = [_to_gemini_tool(tools)]

        contents: list[Any] = [prompt]
        loop_guard = ToolCallLoopGuard()

        while True:
            logger.info(f"[GEMINI] streaming call to {self.model_name}")
            accumulated_parts: list[Any] = []
            function_calls: list[Any] = []
            for chunk in self._client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            ):
                if not chunk.candidates or not chunk.candidates[0].content:
                    continue
                for part in chunk.candidates[0].content.parts or []:
                    if getattr(part, "function_call", None):
                        function_calls.append(part.function_call)
                        accumulated_parts.append(part)
                    elif getattr(part, "text", None):
                        yield part.text
                        accumulated_parts.append(part)

            if not function_calls or not tool_handler:
                return

            calls = [
                ToolCall(id=fc.name, name=fc.name, arguments=dict(fc.args or {}))
                for fc in function_calls
            ]
            loop_guard.check(calls)

            contents.append(types.Content(role="model", parts=accumulated_parts))
            response_parts = []
            for call in calls:
                logger.info(f"[GEMINI] tool call: {call.name}({call.arguments})")
                result_text = tool_handler(call)
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=call.name, response={"result": result_text}
                        )
                    )
                )
            contents.append(types.Content(role="user", parts=response_parts))
