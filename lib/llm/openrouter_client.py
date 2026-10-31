"""OpenRouter client, with agentic tool-calling support.

Plain `httpx`-based `chat/completions` calls against OpenRouter's
OpenAI-compatible API. Note for whoever changes `DEFAULT_MODEL` next: model
catalogs and routing prefixes drift — check `GET /api/v1/models` before
assuming a model string from another provider or an older doc is still
valid.

Synchronous (`httpx.Client`, not `httpx.AsyncClient`; `SSEParser` iterates
`response.iter_bytes()`, not `aiter_bytes()`), matching `AnthropicClient`/
`GeminiClient` and this app's own synchronous Streamlit rerun model.
Tool-calling requires streaming on OpenRouter's API even for a
"non-streaming" caller — `generate_response` routes through the same
streaming/tool-call loop as `generate_response_stream` whenever `tools` is
given.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
from tenacity import Retrying, stop_after_attempt, wait_exponential

from ..core.env import OPENROUTER_API_KEY
from ..utils.io import get_logger
from .tool_types import ToolCall, ToolCallLoopGuard

logger = get_logger("llm")

DEFAULT_MODEL = "google/gemini-3.7-flash"
BASE_URL = "https://openrouter.ai/api/v1"


def _log_retry_attempt(retry_state) -> None:
    logger.debug(f"[LLM] attempt {retry_state.attempt_number} failed, retrying...")


class SSEParser:
    """Buffers and parses an OpenAI-style SSE stream, extracting content
    deltas and tool-call fragments. Synchronous — see module docstring.

    Tool-call fragments are keyed by the delta's own `index`, not `id` —
    confirmed live against OpenRouter (google/gemini-3.1-flash-lite): only
    the *first* fragment of a tool call carries `id`; later fragments
    carrying the rest of `arguments` omit it and carry only `index`, which
    stays stable across every fragment of the same call. Keying by `id`
    instead splits one tool call's fragments across two different buffer
    entries — one with a name and no arguments, one with arguments and no
    name — so neither ever passes `get_tool_calls()`'s completeness check
    and the call is silently lost.
    """

    def __init__(self) -> None:
        self.response_text = ""
        self._tool_call_buffer: dict[int, dict[str, str]] = {}
        self._json_buffer = ""

    def parse(self, response: httpx.Response) -> Iterator[str]:
        """Parse the SSE stream, yielding each content delta as it's
        decoded — so a streaming caller can observe text live — while
        still populating `response_text`/the tool-call buffer as a side
        effect. A caller that only wants the accumulated result just has
        to fully drain the generator, e.g. `for _ in parser.parse(response):
        pass`, rather than calling it as a plain statement — `parse()`
        being a generator function means nothing inside it runs at all
        until iterated.
        """
        for chunk in response.iter_bytes(chunk_size=4096):
            self._json_buffer += chunk.decode("utf-8")
            while True:
                line_end = self._json_buffer.find("\n")
                if line_end == -1:
                    break
                line = self._json_buffer[:line_end].strip()
                self._json_buffer = self._json_buffer[line_end + 1:]
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except json.JSONDecodeError:
                    self._json_buffer = f"data: {data}\n" + self._json_buffer
                    break
                if content := delta.get("content"):
                    self.response_text += content
                    yield content
                for tc in delta.get("tool_calls") or []:
                    position = tc.get("index", 0)
                    buf = self._tool_call_buffer.setdefault(
                        position, {"id": "", "name": "", "arguments_buffer": ""}
                    )
                    if tc.get("id"):
                        buf["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name") and not buf["name"]:
                        buf["name"] = fn["name"]
                    if fn.get("arguments"):
                        buf["arguments_buffer"] += fn["arguments"]

    def get_tool_calls(self) -> list[ToolCall]:
        calls = []
        for position, buf in self._tool_call_buffer.items():
            if not buf["name"] or not buf["arguments_buffer"].strip():
                continue
            try:
                args = json.loads(buf["arguments_buffer"])
            except json.JSONDecodeError:
                continue
            call_id = buf["id"] or f"tool_{position}"
            calls.append(ToolCall(id=call_id, name=buf["name"], arguments=args))
        return calls


def _to_openrouter_tools(definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d["description"],
                "parameters": d["parameters"],
            },
        }
        for d in definitions
    ]


class OpenRouterClient:
    """Client for OpenRouter's OpenAI-compatible chat completions API."""

    def __init__(
        self, api_key: str | None = None, model: str = DEFAULT_MODEL, base_url: str = BASE_URL
    ):
        api_key = api_key or OPENROUTER_API_KEY
        if not api_key:
            raise ValueError(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY in "
                ".streamlit/secrets.toml."
            )
        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://talentbook.dev",
            "X-Title": "Chat with Talentbook",
        }

    def generate_response(
        self,
        system_instruction: str,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_handler: Callable[[ToolCall], str] | None = None,
    ) -> str | dict[str, Any]:
        """Generate a response. See `AnthropicClient.generate_response` for
        the shared contract. Routes through the streaming/tool-call loop
        only when `tools` is given — a plain non-streaming call otherwise."""
        final_prompt = prompt
        if response_schema is not None:
            schema_str = json.dumps(response_schema, indent=2)
            final_prompt = (
                f"{prompt}\n\nRespond ONLY with a valid JSON object matching "
                f"this schema exactly. No markdown fences, no explanation:\n"
                f"{schema_str}"
            )

        retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            before_sleep=_log_retry_attempt,
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                if tools and tool_handler:
                    text_content = self._run_with_tools(
                        system_instruction, final_prompt, tools, tool_handler
                    )
                else:
                    text_content = self._generate_once(system_instruction, final_prompt)

                if not text_content:
                    raise ValueError("OpenRouter returned an empty response")

                if response_schema is None:
                    logger.info("[OPENROUTER] response received")
                    return text_content

                parsed = json.loads(text_content)
                logger.info("[OPENROUTER] response received and parsed as JSON")
                return parsed

        raise RuntimeError("unreachable: Retrying always returns or raises")  # pragma: no cover

    def _generate_once(self, system_instruction: str, prompt: str) -> str:
        logger.info(f"[OPENROUTER] calling {self.model_name}")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        with httpx.Client(timeout=httpx.Timeout(120)) as client:
            response = client.post(
                f"{self.base_url}/chat/completions", headers=self._headers(), json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def _run_with_tools(
        self,
        system_instruction: str,
        prompt: str,
        tools: list[dict[str, Any]],
        tool_handler: Callable[[ToolCall], str],
    ) -> str:
        """The agentic loop: stream, parse tool calls, execute them, feed
        results back, repeat until the model stops asking for tools.

        Follow-up messages must use the real OpenAI tool-calling protocol —
        an `assistant` turn carrying the `tool_calls` it made, then one
        `tool` turn per call keyed by `tool_call_id` — not a plain `user`
        message describing the result as text. Confirmed live: with the
        latter, the model never recognises its own tool call as answered
        and just calls the same tool again — indefinitely, until the
        connection gets dropped server-side. `ToolCallLoopGuard`
        (`lib/llm/tool_types.py`) is the backstop in case a model still
        won't converge even with the correct protocol — it also catches
        that exact failure mode directly (the same tool + arguments
        repeating back to back), not just a blunt round count.
        """
        openrouter_tools = _to_openrouter_tools(tools)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
        loop_guard = ToolCallLoopGuard()

        while True:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "tools": openrouter_tools,
                "stream": True,
            }
            with httpx.Client(timeout=httpx.Timeout(300)) as client:
                with client.stream(
                    "POST", f"{self.base_url}/chat/completions",
                    headers=self._headers(), json=payload,
                ) as response:
                    response.raise_for_status()
                    parser = SSEParser()
                    # Drain fully — parse() is now a generator (streaming
                    # needs it to yield deltas live); this loop only cares
                    # about the accumulated response_text/tool-call buffer
                    # it leaves behind as a side effect, same as before.
                    for _ in parser.parse(response):
                        pass

            tool_calls = parser.get_tool_calls()
            if not tool_calls:
                return parser.response_text
            loop_guard.check(tool_calls)

            messages.append({
                "role": "assistant",
                "content": parser.response_text or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in tool_calls
                ],
            })
            for call in tool_calls:
                logger.info(f"[OPENROUTER] tool call: {call.name}({call.arguments})")
                result_text = tool_handler(call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_text,
                })

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

        Same agentic loop as `_run_with_tools`, and the same real
        OpenAI-protocol follow-up shape that method's own docstring
        explains is required (an `assistant` turn carrying `tool_calls`,
        then one `tool` turn per call) — just `yield from`ing `SSEParser`'s
        now-generator `parse()` instead of draining it, so a caller sees
        each round's text live instead of only the accumulated
        `response_text` once the round ends.
        """
        openrouter_tools = _to_openrouter_tools(tools) if tools else None
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
        loop_guard = ToolCallLoopGuard()

        while True:
            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
            }
            if openrouter_tools:
                payload["tools"] = openrouter_tools

            logger.info(f"[OPENROUTER] streaming call to {self.model_name}")
            with httpx.Client(timeout=httpx.Timeout(300)) as client:
                with client.stream(
                    "POST", f"{self.base_url}/chat/completions",
                    headers=self._headers(), json=payload,
                ) as response:
                    response.raise_for_status()
                    parser = SSEParser()
                    yield from parser.parse(response)

            tool_calls = parser.get_tool_calls()
            if not tool_calls or not tool_handler:
                return
            loop_guard.check(tool_calls)

            messages.append({
                "role": "assistant",
                "content": parser.response_text or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in tool_calls
                ],
            })
            for call in tool_calls:
                logger.info(f"[OPENROUTER] tool call: {call.name}({call.arguments})")
                result_text = tool_handler(call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_text,
                })
