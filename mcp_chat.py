"""Bridges Streamlit's synchronous rerun model to a running Talentbook
backend's MCP connector, so `app_st_mcp_chat.py` can run a genuine LLM
agentic-tool-use loop — `lib.llm.get_llm_client()` — over that connector's
tools, with the LLM itself deciding which tools to call from the
conversation.

No default credential and nothing read from a config file: the viewer
pastes their own `tb_pat_...` access token into the sidebar — personal or
org-scoped, whichever that token's own scopes grant decides what tools show
up, this module doesn't need to know which kind it is.

The MCP server's address is also entered in the sidebar rather than fixed —
`lib.core.env.BACKEND_BASE_URL` (from `.streamlit/secrets.toml`'s
`REFLEX_BE_BASE_URL`) is only this page's pre-filled default for that box. The sidebar takes the bare address (no `/mcp/` suffix); `_mcp_url()`
below appends the endpoint path.

The MCP SDK is async-only; `lib.llm`'s provider clients call `tool_handler`
synchronously, inline, mid-stream. Rather than reconnect per tool call (slow)
or fight asyncio's no-reentrant-loop rule, one `McpBridge` opens a single
persistent session on a dedicated background thread with its own event
loop, and the main (synchronous) Streamlit thread drives it with plain
blocking calls via `asyncio.run_coroutine_threadsafe`. Bridges are cached
with `st.cache_resource`, keyed on `(token, server_address)`, so a
different pasted-in PAT or a different server gets its own connection
rather than reusing — or being overwritten by — whatever was entered
before it.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AsyncExitStack
from typing import Any

import httpx2
import streamlit as st
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool

from lib.core.time import utcnow
from lib.llm import get_llm_client
from lib.llm.tool_types import ToolCall


def _mcp_url(server_address: str) -> str:
    """A bare server address (e.g. "http://localhost:8888", no trailing
    slash expected from the sidebar) to its MCP endpoint URL. The trailing
    slash matters — Starlette's Mount always matches against
    "<prefix>/{path:path}", so a bare "/mcp" with nothing after it 404s
    regardless of the sub-app's own path= setting."""
    return server_address.rstrip("/") + "/mcp/"


class McpConnectionError(RuntimeError):
    """The MCP server couldn't be reached, or rejected the token —
    surfaced as a friendly page-level message rather than a raw traceback
    with an asyncio stack in it."""


class McpBridge:
    """One persistent MCP `ClientSession`, owned by a dedicated background
    thread's event loop, exposed to the main Streamlit thread as plain
    blocking calls."""

    def __init__(self, url: str, token: str):
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()
        ready.wait()

        # Entered once and never exited — this bridge lives for the whole
        # process (cached via get_bridge() below), so there's no natural
        # point to close the stack; the daemon thread dies with the process.
        self._stack = AsyncExitStack()
        self._session = self._run(self._connect(url, token))
        self.tools: list[Tool] = self._run(self._session.list_tools()).tools

    async def _connect(self, url: str, token: str) -> ClientSession:
        http_client = await self._stack.enter_async_context(
            httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"})
        )
        read, write = await self._stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call one tool and return its result as text, retrying briefly on
        the MCP endpoint's own rate limit — an agentic turn can make several
        back-to-back calls and hit a transient -32000 for real."""

        async def _call():
            for _ in range(20):
                try:
                    return await self._session.call_tool(name, arguments)
                except Exception as exc:
                    if "Rate limit exceeded" in str(exc):
                        await asyncio.sleep(1.1)
                        continue
                    raise
            raise RuntimeError(f"Still rate-limited after retrying {name!r}.")

        result = self._run(_call())
        texts = [t for b in result.content if (t := getattr(b, "text", None)) is not None]
        return "\n".join(texts) if texts else "(no output)"


@st.cache_resource(show_spinner="Connecting to the Talentbook MCP server…")
def get_bridge(token: str, server_address: str) -> McpBridge:
    """One persistent MCP connection for this exact `(token, server_address)`
    pair — cached so a page rerun (every keystroke elsewhere on the page)
    reuses it instead of reconnecting. Both are required; there's no
    file-backed default to fall back to."""
    url = _mcp_url(server_address)
    last_exc: Exception | None = None
    # One retry: the background-thread startup (a fresh loop + a fresh
    # anyio-managed Streamable-HTTP connection) occasionally races on its
    # very first attempt and surfaces as a bare CancelledError rather than
    # a clean connection error — a second attempt on a brand new thread has
    # reliably succeeded in testing, so it's cheap insurance rather than a
    # real fix for a specific diagnosed fault.
    for attempt in range(2):
        try:
            return McpBridge(url, token)
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(0.5)
    raise McpConnectionError(
        f"Couldn't connect to the Talentbook MCP server at {url} ({last_exc}). "
        f"Double-check the token and server address, and make sure the backend "
        f"is running."
    ) from last_exc


@st.cache_data(show_spinner=False)
def whoami(token: str, server_address: str) -> dict | None:
    """The account this token belongs to, if it's a personal token — one
    real tool call, both a connectivity smoke test and something worth
    showing the user. `None` (not an error) for an org token, which has
    no `read:profile` scope to call `get_my_profile` with — the caller
    falls back to a generic "connected" status rather than a name/email.
    A real connection failure still raises, via `get_bridge`."""
    bridge = get_bridge(token, server_address)
    if not any(t.name == "get_my_profile" for t in bridge.tools):
        return None
    try:
        return json.loads(bridge.call_tool("get_my_profile", {}))
    except Exception as exc:
        raise McpConnectionError(f"Connected, but get_my_profile failed: {exc}") from exc


def _tool_definitions(tools: list[Tool]) -> list[dict[str, Any]]:
    """MCP's `Tool` shape to the provider-neutral shape
    `generate_response_stream` expects, so it doesn't need to know these
    tools came from MCP rather than a local dispatcher."""
    return [
        {"name": t.name, "description": t.description or "", "parameters": t.input_schema}
        for t in tools
    ]


def _make_tool_handler(bridge: McpBridge) -> Callable[[ToolCall], str]:
    def handler(call: ToolCall) -> str:
        try:
            return bridge.call_tool(call.name, call.arguments)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return handler


SYSTEM_INSTRUCTION = """\
You are the Talentbook assistant, working over a single access token's own
MCP tools — whatever that token is scoped for. A personal token's tools
cover its owner's own professional development data: profile, goals,
skills, OKRs, journal, Business Model You canvas, and consultations. An
organisation token's tools instead cover that whole organisation: roster,
role baselines, and work-job templates — read each tool's own description
to tell which kind you're working with, and never blend "you" (a person)
with "the org" (everyone) in an answer.

You have live tools over MCP — call them whenever a question needs real
data instead of guessing or relying on earlier turns alone, since the data
can change between messages. Where a write tool is available and the
person clearly asks you to create or update something, use it; otherwise
default to read-only. If a tool call fails or a scope isn't granted to this
token, say so plainly instead of inventing an answer.

Be concise, warm, and concrete — reference actual goals, skills, people,
and entries by name rather than speaking in generalities. This is a chat,
not a report: short paragraphs, not headers and bullet essays, unless the
question is itself a request for a list."""


def _format_transcript(history: list[dict[str, str]], latest: str) -> str:
    lines = [f"Today's date: {utcnow().date().isoformat()}", ""]
    for turn in history:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['content']}")
    lines.append(f"User: {latest}")
    return "\n\n".join(lines)


def stream_reply(messages: list[dict[str, str]], bridge: McpBridge) -> Iterator[str]:
    """Stream the assistant's reply to the newest turn in `messages` (the
    full chat history so far, newest user message last), using `bridge`'s
    tools — whatever the connected PAT is scoped for.

    `generate_response_stream` takes one `prompt` string per call rather
    than a messages list — every `lib/llm` provider client shares that
    shape, built for single-shot jobs, not multi-turn chat — so each turn
    re-renders the whole transcript into `prompt` instead of reshaping all
    three provider clients for one feature. Good enough for a chat this
    size; a model with a large context window reads a plain transcript
    just as well as structured turns.
    """
    *history, current = messages
    client = get_llm_client()
    yield from client.generate_response_stream(
        SYSTEM_INSTRUCTION,
        _format_transcript(history, current["content"]),
        tools=_tool_definitions(bridge.tools),
        tool_handler=_make_tool_handler(bridge),
    )
