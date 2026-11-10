"""Minimal MCP chat client — connects to a Streamable-HTTP MCP server with a
bearer token, lists its tools, and lets you call them, either one-shot from
the command line or interactively.

Usage:
    TALENTBOOK_PAT=tb_pat_... python app_mcp_repl_simple.py
    TALENTBOOK_PAT=tb_pat_... python app_mcp_repl_simple.py list
    TALENTBOOK_PAT=tb_pat_... python app_mcp_repl_simple.py call get_my_goals {}
    TALENTBOOK_PAT=tb_pat_... python app_mcp_repl_simple.py --server https://talentbook.example.com list

--server takes just the app's base URL (default http://localhost:8888) — the
/mcp/ path (note the trailing slash — Starlette's Mount 404s without it) is
appended automatically, so switching between a local dev server and prod is
one flag, not a full URL to retype. --server/--token override the env vars.

Interactive commands once in the REPL (no subcommand, or `repl`):
    list                        show every tool this token can see
    call <tool_name> <json>     call a tool, e.g. call get_my_goals {}
    quit                        exit
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from contextlib import asynccontextmanager

import httpx2
import typer
from mcp import ClientSession, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool

app = typer.Typer(add_completion=False, help=__doc__)


@asynccontextmanager
async def _connect(url: str, token: str):
    http_client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"})
    async with streamable_http_client(url, http_client=http_client) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _mcp_error_message(exc: BaseException) -> str | None:
    """Unwrap a (possibly nested) ExceptionGroup to find the underlying
    MCPError and build an actionable one-line message. The SDK's own
    streamable_http client collapses *any* non-2xx HTTP response (401,
    403, 429, 500...) into a generic "Server returned an error response"
    with no distinguishing status code reaching this far — so this can't
    name the exact cause, only the likely one (a bad token, since that's
    by far the most common way to hit this)."""
    if isinstance(exc, MCPError):
        return (
            f"The server rejected the connection ({exc.message}). This usually "
            "means --token/TALENTBOOK_PAT is invalid, expired, or revoked. "
            "Check Connected Agents in the app and issue a fresh one if needed."
        )
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            message = _mcp_error_message(sub)
            if message:
                return message
    return None


def _run_async(coro) -> None:
    """Run an MCP command's coroutine, turning a connection/auth failure
    into one clean line instead of the SDK's own deeply nested
    ExceptionGroup traceback (session.initialize() raises inside two
    layers of anyio TaskGroup — streamable_http_client's and
    ClientSession's own — so a bare `await` lets both re-wrap it on the
    way out)."""
    try:
        asyncio.run(coro)
    except (MCPError, BaseExceptionGroup) as exc:
        print(f"\n{_mcp_error_message(exc) or exc}")
        raise typer.Exit(1)


def _print_tool_list(tools: list[Tool]) -> None:
    name_width = max((len(t.name) for t in tools), default=0)
    for t in tools:
        # Collapse the docstring's own line breaks so wrapping is driven
        # only by terminal width, not source formatting.
        desc = " ".join((t.description or "").split())
        wrapped = textwrap.wrap(desc, width=88) or [""]
        print(f"  {t.name.ljust(name_width)}  {wrapped[0]}")
        for extra in wrapped[1:]:
            print(f"  {' ' * name_width}  {extra}")


async def _print_call_result(session: ClientSession, tool_name: str, arguments: dict) -> None:
    result = await session.call_tool(tool_name, arguments)
    for block in result.content:
        print(getattr(block, "text", block))


async def _repl(url: str, token: str) -> None:
    async with _connect(url, token) as session:
        print(f"Connected to {url}. Type 'list', 'call <tool> <json>', or 'quit'.")

        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line or line == "quit":
                break

            if line == "list":
                tools = await session.list_tools()
                _print_tool_list(tools.tools)
                continue

            if line.startswith("call "):
                _, _, rest = line.partition(" ")
                name, _, raw_args = rest.partition(" ")
                try:
                    arguments = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError as exc:
                    print(f"  bad JSON: {exc}")
                    continue
                await _print_call_result(session, name, arguments)
                continue

            print("  unknown command — try 'list', 'call <tool> <json>', or 'quit'")


_ServerOption = typer.Option(
    "http://localhost:8888",
    "--server",
    envvar="TALENTBOOK_SERVER",
    help="App base URL, e.g. http://localhost:8888 for dev or a prod host — /mcp/ is appended.",
)
_TokenOption = typer.Option(
    ...,
    "--token",
    envvar="TALENTBOOK_PAT",
    help="Bearer PAT for the MCP connector.",
)


@app.callback(invoke_without_command=True)
def cli(
    ctx: typer.Context,
    server: str = _ServerOption,
    token: str = _TokenOption,
) -> None:
    """Connect to a Talentbook MCP server and list or call its tools."""
    url = f"{server.rstrip('/')}/mcp/"
    ctx.obj = (url, token)
    if ctx.invoked_subcommand is None:
        _run_async(_repl(url, token))


@app.command("list")
def list_tools(ctx: typer.Context) -> None:
    """Show every tool this token can see."""
    url, token = ctx.obj

    async def _run() -> None:
        async with _connect(url, token) as session:
            tools = await session.list_tools()
            _print_tool_list(tools.tools)

    _run_async(_run())


@app.command()
def call(
    ctx: typer.Context,
    tool_name: str = typer.Argument(..., help="MCP tool name, e.g. get_my_goals"),
    arguments: str = typer.Argument("{}", help="JSON object of arguments"),
) -> None:
    """Call one tool once and print its result."""
    url, token = ctx.obj
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as exc:
        typer.echo(f"bad JSON: {exc}", err=True)
        raise typer.Exit(1)

    async def _run() -> None:
        async with _connect(url, token) as session:
            await _print_call_result(session, tool_name, parsed_arguments)

    _run_async(_run())


@app.command()
def repl(ctx: typer.Context) -> None:
    """Connect once and keep calling tools interactively (same as no subcommand)."""
    url, token = ctx.obj
    _run_async(_repl(url, token))


if __name__ == "__main__":
    app()
