"""Minimal MCP chat client — connects to a Streamable-HTTP MCP server with a
bearer token, lists its tools, and lets you call them, either one-shot from
the command line or interactively.

Usage:
    TALENTBOOK_PAT=tb_pat_... python app_mcp_repl.py
    TALENTBOOK_PAT=tb_pat_... python app_mcp_repl.py list
    TALENTBOOK_PAT=tb_pat_... python app_mcp_repl.py call get_my_goals {}
    TALENTBOOK_PAT=tb_pat_... python app_mcp_repl.py consult
    TALENTBOOK_PAT=tb_pat_... python app_mcp_repl.py --server https://talentbook.example.com list

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
import uuid
from contextlib import asynccontextmanager

import httpx2
import typer
from mcp import ClientSession
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


def _parsed(result) -> object:
    """A tool-call result's first content block, JSON-decoded when
    possible — every tool in lib/mcp_tools.py returns a dict/list via
    its own `_dump()`, same reasoning as tests/test_mcp_connector.py's
    `_tool_call_result`."""
    for block in result.content:
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return None


async def _call_json(
    session: ClientSession, tool_name: str, arguments: dict, retry_on_rate_limit: bool = True,
) -> object:
    """Call a tool and return its parsed result. The MCP endpoint rate-
    limits (api/mcp.py — a short burst, then 1/second); a guided
    consultation flow makes enough back-to-back calls to hit that for
    real, so this retries with backoff instead of failing the whole
    flow over a transient -32000."""
    for _ in range(20):
        try:
            result = await session.call_tool(tool_name, arguments)
        except Exception as exc:
            if retry_on_rate_limit and "Rate limit exceeded" in str(exc):
                await asyncio.sleep(1.1)
                continue
            raise
        return _parsed(result)
    raise RuntimeError(f"Still rate-limited after retrying {tool_name!r} — try again shortly.")


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


async def _consult(url: str, token: str, spec_key: str) -> None:
    """Drive one consultation start to finish — the same guided,
    question-at-a-time flow app_rx/state/consultation_state.py's
    ConsultationState walks a person through, here driven from stdin
    instead of the browser. See _pm/MCP_CONNECTOR_USAGE_NOTES.md §1 for
    the tool-by-tool shape this follows."""
    async with _connect(url, token) as session:
        if not spec_key:
            available = await _call_json(session, "get_available_consultations", {})
            print("Available consultations:")
            for spec in available:
                for child in spec.get("children") or [spec]:
                    print(f"  {child['spec_key']:<32} {child['title']}")
            spec_key = input("\nspec_key> ").strip()
            if not spec_key:
                print("No spec_key given — aborting.")
                return

        started = await _call_json(
            session, "start_consultation",
            {"idempotency_key": str(uuid.uuid4()), "spec_key": spec_key},
        )
        if isinstance(started, dict) and started.get("error"):
            print(f"  {started['error']}")
            return
        consultation_id = started["consultation_id"]
        print(f"Started consultation {consultation_id} ({spec_key}).\n")

        while True:
            question = await _call_json(
                session, "get_next_consultation_question",
                {"consultation_id": consultation_id},
            )
            if question.get("done"):
                break

            print(f"[{question['progress_current']}/{question['progress_total']}] {question['question']}")
            if question.get("help"):
                print(f"  ({question['help']})")
            if question.get("options"):
                suffix = " — comma-separate for multiple" if question["type"] == "multiple" else ""
                print(f"  options: {', '.join(question['options'])}{suffix}")
            raw = input("> ").strip()
            value: str | list[str] = raw
            if question["type"] == "multiple":
                value = [v.strip() for v in raw.split(",") if v.strip()]

            answered = await _call_json(
                session, "answer_consultation_question",
                {
                    "idempotency_key": str(uuid.uuid4()),
                    "consultation_id": consultation_id,
                    "wm_key": question["wm"],
                    "question": question["question"],
                    "value": value,
                },
            )
            if isinstance(answered, dict) and answered.get("error"):
                print(f"  {answered['error']}")
                return
            print()

        print("All questions answered — generating your report...")
        generated = await _call_json(
            session, "generate_consultation_report",
            {"idempotency_key": str(uuid.uuid4()), "consultation_id": consultation_id},
        )
        if isinstance(generated, dict) and generated.get("error"):
            print(f"  {generated['error']}")
            return

        while True:
            result = await _call_json(
                session, "get_consultation_result", {"consultation_id": consultation_id},
            )
            if result.get("status") != "in_progress":
                break
            print(".", end="", flush=True)
            await asyncio.sleep(1.5)

        print()
        if result["status"] == "complete":
            print(json.dumps(result["result"], indent=2))
        else:
            print(f"Report generation failed: {result.get('error_message', '(no message)')}")


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
        asyncio.run(_repl(url, token))


@app.command("list")
def list_tools(ctx: typer.Context) -> None:
    """Show every tool this token can see."""
    url, token = ctx.obj

    async def _run() -> None:
        async with _connect(url, token) as session:
            tools = await session.list_tools()
            _print_tool_list(tools.tools)

    asyncio.run(_run())


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

    asyncio.run(_run())


@app.command()
def repl(ctx: typer.Context) -> None:
    """Connect once and keep calling tools interactively (same as no subcommand)."""
    url, token = ctx.obj
    asyncio.run(_repl(url, token))


@app.command()
def consult(
    ctx: typer.Context,
    spec_key: str = typer.Argument("", help="Consultation spec_key to start; omit to pick from a list"),
) -> None:
    """Start a consultation (or pick one), answer every question from
    stdin, then generate and wait for its report. Needs write:
    consultations (and read:consultations to list specs when spec_key
    is omitted)."""
    url, token = ctx.obj
    asyncio.run(_consult(url, token, spec_key))


if __name__ == "__main__":
    app()
