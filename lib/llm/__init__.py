"""LLM provider clients, for the chat app's agentic MCP tool-calling loop.

Three provider clients — `anthropic_client.py`, `gemini_client.py`,
`openrouter_client.py` — sharing one `generate_response(system_instruction,
prompt, response_schema=None, tools=None, tool_handler=None)` /
`generate_response_stream(...)` shape, so a caller doesn't need
provider-specific branching beyond picking which class to instantiate.
`tool_types.py` defines the provider-neutral `ToolCall` shape and a loop
guard shared by all three; the tools themselves come from whatever MCP
server is connected (see `mcp_chat.py`), not from this package.

`get_llm_client()` below is that "picking which class to instantiate" step,
driven by `LLM_PROVIDER` (`.streamlit/secrets.toml`) — the caller calls it
instead of importing a specific client class directly, so changing one
setting switches provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.env import LLM_PROVIDER

if TYPE_CHECKING:
    from .anthropic_client import AnthropicClient
    from .gemini_client import GeminiClient
    from .openrouter_client import OpenRouterClient

_PROVIDERS = ("anthropic", "gemini", "openrouter")


def get_llm_client() -> "AnthropicClient | GeminiClient | OpenRouterClient":  # noqa: UP037
    """The provider client the chat app should call — chosen by
    `LLM_PROVIDER`, defaulting to Anthropic. Each client class is imported
    lazily inside its own branch rather than at module load, so picking
    one provider doesn't require every provider's SDK to be importable
    (e.g. `google-genai` need not be installed just to run Anthropic-only)."""
    if LLM_PROVIDER == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient()
    if LLM_PROVIDER == "gemini":
        from .gemini_client import GeminiClient

        return GeminiClient()
    if LLM_PROVIDER == "openrouter":
        from .openrouter_client import OpenRouterClient

        return OpenRouterClient()
    raise ValueError(
        f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r} — must be one of {_PROVIDERS}"
    )
