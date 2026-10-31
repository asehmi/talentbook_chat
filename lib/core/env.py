"""Configuration for Chat with Talentbook.

Reads from Streamlit's own secrets mechanism (`st.secrets`) rather than a
`.env` file — locally backed by `.streamlit/secrets.toml` (gitignored; copy
`.streamlit/secrets.toml.example` to create it), and on Streamlit Cloud by
the app's own Secrets settings. Same TOML keys either way, so nothing here
changes between local dev and a cloud deploy.

Just what this standalone MCP chat client actually needs: which LLM
provider/API key to use, the MCP server address's sidebar default, and the
log level. No Auth0, Stripe, or database settings — this app never touches
a database, it only talks to a running Talentbook backend's `/mcp` endpoint
over HTTP, using whichever access token the viewer pastes into the sidebar.
"""

from __future__ import annotations

import streamlit as st


def _secret(name: str, default: str = "") -> str:
    """A secrets.toml value, or `default` if it isn't set. `st.secrets`
    raises on a missing top-level key via `[...]` (and on no secrets file
    at all), so every read goes through `.get()` instead — this app runs
    fine with nothing configured yet, surfacing that as a clear "API key
    not configured" error from the LLM client rather than a crash here."""
    return st.secrets.get(name, default)


def _flag(name: str, default: bool = False) -> bool:
    raw = st.secrets.get(name, None)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("true", "1", "yes")


# ===== Deployment =====
# True only on the one deployment this is meant for — set in that
# deploy's own Streamlit Cloud Secrets, never in
# .streamlit/secrets.toml.example or a local secrets.toml (so it
# defaults to False everywhere else). Gates whether the sidebar lets a
# viewer override BACKEND_BASE_URL at all — see app_mcp_chat_streamlit.py.
IS_PROD = _flag("IS_PROD", default=False)

# ===== MCP server =====
# Not IS_PROD: only this page's pre-filled default for the sidebar's
# server-address box — the real address is whatever's typed in there.
# IS_PROD: the sidebar box is hidden and this is the only address ever
# used, so a viewer can't repoint the app at a different backend.
BACKEND_BASE_URL = _secret("REFLEX_BE_BASE_URL", "http://localhost:8000")

# ===== LLM provider =====
# Which provider client lib.llm.get_llm_client() returns — "anthropic"
# (default), "gemini", or "openrouter". All three share generate_response()'s
# shape (lib/llm/__init__.py), so only this one setting needs to change to
# switch provider.
LLM_PROVIDER = _secret("LLM_PROVIDER", "anthropic").strip().lower()
ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
GEMINI_API_KEY = _secret("GOOGLE_API_KEY") or _secret("GEMINI_API_KEY")
OPENROUTER_API_KEY = _secret("OPENROUTER_API_KEY") or _secret("OR_API_KEY")

LOG_LEVEL = _secret("LOG_LEVEL", "info").lower()
