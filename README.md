# talentbook_chat

Streamlit MCP chat client: secure agent interaction with Talentbook.

A real LLM agentic-tool-use loop over a running [Talentbook](https://github.com/)
backend's MCP connector — the LLM decides for itself which tools to call,
based on whatever access token you paste into the sidebar. Nothing is
pinned to one account: a personal `tb_pat_...` token exposes your own
goals/skills/OKRs/journal/canvas, an org-scoped one exposes that
organisation's roster and role baselines instead, and the chat adapts to
whichever scopes the token actually grants.

This repo is a standalone copy of Talentbook's `app_st/app_st_mcp_chat.py`
page (now `app_mcp_chat_streamlit.py` here), trimmed to just its own
dependencies (no database, no Reflex, no other services) so it can run and
deploy on its own. `app_mcp_repl.py` is a second, CLI-based way to use the
same MCP connector — handy for scripting or debugging a token's tools
without opening the chat UI.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in an LLM provider API key
streamlit run app_mcp_chat_streamlit.py
```

The Talentbook backend you want to chat against must already be running
somewhere reachable (its MCP connector mounted at `/mcp`) — type its base
URL into the sidebar's "Server address" box, and paste in an access token
issued from that backend's Connected Agents page.

## CLI client (app_mcp_repl.py)

For scripting or quick debugging without the chat UI, `app_mcp_repl.py`
lists and calls the same MCP tools from the command line — it doesn't read
`.streamlit/secrets.toml` or any `lib/` code, just plain env vars:

```bash
TALENTBOOK_PAT=tb_pat_... python app_mcp_repl.py list
TALENTBOOK_PAT=tb_pat_... python app_mcp_repl.py call get_my_goals {}
TALENTBOOK_PAT=tb_pat_... python app_mcp_repl.py consult
TALENTBOOK_PAT=tb_pat_... python app_mcp_repl.py --server https://talentbook.example.com list
```

With no subcommand it drops into an interactive REPL (`list`, `call <tool>
<json>`, `quit`). `--server`/`--token` override the env vars. See `python
app_mcp_repl.py --help` for the full command list.

`app_mcp_repl_simple.py` is an earlier, simpler draft of the same idea
(just `list`/`call`/`repl`, no `consult`) — kept for reference, but
`app_mcp_repl.py` is the one to actually use.

## Configuration

All configuration is read via Streamlit's own secrets mechanism
(`st.secrets`), never a `.env` file — see `lib/core/env.py`. Locally that
means `.streamlit/secrets.toml` (gitignored, created by copying
`.streamlit/secrets.toml.example`); on Streamlit Cloud it means the app's
own **Settings -> Secrets**, pasted in as the same TOML. Nothing in the code
differs between the two.

| Key | Required | Purpose |
|---|---|---|
| `LLM_PROVIDER` | No (default `anthropic`) | `anthropic`, `gemini`, or `openrouter` — which client `lib.llm.get_llm_client()` returns. |
| `ANTHROPIC_API_KEY` | If using `anthropic` | Anthropic API key. |
| `GOOGLE_API_KEY` | If using `gemini` | Gemini API key (`GEMINI_API_KEY` also accepted). |
| `OPENROUTER_API_KEY` | If using `openrouter` | OpenRouter API key (`OR_API_KEY` also accepted). |
| `REFLEX_BE_BASE_URL` | No (default `http://localhost:8000`) | Pre-fills the sidebar's "Server address" box; the viewer can always type a different one — unless `IS_PROD` is true, in which case it's the only address ever used. |
| `IS_PROD` | No (default `false`) | Set `true` only on the prod deployment. Hides the sidebar's "Server address" box and fixes it to `REFLEX_BE_BASE_URL`, so a viewer's own token can't be used to repoint prod at a different backend. |

The Talentbook access token used for a given chat is never stored here —
each visitor pastes their own in the sidebar, kept only in that browser
session. The server address isn't either, except on the prod deployment,
where it's intentionally fixed rather than left to the viewer.

`.streamlit/config.toml` holds Streamlit's own runtime settings (usage-stats
opt-out, headless mode) — not secrets, and safe to commit as-is.

## Deploy to Streamlit Cloud

Point a new Streamlit Cloud app at this repo's `app_mcp_chat_streamlit.py`
as the main file, then fill in the table above under **Settings ->
Secrets**. No code or config change is needed between local dev and the
cloud deploy.
