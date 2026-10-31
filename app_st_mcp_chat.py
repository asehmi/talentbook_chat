"""Chat with Talentbook — standalone, real LLM chat, not a manual tool REPL.

A genuine LLM agentic-tool-use loop (`lib.llm.get_llm_client()`) over a
running Talentbook backend's MCP connector, through `mcp_chat.py` — the LLM
itself decides which tools to call from the conversation, based on whatever
access token the viewer pastes in.

There's no default credential: the sidebar requires a viewer to paste their
own `tb_pat_...` token (personal or org-scoped — whichever that token's own
scopes grant decides what shows up) and a server address before the chat
does anything. An optional name field next to the token exists only to help
the browser's own password manager label a saved token.

Run with the Talentbook backend already running elsewhere (its MCP
connector mounted at `/mcp`):

    streamlit run app_st_mcp_chat.py
"""

from __future__ import annotations

import streamlit as st

from lib.core.env import BACKEND_BASE_URL
from mcp_chat import McpConnectionError, get_bridge, stream_reply, whoami

st.set_page_config(
    page_title="Chat with Talentbook",
    page_icon=":material/forum:",
    layout="wide",
)

st.markdown("""
<style>
div[data-testid="stMainBlockContainer"] {
    padding-top: 2rem;
    padding-right: 70rem;
    padding-left: 5rem;
    padding-bottom: 2rem;
}
div[data-testid="stChatInput"] {
    padding-top: 2rem;
    padding-right: 60rem;
    padding-left: 0rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


def _safe_stream(chunks):
    """Let a mid-stream failure (MCP connection drop, LLM API error, ...)
    show up as part of the assistant's own message instead of crashing the
    page — the user is mid-conversation, not loading a fresh screen."""
    try:
        yield from chunks
    except Exception as exc:
        yield f"\n\n:material/error: Something went wrong: {exc}"


def _apply_token_switch(new_token: str, new_server_address: str) -> None:
    """Make the typed-in (token, server_address) the active one. Appends a
    boundary message rather than clearing history — the earlier turns stay
    visible on purpose, so the LLM actually has to honor the boundary
    instead of it being moot because there's nothing left to misread. No
    confirmation needed: this message is the fix, worded for the LLM itself
    rather than just the human, so there's nothing left for a confirmation
    to protect against."""
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            # Read back via _format_transcript on every later turn, so this
            # is an instruction to the LLM, not just a notice to the human —
            # everything above this line was said under a different token's
            # scopes and may no longer hold.
            "content": (
                ":material/swap_horiz: **Token changed.** Everything above this "
                "point was said under a different access token, possibly with "
                "different scopes — don't treat it as still true. Check this "
                "token's own available tools before answering what it can or "
                "can't do."
            ),
        }
    )
    st.session_state.active_token = new_token
    st.session_state.active_server_address = new_server_address


def _render() -> None:
    with st.sidebar:
        st.markdown(":material/key: **Talentbook access token**")
        st.caption(
            "Paste a `tb_pat_...` personal access token — personal or "
            "org-scoped, either works. Nothing connects until one is here."
        )
        # The name field is never read by this page — it exists purely so
        # the browser's own password manager has something to label the
        # saved credential with (autocomplete="username" pairs it with the
        # password field below), since someone chatting as several
        # different tokens over time would otherwise see them all saved
        # under this one page's address with nothing to tell them apart.
        st.text_input(
            "Name (optional)",
            key="token_name",
            placeholder="e.g. Priya personal",
            autocomplete="username",
            help="Only used to label this token in your browser's saved "
            "passwords — never sent anywhere or read by this app.",
        )
        token = st.text_input(
            "Token",
            type="password",
            key="token",
            placeholder="tb_pat_...",
            autocomplete="current-password",
        )

        st.divider()
        st.markdown(":material/dns: **MCP server address**")
        server_address = st.text_input(
            "Server address",
            key="server_address",
            value=BACKEND_BASE_URL,
            placeholder="http://localhost:8888",
            help="Base URL only — the /mcp/ endpoint path is added automatically.",
        )

    st.title(":material/forum: Chat with Talentbook")
    st.caption(
        "A live conversation with an LLM that can read your own Talentbook data over "
        "MCP — profile, goals, skills, OKRs, journal, canvas, and consultations — "
        "deciding for itself which tools to call, the same way an agent like Claude "
        "Desktop would."
    )

    server_address = server_address.strip()
    if not token or not server_address:
        st.info(
            "Enter a Talentbook access token and server address in the sidebar "
            "to start chatting — issue a token from the Connected Agents page.",
            icon=":material/key:",
        )
        st.stop()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "active_token" not in st.session_state:
        st.session_state.active_token = None
        st.session_state.active_server_address = None

    if st.session_state.active_token is None:
        # First-ever connection this session — nothing to clear.
        st.session_state.active_token = token
        st.session_state.active_server_address = server_address
    elif (token, server_address) != (
        st.session_state.active_token,
        st.session_state.active_server_address,
    ):
        _apply_token_switch(token, server_address)

    active_token = st.session_state.active_token
    active_server_address = st.session_state.active_server_address

    try:
        bridge = get_bridge(active_token, active_server_address)
        me = whoami(active_token, active_server_address)
    except McpConnectionError as exc:
        st.error(str(exc), icon=":material/link_off:")
        st.stop()

    with st.container(horizontal=True, vertical_alignment="center"):
        if me is not None:
            status = (
                f":material/link: Connected as **{me.get('name', 'unknown')}** "
                f"(`{me.get('email', '?')}`)"
            )
        else:
            status = f":material/link: Connected — {len(bridge.tools)} org tool(s) available"
        st.caption(status)
        with st.container(horizontal_alignment="right"):
            if st.button("New chat", icon=":material/add_comment:"):
                st.session_state.chat_messages = []
                st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Personal-flavored — only offered for a personal token (me is not
    # None); an org token has no "my goals" of its own to suggest.
    SUGGESTIONS = {
        ":material/flag: My current goals": (
            "What are my current goals, and which one should I focus on next?"
        ),
        ":material/target: My biggest skill gaps": "Where are my biggest skill gaps right now?",
        ":material/edit_note: Recent journal themes": (
            "Summarize what I've been writing about in my journal lately."
        ),
        ":material/dashboard: My canvas, explained": "Walk me through my Business Model You canvas.",
    }

    prompt = None
    if not st.session_state.chat_messages and me is not None:
        selected = st.pills(
            "Try asking:", list(SUGGESTIONS.keys()), label_visibility="collapsed"
        )
        if selected:
            prompt = SUGGESTIONS[selected]

    typed = st.chat_input(
        "Ask about your goals, skills, OKRs, journal, or canvas…",
        submit_mode="disable",
    )
    if typed:
        prompt = typed

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            reply = st.write_stream(
                _safe_stream(stream_reply(st.session_state.chat_messages, bridge))
            )
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})


_render()
