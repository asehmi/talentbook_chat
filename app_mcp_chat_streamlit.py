"""Chat with Talentbook — standalone, real LLM chat, not a manual tool REPL.

A genuine LLM agentic-tool-use loop (`lib.llm.get_llm_client()`) over a
running Talentbook backend's MCP connector, through `mcp_st_bridge.py` — the
LLM itself decides which tools to call from the conversation, based on
whatever access token the viewer pastes in.

There's no default credential: the sidebar requires a viewer to paste their
own `tb_pat_...` token (personal or org-scoped — whichever that token's own
scopes grant decides what shows up) before the chat does anything. The
name/token fields sit in their own form with a submit button — needed
because a browser-autofilled credential fires no on_change event, so
without an explicit submit action nothing would tell Streamlit the fields
changed. An optional name field next to the token exists only to help the
browser's own password manager label a saved token.

A second page, Instructions, walks a new visitor through getting a token
and trying the chat. Both pages are registered with `st.navigation(...,
position="hidden")` so Streamlit's own page list never appears — the only
way between them is the page_link each one renders itself, placed exactly
where that page wants it (the sidebar's bottom for Chat -> Instructions).

The MCP server address is normally also editable in the sidebar (prefilled
from `BACKEND_BASE_URL`), so the same deploy can be pointed at a different
backend. On the prod deployment (`IS_PROD` secret), that box is hidden and
the address is fixed to `BACKEND_BASE_URL` — a viewer's own token doesn't
let them repoint prod at a different backend.

Run with the Talentbook backend already running elsewhere (its MCP
connector mounted at `/mcp`):

    streamlit run app_mcp_chat_streamlit.py
"""

from __future__ import annotations

import streamlit as st
from streamlit import session_state as state

from lib.core.env import IS_PROD, BACKEND_BASE_URL, FRONTEND_BASE_URL
from lib.core.time import utcnow
from mcp_st_bridge import McpConnectionError, get_bridge, stream_reply, whoami

st.set_page_config(
    page_title="Chat with Talentbook",
    page_icon=":material/forum:",
    layout="centered",
)

# The same wordmark used in Talentbook's own HTML articles/pitch decks
# (assets/images/talentbook-name-logo-indigo-transparent.png in the main
# repo) — st.logo() puts it at the top of the sidebar (and the app header)
# on its own, so it has to be called early, before the sidebar is built.
st.logo("assets/talentbook-logo.png", size="large")

st.markdown("""
<style>
div[data-testid="stMainBlockContainer"] {
    padding-top: 2rem;
    /* padding-right: 70rem; */
    /* padding-left: 5rem; */
    padding-bottom: 2rem;
}
div[data-testid="stChatInput"] {
    padding-top: 2rem;
    /* padding-right: 60rem; */
    /* padding-left: 0rem; */
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


def _transcript_markdown(messages: list[dict[str, str]]) -> str:
    """The full chat history as one Markdown document, for the Export chat
    button — plain speaker headings so it reads naturally pasted into a
    doc or opened as a .md file, rather than a raw JSON dump."""
    lines = [f"# Chat with Talentbook — exported {utcnow():%Y-%m-%d %H:%M} UTC", ""]
    for msg in messages:
        speaker = "You" if msg["role"] == "user" else "Assistant"
        lines += [f"## {speaker}", "", msg["content"], ""]
    return "\n".join(lines)


def _sidebar_instructions_link() -> None:
    """Called from both the token-gate's early-stop path and the bottom of
    the connected sidebar — a new visitor with no token yet still needs to
    find this before they can get one, so it can't live only behind the
    Example questions block, which needs a live connection to populate."""
    st.divider()
    st.page_link(_instructions_page, icon=":material/menu_book:")


def _export_button(messages: list[dict[str, str]]) -> None:
    st.download_button(
        "Export chat",
        data=_transcript_markdown(messages),
        file_name=f"talentbook-chat-{utcnow():%Y%m%d-%H%M}.md",
        mime="text/markdown",
        icon=":material/download:",
    )


def _apply_token_switch(new_token: str, new_server_address: str) -> None:
    """Make the typed-in (token, server_address) the active one. Appends a
    boundary message rather than clearing history — the earlier turns stay
    visible on purpose, so the LLM actually has to honor the boundary
    instead of it being moot because there's nothing left to misread. No
    confirmation needed: this message is the fix, worded for the LLM itself
    rather than just the human, so there's nothing left for a confirmation
    to protect against.

    The boundary message itself is skipped when there's no history yet
    (e.g. the token was changed before ever sending a message, or an
    earlier connection attempt failed) — there's nothing for it to bound,
    so it would just be a stray assistant message sitting above an empty
    chat. The (token, server_address) switch below always happens
    regardless — it's what makes a resubmitted form actually take effect,
    not just the message."""
    if state.chat_messages:
        state.chat_messages.append(
            {
                "role": "assistant",
                # Read back via _format_transcript on every later turn, so
                # this is an instruction to the LLM, not just a notice to
                # the human — everything above this line was said under a
                # different token's scopes and may no longer hold.
                "content": (
                    ":material/swap_horiz: **Token changed.** Everything above this "
                    "point was said under a different access token, possibly with "
                    "different scopes — don't treat it as still true. Check this "
                    "token's own available tools before answering what it can or "
                    "can't do."
                ),
            }
        )
    state.active_token = new_token
    state.active_server_address = new_server_address


def _render_chat() -> None:
    # Streamlit clears a widget's session_state entry on any run where
    # that widget isn't instantiated. The credentials form and the server
    # address box below only render on *this* page — so a round trip
    # through the Instructions page (which never renders them) wipes
    # both, and returning here used to re-trigger the "enter a token"
    # gate below even though the connection, and the chat history under
    # it, was still live. `chat_messages`/`active_token` themselves are
    # plain session_state, not widget state, so they're never touched by
    # this — only the form fields lose their value, and the gate was the
    # thing hiding the still-intact history behind them. Re-seed both
    # keys from the already-active connection before the widgets below
    # are created, so a return trip shows what's actually connected
    # instead of an empty form.
    if "chat_messages" not in state:
        state.chat_messages = []
    if "active_token" not in state:
        state.active_token = None
        state.active_server_address = None
    if "token" not in state and state.active_token:
        state.token = state.active_token
    if "server_address" not in state:
        state.server_address = state.active_server_address or BACKEND_BASE_URL
    # token_name has no other state backing it (it's never read for
    # anything but the browser's own password manager — see the form
    # below), so without its own persisted copy it would reappear blank
    # after a round trip even once token/server_address were fixed.
    if "token_name" not in state and state.get("active_token_name"):
        state.token_name = state.active_token_name

    with st.sidebar:
        st.subheader(
            ":material/key: Access token",
            help=(
                "Paste a `tb_pat_...` personal access token — personal or "
                "org-scoped, either works. Nothing connects until one is here."
            ),
        )
        # In a form so nothing connects on every keystroke, and so a
        # browser-autofilled Name+Token pair (which fires no on_change
        # event at all — Streamlit never sees it until something inside
        # the form submits) has an obvious, visible action to trigger it:
        # click Submit, or hit Enter while either field has focus, same as
        # any other HTML form. Without the form, autofill silently left
        # the token unset until you clicked into it and pressed Enter
        # yourself, with nothing on the page suggesting that was needed.
        with st.form("credentials_form", border=False):
            # The name field is never read by this page — it exists purely
            # so the browser's own password manager has something to label
            # the saved credential with (autocomplete="username" pairs it
            # with the password field below), since someone chatting as
            # several different tokens over time would otherwise see them
            # all saved under this one page's address with nothing to tell
            # them apart.
            st.text_input(
                "Name (optional)",
                key="token_name",
                placeholder="e.g. TB-PAT (for read-only)",
                autocomplete="username",
                help="Only used to label this token in your browser's saved "
                "passwords — never sent anywhere or read by this app.",
            )
            st.text_input(
                "Token",
                type="password",
                key="token",
                placeholder="tb_pat_...",
                autocomplete="current-password",
            )
            authenticated = st.form_submit_button(
                "Authenticate", icon=":material/login:", width="stretch", type="primary"
            )
        token = state.get("token", "")
        if state.get("token_name"):
            state.active_token_name = state.token_name

        # On the prod deployment (IS_PROD secret), the server address is
        # fixed to BACKEND_BASE_URL and never shown — a viewer pasting in
        # their own token still can't repoint this deployment at a
        # different backend. Everywhere else (local dev, a staging
        # deploy), it's an editable box defaulting to BACKEND_BASE_URL,
        # since pointing at a different backend is the whole point there.
        if IS_PROD:
            server_address = BACKEND_BASE_URL
        else:
            st.subheader(":material/dns: MCP server address")
            server_address = st.text_input(
                "Server address",
                key="server_address",
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
        prompt = (
            "Enter a Talentbook access token in the sidebar to start chatting"
            if IS_PROD
            else "Enter a Talentbook access token and server address in the sidebar "
                 "to start chatting"
        )
        st.info(
            f"{prompt} — issue a token from the Connected Agents page.",
            icon=":material/key:",
        )
        with st.sidebar:
            _sidebar_instructions_link()
        st.stop()

    if authenticated:
        # get_bridge/whoami are st.cache_resource/st.cache_data, keyed on
        # (token, server_address) — without this, clicking Authenticate
        # with the exact same token and server address (e.g. retrying
        # after the backend's MCP session idle-timed-out or the backend
        # restarted) would just hand back the same cached bridge and its
        # now-dead session, forever, since nothing about the cache key
        # changed. Clearing on every click makes Authenticate double as
        # an explicit reconnect, not just a credential-switch detector.
        get_bridge.clear(token, server_address)
        whoami.clear(token, server_address)

    if state.active_token is None:
        # First-ever connection this session — nothing to clear.
        state.active_token = token
        state.active_server_address = server_address
    elif (token, server_address) != (
        state.active_token,
        state.active_server_address,
    ):
        _apply_token_switch(token, server_address)

    active_token = state.active_token
    active_server_address = state.active_server_address

    try:
        bridge = get_bridge(active_token, active_server_address)
        me = whoami(active_token, active_server_address)
    except McpConnectionError as exc:
        st.error(str(exc), icon=":material/link_off:")
        st.stop()

    # get_my_profile is personal-only — present in the tool catalog iff
    # this token carries read:profile, which an org-scoped token can
    # never be granted (see lib/core/models.py's MCP_ORG_SCOPES) — so its
    # presence is a direct, reliable personal-vs-org signal, independent
    # of whether whoami() below found an identity to report at all.
    is_personal = any(t.name == "get_my_profile" for t in bridge.tools)

    with st.container(horizontal=True, vertical_alignment="center"):
        if me is not None:
            # whoami's server-side tool reports the real account either
            # way — this token's own owner for a personal token, the
            # issuing admin's for an org one — so the same "Connected as
            # NAME (EMAIL)" shape now applies to both, not just personal.
            display_name = me.get("name") or me.get("email", "unknown")
            status = (
                f":material/link: Connected as **{display_name}** (`{me.get('email', '?')}`) "
                f"— {len(bridge.tools)} tool(s) available"
            )
        else:
            # Only when talking to a server old enough to predate the
            # whoami tool — no identity to report, just the tool count.
            status = f":material/link: Connected — {len(bridge.tools)} tool(s) available"
        st.caption(status)
        with st.container(horizontal=True, horizontal_alignment="right"):
            # A placeholder rather than rendering the button here directly
            # — at this point in the script we don't know yet whether
            # this run ends with any messages (a prompt sent this run can
            # still fail, and a fresh chat has none at all), so there's
            # nothing valid to decide "show it or not" with yet. Filled in
            # once that's known, at the end of the function.
            export_placeholder = st.empty()
            if st.button("New chat", icon=":material/add_comment:"):
                state.chat_messages = []
                st.rerun()

    # Two flavors, picked below by `is_personal` — each tool catalog
    # covers different data, so the example questions have to too.
    # "Tasks you can perform" leads both: whichever kind of token this
    # is, it's the one question that always makes sense before the
    # viewer knows what's actually on offer.
    PERSONAL_SUGGESTIONS = {
        ":material/help: Tasks you can perform": (
            "What kinds of things can you help me with?"
        ),
        ":material/flag: My current goals": (
            "What are my current goals, and which one should I focus on next?"
        ),
        ":material/target: My biggest skill gaps": "Where are my biggest skill gaps right now?",
        ":material/edit_note: Recent journal themes": (
            "Summarize what I've been writing about in my journal lately."
        ),
        ":material/dashboard: My canvas, explained": "Walk me through my Business Model You canvas.",
    }
    ORG_SUGGESTIONS = {
        ":material/help: Tasks you can perform": (
            "What kinds of things can you help me with for this organisation?"
        ),
        ":material/groups: Our org roster": "Give me an overview of our organisation's roster.",
        ":material/target: Role baselines": (
            "What role baselines do we have defined, and what do they require?"
        ),
        ":material/work: Work-job templates": (
            "Walk me through our work-job templates and the skills they need."
        ),
    }

    # In the sidebar, not the main column, and always rendered rather than
    # only before the first message — a plain main-column widget would
    # disappear the moment chat_messages stops being empty (it did, before
    # this change), which both hides a feature that's still useful mid-
    # conversation and means its on-screen presence depends on history
    # state surviving reruns exactly right. The sidebar has no such
    # condition to get wrong.
    with st.sidebar:
        st.subheader(":material/lightbulb: Example questions")
        suggestions = PERSONAL_SUGGESTIONS if is_personal else ORG_SUGGESTIONS
        pills_key = f"suggestion_pills_{len(state.chat_messages)}"

        def _apply_suggestion() -> None:
            # Runs as st.pills' on_change callback — before this rerun's
            # script body executes — so state.pending_prompt is already
            # set by the time it's read below. Reading st.pills' own
            # return value directly used to need an extra click: Streamlit
            # schedules the rerun as soon as the click lands, but a value
            # read inline further down the script isn't guaranteed to
            # reflect that same click until the callback has run first.
            picked = state.get(pills_key)
            if picked:
                state.pending_prompt = suggestions[picked]

        st.pills(
            "Try asking:",
            list(suggestions.keys()),
            label_visibility="collapsed",
            # st.pills has no "fire once" mode — its selection otherwise
            # stays stuck after being read once, re-sending the same
            # question on every later rerun. A key that changes with the
            # message count makes each send start a fresh, unselected
            # widget instead of carrying the old selection forward.
            key=pills_key,
            on_change=_apply_suggestion,
        )
        _sidebar_instructions_link()
    suggested_prompt = state.pop("pending_prompt", None)

    typed = st.chat_input(
        "Ask about your goals, skills, OKRs, journal, or canvas…",
        submit_mode="disable",
    )
    prompt = typed or suggested_prompt

    messages = state.chat_messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt:
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            reply = st.write_stream(_safe_stream(stream_reply(messages, bridge)))
            messages.append({"role": "assistant", "content": reply})

    with export_placeholder:
        if messages:
            _export_button(messages)


def _render_instructions() -> None:
    with st.sidebar:
        st.page_link(_chat_page, icon=":material/forum:")

    st.title(":material/menu_book: Instructions")
    st.caption("Everything you need to get a token and start chatting with your own Talentbook data.")

    st.subheader("1. Get two access tokens")
    st.markdown(
        "Log into your own Talentbook account, then open "
        f'<a href="{FRONTEND_BASE_URL.rstrip("/")}/app/connected-agents" '
        'target="_blank" rel="noopener">Connected Agents</a> (opens in a new '
        "tab, so you stay logged in there and don't lose this chat). This "
        "page issues two different kinds of token, and there's no reason "
        "not to create both now — they cover different data, and later steps "
        "here use each one:\n\n"
        "- **Personal token** (under *Your tokens*) — your own goals, skills, "
        "OKRs, journal, canvas, and consultations.\n"
        "- **Org token** (under *Org Integrations*) — your organisation's "
        "roster, role baselines, and work-job templates. Not part of a "
        "company org? Talentbook still gives you an \"org of one\" scoped to "
        "just your own catalog, so this applies to you too.\n\n"
        "For each one, click **Create token**, give it a name, and check the "
        "scopes you want it to have — nothing is selected by default, and the "
        "token needs at least one. Copy the token (`tb_pat_...`) as soon as "
        "it's shown — it isn't shown again, and revoking it later is free if "
        "you're just testing.",
        unsafe_allow_html=True,
    )

    st.subheader("2. Connect")
    st.markdown(
        "Paste a name and one of your tokens into this app's sidebar and "
        "click **Authenticate**. You should see *\"Connected as YOU (your "
        "email) — N tool(s) available\"* above the chat box. You can switch "
        "which token is active anytime — paste the other one in and click "
        "**Authenticate** again; your chat history stays, and the assistant "
        "is told not to treat anything said before the switch as still true "
        "under the new token's scopes."
    )

    st.subheader("3. Set up your skills inventory first")
    st.markdown(
        "The example prompt below compares two canvases, so build at least "
        "one in Talentbook before trying it. The usual way is onboarding: go "
        "through it and rate your proficiency across the skill taxonomy, "
        "then save it as a canvas (for example, **Current Role**). Create a "
        "second canvas for a different snapshot — a target role, a future "
        "scenario, or a re-rating after some time — so you have two to "
        "compare (for example, **Target Role**)."
    )
    st.markdown(
        "You don't have to go through onboarding by hand, though — as an "
        "admin of your own catalog, the same built-in designer is reachable "
        "from chat. Connect with your **Org token** (with read and write "
        "scopes on organisation roles and work-job templates), then ask:"
    )
    st.markdown(
        "- *\"Show me what skills catalogs are available?\"* — lists the "
        "catalogs you can set yourself up against.\n"
        "- *\"Set me up as 'Solo Entrepreneur'\"* — or name one from the list "
        "the previous answer gave you.\n"
        "- *\"Show me the skills targets, and set all my critical skills as "
        "tracked.\"*"
    )

    st.subheader("4. Try this prompt")
    st.markdown(
        "Switch back to your **Personal token** first — with all its Read "
        "scopes checked (profile, goals, skills, OKRs, journal, canvas, "
        "consultations) — since this prompt reads across your own canvases, "
        "not organisation data."
    )
    st.code(
        "Compare the CANVAS-1 and CANVAS-2 canvases and tell me what skills I "
        "need to drop, adopt anew, and work on. Give me work strategies for "
        "addressing the gaps in proficiency, and perhaps skills that are deemed "
        "not critical that I should track nonetheless. Also advise on "
        "consultations that I can take to help close the gap.",
        language=None,
    )
    st.caption("Swap CANVAS-1 and CANVAS-2 for your own canvas names.")

    st.subheader("5. Keep your tokens safe")
    st.markdown(
        "Treat each one like a password — don't share it, and revoke it from "
        "Connected Agents once you're done testing."
    )


_chat_page = st.Page(
    _render_chat, title="Chat with Talentbook", icon=":material/forum:", default=True
)
_instructions_page = st.Page(
    _render_instructions, title="Instructions", icon=":material/menu_book:"
)
# position="hidden" suppresses Streamlit's own page list (which would
# otherwise sit at the top of the sidebar, ahead of the credentials form) —
# each page links to the other itself, exactly where that page wants it.
st.navigation([_chat_page, _instructions_page], position="hidden").run()
