"""Streamlit UI components for the split-screen Voice-Only Agent.

Responsibility: Render all visual elements — sidebar config, user panel
    (mic button + transcript), agent panel (avatar + transcript), and
    status / debug displays.  No text-input chat fields exist.
Pattern: Presentational components — each function receives data and
    renders it; none contain WebSocket or audio logic.
Why: Keeping rendering separate from state management makes the UI
    testable, readable, and easy to reskin.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from app.models.enums import AgentSpeakingState, ConnectionState, MicState, Modality, Voice

# ---------------------------------------------------------------------------
# Default avatar placeholder (inline SVG data URI)
# ---------------------------------------------------------------------------
_DEFAULT_AVATAR_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'>"
    "<rect width='120' height='120' rx='60' fill='%234F46E5'/>"
    "<text x='50%25' y='54%25' text-anchor='middle' dominant-baseline='middle' "
    "font-family='sans-serif' font-size='48' fill='white'>AI</text>"
    "</svg>"
)


def render_sidebar_config() -> dict[str, Any]:
    """Render the configuration sidebar and return validated settings.

    Returns a dict with keys matching RealtimeConfig fields.
    The OpenAI API key is loaded from the environment (OPENAI_API_KEY)
    or a ``.env`` file — it is **not** entered in the sidebar.
    """
    st.sidebar.header("Session Configuration")

    model = st.sidebar.selectbox(
        "Model",
        options=[
            "gpt-4o-realtime-preview-2024-12-17",
            "gpt-4o-realtime-preview",
            "gpt-4o-mini-realtime-preview-2024-12-17",
            "gpt-4o-mini-realtime-preview",
        ],
        index=0,
    )

    voice = st.sidebar.selectbox(
        "Voice",
        options=[v.value for v in Voice],
        index=0,
    )

    modality_options = st.sidebar.multiselect(
        "Modalities",
        options=[m.value for m in Modality],
        default=["text", "audio"],
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.8,
        step=0.1,
    )

    instructions = st.sidebar.text_area(
        "System Instructions",
        value="You are a helpful, concise voice assistant.",
        height=100,
    )

    st.sidebar.subheader("Turn Detection")
    turn_mode = st.sidebar.selectbox(
        "Mode",
        options=["manual", "server_vad", "semantic_vad"],
        index=0,
        help="'manual' = push-to-talk (recommended). VAD modes auto-detect speech.",
    )

    turn_config = None
    if turn_mode != "manual":
        threshold = st.sidebar.slider("VAD Threshold", 0.0, 1.0, 0.5, 0.05)
        silence_ms = st.sidebar.slider("Silence Duration (ms)", 100, 5000, 500, 100)
        prefix_ms = st.sidebar.slider("Prefix Padding (ms)", 0, 2000, 300, 50)
        turn_config = {
            "type": turn_mode,
            "threshold": threshold,
            "silence_duration_ms": silence_ms,
            "prefix_padding_ms": prefix_ms,
        }

    # Agent avatar upload
    st.sidebar.subheader("Agent Avatar")
    uploaded = st.sidebar.file_uploader(
        "Upload avatar image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Displayed in the Agent panel. Optional.",
    )
    if uploaded is not None:
        st.session_state["agent_avatar"] = uploaded.getvalue()

    return {
        "model": model,
        "voice": voice,
        "modalities": modality_options or ["text", "audio"],
        "temperature": temperature,
        "instructions": instructions,
        "turn_detection": turn_config,
    }


def render_connection_controls(state: ConnectionState) -> tuple[bool, bool]:
    """Render Connect/Disconnect buttons. Returns (connect_clicked, disconnect_clicked)."""
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        connect = st.button(
            "Connect",
            disabled=state == ConnectionState.CONNECTED,
            type="primary",
        )
    with col2:
        disconnect = st.button(
            "Disconnect",
            disabled=state == ConnectionState.DISCONNECTED,
        )
    with col3:
        status_colors = {
            ConnectionState.CONNECTED: "🟢",
            ConnectionState.CONNECTING: "🟡",
            ConnectionState.RECONNECTING: "🟠",
            ConnectionState.CIRCUIT_OPEN: "🔴",
            ConnectionState.DISCONNECTED: "⚪",
        }
        icon = status_colors.get(state, "⚪")
        st.markdown(f"**Status:** {icon} {state.value.replace('_', ' ').title()}")

    return connect, disconnect


# ---------------------------------------------------------------------------
# Split-screen panels
# ---------------------------------------------------------------------------


def render_user_panel(
    mic_state: MicState,
    user_transcript: list[str],
    is_connected: bool,
) -> Any:
    """Render the USER (left) panel with ``st.audio_input`` for push-to-talk.

    Contains:
    - "USER" header
    - ``st.audio_input`` widget (browser mic capture)
    - Mic status indicator
    - Read-only transcript window

    Returns the recorded audio (``UploadedFile``) or ``None``.
    """
    st.markdown("### 🎙 USER")

    # Mic status indicator
    mic_labels = {
        MicState.IDLE: ("⚪ Ready", "secondary"),
        MicState.RECORDING: ("🔴 Recording...", "primary"),
        MicState.SENDING: ("🟡 Sending...", "secondary"),
    }
    label, _ = mic_labels.get(mic_state, ("⚪ Ready", "secondary"))
    st.caption(f"Mic status: **{label}**")

    # Audio input — browser-native microphone capture
    audio_data = st.audio_input(
        "🎤 Push to Talk",
        key="mic_input",
        disabled=not is_connected,
    )

    # Read-only transcript
    st.markdown("**Your utterances:**")
    transcript_container = st.container(height=280)
    with transcript_container:
        if not user_transcript:
            st.caption("Speak into your microphone to see transcripts here.")
        else:
            for utterance in user_transcript:
                st.markdown(f"> {utterance}")

    return audio_data


def render_agent_panel(
    agent_state: AgentSpeakingState,
    agent_transcript: list[str],
    buffer_depth_sec: float,
) -> None:
    """Render the AGENT (right) panel.

    Contains:
    - "AGENT" header
    - Agent avatar image
    - Agent speaking status indicator
    - Read-only transcript window
    """
    st.markdown("### 🤖 AGENT")

    # Agent avatar
    avatar_bytes = st.session_state.get("agent_avatar")
    if avatar_bytes:
        st.image(avatar_bytes, width=100)
    else:
        st.image(_DEFAULT_AVATAR_SVG, width=100)

    # Speaking status indicator
    agent_labels = {
        AgentSpeakingState.IDLE: "⚪ Idle",
        AgentSpeakingState.THINKING: "🟡 Thinking...",
        AgentSpeakingState.SPEAKING: "🟢 Speaking...",
    }
    status_text = agent_labels.get(agent_state, "⚪ Idle")
    st.caption(f"Agent status: **{status_text}**")

    if buffer_depth_sec > 0:
        st.caption(f"Audio buffer: {buffer_depth_sec:.2f}s")

    # Read-only transcript
    st.markdown("**Agent responses:**")
    transcript_container = st.container(height=280)
    with transcript_container:
        if not agent_transcript:
            st.caption("Agent responses will appear here.")
        else:
            for response in agent_transcript:
                st.markdown(f"💬 {response}")


def render_audio_status(buffer_depth_sec: float, last_event_time: float) -> None:
    """Render audio pipeline status metrics."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Buffer Depth", f"{buffer_depth_sec:.2f}s")
    with col2:
        if last_event_time > 0:
            ago = time.time() - last_event_time
            st.metric("Last Event", f"{ago:.1f}s ago")
        else:
            st.metric("Last Event", "N/A")


def render_debug_log(events: list[dict[str, Any]]) -> None:
    """Render the scrollable debug event log."""
    with st.expander("Debug Log (last 50 events)", expanded=False):
        if not events:
            st.caption("No events received yet.")
        else:
            for ev in reversed(events[-50:]):
                event_type = ev.get("type", "unknown")
                st.text(f"[{event_type}]")


def render_errors(errors: list[str]) -> None:
    """Display accumulated errors."""
    if errors:
        with st.expander(f"Errors ({len(errors)})", expanded=True):
            for err in errors[-10:]:
                st.error(err)
