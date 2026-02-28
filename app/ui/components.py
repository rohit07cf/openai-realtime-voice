"""Streamlit UI components for the Realtime Voice Agent.

Responsibility: Render all visual elements — config sidebar, transcript,
    audio status, debug log — using data from the VoiceAgentBridge.
Pattern: Presentational components — each function receives data and
    renders it; none contain WebSocket or audio logic.
Why: Keeping rendering separate from state management makes the UI
    testable, readable, and easy to reskin.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from app.models.enums import ConnectionState, Modality, Voice


def render_sidebar_config() -> dict[str, Any]:
    """Render the configuration sidebar and return validated settings.

    Returns a dict with keys matching RealtimeConfig fields.
    """
    st.sidebar.header("Session Configuration")

    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Required. Your OpenAI API key with Realtime access.",
    )

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
        options=["server_vad", "semantic_vad", "manual"],
        index=0,
        help="'manual' disables auto turn detection (push-to-talk).",
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

    return {
        "api_key": api_key,
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


def render_transcript(messages: list[dict[str, str]]) -> None:
    """Render the conversation transcript."""
    st.subheader("Conversation")
    container = st.container(height=400)
    with container:
        if not messages:
            st.caption("No messages yet. Connect and start talking or use Demo Mode.")
        for msg in messages:
            role = msg["role"]
            text = msg["text"]
            if role == "user":
                st.chat_message("user").write(text)
            else:
                st.chat_message("assistant").write(text)


def render_demo_input() -> str | None:
    """Render text input for demo mode (no mic)."""
    st.subheader("Demo Mode (Text Input)")
    st.caption("Send text messages to test the agent without a microphone.")
    with st.form("demo_form", clear_on_submit=True):
        text = st.text_input("Your message", placeholder="Type something...")
        submitted = st.form_submit_button("Send", type="primary")
    if submitted and text.strip():
        return text.strip()
    return None


def render_audio_status(buffer_depth_sec: float, last_event_time: float) -> None:
    """Render audio pipeline status metrics."""
    st.subheader("Audio Status")
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
