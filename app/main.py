"""Streamlit entrypoint for the OpenAI Realtime Voice Agent (Voice-Only).

Run with:
    streamlit run app/main.py

This module wires together the split-screen UI and the VoiceAgentBridge.
It contains *zero* WebSocket or audio logic — all of that lives behind
the bridge adapter.

Layout:
    LEFT column  = USER panel  (audio input, user transcript)
    RIGHT column = AGENT panel (avatar, agent status, agent transcript)
"""

from __future__ import annotations

import logging

import streamlit as st

from app.models.config import AppSettings, RealtimeConfig, TurnDetectionConfig
from app.models.enums import ConnectionState, Modality, Voice
from app.ui.bridge import VoiceAgentBridge
from app.ui.components import (
    render_agent_panel,
    render_audio_status,
    render_connection_controls,
    render_debug_log,
    render_errors,
    render_sidebar_config,
    render_user_panel,
)
from app.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OpenAI Realtime Voice Agent",
    page_icon="🎙️",
    layout="wide",
)

setup_logging("INFO")

# ---------------------------------------------------------------------------
# Session state: keep the bridge alive across Streamlit re-runs
# ---------------------------------------------------------------------------
if "bridge" not in st.session_state:
    st.session_state.bridge = VoiceAgentBridge()

bridge: VoiceAgentBridge = st.session_state.bridge

# ---------------------------------------------------------------------------
# Sidebar — configuration panel (API key loaded from env / .env)
# ---------------------------------------------------------------------------
cfg = render_sidebar_config()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("OpenAI Realtime Voice Agent")
st.caption("Voice-only interaction — record a message and it will be sent to the agent")

# ---------------------------------------------------------------------------
# Connection controls
# ---------------------------------------------------------------------------
connect_clicked, disconnect_clicked = render_connection_controls(bridge.connection_state)

if connect_clicked:
    try:
        settings = AppSettings()
    except Exception:
        st.error(
            "**OPENAI_API_KEY** is not configured. "
            "Set it as an environment variable or in a `.env` file."
        )
        settings = None

    if settings is not None:
        turn_detection = None
        if cfg["turn_detection"]:
            turn_detection = TurnDetectionConfig(**cfg["turn_detection"])
        realtime_cfg = RealtimeConfig(
            model=cfg["model"],
            voice=Voice(cfg["voice"]),
            modalities=[Modality(m) for m in cfg["modalities"]],
            temperature=cfg["temperature"],
            instructions=cfg["instructions"],
            turn_detection=turn_detection,
        )
        with st.spinner("Connecting to OpenAI Realtime API..."):
            success = bridge.connect(settings, realtime_cfg)
        if success:
            st.success("Connected!")
            st.rerun()
        else:
            st.error("Connection failed. Check your API key and network.")

if disconnect_clicked:
    bridge.disconnect()
    st.session_state.pop("_last_audio_bytes", None)
    st.rerun()

# ---------------------------------------------------------------------------
# Split-screen: USER (left) | AGENT (right)
# ---------------------------------------------------------------------------
col_user, col_agent = st.columns(2)

is_connected = bridge.connection_state == ConnectionState.CONNECTED

with col_user:
    audio_data = render_user_panel(
        mic_state=bridge.mic_state,
        user_transcript=bridge.user_transcript,
        is_connected=is_connected,
    )

with col_agent:
    render_agent_panel(
        agent_state=bridge.agent_state,
        agent_transcript=bridge.agent_transcript,
        buffer_depth_sec=bridge.audio_buffer.depth_seconds,
    )

# ---------------------------------------------------------------------------
# Process new audio recording (deduplicate across Streamlit reruns)
# ---------------------------------------------------------------------------
if audio_data is not None and is_connected:
    audio_bytes = audio_data.getvalue()
    if audio_bytes != st.session_state.get("_last_audio_bytes"):
        st.session_state["_last_audio_bytes"] = audio_bytes
        with st.spinner("Sending audio to agent..."):
            try:
                bridge.send_wav_audio(audio_bytes)
            except Exception:
                logger.exception("Failed to send audio")
                st.error("Failed to send audio. Please try again.")
        st.rerun()

# ---------------------------------------------------------------------------
# Footer: audio status + debug log
# ---------------------------------------------------------------------------
st.divider()
footer_left, footer_right = st.columns(2)
with footer_left:
    render_audio_status(
        buffer_depth_sec=bridge.audio_buffer.depth_seconds,
        last_event_time=bridge.last_event_time,
    )
with footer_right:
    render_errors(bridge.errors)
    render_debug_log(bridge.event_log)

# ---------------------------------------------------------------------------
# Auto-refresh while connected (poll for new events)
# ---------------------------------------------------------------------------
if bridge.connection_state == ConnectionState.CONNECTED:
    import time

    time.sleep(0.5)
    st.rerun()
