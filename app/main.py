"""Streamlit entrypoint for the OpenAI Realtime Voice Agent (Voice-Only).

Run with:
    streamlit run app/main.py

This module wires together the split-screen UI and the VoiceAgentBridge.
It contains *zero* WebSocket or audio logic — all of that lives behind
the bridge adapter.

Layout:
    LEFT column  = USER panel  (mic button, user transcript)
    RIGHT column = AGENT panel (avatar, agent status, agent transcript)
"""

from __future__ import annotations

import streamlit as st

from app.models.config import AppSettings, RealtimeConfig, TurnDetectionConfig
from app.models.enums import ConnectionState, MicState, Modality, Voice
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
# Sidebar — configuration panel
# ---------------------------------------------------------------------------
cfg = render_sidebar_config()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("OpenAI Realtime Voice Agent")
st.caption("Voice-only interaction — press the mic button to speak")

# ---------------------------------------------------------------------------
# Connection controls
# ---------------------------------------------------------------------------
connect_clicked, disconnect_clicked = render_connection_controls(bridge.connection_state)

if connect_clicked:
    api_key = cfg["api_key"]
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        settings = AppSettings(openai_api_key=api_key)
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
    st.rerun()

# ---------------------------------------------------------------------------
# Split-screen: USER (left) | AGENT (right)
# ---------------------------------------------------------------------------
col_user, col_agent = st.columns(2)

is_connected = bridge.connection_state == ConnectionState.CONNECTED

with col_user:
    mic_clicked = render_user_panel(
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
# Mic button logic (push-to-talk toggle)
# ---------------------------------------------------------------------------
if mic_clicked and is_connected:
    if bridge.mic_state == MicState.IDLE:
        # Start recording
        bridge.mic_state = MicState.RECORDING
        st.rerun()
    elif bridge.mic_state == MicState.RECORDING:
        # Stop recording -> commit audio -> request response
        bridge.commit_audio()
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
