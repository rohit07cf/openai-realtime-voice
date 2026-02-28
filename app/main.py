"""Streamlit entrypoint for the OpenAI Realtime Voice Agent.

Run with:
    streamlit run app/main.py

This module wires together the UI components and the VoiceAgentBridge.
It contains *zero* WebSocket or audio logic — all of that lives behind
the bridge adapter.
"""

from __future__ import annotations

import streamlit as st

from app.models.config import AppSettings, RealtimeConfig, TurnDetectionConfig
from app.models.enums import ConnectionState, Modality, Voice
from app.ui.bridge import VoiceAgentBridge
from app.ui.components import (
    render_audio_status,
    render_connection_controls,
    render_debug_log,
    render_demo_input,
    render_errors,
    render_sidebar_config,
    render_transcript,
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
st.caption("Production-ready voice agent powered by OpenAI's Realtime API")

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
# Main content area
# ---------------------------------------------------------------------------
col_main, col_side = st.columns([3, 1])

with col_main:
    # Transcript
    render_transcript(bridge.transcript)

    # Demo mode text input
    text = render_demo_input()
    if text and bridge.connection_state == ConnectionState.CONNECTED:
        bridge.send_text_message(text)
        st.rerun()
    elif text and bridge.connection_state != ConnectionState.CONNECTED:
        st.warning("Please connect first.")

with col_side:
    # Audio status
    render_audio_status(
        buffer_depth_sec=bridge.audio_buffer.depth_seconds,
        last_event_time=bridge.last_event_time,
    )
    # Errors
    render_errors(bridge.errors)
    # Debug log
    render_debug_log(bridge.event_log)

# ---------------------------------------------------------------------------
# Auto-refresh while connected (poll for new events)
# ---------------------------------------------------------------------------
if bridge.connection_state == ConnectionState.CONNECTED:
    import time

    time.sleep(0.5)
    st.rerun()
