"""Streamlit UI components for the Realtime API Dashboard.

Styled to match a dark, minimal dashboard inspired by the OpenAI
Realtime API reference design.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from app.models.enums import AgentSpeakingState, ConnectionState, MicState, Modality, Voice

# ---------------------------------------------------------------------------
# Global CSS — injected once at the top of the page
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ---- root overrides ---- */
html, body, [data-testid="stAppViewContainer"], .main .block-container {
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0;
}
.main .block-container { padding-top: 1rem; padding-bottom: 0; max-width: 100%; }
header[data-testid="stHeader"] { display: none; }

/* ---- custom header bar ---- */
.rt-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 1.5rem;
    border-bottom: 1px solid #2D2D2D;
    margin: -1rem -1rem 1rem -1rem;
}
.rt-header-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: #fff;
}
.rt-conn-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.2rem 0.7rem;
    border: 1px solid #2D2D2D; border-radius: 4px;
    font-size: 0.6rem; font-family: monospace; letter-spacing: 0.12em;
    color: #94a3b8;
}
.rt-conn-dot {
    width: 6px; height: 6px; border-radius: 50%;
}
.rt-conn-dot.green  { background: #22c55e; }
.rt-conn-dot.yellow { background: #eab308; }
.rt-conn-dot.red    { background: #ef4444; }
.rt-conn-dot.gray   { background: #64748b; }

/* ---- section headers ---- */
.panel-label {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: #64748b; margin-bottom: 1rem;
}

/* ---- transcript items ---- */
.transcript-item {
    font-size: 1.15rem; font-weight: 300; line-height: 1.6;
    color: #fff; margin-bottom: 1.2rem;
}
.transcript-item.prev {
    opacity: 0.35; font-style: italic; color: #cbd5e1;
}
.transcript-hint {
    font-size: 0.75rem; color: #475569;
}

/* ---- status chips ---- */
.status-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase;
}
.status-chip.listening { color: #10a37f; }
.status-chip.thinking  { color: #eab308; }
.status-chip.speaking  { color: #10a37f; }
.status-chip.idle      { color: #64748b; }

/* ---- footer bar ---- */
.rt-footer {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.5rem 1.5rem;
    border-top: 1px solid #2D2D2D;
    margin: 0.5rem -1rem -1rem -1rem;
    font-size: 0.6rem; font-family: monospace; color: #64748b;
}
.rt-footer span.val { color: #fff; }
.rt-footer span.active { color: #22c55e; font-weight: 700; }

/* ---- mic area ---- */
.mic-area {
    display: flex; flex-direction: column; align-items: center;
    padding: 1.5rem 0 0.5rem 0;
    border-top: 1px solid rgba(45,45,45,0.5);
}
.mic-label {
    font-size: 0.65rem; font-weight: 900; letter-spacing: 0.25em;
    text-transform: uppercase; color: #fff; margin-top: 0.8rem;
}
.mic-sublabel {
    font-size: 0.6rem; color: #64748b; margin-top: 0.3rem;
}

/* ---- agent badge ---- */
.agent-badge {
    display: inline-block; padding: 0.15rem 0.6rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid #2D2D2D; border-radius: 4px;
    font-size: 0.6rem; font-family: monospace; color: #94a3b8;
    margin-bottom: 0.8rem;
}

/* ---- hide streamlit audio_input label ---- */
[data-testid="stAudioInput"] label { display: none !important; }

/* ---- divider override ---- */
hr { border-color: #2D2D2D !important; }
</style>
"""


def inject_custom_css() -> None:
    """Inject the global dashboard CSS once per page render."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header(state: ConnectionState) -> None:
    """Render the top header bar with title and connection badge."""
    dot_class = {
        ConnectionState.CONNECTED: "green",
        ConnectionState.CONNECTING: "yellow",
        ConnectionState.RECONNECTING: "yellow",
        ConnectionState.CIRCUIT_OPEN: "red",
        ConnectionState.DISCONNECTED: "gray",
    }.get(state, "gray")
    label = state.value.replace("_", " ").upper()

    st.markdown(f"""
    <div class="rt-header">
        <span class="rt-header-title">Realtime API</span>
        <div class="rt-conn-badge">
            <span class="rt-conn-dot {dot_class}"></span>
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar_config() -> dict[str, Any]:
    """Render the configuration sidebar and return validated settings."""
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
        "Temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.1,
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
        help="'manual' = push-to-talk (recommended).",
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

    st.sidebar.subheader("Audio Playback")
    enable_tts = st.sidebar.toggle(
        "Enable TTS (speak responses)",
        value=True,
        help="When enabled, the agent's audio response is played automatically.",
    )

    return {
        "model": model,
        "voice": voice,
        "modalities": modality_options or ["text", "audio"],
        "temperature": temperature,
        "instructions": instructions,
        "turn_detection": turn_config,
        "enable_tts": enable_tts,
    }


# ---------------------------------------------------------------------------
# Connection controls
# ---------------------------------------------------------------------------

def render_connection_controls(state: ConnectionState) -> tuple[bool, bool]:
    """Render Connect / Disconnect buttons."""
    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        connect = st.button(
            "Connect", disabled=state == ConnectionState.CONNECTED, type="primary",
        )
    with col2:
        disconnect = st.button(
            "Disconnect", disabled=state == ConnectionState.DISCONNECTED,
        )
    return connect, disconnect


# ---------------------------------------------------------------------------
# User panel (left)
# ---------------------------------------------------------------------------

def render_user_panel(
    mic_state: MicState,
    user_transcript: list[str],
    is_connected: bool,
) -> Any:
    """Render the User Input (Audio) panel.

    Returns the recorded audio (``UploadedFile``) or ``None``.
    """
    st.markdown('<div class="panel-label">User Input (Audio)</div>', unsafe_allow_html=True)

    # Status chip
    chip_map = {
        MicState.IDLE: ("Ready", "idle"),
        MicState.RECORDING: ("Listening...", "listening"),
        MicState.SENDING: ("Sending...", "thinking"),
    }
    text, cls = chip_map.get(mic_state, ("Ready", "idle"))
    st.markdown(f'<div class="status-chip {cls}">{text}</div>', unsafe_allow_html=True)

    # Transcript area
    transcript_container = st.container(height=300)
    with transcript_container:
        if not user_transcript:
            hint = "Speak into your microphone to see transcripts here."
            st.markdown(
                f'<p class="transcript-hint">{hint}</p>',
                unsafe_allow_html=True,
            )
        else:
            # Show all-but-last as "previous" style, last as "current"
            for utt in user_transcript[:-1]:
                st.markdown(
                    f'<p class="transcript-item prev">"{utt}"</p>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<p class="transcript-item">"{user_transcript[-1]}"</p>',
                unsafe_allow_html=True,
            )

    # Mic / audio input area
    st.markdown('<div class="mic-area">', unsafe_allow_html=True)
    audio_data = st.audio_input(
        "Push to Talk",
        key="mic_input",
        disabled=not is_connected,
    )
    st.markdown(
        '<span class="mic-label">PUSH TO TALK</span>'
        '<span class="mic-sublabel">Click to start speaking, click again to send</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    return audio_data


# ---------------------------------------------------------------------------
# Agent panel (right)
# ---------------------------------------------------------------------------

def render_agent_panel(
    agent_state: AgentSpeakingState,
    agent_transcript: list[str],
    buffer_depth_sec: float,
) -> None:
    """Render the Model Output (Text/Audio) panel."""
    label = "Model Output (Text / Audio)"
    st.markdown(
        f'<div class="panel-label">{label}</div>',
        unsafe_allow_html=True,
    )

    # Status chip
    chip_map = {
        AgentSpeakingState.IDLE: ("Idle", "idle"),
        AgentSpeakingState.THINKING: ("Thinking...", "thinking"),
        AgentSpeakingState.SPEAKING: ("Speaking...", "speaking"),
    }
    text, cls = chip_map.get(agent_state, ("Idle", "idle"))
    st.markdown(f'<div class="status-chip {cls}">{text}</div>', unsafe_allow_html=True)

    # Transcript area
    st.markdown('<span class="agent-badge">assistant</span>', unsafe_allow_html=True)

    transcript_container = st.container(height=300)
    with transcript_container:
        if not agent_transcript:
            st.markdown(
                '<p class="transcript-hint">Agent responses will appear here.</p>',
                unsafe_allow_html=True,
            )
        else:
            for resp in agent_transcript[:-1]:
                st.markdown(
                    f'<p class="transcript-item prev">{resp}</p>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<p class="transcript-item">{agent_transcript[-1]}</p>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def render_footer(
    buffer_depth_sec: float,
    last_event_time: float,
    is_connected: bool,
) -> None:
    """Render the bottom status bar with metrics."""
    if last_event_time > 0:
        ago = time.time() - last_event_time
        event_str = f"{ago:.0f}s ago"
    else:
        event_str = "N/A"

    stream_status = '<span class="active">ACTIVE</span>' if is_connected else "INACTIVE"

    st.markdown(f"""
    <div class="rt-footer">
        <div style="display:flex; gap:2rem;">
            <div>BUFFER: <span class="val">{buffer_depth_sec:.2f}s</span></div>
            <div>LAST EVENT: <span class="val">{event_str}</span></div>
            <div>EVENT STREAM: {stream_status}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


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
