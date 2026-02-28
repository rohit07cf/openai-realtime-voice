"""Enumeration types for the Realtime Voice Agent.

Responsibility: Define constrained value sets used across config and events.
Pattern: Python StrEnum for JSON-friendly serialization.
Why: Enums prevent typo-driven runtime errors in voice names, modality
     strings, and connection states — a critical safety net when building
     against a fast-moving external API.
"""

from enum import StrEnum


class Voice(StrEnum):
    """Voices supported by the OpenAI Realtime API."""

    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"


class Modality(StrEnum):
    """Content modalities the session can produce."""

    TEXT = "text"
    AUDIO = "audio"


class TurnDetectionType(StrEnum):
    """Turn-detection modes for the Realtime session."""

    SERVER_VAD = "server_vad"
    SEMANTIC_VAD = "semantic_vad"


class AudioFormat(StrEnum):
    """Audio encoding formats accepted by the Realtime API."""

    PCM16 = "pcm16"
    G711_ULAW = "g711_ulaw"
    G711_ALAW = "g711_alaw"


class ConnectionState(StrEnum):
    """Observable connection states surfaced in the UI.

    Aligned with the Circuit Breaker pattern states.
    """

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CIRCUIT_OPEN = "circuit_open"


class MicState(StrEnum):
    """Microphone states for the user panel in the voice-only UI.

    Drives the push-to-talk UX:
    - IDLE: mic is off, waiting for user to press
    - RECORDING: mic is actively capturing audio frames
    - SENDING: audio committed, waiting for server acknowledgement
    """

    IDLE = "idle"
    RECORDING = "recording"
    SENDING = "sending"


class AgentSpeakingState(StrEnum):
    """Agent panel states reflecting the model's response lifecycle.

    - IDLE: no active response
    - THINKING: audio committed, server is generating (before first audio delta)
    - SPEAKING: response.audio.delta events are streaming
    """

    IDLE = "idle"
    THINKING = "thinking"
    SPEAKING = "speaking"
