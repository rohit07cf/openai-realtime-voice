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
