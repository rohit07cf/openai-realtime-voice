"""Pydantic configuration models with strict validation.

Responsibility: Validate *all* user-facing and environment-sourced
    configuration before it reaches the WebSocket engine.
Pattern: Pydantic Settings (env-driven) + domain model (RealtimeConfig).
Why: In a streaming system, an invalid sample-rate or missing API key
    discovered *after* the WebSocket handshake wastes time and confuses
    error diagnostics.  Pydantic catches these at the boundary.

OpenAI Realtime API Configuration:
This module defines the configuration models for the OpenAI Realtime API,
ensuring all settings are validated before use. The Realtime API requires
specific session configurations sent via WebSocket events (e.g., session.update)
to control voice, audio formats, and modalities for real-time interactions.

WebSockets Integration:
Configurations are serialized and sent over the WebSocket connection to
OpenAI's servers. For example, the RealtimeConfig's to_session_payload method
creates the JSON payload for the session.update event, which configures the
AI session in real-time without restarting the connection.

WebRTC Context:
Audio formats (e.g., PCM16) specified here must align with WebRTC audio
streams used in the client-side UI. WebRTC handles browser-based audio
capture/playback, and these configs ensure compatibility for seamless
voice input/output in real-time conversations.

Real-Time Operation:
Settings like turn detection, modalities (text/audio), and voice control
how the AI responds in real-time. For instance, server VAD enables automatic
turn-taking in voice chats, while audio formats ensure low-latency streaming
over WebSockets, supporting natural, uninterrupted AI voice interactions.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.enums import AudioFormat, Modality, TurnDetectionType, Voice

# ---------------------------------------------------------------------------
# Application-level settings (loaded from env / .env)
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """Environment-driven application settings.

    Responsibility: Single source of truth for secrets and tunables.
    Pattern: Pydantic Settings — validates env vars at import time.

    OpenAI Realtime API Settings:
    These settings load from environment variables or .env files, providing
    the API key for authentication and defaults for the real-time session.
    The API key is essential for establishing WebSocket connections to
    OpenAI's Realtime API, enabling secure, authenticated voice streaming.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(..., min_length=1, description="OpenAI API key (required)")
    realtime_model: str = Field(
        default="gpt-realtime",
        description="Realtime model identifier",
    )
    voice: Voice = Field(default=Voice.ALLOY, description="Default TTS voice")
    modalities: list[Modality] = Field(default=[Modality.AUDIO])
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


# ---------------------------------------------------------------------------
# Turn-detection sub-model
# ---------------------------------------------------------------------------


class TurnDetectionConfig(BaseModel):
    """Voice Activity Detection / turn-detection parameters.

    Maps directly to the ``turn_detection`` object in ``session.update``.

    Real-Time Turn Detection:
    Configures how the OpenAI Realtime API detects when the user stops speaking,
    allowing the AI to respond automatically. Server VAD (Voice Activity Detection)
    analyzes audio in real-time over the WebSocket, enabling natural conversation
    flow without manual triggers. Parameters like threshold and silence duration
    fine-tune sensitivity for low-latency turn-taking in voice interactions.
    """

    type: TurnDetectionType = TurnDetectionType.SERVER_VAD
    threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    prefix_padding_ms: Annotated[int, Field(ge=0, le=5000)] = 300
    silence_duration_ms: Annotated[int, Field(ge=0, le=10000)] = 500


# ---------------------------------------------------------------------------
# Realtime session configuration
# ---------------------------------------------------------------------------


class RealtimeConfig(BaseModel):
    """Validated session-level configuration sent via ``session.update``.

    Responsibility: Encode every knob the Realtime API exposes for a
        session, with sensible defaults and tight ranges.
    Pattern: Value Object — immutable after creation.
    Why: A single typo in ``input_audio_format`` could silently cause
        garbled playback.  Strict validation eliminates that risk.

    OpenAI Realtime API Session Config:
    This model encapsulates all configurable aspects of a Realtime API session,
    such as voice, temperature, and audio formats. It's sent via the session.update
    event over WebSocket to dynamically configure the AI's behavior during
    real-time conversations, allowing adjustments without disconnecting.
    """

    model: str = "gpt-realtime"
    modalities: list[Modality] = Field(default=[Modality.AUDIO])
    instructions: str = Field(
        default="You are a helpful, concise voice assistant.",
        max_length=4096,
    )
    voice: Voice = Voice.ALLOY
    input_audio_format: AudioFormat = AudioFormat.PCM16
    output_audio_format: AudioFormat = AudioFormat.PCM16
    input_audio_transcription_model: str | None = "whisper-1"
    turn_detection: TurnDetectionConfig | None = Field(default_factory=TurnDetectionConfig)
    max_output_tokens: int | str = Field(default="inf")
    audio_sample_rate: int = Field(default=24000, ge=8000, le=48000)
    output_voice_speed: Annotated[float, Field(ge=0.25, le=4.0)] = 1.0

    @field_validator("max_output_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int | str) -> int | str:
        if isinstance(v, str) and v != "inf":
            raise ValueError("String value must be 'inf'")
        if isinstance(v, int) and v < 1:
            raise ValueError("Must be a positive integer or 'inf'")
        return v

    def _format_payload(self) -> dict:
        # GA expects a structured format object. PCM16 maps to audio/pcm with
        # the negotiated sample rate; G711 codecs use their MIME types.
        if self.input_audio_format == AudioFormat.PCM16:
            return {"type": "audio/pcm", "rate": self.audio_sample_rate}
        if self.input_audio_format == AudioFormat.G711_ULAW:
            return {"type": "audio/pcmu"}
        return {"type": "audio/pcma"}

    def to_session_payload(self) -> dict:
        """Serialize to the JSON body expected by ``session.update`` (GA shape).

        The GA Realtime API restructured the session object: voice, audio
        formats, turn detection, and transcription all moved under a single
        ``audio: {input: {...}, output: {...}}`` block, ``modalities`` became
        ``output_modalities``, and ``max_response_output_tokens`` became
        ``max_output_tokens``. The top-level ``type: 'realtime'`` discriminator
        is also required.
        """
        audio_input: dict = {"format": self._format_payload()}
        if self.turn_detection:
            audio_input["turn_detection"] = {
                "type": self.turn_detection.type.value,
                "threshold": self.turn_detection.threshold,
                "prefix_padding_ms": self.turn_detection.prefix_padding_ms,
                "silence_duration_ms": self.turn_detection.silence_duration_ms,
            }
        else:
            audio_input["turn_detection"] = None
        if self.input_audio_transcription_model:
            audio_input["transcription"] = {"model": self.input_audio_transcription_model}

        audio_output: dict = {
            "format": self._format_payload(),
            "voice": self.voice.value,
            "speed": self.output_voice_speed,
        }

        return {
            "type": "realtime",
            "output_modalities": [m.value for m in self.modalities],
            "instructions": self.instructions,
            "audio": {"input": audio_input, "output": audio_output},
            "max_output_tokens": self.max_output_tokens,
        }
