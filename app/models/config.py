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
        default="gpt-4o-realtime-preview-2024-12-17",
        description="Realtime model identifier",
    )
    voice: Voice = Field(default=Voice.ALLOY, description="Default TTS voice")
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.8
    modalities: list[Modality] = Field(default=[Modality.TEXT, Modality.AUDIO])
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

    model: str = "gpt-4o-realtime-preview-2024-12-17"
    modalities: list[Modality] = Field(default=[Modality.TEXT, Modality.AUDIO])
    instructions: str = Field(
        default="You are a helpful, concise voice assistant.",
        max_length=4096,
    )
    voice: Voice = Voice.ALLOY
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.8
    input_audio_format: AudioFormat = AudioFormat.PCM16
    output_audio_format: AudioFormat = AudioFormat.PCM16
    input_audio_transcription_model: str | None = "whisper-1"
    turn_detection: TurnDetectionConfig | None = Field(default_factory=TurnDetectionConfig)
    max_response_output_tokens: int | str = Field(default=4096)

    @field_validator("max_response_output_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int | str) -> int | str:
        if isinstance(v, str) and v != "inf":
            raise ValueError("String value must be 'inf'")
        if isinstance(v, int) and v < 1:
            raise ValueError("Must be a positive integer or 'inf'")
        return v

    def to_session_payload(self) -> dict:
        """Serialize to the JSON body expected by ``session.update``.

        WebSocket Session Update:
        This method converts the RealtimeConfig into a JSON payload sent
        over the WebSocket as a session.update event. The OpenAI Realtime API
        uses this to configure the AI session in real-time, allowing dynamic
        changes to voice, modalities, and audio settings without reconnecting.

        Real-Time Configuration:
        Sending this payload enables immediate application of settings like
        turn detection and audio formats, ensuring the AI adapts instantly
        to user preferences during ongoing voice conversations. This supports
        seamless real-time interactions by aligning server-side processing
        with client expectations.
        """
        payload: dict = {
            "modalities": [m.value for m in self.modalities],
            "instructions": self.instructions,
            "voice": self.voice.value,
            "temperature": self.temperature,
            "input_audio_format": self.input_audio_format.value,
            "output_audio_format": self.output_audio_format.value,
            "turn_detection": (
                {
                    "type": self.turn_detection.type.value,
                    "threshold": self.turn_detection.threshold,
                    "prefix_padding_ms": self.turn_detection.prefix_padding_ms,
                    "silence_duration_ms": self.turn_detection.silence_duration_ms,
                }
                if self.turn_detection
                else None
            ),
        }
        if self.input_audio_transcription_model:
            payload["input_audio_transcription"] = {
                "model": self.input_audio_transcription_model,
            }
        if isinstance(self.max_response_output_tokens, str):
            payload["max_response_output_tokens"] = self.max_response_output_tokens
        else:
            payload["max_response_output_tokens"] = self.max_response_output_tokens
        return payload
