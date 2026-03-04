"""Pydantic event models — discriminated unions for type-safe event routing.

Responsibility: Parse raw JSON from/to the OpenAI Realtime WebSocket into
    strongly-typed Python objects.
Pattern: Discriminated Union (Pydantic ``Literal`` discriminator on ``type``).
Why: The Realtime API transmits dozens of event types over a single WebSocket.
    Without a discriminated union, every handler must defensively check keys.
    With it, ``model_validate`` gives you the *exact* subclass — or a clear
    ``ValidationError`` — eliminating an entire class of silent-corruption bugs.

OpenAI Realtime API Events:
This module defines all event types used in the OpenAI Realtime API for
bidirectional communication over WebSockets. Client events (e.g., session.update,
input_audio_buffer.append) are sent to configure and stream audio to the AI.
Server events (e.g., response.audio.delta, error) are received in response,
enabling real-time voice interactions.

WebSockets and Event Streaming:
Events are serialized to/from JSON and transmitted over the persistent WebSocket
connection. This allows low-latency, asynchronous exchange of control messages,
audio data, and status updates without HTTP polling, crucial for real-time apps.

WebRTC Integration:
Audio-related events, such as ResponseAudioDelta (containing base64-encoded
PCM audio), can be decoded and played via WebRTC in the browser UI. WebRTC
handles client-side audio capture/playback, integrating seamlessly with these
events for end-to-end voice streaming.

Real-Time Operation:
The system relies on frequent events like audio deltas arriving multiple times
per second during AI speech. Parsing and dispatching these immediately ensures
low-latency responses, supporting natural conversations with interruptions and
dynamic audio streaming.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Client → Server events
# ============================================================================


class _ClientBase(BaseModel):
    """Common base for all client-bound events."""

    event_id: str | None = None


class SessionUpdateEvent(_ClientBase):
    """``session.update`` — push validated config to the server.

    OpenAI Realtime API Session Configuration:
    Sent over WebSocket to configure the AI session (e.g., voice, modalities).
    This event allows dynamic setup without reconnecting, enabling real-time
    adjustments during conversations.
    """

    type: Literal["session.update"] = "session.update"
    session: dict = Field(..., description="Session configuration payload")


class InputAudioBufferAppend(_ClientBase):
    """``input_audio_buffer.append`` — stream a chunk of PCM16 audio.

    Real-Time Audio Streaming:
    Sends base64-encoded audio chunks from the user's microphone over WebSocket.
    This enables continuous voice input to the OpenAI Realtime API, supporting
    live conversations with low-latency processing.
    """

    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str = Field(..., description="Base64-encoded audio bytes")

    @classmethod
    def from_bytes(cls, raw: bytes, event_id: str | None = None) -> InputAudioBufferAppend:
        """Convenience: encode raw PCM bytes to base64 for the wire."""
        return cls(audio=base64.b64encode(raw).decode("ascii"), event_id=event_id)


class InputAudioBufferCommit(_ClientBase):
    """``input_audio_buffer.commit`` — finalize the current audio turn."""

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class InputAudioBufferClear(_ClientBase):
    """``input_audio_buffer.clear`` — discard buffered audio."""

    type: Literal["input_audio_buffer.clear"] = "input_audio_buffer.clear"


class ResponseCreate(_ClientBase):
    """``response.create`` — request the model to generate a response."""

    type: Literal["response.create"] = "response.create"
    response: dict | None = None


class ResponseCancel(_ClientBase):
    """``response.cancel`` — abort an in-progress response."""

    type: Literal["response.cancel"] = "response.cancel"


class ConversationItemCreate(_ClientBase):
    """``conversation.item.create`` — inject an item (e.g. text message)."""

    type: Literal["conversation.item.create"] = "conversation.item.create"
    item: dict = Field(..., description="Conversation item payload")


# Union of all client events
ClientEvent = Annotated[
    SessionUpdateEvent
    | InputAudioBufferAppend
    | InputAudioBufferCommit
    | InputAudioBufferClear
    | ResponseCreate
    | ResponseCancel
    | ConversationItemCreate,
    Field(discriminator="type"),
]


# ============================================================================
# Server → Client events
# ============================================================================


class _ServerBase(BaseModel):
    """Common base for all server-bound events."""

    event_id: str | None = None


class SessionCreatedEvent(_ServerBase):
    """``session.created`` — handshake complete."""

    type: Literal["session.created"] = "session.created"
    session: dict = Field(default_factory=dict)


class SessionUpdatedEvent(_ServerBase):
    """``session.updated`` — config acknowledged."""

    type: Literal["session.updated"] = "session.updated"
    session: dict = Field(default_factory=dict)


class ErrorEvent(_ServerBase):
    """``error`` — server-side error."""

    type: Literal["error"] = "error"
    error: dict = Field(default_factory=dict)

    @property
    def message(self) -> str:
        return self.error.get("message", "Unknown error")

    @property
    def code(self) -> str | None:
        return self.error.get("code")


class InputAudioBufferCommitted(_ServerBase):
    """``input_audio_buffer.committed``."""

    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    item_id: str | None = None


class InputAudioBufferSpeechStarted(_ServerBase):
    """``input_audio_buffer.speech_started`` — VAD detected voice."""

    type: Literal["input_audio_buffer.speech_started"] = "input_audio_buffer.speech_started"
    audio_start_ms: int = 0
    item_id: str | None = None


class InputAudioBufferSpeechStopped(_ServerBase):
    """``input_audio_buffer.speech_stopped`` — VAD silence detected."""

    type: Literal["input_audio_buffer.speech_stopped"] = "input_audio_buffer.speech_stopped"
    audio_end_ms: int = 0
    item_id: str | None = None


class ConversationItemCreated(_ServerBase):
    """``conversation.item.created`` — new item in the conversation."""

    type: Literal["conversation.item.created"] = "conversation.item.created"
    item: dict = Field(default_factory=dict)


class ConversationItemTranscriptionCompleted(_ServerBase):
    """``conversation.item.input_audio_transcription.completed``."""

    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    item_id: str | None = None
    transcript: str = ""


class ResponseCreatedEvent(_ServerBase):
    """``response.created`` — model started generating."""

    type: Literal["response.created"] = "response.created"
    response: dict = Field(default_factory=dict)


class ResponseDoneEvent(_ServerBase):
    """``response.done`` — model finished generating."""

    type: Literal["response.done"] = "response.done"
    response: dict = Field(default_factory=dict)


class ResponseAudioDelta(_ServerBase):
    """``response.audio.delta`` — a chunk of base64 audio.

    This is the hottest event in the pipeline — arrives many times per
    second during playback.

    Real-Time Audio Streaming:
    Delivers incremental AI-generated audio over WebSocket. Frequent deltas
    enable smooth, low-latency playback, supporting real-time voice responses.
    """

    type: Literal["response.audio.delta"] = "response.audio.delta"
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    delta: str = ""  # base64-encoded audio

    def decode_audio(self) -> bytes:
        """Decode the base64 delta into raw PCM bytes.

        Audio Decoding for WebRTC:
        Converts base64 audio to raw bytes for playback via WebRTC or audio
        libraries, ensuring real-time audio output.
        """
        return base64.b64decode(self.delta)


class ResponseAudioDone(_ServerBase):
    """``response.audio.done`` — audio stream complete for this part."""

    type: Literal["response.audio.done"] = "response.audio.done"
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0


class ResponseAudioTranscriptDelta(_ServerBase):
    """``response.audio_transcript.delta`` — incremental transcript."""

    type: Literal["response.audio_transcript.delta"] = "response.audio_transcript.delta"
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


class ResponseAudioTranscriptDone(_ServerBase):
    """``response.audio_transcript.done`` — final transcript for a part."""

    type: Literal["response.audio_transcript.done"] = "response.audio_transcript.done"
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    transcript: str = ""


class ResponseTextDelta(_ServerBase):
    """``response.text.delta`` — incremental text content."""

    type: Literal["response.text.delta"] = "response.text.delta"
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    delta: str = ""


class ResponseTextDone(_ServerBase):
    """``response.text.done`` — text generation complete."""

    type: Literal["response.text.done"] = "response.text.done"
    response_id: str | None = None
    item_id: str | None = None
    output_index: int = 0
    content_index: int = 0
    text: str = ""


class RateLimitsUpdated(_ServerBase):
    """``rate_limits.updated`` — current rate-limit snapshot."""

    type: Literal["rate_limits.updated"] = "rate_limits.updated"
    rate_limits: list[dict] = Field(default_factory=list)


class GenericServerEvent(_ServerBase):
    """Catch-all for unmodeled server events.

    Keeps the pipeline from crashing on new/unknown event types.
    """

    type: str = "unknown"
    raw: dict[str, Any] = Field(default_factory=dict)


# Discriminated union of all *known* server events.
# GenericServerEvent is intentionally excluded from the union —
# it's used as a fallback in parse_server_event().
ServerEvent = Annotated[
    SessionCreatedEvent
    | SessionUpdatedEvent
    | ErrorEvent
    | InputAudioBufferCommitted
    | InputAudioBufferSpeechStarted
    | InputAudioBufferSpeechStopped
    | ConversationItemCreated
    | ConversationItemTranscriptionCompleted
    | ResponseCreatedEvent
    | ResponseDoneEvent
    | ResponseAudioDelta
    | ResponseAudioDone
    | ResponseAudioTranscriptDelta
    | ResponseAudioTranscriptDone
    | ResponseTextDelta
    | ResponseTextDone
    | RateLimitsUpdated,
    Field(discriminator="type"),
]

# Map from event-type string to its model for manual dispatch
_SERVER_EVENT_MODELS: dict[str, type[BaseModel]] = {
    "session.created": SessionCreatedEvent,
    "session.updated": SessionUpdatedEvent,
    "error": ErrorEvent,
    "input_audio_buffer.committed": InputAudioBufferCommitted,
    "input_audio_buffer.speech_started": InputAudioBufferSpeechStarted,
    "input_audio_buffer.speech_stopped": InputAudioBufferSpeechStopped,
    "conversation.item.created": ConversationItemCreated,
    "conversation.item.input_audio_transcription.completed": ConversationItemTranscriptionCompleted,
    "response.created": ResponseCreatedEvent,
    "response.done": ResponseDoneEvent,
    "response.audio.delta": ResponseAudioDelta,
    "response.audio.done": ResponseAudioDone,
    "response.audio_transcript.delta": ResponseAudioTranscriptDelta,
    "response.audio_transcript.done": ResponseAudioTranscriptDone,
    "response.text.delta": ResponseTextDelta,
    "response.text.done": ResponseTextDone,
    "rate_limits.updated": RateLimitsUpdated,
}


def parse_server_event(data: dict[str, Any]) -> _ServerBase:
    """Parse a raw JSON dict into a typed ServerEvent or GenericServerEvent.

    Uses the ``type`` field to look up the correct Pydantic model.
    Falls back to ``GenericServerEvent`` for unrecognized types so the
    pipeline never drops data silently.

    Event Parsing for Real-Time Processing:
    Converts incoming WebSocket JSON into typed objects for safe, immediate
    dispatching. This ensures real-time handling of events like audio deltas
    without type errors, maintaining low-latency voice interactions.
    """
    event_type = data.get("type", "unknown")
    model_cls = _SERVER_EVENT_MODELS.get(event_type)
    if model_cls is not None:
        return model_cls.model_validate(data)
    return GenericServerEvent(type=event_type, raw=data)
