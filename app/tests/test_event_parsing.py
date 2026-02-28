"""Tests for event parsing and discriminated-union validation.

Verifies that raw JSON dicts are correctly parsed into the right Pydantic
model subclass, and that malformed payloads fall back gracefully.
"""

from __future__ import annotations

import base64

from app.models.events import (
    ConversationItemCreated,
    ErrorEvent,
    GenericServerEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    ResponseAudioDelta,
    ResponseAudioTranscriptDelta,
    ResponseCreate,
    SessionCreatedEvent,
    SessionUpdateEvent,
    parse_server_event,
)


class TestServerEventParsing:
    """Verify parse_server_event dispatches to correct model."""

    def test_session_created(self):
        raw = {"type": "session.created", "event_id": "evt_1", "session": {"id": "sess_1"}}
        event = parse_server_event(raw)
        assert isinstance(event, SessionCreatedEvent)
        assert event.session["id"] == "sess_1"

    def test_error_event(self):
        raw = {
            "type": "error",
            "error": {"message": "Invalid API key", "code": "invalid_api_key"},
        }
        event = parse_server_event(raw)
        assert isinstance(event, ErrorEvent)
        assert event.message == "Invalid API key"
        assert event.code == "invalid_api_key"

    def test_response_audio_delta(self):
        audio_bytes = b"\x00\x01\x02\x03"
        b64 = base64.b64encode(audio_bytes).decode()
        raw = {
            "type": "response.audio.delta",
            "delta": b64,
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
        }
        event = parse_server_event(raw)
        assert isinstance(event, ResponseAudioDelta)
        assert event.decode_audio() == audio_bytes

    def test_audio_transcript_delta(self):
        raw = {
            "type": "response.audio_transcript.delta",
            "delta": "Hello",
            "response_id": "resp_1",
        }
        event = parse_server_event(raw)
        assert isinstance(event, ResponseAudioTranscriptDelta)
        assert event.delta == "Hello"

    def test_conversation_item_created(self):
        raw = {
            "type": "conversation.item.created",
            "item": {"id": "item_1", "type": "message", "role": "assistant"},
        }
        event = parse_server_event(raw)
        assert isinstance(event, ConversationItemCreated)
        assert event.item["role"] == "assistant"

    def test_unknown_event_type_falls_back(self):
        raw = {"type": "some.future.event", "data": "hello"}
        event = parse_server_event(raw)
        assert isinstance(event, GenericServerEvent)
        assert event.type == "some.future.event"
        assert event.raw["data"] == "hello"

    def test_missing_type_field(self):
        raw = {"data": "no type"}
        event = parse_server_event(raw)
        assert isinstance(event, GenericServerEvent)
        assert event.type == "unknown"

    def test_error_event_properties(self):
        raw = {"type": "error", "error": {"message": "rate limited", "code": "rate_limit"}}
        event = parse_server_event(raw)
        assert isinstance(event, ErrorEvent)
        assert "rate limited" in event.message

    def test_error_event_missing_fields(self):
        raw = {"type": "error", "error": {}}
        event = parse_server_event(raw)
        assert isinstance(event, ErrorEvent)
        assert event.message == "Unknown error"
        assert event.code is None


class TestClientEvents:
    """Verify client event construction and serialization."""

    def test_session_update(self):
        event = SessionUpdateEvent(session={"voice": "alloy"})
        data = event.model_dump(exclude_none=True)
        assert data["type"] == "session.update"
        assert data["session"]["voice"] == "alloy"

    def test_input_audio_buffer_append_from_bytes(self):
        raw_audio = b"\x00\x10\x20\x30"
        event = InputAudioBufferAppend.from_bytes(raw_audio)
        assert event.type == "input_audio_buffer.append"
        decoded = base64.b64decode(event.audio)
        assert decoded == raw_audio

    def test_input_audio_buffer_commit(self):
        event = InputAudioBufferCommit()
        assert event.model_dump(exclude_none=True)["type"] == "input_audio_buffer.commit"

    def test_response_create_empty(self):
        event = ResponseCreate()
        data = event.model_dump(exclude_none=True)
        assert data["type"] == "response.create"
        assert "response" not in data

    def test_response_create_with_config(self):
        event = ResponseCreate(response={"modalities": ["text"]})
        data = event.model_dump(exclude_none=True)
        assert data["response"]["modalities"] == ["text"]
