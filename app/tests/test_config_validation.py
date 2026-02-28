"""Tests for Pydantic configuration validation.

Validates that RealtimeConfig and AppSettings reject invalid inputs
and accept valid ones — the first line of defense against misconfigured
sessions reaching the WebSocket.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.config import AppSettings, RealtimeConfig, TurnDetectionConfig
from app.models.enums import AudioFormat, Modality, TurnDetectionType, Voice


class TestRealtimeConfig:
    """Validate RealtimeConfig constraints."""

    def test_default_config_is_valid(self):
        cfg = RealtimeConfig()
        assert cfg.voice == Voice.ALLOY
        assert cfg.temperature == 0.8
        assert Modality.AUDIO in cfg.modalities

    def test_temperature_out_of_range(self):
        with pytest.raises(ValidationError):
            RealtimeConfig(temperature=3.0)

    def test_temperature_negative(self):
        with pytest.raises(ValidationError):
            RealtimeConfig(temperature=-0.1)

    def test_valid_voices(self):
        for voice in Voice:
            cfg = RealtimeConfig(voice=voice)
            assert cfg.voice == voice

    def test_invalid_voice(self):
        with pytest.raises(ValidationError):
            RealtimeConfig(voice="nonexistent_voice")

    def test_max_tokens_positive_int(self):
        cfg = RealtimeConfig(max_response_output_tokens=1000)
        assert cfg.max_response_output_tokens == 1000

    def test_max_tokens_inf_string(self):
        cfg = RealtimeConfig(max_response_output_tokens="inf")
        assert cfg.max_response_output_tokens == "inf"

    def test_max_tokens_invalid_string(self):
        with pytest.raises(ValidationError):
            RealtimeConfig(max_response_output_tokens="unlimited")

    def test_max_tokens_zero(self):
        with pytest.raises(ValidationError):
            RealtimeConfig(max_response_output_tokens=0)

    def test_session_payload_structure(self):
        cfg = RealtimeConfig(
            voice=Voice.CORAL,
            modalities=[Modality.TEXT, Modality.AUDIO],
            temperature=0.6,
        )
        payload = cfg.to_session_payload()
        assert payload["voice"] == "coral"
        assert payload["temperature"] == 0.6
        assert "text" in payload["modalities"]
        assert "audio" in payload["modalities"]
        assert payload["turn_detection"] is not None
        assert payload["turn_detection"]["type"] == "server_vad"

    def test_session_payload_no_turn_detection(self):
        cfg = RealtimeConfig(turn_detection=None)
        payload = cfg.to_session_payload()
        assert payload["turn_detection"] is None

    def test_audio_format_values(self):
        cfg = RealtimeConfig(
            input_audio_format=AudioFormat.G711_ULAW,
            output_audio_format=AudioFormat.G711_ALAW,
        )
        payload = cfg.to_session_payload()
        assert payload["input_audio_format"] == "g711_ulaw"
        assert payload["output_audio_format"] == "g711_alaw"

    def test_instructions_max_length(self):
        with pytest.raises(ValidationError):
            RealtimeConfig(instructions="x" * 4097)


class TestTurnDetectionConfig:
    """Validate turn-detection sub-model."""

    def test_defaults(self):
        td = TurnDetectionConfig()
        assert td.type == TurnDetectionType.SERVER_VAD
        assert 0.0 <= td.threshold <= 1.0

    def test_threshold_out_of_range(self):
        with pytest.raises(ValidationError):
            TurnDetectionConfig(threshold=1.5)

    def test_silence_duration_negative(self):
        with pytest.raises(ValidationError):
            TurnDetectionConfig(silence_duration_ms=-1)

    def test_semantic_vad(self):
        td = TurnDetectionConfig(type=TurnDetectionType.SEMANTIC_VAD)
        assert td.type == TurnDetectionType.SEMANTIC_VAD


class TestAppSettings:
    """Validate env-driven settings (mocked via constructor)."""

    def test_missing_api_key(self):
        with pytest.raises(ValidationError):
            AppSettings(openai_api_key="")

    def test_valid_settings(self):
        s = AppSettings(openai_api_key="sk-test-key-123")
        assert s.realtime_model == "gpt-4o-realtime-preview-2024-12-17"
        assert s.voice == Voice.ALLOY

    def test_invalid_log_level(self):
        with pytest.raises(ValidationError):
            AppSettings(openai_api_key="sk-test", log_level="VERBOSE")

    def test_valid_log_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            s = AppSettings(openai_api_key="sk-test", log_level=level)
            assert s.log_level == level
