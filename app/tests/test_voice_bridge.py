"""Tests for the voice-only VoiceAgentBridge state management.

Verifies that mic state and agent speaking state transitions work
correctly, and that transcripts are split into user/agent channels.
"""

from __future__ import annotations

import pytest

from app.models.enums import AgentSpeakingState, MicState
from app.models.events import (
    ConversationItemTranscriptionCompleted,
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseAudioTranscriptDone,
    ResponseCreatedEvent,
    ResponseDoneEvent,
)
from app.ui.bridge import VoiceAgentBridge


@pytest.fixture
def bridge():
    """Create a fresh bridge instance (not connected)."""
    return VoiceAgentBridge()


class TestMicState:
    """Verify mic state transitions in the bridge."""

    def test_initial_mic_state_is_idle(self, bridge: VoiceAgentBridge):
        assert bridge.mic_state == MicState.IDLE

    def test_mic_state_can_be_set_to_recording(self, bridge: VoiceAgentBridge):
        bridge.mic_state = MicState.RECORDING
        assert bridge.mic_state == MicState.RECORDING

    def test_mic_state_can_be_set_to_sending(self, bridge: VoiceAgentBridge):
        bridge.mic_state = MicState.SENDING
        assert bridge.mic_state == MicState.SENDING

    def test_mic_state_cycle(self, bridge: VoiceAgentBridge):
        """IDLE -> RECORDING -> SENDING -> IDLE."""
        assert bridge.mic_state == MicState.IDLE
        bridge.mic_state = MicState.RECORDING
        assert bridge.mic_state == MicState.RECORDING
        bridge.mic_state = MicState.SENDING
        assert bridge.mic_state == MicState.SENDING
        bridge.mic_state = MicState.IDLE
        assert bridge.mic_state == MicState.IDLE

    def test_disconnect_resets_mic_state(self, bridge: VoiceAgentBridge):
        bridge.mic_state = MicState.RECORDING
        bridge.disconnect()
        assert bridge.mic_state == MicState.IDLE


class TestAgentSpeakingState:
    """Verify agent state transitions driven by event handlers."""

    def test_initial_agent_state_is_idle(self, bridge: VoiceAgentBridge):
        assert bridge.agent_state == AgentSpeakingState.IDLE

    @pytest.mark.asyncio
    async def test_response_created_sets_thinking(self, bridge: VoiceAgentBridge):
        event = ResponseCreatedEvent(response={"id": "r1"})
        await bridge._on_response_created(event)
        assert bridge.agent_state == AgentSpeakingState.THINKING

    @pytest.mark.asyncio
    async def test_audio_delta_sets_speaking(self, bridge: VoiceAgentBridge):
        # First go to thinking
        await bridge._on_response_created(ResponseCreatedEvent(response={}))
        assert bridge.agent_state == AgentSpeakingState.THINKING

        # Audio delta transitions to speaking
        event = ResponseAudioDelta(delta="AAAA")
        await bridge._on_audio_delta(event)
        assert bridge.agent_state == AgentSpeakingState.SPEAKING

    @pytest.mark.asyncio
    async def test_response_done_resets_to_idle(self, bridge: VoiceAgentBridge):
        await bridge._on_response_created(ResponseCreatedEvent(response={}))
        await bridge._on_audio_delta(ResponseAudioDelta(delta="AAAA"))
        assert bridge.agent_state == AgentSpeakingState.SPEAKING

        await bridge._on_response_done(ResponseDoneEvent(response={}))
        assert bridge.agent_state == AgentSpeakingState.IDLE

    @pytest.mark.asyncio
    async def test_audio_done_does_not_reset_state(self, bridge: VoiceAgentBridge):
        """audio.done fires per-part, not per-response. State stays SPEAKING."""
        await bridge._on_response_created(ResponseCreatedEvent(response={}))
        await bridge._on_audio_delta(ResponseAudioDelta(delta="AAAA"))
        await bridge._on_audio_done(ResponseAudioDone())
        # Still speaking until response.done
        assert bridge.agent_state == AgentSpeakingState.SPEAKING

    def test_disconnect_resets_agent_state(self, bridge: VoiceAgentBridge):
        bridge._agent_state = AgentSpeakingState.SPEAKING
        bridge.disconnect()
        assert bridge.agent_state == AgentSpeakingState.IDLE


class TestSplitTranscript:
    """Verify user and agent transcripts are separate read-only lists."""

    def test_initial_transcripts_empty(self, bridge: VoiceAgentBridge):
        assert bridge.user_transcript == []
        assert bridge.agent_transcript == []

    @pytest.mark.asyncio
    async def test_user_transcription_appends_to_user(self, bridge: VoiceAgentBridge):
        event = ConversationItemTranscriptionCompleted(
            transcript="Hello from user",
            item_id="i1",
        )
        await bridge._on_user_transcription(event)
        assert len(bridge.user_transcript) == 1
        assert bridge.user_transcript[0] == "Hello from user"
        assert bridge.agent_transcript == []

    @pytest.mark.asyncio
    async def test_agent_transcript_appends_to_agent(self, bridge: VoiceAgentBridge):
        event = ResponseAudioTranscriptDone(transcript="Hello from agent")
        await bridge._on_transcript_done(event)
        assert len(bridge.agent_transcript) == 1
        assert bridge.agent_transcript[0] == "Hello from agent"
        assert bridge.user_transcript == []

    @pytest.mark.asyncio
    async def test_empty_transcription_ignored(self, bridge: VoiceAgentBridge):
        event = ConversationItemTranscriptionCompleted(transcript="   ", item_id="i2")
        await bridge._on_user_transcription(event)
        assert bridge.user_transcript == []

    def test_clear_transcript(self, bridge: VoiceAgentBridge):
        bridge._user_transcript = ["a", "b"]
        bridge._agent_transcript = ["c"]
        bridge._errors = ["e"]
        bridge.clear_transcript()
        assert bridge.user_transcript == []
        assert bridge.agent_transcript == []
        assert bridge.errors == []


class TestMicStateEnum:
    """Validate MicState enum values."""

    def test_all_values(self):
        assert MicState.IDLE == "idle"
        assert MicState.RECORDING == "recording"
        assert MicState.SENDING == "sending"

    def test_member_count(self):
        assert len(MicState) == 3


class TestAgentSpeakingStateEnum:
    """Validate AgentSpeakingState enum values."""

    def test_all_values(self):
        assert AgentSpeakingState.IDLE == "idle"
        assert AgentSpeakingState.THINKING == "thinking"
        assert AgentSpeakingState.SPEAKING == "speaking"

    def test_member_count(self):
        assert len(AgentSpeakingState) == 3
