"""VoiceAgentBridge — Adapter between the async engine and Streamlit UI.

Responsibility: Expose a clean, synchronous-friendly API that the
    Streamlit script can call without importing asyncio, websockets,
    or Pydantic event internals.  This is a *voice-only* bridge — there
    is no text-input path.
Pattern: Adapter / Bridge — translates between two incompatible
    interfaces (async engine <-> sync Streamlit re-run model).
Why: Streamlit re-runs the entire script on every widget interaction.
    Async operations must be hidden behind simple method calls that
    schedule coroutines on a background event loop.  The bridge also
    decodes audio deltas into the thread-safe AudioBuffer, keeping the
    UI layer completely unaware of base64 or PCM16 details.

OpenAI Realtime API Integration:
This bridge connects the Streamlit UI to the OpenAI Realtime API via the
RealtimeManager, handling voice-only interactions. It sends audio input
and receives AI responses in real-time, managing session lifecycle and
event dispatching.

WebSockets and Real-Time Communication:
The bridge schedules async operations on a background event loop to
communicate with the WebSocket connection to OpenAI. Events like audio
deltas are received asynchronously and processed immediately for low-latency
voice streaming.

WebRTC Context:
While WebRTC is not directly used here, the bridge handles audio data
that can be integrated with WebRTC for browser-based capture/playback.
Audio chunks are decoded from base64 (from WebSocket events) and buffered
for playback, enabling seamless voice I/O in real-time applications.

Real-Time Operation:
The bridge maintains UI states (mic, agent speaking) and accumulates
transcripts/audio in real-time. Handlers process incoming events instantly,
ensuring responsive voice conversations with immediate audio playback and
transcript updates.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from app.core.dispatcher import EventDispatcher
from app.core.manager import RealtimeManager
from app.models.config import AppSettings, RealtimeConfig
from app.models.enums import AgentSpeakingState, ConnectionState, MicState
from app.models.events import (
    ConversationItemCreate,
    ConversationItemTranscriptionCompleted,
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferClear,
    InputAudioBufferCommit,
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseAudioTranscriptDelta,
    ResponseAudioTranscriptDone,
    ResponseCreate,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    _ServerBase,
)
from app.utils.audio import AudioBuffer

logger = logging.getLogger(__name__)


class VoiceAgentBridge:
    """Adapter layer that the Streamlit UI interacts with (voice-only).

    Manages:
    - A background asyncio event loop (in a daemon thread)
    - The RealtimeManager singleton
    - An AudioBuffer for playback data
    - Mic and agent speaking state for the split-screen UI
    - Transcript accumulation (read-only in the UI)

    All public methods are synchronous and safe to call from Streamlit.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._manager: RealtimeManager | None = None
        self._audio_buffer = AudioBuffer()

        # Transcript state (written by async handlers, read by Streamlit)
        self._lock = threading.Lock()
        self._user_transcript: list[str] = []
        self._agent_transcript: list[str] = []
        self._assistant_text_buffer: str = ""
        self._errors: list[str] = []

        # Voice UX states
        self._mic_state = MicState.IDLE
        self._agent_state = AgentSpeakingState.IDLE

        # Playback: accumulate response audio, then convert to WAV
        self._response_audio_chunks: list[bytes] = []
        self._playback_audio: bytes | None = None

    # -- Event loop management -----------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start the background event loop if not already running."""
        if self._loop is not None and self._loop.is_running():
            return self._loop

        self._loop = asyncio.new_event_loop()

        def _run(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._thread = threading.Thread(target=_run, args=(self._loop,), daemon=True)
        self._thread.start()
        return self._loop

    def _run_coro(self, coro: Any) -> Any:
        """Schedule a coroutine on the background loop and block until done."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=30)

    # -- Connection ----------------------------------------------------------

    def connect(self, settings: AppSettings, config: RealtimeConfig) -> bool:
        """Connect to OpenAI Realtime API. Returns True on success.

        Establishes WebSocket Connection:
        Initializes the RealtimeManager with API key and config, then connects
        to OpenAI's Realtime API via WebSocket. Registers event handlers for
        real-time processing of audio, transcripts, and errors.

        Real-Time Session Setup:
        Upon connection, the session is configured for voice interactions,
        enabling immediate bidirectional audio streaming and event exchange.
        """
        RealtimeManager.reset_singleton()
        self._manager = RealtimeManager(api_key=settings.openai_api_key, config=config)
        self._register_handlers(self._manager.dispatcher)
        return self._run_coro(self._manager.connect())

    def disconnect(self) -> None:
        """Disconnect from the API."""
        if self._manager:
            self._run_coro(self._manager.disconnect())
        with self._lock:
            self._mic_state = MicState.IDLE
            self._agent_state = AgentSpeakingState.IDLE
            self._response_audio_chunks.clear()
            self._playback_audio = None

    @property
    def connection_state(self) -> ConnectionState:
        if self._manager:
            return self._manager.state
        return ConnectionState.DISCONNECTED

    @property
    def event_log(self) -> list[dict[str, Any]]:
        if self._manager:
            return self._manager.event_log
        return []

    @property
    def last_event_time(self) -> float:
        if self._manager:
            return self._manager.last_event_time
        return 0.0

    # -- Mic state -----------------------------------------------------------

    @property
    def mic_state(self) -> MicState:
        with self._lock:
            return self._mic_state

    @mic_state.setter
    def mic_state(self, value: MicState) -> None:
        with self._lock:
            self._mic_state = value

    # -- Agent speaking state ------------------------------------------------

    @property
    def agent_state(self) -> AgentSpeakingState:
        with self._lock:
            return self._agent_state

    # -- Audio ---------------------------------------------------------------

    @property
    def audio_buffer(self) -> AudioBuffer:
        return self._audio_buffer

    def send_audio_chunk(self, pcm_bytes: bytes) -> None:
        """Send a chunk of PCM16 audio to the server.

        Real-Time Audio Streaming:
        Encodes PCM audio bytes to base64 and sends them over WebSocket as
        an input_audio_buffer.append event. This enables continuous voice
        input to the OpenAI Realtime API, supporting live, low-latency
        conversations.
        """
        if not self._manager:
            return
        event = InputAudioBufferAppend.from_bytes(pcm_bytes)
        self._run_coro(self._manager.send(event))

    def commit_audio(self) -> None:
        """Commit the audio buffer and request a response.

        Transitions: mic -> SENDING, agent -> THINKING.
        """
        if not self._manager:
            return
        with self._lock:
            self._mic_state = MicState.SENDING
            self._agent_state = AgentSpeakingState.THINKING
        self._run_coro(self._manager.send(InputAudioBufferCommit()))
        self._run_coro(self._manager.send(ResponseCreate()))

    def send_wav_audio(self, wav_bytes: bytes) -> None:
        """Decode WAV, create a user audio conversation item, and request a response.

        Uses ``conversation.item.create`` to send the entire recording as a
        single conversation item.  This bypasses the streaming input buffer
        (and server-side VAD) so the audio is never fragmented — fixing the
        garbled-transcription issue that occurs when batch-sent audio is
        split by ``server_vad``.
        """
        import base64 as b64

        from app.utils.audio import wav_bytes_to_pcm16_24k

        if not self._manager:
            return

        pcm = wav_bytes_to_pcm16_24k(wav_bytes)
        audio_b64 = b64.b64encode(pcm).decode("ascii")

        # Create a complete user audio item (bypasses streaming buffer / VAD)
        item_event = ConversationItemCreate(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "audio": audio_b64}],
            }
        )
        self._run_coro(self._manager.send(item_event))

        # Set state and request response
        with self._lock:
            self._mic_state = MicState.SENDING
            self._agent_state = AgentSpeakingState.THINKING
        self._run_coro(self._manager.send(ResponseCreate()))

    def clear_audio_input(self) -> None:
        """Clear the server-side input audio buffer."""
        if not self._manager:
            return
        self._run_coro(self._manager.send(InputAudioBufferClear()))

    # -- Transcript (read-only) ----------------------------------------------

    @property
    def user_transcript(self) -> list[str]:
        """Last N user utterances (read-only in UI)."""
        with self._lock:
            return list(self._user_transcript)

    @property
    def agent_transcript(self) -> list[str]:
        """Last N agent responses (read-only in UI)."""
        with self._lock:
            return list(self._agent_transcript)

    @property
    def errors(self) -> list[str]:
        with self._lock:
            return list(self._errors)

    def clear_transcript(self) -> None:
        with self._lock:
            self._user_transcript.clear()
            self._agent_transcript.clear()
            self._assistant_text_buffer = ""
            self._errors.clear()

    # -- Handler registration ------------------------------------------------

    def _register_handlers(self, dispatcher: EventDispatcher) -> None:
        """Wire up event handlers for audio, transcript, and error events."""
        dispatcher.register("response.audio.delta", self._on_audio_delta)
        dispatcher.register("response.audio.done", self._on_audio_done)
        dispatcher.register("response.audio_transcript.delta", self._on_transcript_delta)
        dispatcher.register("response.audio_transcript.done", self._on_transcript_done)
        dispatcher.register("response.created", self._on_response_created)
        dispatcher.register("response.done", self._on_response_done)
        dispatcher.register(
            "conversation.item.input_audio_transcription.completed",
            self._on_user_transcription,
        )
        dispatcher.register("error", self._on_error)

    def get_playback_audio(self) -> bytes | None:
        """Return accumulated agent audio as WAV bytes (and clear it).

        Called by the UI layer to feed ``st.audio(autoplay=True)``.
        Returns ``None`` when no new audio is available.
        """
        with self._lock:
            audio = self._playback_audio
            self._playback_audio = None
            return audio

    async def _on_audio_delta(self, event: _ServerBase) -> None:
        """Decode audio delta and push into the playback buffer.

        Transitions agent state to SPEAKING on the first delta.

        Real-Time Audio Playback:
        Decodes base64 audio from WebSocket events into PCM bytes and buffers
        them for immediate playback. This handles frequent audio deltas from
        the OpenAI Realtime API, enabling smooth, real-time AI voice output.
        Updates UI state to reflect active speaking.
        """
        assert isinstance(event, ResponseAudioDelta)
        pcm = event.decode_audio()
        self._audio_buffer.append(pcm)
        with self._lock:
            self._response_audio_chunks.append(pcm)
            if self._agent_state != AgentSpeakingState.SPEAKING:
                self._agent_state = AgentSpeakingState.SPEAKING

    async def _on_audio_done(self, event: _ServerBase) -> None:
        """Audio stream complete for this response part."""
        assert isinstance(event, ResponseAudioDone)
        # Agent will go IDLE when response.done fires

    async def _on_response_created(self, event: _ServerBase) -> None:
        """Server acknowledged it is generating a response."""
        assert isinstance(event, ResponseCreatedEvent)
        with self._lock:
            self._mic_state = MicState.IDLE
            if self._agent_state == AgentSpeakingState.IDLE:
                self._agent_state = AgentSpeakingState.THINKING

    async def _on_response_done(self, event: _ServerBase) -> None:
        """Response generation complete — build playback WAV, agent goes idle."""
        assert isinstance(event, ResponseDoneEvent)
        with self._lock:
            self._agent_state = AgentSpeakingState.IDLE
            if self._response_audio_chunks:
                all_pcm = b"".join(self._response_audio_chunks)
                self._response_audio_chunks.clear()
                from app.utils.audio import pcm16_to_wav_bytes

                self._playback_audio = pcm16_to_wav_bytes(all_pcm)

    async def _on_transcript_delta(self, event: _ServerBase) -> None:
        """Accumulate assistant audio-transcript deltas."""
        assert isinstance(event, ResponseAudioTranscriptDelta)
        with self._lock:
            self._assistant_text_buffer += event.delta

    async def _on_transcript_done(self, event: _ServerBase) -> None:
        """Finalize the assistant transcript turn."""
        assert isinstance(event, ResponseAudioTranscriptDone)
        with self._lock:
            text = event.transcript or self._assistant_text_buffer
            if text.strip():
                self._agent_transcript.append(text)
                if len(self._agent_transcript) > 50:
                    self._agent_transcript = self._agent_transcript[-50:]
            self._assistant_text_buffer = ""

    async def _on_user_transcription(self, event: _ServerBase) -> None:
        """Add the user's speech transcription to the user transcript."""
        assert isinstance(event, ConversationItemTranscriptionCompleted)
        with self._lock:
            if event.transcript.strip():
                self._user_transcript.append(event.transcript)
                if len(self._user_transcript) > 50:
                    self._user_transcript = self._user_transcript[-50:]

    async def _on_error(self, event: _ServerBase) -> None:
        """Log and surface errors."""
        assert isinstance(event, ErrorEvent)
        msg = f"[{event.code or 'error'}] {event.message}"
        logger.error("Server error: %s", msg)
        with self._lock:
            self._errors.append(msg)
            if len(self._errors) > 20:
                self._errors = self._errors[-20:]
