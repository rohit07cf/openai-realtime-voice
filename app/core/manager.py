"""RealtimeManager — Singleton orchestrator for the WebSocket session.

Responsibility: Own the entire connection lifecycle (connect, configure,
    receive-loop, send, disconnect, reconnect) while remaining agnostic
    to UI and audio details.
Pattern: Singleton — exactly one manager per process.  Ensures a single
    WebSocket and a single receive loop regardless of how many Streamlit
    re-runs occur.
Why: Streamlit re-executes the script on every interaction.  If we
    created a new manager each time, we'd leak connections and tasks.
    The Singleton guarantees one durable session.

OpenAI Realtime API Orchestration:
This manager coordinates the entire interaction with the OpenAI Realtime API,
managing the WebSocket connection for bidirectional event streaming. It handles
session configuration (via session.update events), sends client events (e.g.,
audio input, control commands), and receives server events (e.g., audio responses,
transcripts). The API enables real-time conversational AI with low-latency voice
streaming.

WebSockets in Action:
The manager uses a persistent WebSocket connection to OpenAI's servers for
continuous, asynchronous data exchange. Unlike HTTP, WebSockets allow instant
sending and receiving of events without polling, crucial for real-time voice
applications. The connection supports large payloads (e.g., audio data) and
handles reconnections with resilience patterns.

WebRTC Integration Context:
While WebRTC is not directly managed here, this manager interfaces with audio
data that may originate from WebRTC in the UI layer (e.g., browser microphone
capture). WebRTC provides peer-to-peer audio streaming, which can feed into
this manager's event system for seamless voice input to the OpenAI API.

Real-Time Operation:
The manager runs a background receive loop that continuously reads events from
the WebSocket and dispatches them immediately. This ensures real-time processing
of AI responses, user interruptions, and audio streaming. Combined with concurrent
dispatching, it supports natural, low-latency voice conversations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.core.connection import RealtimeConnection
from app.core.dispatcher import EventDispatcher
from app.core.resilience import CircuitBreaker, CircuitState, retry_with_backoff
from app.models.config import RealtimeConfig
from app.models.enums import ConnectionState
from app.models.events import (
    _ClientBase,
    _ServerBase,
    parse_server_event,
)

logger = logging.getLogger(__name__)


class RealtimeManager:
    """Singleton manager for a persistent OpenAI Realtime session.

    Coordinates:
    - ``RealtimeConnection`` (transport)
    - ``EventDispatcher`` (observer fan-out)
    - ``CircuitBreaker`` (resilience)

    Usage::

        mgr = RealtimeManager(api_key="sk-...", config=my_config)
        mgr.dispatcher.register("response.audio.delta", my_handler)
        await mgr.connect()
        await mgr.send(some_client_event)
        # ...later...
        await mgr.disconnect()
    """

    _instance: RealtimeManager | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> RealtimeManager:
        """Singleton enforcement — reuse existing instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        api_key: str = "",
        config: RealtimeConfig | None = None,
    ) -> None:
        if self._initialized:
            return
        self._api_key = api_key
        self._config = config or RealtimeConfig()
        self._conn: RealtimeConnection | None = None
        self._dispatcher = EventDispatcher()
        self._circuit = CircuitBreaker(failure_threshold=3, cooldown_seconds=30)
        self._receive_task: asyncio.Task[None] | None = None
        self._state = ConnectionState.DISCONNECTED
        self._last_event_time: float = 0.0
        self._event_log: list[dict[str, Any]] = []
        self._initialized = True

    # -- Public properties ---------------------------------------------------

    @property
    def dispatcher(self) -> EventDispatcher:
        return self._dispatcher

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def circuit_state(self) -> CircuitState:
        return self._circuit.state

    @property
    def last_event_time(self) -> float:
        return self._last_event_time

    @property
    def event_log(self) -> list[dict[str, Any]]:
        """Last N raw events (capped at 50) for the debug panel."""
        return self._event_log[-50:]

    # -- Configuration update ------------------------------------------------

    def update_config(self, config: RealtimeConfig) -> None:
        self._config = config

    def update_api_key(self, api_key: str) -> None:
        self._api_key = api_key

    # -- Connection lifecycle ------------------------------------------------

    async def connect(self) -> bool:
        """Establish the WebSocket and send initial session.update.

        Returns True on success, False if the circuit breaker blocks.

        WebSocket Connection Establishment:
        This method initiates a secure WebSocket connection to OpenAI's Realtime API
        using the RealtimeConnection class. It authenticates with the API key and
        specifies the model (e.g., GPT-4o Realtime). The connection is persistent,
        enabling continuous bidirectional communication.

        OpenAI Realtime API Session Initialization:
        Upon successful connection, it sends a session.update event to configure
        the AI session (e.g., voice settings, tools, modalities). This sets up
        the real-time conversational environment, allowing for voice input/output
        and dynamic responses.

        Real-Time Readiness:
        Starting the background receive loop ensures the system is prepared to
        handle incoming events immediately. This loop continuously listens for
        server responses, dispatching them for real-time processing (e.g., audio
        playback, UI updates).

        Resilience Features:
        Uses a circuit breaker to prevent repeated connection attempts on failure,
        and handles errors gracefully to maintain system stability.
        """
        if self._state == ConnectionState.CONNECTED:
            logger.info("Already connected")
            return True

        if not self._circuit.allow_request():
            self._state = ConnectionState.CIRCUIT_OPEN
            logger.warning("Circuit breaker is OPEN — connect blocked")
            return False

        self._state = ConnectionState.CONNECTING
        self._conn = RealtimeConnection(api_key=self._api_key, model=self._config.model)

        try:
            await self._conn.connect()
        except Exception as exc:
            logger.error("Connection failed: %s", exc)
            self._circuit.record_failure()
            self._state = (
                ConnectionState.CIRCUIT_OPEN
                if not self._circuit.allow_request()
                else ConnectionState.DISCONNECTED
            )
            return False

        self._circuit.record_success()
        self._state = ConnectionState.CONNECTED

        # Start background receive loop first (so we can receive errors)
        self._receive_task = asyncio.create_task(self._receive_loop())

        # Send session configuration — may fail if the server rejects
        # the key/model during the handshake (close code 3000).
        try:
            await self._send_session_update()
        except Exception as exc:
            logger.error("Session update failed: %s", exc)
            self._circuit.record_failure()
            await self.disconnect()
            return False

        return True

    async def disconnect(self) -> None:
        """Gracefully tear down the session."""
        self._state = ConnectionState.DISCONNECTED
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._conn:
            await self._conn.close()
            self._conn = None
        logger.info("Manager disconnected")

    async def reconnect(self) -> bool:
        """Disconnect then reconnect with backoff + circuit breaker."""
        self._state = ConnectionState.RECONNECTING
        await self.disconnect()

        success = await retry_with_backoff(
            lambda: self.connect(),
            max_retries=5,
            base_delay=1.0,
            max_delay=30.0,
            circuit_breaker=self._circuit,
        )
        if not success:
            self._state = (
                ConnectionState.CIRCUIT_OPEN
                if not self._circuit.allow_request()
                else ConnectionState.DISCONNECTED
            )
        return success

    # -- Sending events ------------------------------------------------------

    async def send(self, event: _ClientBase) -> None:
        """Serialize and send a ClientEvent to the server.

        OpenAI Realtime API Client Events:
        This method sends events to the OpenAI Realtime API over the WebSocket,
        such as audio input buffers, session updates, or control commands.
        Events are serialized to JSON and transmitted asynchronously, enabling
        real-time interaction. For example, streaming user voice data as it
        is captured allows for immediate AI processing and responses.

        WebSocket Transmission:
        Utilizes the underlying WebSocket connection for low-latency delivery.
        The persistent connection ensures events are sent instantly without
        establishing new HTTP requests, supporting continuous voice conversations.

        Real-Time Voice Input:
        Critical for sending audio chunks in real-time, allowing the AI to
        respond dynamically to user speech, including handling interruptions
        or follow-up questions seamlessly.
        """
        if self._conn is None or not self._conn.is_open:
            raise RuntimeError("Cannot send — not connected")
        payload = event.model_dump(exclude_none=True)
        await self._conn.send(payload)
        logger.debug("Sent %s", payload.get("type"))

    # -- Internal ------------------------------------------------------------

    async def _send_session_update(self) -> None:
        """Push the current RealtimeConfig as a session.update event."""
        from app.models.events import SessionUpdateEvent

        event = SessionUpdateEvent(session=self._config.to_session_payload())
        await self.send(event)
        logger.info("Sent session.update with config")

    async def _receive_loop(self) -> None:
        """Background task: read events from the WebSocket and dispatch.

        Real-Time Event Reception:
        This background loop continuously reads incoming events from the OpenAI
        Realtime API via the WebSocket connection. It runs asynchronously,
        ensuring that server responses (e.g., AI audio, transcripts) are processed
        immediately upon arrival, maintaining low-latency real-time interactions.

        OpenAI Realtime API Server Events:
        Events received include audio deltas for streaming AI voice responses,
        transcript updates, error notifications, and session acknowledgments.
        Each event is parsed from JSON into typed objects and dispatched to
        registered handlers for processing.

        WebSocket Continuous Listening:
        Unlike HTTP polling, the WebSocket allows persistent listening for
        events. The loop blocks on receive() until data arrives, then dispatches
        concurrently, enabling simultaneous handling of multiple event types
        (e.g., audio playback and UI updates).

        Resilience and Error Handling:
        If the connection fails, it triggers reconnection with backoff and
        circuit breaker protection. Errors in individual events are logged,
        but the loop continues to maintain real-time operation.

        WebRTC Audio Integration:
        Received audio events can be integrated with WebRTC for playback
        in the browser, creating a seamless voice pipeline from AI generation
        to user hearing.
        """
        assert self._conn is not None
        try:
            while self._state == ConnectionState.CONNECTED:
                try:
                    raw = await self._conn.receive()
                except Exception as exc:
                    if self._state != ConnectionState.CONNECTED:
                        break
                    logger.error("Receive error: %s", exc)
                    self._circuit.record_failure()
                    # Trigger reconnect in a separate task to avoid blocking
                    asyncio.create_task(self.reconnect())
                    return

                self._last_event_time = time.time()

                # Append to debug log (capped)
                self._event_log.append(raw)
                if len(self._event_log) > 60:
                    self._event_log = self._event_log[-50:]

                # Parse and dispatch
                event: _ServerBase = parse_server_event(raw)
                await self._dispatcher.dispatch(event)

        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")

    @classmethod
    def reset_singleton(cls) -> None:
        """Destroy the singleton (for testing only)."""
        cls._instance = None
