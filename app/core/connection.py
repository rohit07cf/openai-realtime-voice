"""Low-level WebSocket connection handler for the OpenAI Realtime API.

Responsibility: Manage a single WebSocket connection lifecycle — open,
    send, receive, close — without any knowledge of event semantics.
Pattern: Adapter — wraps the ``websockets`` library behind a minimal
    async interface that the RealtimeManager consumes.
Why: Isolating the raw transport lets us swap libraries (e.g. aiohttp)
    or inject a mock connection in tests without touching business logic.

OpenAI Realtime API Overview:
The OpenAI Realtime API enables real-time conversational AI interactions,
particularly for voice-based applications. It allows bidirectional streaming
of audio data and events over WebSockets, enabling low-latency voice
conversations with AI models like GPT-4o Realtime. Key features include:
- Streaming audio input/output for natural conversations
- Real-time event handling (e.g., session updates, audio deltas)
- Support for interruptions and dynamic responses
This API is designed for applications requiring immediate, interactive AI responses.

WebSockets Explanation:
WebSockets provide a full-duplex communication protocol over a single TCP connection,
allowing both the client and server to send messages asynchronously. Unlike HTTP,
which is request-response based, WebSockets maintain an open connection for
continuous data exchange. This is ideal for real-time applications like voice
streaming, where low latency and persistent connections are crucial.
In this implementation, WebSockets connect to OpenAI's servers to send and receive
JSON-formatted events and audio data in real-time.

Real-Time Operation:
The connection enables real-time interaction by:
1. Establishing a persistent WebSocket link to OpenAI's endpoint
2. Sending audio chunks or control events as they occur
3. Receiving immediate responses, such as AI-generated audio deltas or transcripts
4. Maintaining low-latency communication for seamless voice conversations
This setup supports features like voice interruptions and dynamic AI responses.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)

REALTIME_BASE_URL = "wss://api.openai.com/v1/realtime"


class RealtimeConnection:
    """Thin async wrapper around a WebSocket to the Realtime API.

    Usage::

        conn = RealtimeConnection(api_key="sk-...", model="gpt-4o-realtime-preview-2024-12-17")
        await conn.connect()
        await conn.send({"type": "session.update", "session": {...}})
        msg = await conn.receive()
        await conn.close()
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._ws: ClientConnection | None = None

    @property
    def is_open(self) -> bool:
        """True only if the WebSocket exists and is not already closed."""
        if self._ws is None:
            return False
        try:
            return self._ws.state.name == "OPEN"
        except Exception:
            return self._ws is not None

    async def connect(self) -> None:
        """Open a WebSocket to the OpenAI Realtime endpoint.

        This method establishes a secure WebSocket connection (wss://) to OpenAI's
        Realtime API servers. The connection is authenticated using the API key
        in the Authorization header and specifies the model to use via the URL query parameter.
        The 'OpenAI-Beta' header indicates the use of the realtime API version.

        WebSocket Connection Details:
        - Uses the 'websockets' library for async WebSocket handling.
        - Sets a large max_size (16 MB) to accommodate potentially large audio data payloads.
        - Includes timeouts for opening and closing the connection to handle network issues.
        - Once connected, the WebSocket remains open for bidirectional, real-time communication.

        Real-Time Aspect:
        Establishing this connection enables immediate, low-latency exchange of audio and
        control events, forming the foundation for real-time voice interactions with the AI.
        """
        url = f"{REALTIME_BASE_URL}?model={self._model}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        logger.info("Connecting to %s", url)
        self._ws = await websockets.connect(
            url,
            additional_headers=headers,
            max_size=2**24,  # 16 MB — audio deltas can be large
            open_timeout=15,
            close_timeout=5,
        )
        logger.info("WebSocket connected")

    async def send(self, payload: dict[str, Any]) -> None:
        """Serialize *payload* to JSON and send over the WebSocket.

        This method takes a Python dictionary (payload) representing an event or data
        to send to the OpenAI Realtime API, converts it to JSON format, and transmits
        it over the established WebSocket connection.

        OpenAI Realtime API Events:
        Payloads can include various event types, such as:
        - Session configuration updates (e.g., voice settings, tools)
        - Audio input buffers (streaming user voice data)
        - Control commands (e.g., start/stop responses)
        These events enable real-time control and data streaming to the AI model.

        WebSocket Transmission:
        The JSON data is sent asynchronously over the WebSocket, allowing for
        continuous, low-latency communication. This supports real-time interactions
        where user inputs (like voice) are sent immediately as they occur.
        """
        if self._ws is None:
            raise RuntimeError("Not connected")
        raw = json.dumps(payload)
        await self._ws.send(raw)

    async def receive(self) -> dict[str, Any]:
        """Block until one JSON message arrives from the server.

        Raises ``websockets.ConnectionClosed`` when the peer disconnects.

        This method waits asynchronously for incoming data from the OpenAI Realtime API
        over the WebSocket connection. It receives raw JSON data, parses it into a
        Python dictionary, and returns it for processing.

        OpenAI Realtime API Responses:
        Incoming messages can include:
        - Audio output deltas (streaming AI voice responses)
        - Transcripts of AI speech
        - Event acknowledgments or errors
        - Real-time status updates
        These responses enable the application to react immediately to AI outputs.

        Real-Time Reception:
        By blocking until a message arrives, this method supports the real-time nature
        of the API, ensuring that audio and event data are processed as soon as they
        are received, maintaining low-latency interactions. The WebSocket's full-duplex
        capability allows simultaneous sending and receiving, crucial for conversational AI.
        """
        if self._ws is None:
            raise RuntimeError("Not connected")
        raw = await self._ws.recv()
        return json.loads(raw)

    async def close(self) -> None:
        """Gracefully close the WebSocket.

        This method terminates the WebSocket connection to the OpenAI Realtime API,
        ensuring proper cleanup of resources. It attempts to close the connection
        cleanly and handles any exceptions that may occur during closure.

        WebSocket Closure:
        Closing the WebSocket ends the persistent connection, stopping all real-time
        data exchange. This is important for resource management and to signal the
        end of the session to OpenAI's servers.

        Real-Time Session End:
        Properly closing the connection ensures that any ongoing real-time interactions
        are concluded, preventing data loss or hanging connections. In a voice application,
        this might correspond to ending a conversation or shutting down the voice interface.
        """
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            finally:
                self._ws = None
                logger.info("WebSocket closed")
