"""Low-level WebSocket connection handler for the OpenAI Realtime API.

Responsibility: Manage a single WebSocket connection lifecycle — open,
    send, receive, close — without any knowledge of event semantics.
Pattern: Adapter — wraps the ``websockets`` library behind a minimal
    async interface that the RealtimeManager consumes.
Why: Isolating the raw transport lets us swap libraries (e.g. aiohttp)
    or inject a mock connection in tests without touching business logic.
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
        """Open a WebSocket to the OpenAI Realtime endpoint."""
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
        """Serialize *payload* to JSON and send over the WebSocket."""
        if self._ws is None:
            raise RuntimeError("Not connected")
        raw = json.dumps(payload)
        await self._ws.send(raw)

    async def receive(self) -> dict[str, Any]:
        """Block until one JSON message arrives from the server.

        Raises ``websockets.ConnectionClosed`` when the peer disconnects.
        """
        if self._ws is None:
            raise RuntimeError("Not connected")
        raw = await self._ws.recv()
        return json.loads(raw)

    async def close(self) -> None:
        """Gracefully close the WebSocket."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            finally:
                self._ws = None
                logger.info("WebSocket closed")
