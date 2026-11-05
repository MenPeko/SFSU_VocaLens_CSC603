"""Optional JSON WebSocket broadcaster.

TouchDesigner can consume OSC directly, so this is not on the critical
path. It is useful for:

  * a second live debugger (a web page, `wscat`, etc.)
  * sending the full normalized landmark frame when a consumer wants the
    raw skeleton instead of derived signals
  * feeding other tools (Max/MSP, Unity, a browser visualizer)

Runs inside its own asyncio loop on a background thread so the main
capture loop remains a simple synchronous loop.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict
from typing import Optional

import websockets


class WebsocketBroadcaster:
    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self._host = host
        self._port = port
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[websockets.WebSocketServer] = None
        self._clients: set = set()
        self._started = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started.wait(timeout=3.0)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_until_complete(self._serve_forever())

    async def _serve_forever(self) -> None:
        async def handler(ws):
            self._clients.add(ws)
            try:
                async for _ in ws:
                    # We don't accept inbound messages.
                    pass
            finally:
                self._clients.discard(ws)

        self._server = await websockets.serve(handler, self._host, self._port)
        self._started.set()
        await asyncio.Future()  # run forever

    def broadcast(self, payload_dict: dict) -> None:
        if self._loop is None or not self._clients:
            return
        message = json.dumps(payload_dict)

        async def _send_all():
            if not self._clients:
                return
            await asyncio.gather(
                *(ws.send(message) for ws in list(self._clients)),
                return_exceptions=True,
            )

        asyncio.run_coroutine_threadsafe(_send_all(), self._loop)

    def stop(self) -> None:
        if self._loop and self._server:
            self._loop.call_soon_threadsafe(self._server.close)


def signals_as_dict(signals) -> dict:
    """Convenience: convert a DerivedSignals (or any dataclass) to JSON-safe dict."""
    return asdict(signals)
