"""Tiny HTTP server that serves the live-signals viewer page.

Pairs with ``transport/websocket.py`` — the viewer HTML is a single file
that opens a WebSocket back to the Python process and renders the
DerivedSignals stream.

Runs in a daemon thread so the main capture loop stays synchronous.
"""

from __future__ import annotations

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_VIEWER_DIR = Path(__file__).resolve().parent.parent / "td"


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, websocket_port: int, **kwargs):
        self._ws_port = websocket_port
        super().__init__(*args, directory=str(_VIEWER_DIR), **kwargs)

    def do_GET(self):  # noqa: N802 — stdlib signature
        if self.path in ("/", "/index.html", "/viewer", "/viewer.html"):
            self._serve_viewer()
            return
        if self.path == "/ws-port":
            body = str(self._ws_port).encode("ascii")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def _serve_viewer(self) -> None:
        html_path = _VIEWER_DIR / "viewer.html"
        body = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002, N802 — stdlib signature
        # Silence per-request stderr noise; the main loop already prints
        # all the useful lifecycle info.
        return


class HttpViewerServer:
    def __init__(self, port: int, websocket_port: int, host: str = "127.0.0.1") -> None:
        self._port = port
        self._host = host
        self._ws_port = websocket_port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    def start(self) -> None:
        handler_cls = partial(_Handler, websocket_port=self._ws_port)
        self._server = ThreadingHTTPServer((self._host, self._port), handler_cls)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
