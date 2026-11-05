"""OSC sender.

Flat address schema — every signal is its own OSC address so TouchDesigner
can pick channels one by one via a single OSC In CHOP.

See ``td/OSC_ADDRESSES.md`` for the full contract.
"""

from __future__ import annotations

from typing import Optional

from pythonosc.udp_client import SimpleUDPClient

from .. import config
from ..schema import DerivedSignals, HandSignals


class OscBroadcaster:
    def __init__(
        self,
        host: str = config.DEFAULT_OSC_HOST,
        port: int = config.DEFAULT_OSC_PORT,
        enabled: bool = True,
        root: str = config.OSC_ROOT,
    ) -> None:
        self._enabled = enabled
        self._root = root.rstrip("/")
        self._client: Optional[SimpleUDPClient] = (
            SimpleUDPClient(host, port) if enabled else None
        )

    def _addr(self, path: str) -> str:
        return f"{self._root}{path}"

    def _send(self, path: str, value) -> None:
        if not self._enabled or self._client is None:
            return
        self._client.send_message(self._addr(path), value)

    def broadcast(self, signals: DerivedSignals) -> None:
        if not self._enabled:
            return
        self._send("/heartbeat/frame_index", int(signals.frame_index))
        self._send("/heartbeat/fps", float(signals.fps))
        self._send("/heartbeat/dropped_frames", int(signals.dropped_frames))
        self._send("/heartbeat/timestamp_ms", float(signals.timestamp_ms))

        self._send("/pose/shoulder_center_x", float(signals.shoulder_center_x))
        self._send("/pose/shoulder_center_y", float(signals.shoulder_center_y))
        self._send("/pose/shoulder_width", float(signals.shoulder_width))

        self._broadcast_hand("left", signals.left)
        self._broadcast_hand("right", signals.right)

        self._send("/face/present", int(signals.face.present))
        self._send("/face/mouth_open", float(signals.face.mouth_open))
        self._send("/face/nose_x", float(signals.face.nose_x))
        self._send("/face/nose_y", float(signals.face.nose_y))

        self._send("/session/active", int(signals.session.active))
        self._send("/session/duration_ms", int(signals.session.duration_ms))
        if signals.session.started:
            self._send("/session/event/started", 1)
        if signals.session.ended:
            self._send("/session/event/ended", 1)

    def _broadcast_hand(self, side: str, hand: HandSignals) -> None:
        prefix = f"/hand/{side}"
        self._send(f"{prefix}/present", int(hand.present))
        self._send(f"{prefix}/openness", float(hand.openness))
        self._send(f"{prefix}/palm_facing", float(hand.palm_facing))
        self._send(f"{prefix}/extended_count", float(hand.extended_count))
        self._send(f"{prefix}/pinch_distance", float(hand.pinch_distance))
        self._send(f"{prefix}/pinch_state", int(hand.pinch_state))
        self._send(f"{prefix}/wrist_x", float(hand.wrist_x))
        self._send(f"{prefix}/wrist_y", float(hand.wrist_y))
        self._send(f"{prefix}/wrist_z", float(hand.wrist_z))
        self._send(f"{prefix}/raw_wrist_x", float(hand.raw_wrist_x))
        self._send(f"{prefix}/raw_wrist_y", float(hand.raw_wrist_y))
        self._send(f"{prefix}/wrist_vx", float(hand.wrist_vx))
        self._send(f"{prefix}/wrist_vy", float(hand.wrist_vy))
        self._send(f"{prefix}/gesture_score", float(hand.gesture_score))
        # Gesture label is sent as a string so TD Script DATs can switch on it.
        self._send(f"{prefix}/gesture", str(hand.gesture))

    def close(self) -> None:
        self._client = None
        self._enabled = False
