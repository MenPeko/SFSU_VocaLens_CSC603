"""Rolling-window temporal features.

Port of ``src/sign/sequence.ts`` (``appendRollingWindow``,
``computeSequenceDurationMs``) plus new wrist-velocity derivation for TD.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from ..schema import Point


def append_rolling_window(buffer: Deque, value, max_size: int) -> None:
    """In-place equivalent of the TS ``appendRollingWindow`` helper.
    Uses deque(maxlen=max_size) for O(1) push/pop."""
    if buffer.maxlen != max_size:
        raise ValueError(
            f"append_rolling_window expected deque(maxlen={max_size}), got maxlen={buffer.maxlen}"
        )
    buffer.append(value)


def compute_sequence_duration_ms(timestamps_ms: list[float] | Deque[float]) -> float:
    """Port of ``computeSequenceDurationMs``."""
    if len(timestamps_ms) < 2:
        return 0.0
    first = next(iter(timestamps_ms))
    last_value: float = first
    for value in timestamps_ms:
        last_value = value
    return max(0.0, last_value - first)


@dataclass
class VelocityEstimate:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    speed: float = 0.0


class WristVelocityTracker:
    """Two-point finite-difference velocity for one wrist.

    Returns velocity in shoulder-normalized units per second. Uses a tiny
    EMA (alpha = 0.5) to smooth the frame-to-frame chatter MediaPipe
    produces when the hand is roughly stationary.
    """

    def __init__(self, smoothing: float = 0.5) -> None:
        self._last: Optional[tuple[Point, float]] = None
        self._smoothing = max(0.0, min(1.0, smoothing))
        self._smoothed = VelocityEstimate()

    def update(self, wrist: Optional[Point], timestamp_ms: float) -> VelocityEstimate:
        if wrist is None:
            self._last = None
            self._smoothed = VelocityEstimate()
            return self._smoothed

        if self._last is None:
            self._last = (wrist, timestamp_ms)
            self._smoothed = VelocityEstimate()
            return self._smoothed

        prev_wrist, prev_t = self._last
        dt_s = max(1e-6, (timestamp_ms - prev_t) / 1000.0)
        raw = VelocityEstimate(
            vx=(wrist.x - prev_wrist.x) / dt_s,
            vy=(wrist.y - prev_wrist.y) / dt_s,
            vz=(wrist.z - prev_wrist.z) / dt_s,
        )
        raw.speed = (raw.vx ** 2 + raw.vy ** 2 + raw.vz ** 2) ** 0.5

        a = self._smoothing
        self._smoothed = VelocityEstimate(
            vx=a * raw.vx + (1 - a) * self._smoothed.vx,
            vy=a * raw.vy + (1 - a) * self._smoothed.vy,
            vz=a * raw.vz + (1 - a) * self._smoothed.vz,
        )
        self._smoothed.speed = (
            self._smoothed.vx ** 2 + self._smoothed.vy ** 2 + self._smoothed.vz ** 2
        ) ** 0.5
        self._last = (wrist, timestamp_ms)
        return self._smoothed

    def reset(self) -> None:
        self._last = None
        self._smoothed = VelocityEstimate()
