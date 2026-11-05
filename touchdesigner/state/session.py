"""Hands-up / hands-down session detector.

Port of the state machine embedded in ``useSignLanguage.ts`` around
``captureSessionActiveRef`` and ``handsDownTimeoutRef``:

    while hands visible:
        if not already capturing:
            start a new session, clear buffers
        keep recording
    on hands disappearing:
        wait HANDS_DOWN_DELAY_MS
        if session elapsed >= MIN_SEQUENCE_DURATION_MS:
            emit "session ended"
        else:
            silently drop (reject flicker)

This is broken out of the hook so TD can consume a clean per-frame
SessionSignals (active / started / ended / duration_ms) without the React
refs or canvas timers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..schema import SessionSignals


@dataclass
class SessionEvent:
    """Fine-grained event emitted only on state transitions.

    ``started`` fires the frame the session opens. ``ended`` fires the frame
    the hands-down timer fires AND the duration gate passes. Neither fires
    otherwise.
    """
    started: bool = False
    ended: bool = False


class HandsSessionDetector:
    def __init__(
        self,
        hands_down_delay_ms: int = config.HANDS_DOWN_DELAY_MS,
        min_duration_ms: int = config.MIN_SESSION_DURATION_MS,
    ) -> None:
        self._hands_down_delay_ms = hands_down_delay_ms
        self._min_duration_ms = min_duration_ms
        self._active = False
        self._started_at_ms: float | None = None
        self._last_hands_seen_at_ms: float | None = None

    def update(self, hands_present: bool, timestamp_ms: float) -> tuple[SessionSignals, SessionEvent]:
        event = SessionEvent()

        if hands_present:
            self._last_hands_seen_at_ms = timestamp_ms
            if not self._active:
                self._active = True
                self._started_at_ms = timestamp_ms
                event.started = True
        else:
            if self._active and self._last_hands_seen_at_ms is not None:
                hands_gone_for = timestamp_ms - self._last_hands_seen_at_ms
                if hands_gone_for >= self._hands_down_delay_ms:
                    # Apply the duration gate from MIN_SEQUENCE_DURATION_MS.
                    if (
                        self._started_at_ms is not None
                        and (self._last_hands_seen_at_ms - self._started_at_ms) >= self._min_duration_ms
                    ):
                        event.ended = True
                    self._active = False
                    self._started_at_ms = None

        duration_ms = 0
        if self._active and self._started_at_ms is not None:
            duration_ms = int(max(0, timestamp_ms - self._started_at_ms))

        signals = SessionSignals(
            active=self._active,
            duration_ms=duration_ms,
            started=event.started,
            ended=event.ended,
        )
        return signals, event

    def reset(self) -> None:
        self._active = False
        self._started_at_ms = None
        self._last_hands_seen_at_ms = None
