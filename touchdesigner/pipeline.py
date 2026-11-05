"""Per-frame pipeline: tracking results -> derived signals.

Kept separate from ``main.py`` so it can be unit-tested without a real
camera or OSC stack.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

from . import config
from .features import hands as hand_feats
from .features import face as face_feats
from .features.gestures import classify_hand
from .features.motion import WristVelocityTracker
from .schema import (
    DerivedSignals,
    FaceSignals,
    HandSignals,
    LandmarkFrame,
    Point,
    SessionSignals,
)
from .state.session import HandsSessionDetector
from .tracking.holistic import HolisticFrame


@dataclass
class _FpsMeter:
    """Small EMA-smoothed FPS estimate for the heartbeat channel."""

    alpha: float = 0.1
    _last: float = 0.0
    _fps: float = 0.0

    def tick(self, now_s: float) -> float:
        if self._last <= 0.0:
            self._last = now_s
            return 0.0
        dt = now_s - self._last
        self._last = now_s
        if dt <= 0:
            return self._fps
        instant = 1.0 / dt
        self._fps = self.alpha * instant + (1 - self.alpha) * self._fps
        return self._fps


_CAMERA_GRID_FRAMES = config.MAX_SEQUENCE_FRAMES  # cap matches landmark buffer
_LATEST_GESTURE_HINTS = 8  # how many recent per-hand gesture hints to keep


@dataclass
class PipelineState:
    """Owns everything that persists across frames."""

    landmark_history: Deque[LandmarkFrame] = field(
        default_factory=lambda: deque(maxlen=config.MAX_SEQUENCE_FRAMES)
    )
    timestamp_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=config.MAX_SEQUENCE_FRAMES)
    )
    # Rolling buffer of BGR camera frames captured only while a session is
    # active. We hold references not copies — cv2 returns a new numpy array
    # each read(), so this is safe and cheap.
    camera_history: Deque[Any] = field(
        default_factory=lambda: deque(maxlen=_CAMERA_GRID_FRAMES)
    )
    gesture_history: Deque[dict] = field(
        default_factory=lambda: deque(maxlen=_LATEST_GESTURE_HINTS)
    )
    left_velocity: WristVelocityTracker = field(default_factory=WristVelocityTracker)
    right_velocity: WristVelocityTracker = field(default_factory=WristVelocityTracker)
    session: HandsSessionDetector = field(default_factory=HandsSessionDetector)
    left_pinch_closed: bool = False
    right_pinch_closed: bool = False
    frame_index: int = 0
    dropped_frames: int = 0
    _fps: _FpsMeter = field(default_factory=_FpsMeter)


def reset_session_buffers(state: "PipelineState") -> None:
    """Clear the per-session rolling buffers. Call this when a new hands-up
    session opens so the next translation reflects the current attempt."""
    state.landmark_history.clear()
    state.timestamp_history.clear()
    state.camera_history.clear()
    state.gesture_history.clear()


def record_camera_frame(state: "PipelineState", frame_bgr) -> None:
    """Append a camera frame for the sign-grid. Call per frame; the deque
    truncates automatically. Caller decides when to call (we want frames
    recorded *during* a session, not idle time)."""
    state.camera_history.append(frame_bgr)


def process_frame(state: PipelineState, tracked: HolisticFrame, timestamp_ms: float) -> DerivedSignals:
    """Advance one frame. Mutates `state`, returns the immutable signal snapshot."""
    normalized = tracked.normalized
    state.landmark_history.append(normalized)
    state.timestamp_history.append(timestamp_ms)
    state.frame_index += 1

    left_signals = _compute_hand_signals(
        raw_hand=tracked.left_hand_raw,
        normalized_hand=normalized.leftHand,
        normalized_wrist=normalized.pose.leftWrist,
        raw_wrist_xy=_wrist_raw_xy(tracked.pose_raw, "left"),
        velocity_tracker=state.left_velocity,
        prev_pinch_closed=state.left_pinch_closed,
        timestamp_ms=timestamp_ms,
        usable=tracked.left_hand_usable,
    )
    state.left_pinch_closed = left_signals.pinch_state

    right_signals = _compute_hand_signals(
        raw_hand=tracked.right_hand_raw,
        normalized_hand=normalized.rightHand,
        normalized_wrist=normalized.pose.rightWrist,
        raw_wrist_xy=_wrist_raw_xy(tracked.pose_raw, "right"),
        velocity_tracker=state.right_velocity,
        prev_pinch_closed=state.right_pinch_closed,
        timestamp_ms=timestamp_ms,
        usable=tracked.right_hand_usable,
    )
    state.right_pinch_closed = right_signals.pinch_state

    hands_present = left_signals.present or right_signals.present
    session_signals, _ = state.session.update(hands_present, timestamp_ms)

    # Keep a short rolling list of the most recent per-hand gesture labels.
    # The Gemini interpreter uses these as weak secondary hints.
    if left_signals.present and left_signals.gesture != "none":
        state.gesture_history.append(
            {"name": f"left hand {left_signals.gesture}", "score": float(left_signals.gesture_score)}
        )
    if right_signals.present and right_signals.gesture != "none":
        state.gesture_history.append(
            {"name": f"right hand {right_signals.gesture}", "score": float(right_signals.gesture_score)}
        )

    face_present = bool(normalized.face.lips)
    face_signals = FaceSignals(
        present=face_present,
        mouth_open=face_feats.mouth_open_score(normalized) if face_present else 0.0,
        nose_x=normalized.pose.nose.x if normalized.pose.nose else 0.0,
        nose_y=normalized.pose.nose.y if normalized.pose.nose else 0.0,
    )

    fps = state._fps.tick(time.monotonic())

    return DerivedSignals(
        frame_index=state.frame_index,
        timestamp_ms=timestamp_ms,
        fps=fps,
        dropped_frames=state.dropped_frames,
        shoulder_center_x=tracked.shoulder_center_xy[0],
        shoulder_center_y=tracked.shoulder_center_xy[1],
        shoulder_width=tracked.shoulder_width,
        left=left_signals,
        right=right_signals,
        face=face_signals,
        session=session_signals,
    )


def _wrist_raw_xy(pose_raw: list[Point], side: str) -> tuple[float, float]:
    # mediapipe pose landmark indices: 15 = left wrist, 16 = right wrist.
    # These are the MIRROR of MediaPipe's "anatomical left/right" on a
    # non-selfie image, but because we flip the camera horizontally in
    # capture/webcam.py the caller-facing "left" is indeed landmark 15.
    index = 15 if side == "left" else 16
    if index < len(pose_raw):
        p = pose_raw[index]
        return p.x, p.y
    return 0.0, 0.0


def _compute_hand_signals(
    raw_hand: list[Point],
    normalized_hand: list[Point],
    normalized_wrist: Optional[Point],
    raw_wrist_xy: tuple[float, float],
    velocity_tracker: WristVelocityTracker,
    prev_pinch_closed: bool,
    timestamp_ms: float,
    usable: bool,
) -> HandSignals:
    if not usable or not normalized_hand:
        velocity_tracker.reset()
        return HandSignals(present=False)

    openness = hand_feats.hand_openness(normalized_hand)
    palm = hand_feats.palm_facing_score(normalized_hand)
    extended = hand_feats.hand_extended_count(normalized_hand)
    pinch_d = hand_feats.pinch_distance(normalized_hand)
    pinch_closed = hand_feats.pinch_state(
        normalized_hand,
        previous_closed=prev_pinch_closed,
        close_t=config.PINCH_CLOSE_THRESHOLD,
        open_t=config.PINCH_OPEN_THRESHOLD,
    )
    gesture_name, gesture_score = classify_hand(normalized_hand)

    vel = velocity_tracker.update(normalized_wrist, timestamp_ms)

    return HandSignals(
        present=True,
        openness=openness,
        palm_facing=palm,
        extended_count=extended,
        pinch_distance=pinch_d if pinch_d != float("inf") else 0.0,
        pinch_state=pinch_closed,
        wrist_x=normalized_wrist.x if normalized_wrist else 0.0,
        wrist_y=normalized_wrist.y if normalized_wrist else 0.0,
        wrist_z=normalized_wrist.z if normalized_wrist else 0.0,
        raw_wrist_x=raw_wrist_xy[0],
        raw_wrist_y=raw_wrist_xy[1],
        wrist_vx=vel.vx,
        wrist_vy=vel.vy,
        gesture=gesture_name,
        gesture_score=gesture_score,
    )
