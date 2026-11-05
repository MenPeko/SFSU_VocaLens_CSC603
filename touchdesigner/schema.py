"""Landmark + derived-signal schemas.

The landmark schema is deliberately identical to
``sign_classifier_service/schemas.py`` so the old classifier can consume
these frames unchanged. Derived signals are new, shaped for TD consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# -- Landmark frame (wire-compatible with the existing Python classifier) --

@dataclass
class Point:
    x: float
    y: float
    z: float
    visibility: Optional[float] = None

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass
class PoseSubset:
    nose: Optional[Point] = None
    leftShoulder: Optional[Point] = None
    rightShoulder: Optional[Point] = None
    leftElbow: Optional[Point] = None
    rightElbow: Optional[Point] = None
    leftWrist: Optional[Point] = None
    rightWrist: Optional[Point] = None


@dataclass
class FaceSubset:
    leftEyebrow: list[Point] = field(default_factory=list)
    rightEyebrow: list[Point] = field(default_factory=list)
    leftEye: list[Point] = field(default_factory=list)
    rightEye: list[Point] = field(default_factory=list)
    lips: list[Point] = field(default_factory=list)


@dataclass
class LandmarkFrame:
    timestampMs: float
    pose: PoseSubset = field(default_factory=PoseSubset)
    leftHand: list[Point] = field(default_factory=list)
    rightHand: list[Point] = field(default_factory=list)
    face: FaceSubset = field(default_factory=FaceSubset)


# -- Derived per-frame signals (this is what TD actually consumes) --

HandSide = Literal["left", "right"]


@dataclass
class HandSignals:
    present: bool = False
    openness: float = 0.0         # 0 = fist, 1 = open palm
    palm_facing: float = 0.0      # 0..1 (camera-facing likelihood)
    extended_count: float = 0.0   # fraction of fingers extended, 0..1
    pinch_distance: float = 1.0   # thumb-tip to index-tip in shoulder-normalized units
    pinch_state: bool = False     # Schmitt-triggered close/open
    # Wrist position in shoulder-normalized coords (x/y/z with 0,0,0 =
    # shoulder center, 1 = shoulder width). `raw_*` are the original
    # MediaPipe 0..1 image coords, handy for TD TOP overlays.
    wrist_x: float = 0.0
    wrist_y: float = 0.0
    wrist_z: float = 0.0
    raw_wrist_x: float = 0.0
    raw_wrist_y: float = 0.0
    # Velocity in shoulder-normalized units per second.
    wrist_vx: float = 0.0
    wrist_vy: float = 0.0
    gesture: str = "none"         # open_palm | fist | pointing | victory | thumbs_up | none
    gesture_score: float = 0.0


@dataclass
class FaceSignals:
    present: bool = False
    mouth_open: float = 0.0
    nose_x: float = 0.0
    nose_y: float = 0.0


@dataclass
class SessionSignals:
    """High-level interaction state (derived from hand presence over time)."""

    active: bool = False
    duration_ms: int = 0
    # Fires true for exactly one frame on each transition.
    started: bool = False
    ended: bool = False


@dataclass
class DerivedSignals:
    """Everything a TD patch needs for a single frame."""

    frame_index: int
    timestamp_ms: float
    fps: float
    dropped_frames: int
    shoulder_center_x: float
    shoulder_center_y: float
    shoulder_width: float
    left: HandSignals = field(default_factory=HandSignals)
    right: HandSignals = field(default_factory=HandSignals)
    face: FaceSignals = field(default_factory=FaceSignals)
    session: SessionSignals = field(default_factory=SessionSignals)


# -- Interop with sign_classifier_service ----------------------------------

def to_classifier_payload(frame: LandmarkFrame) -> dict:
    """Convert a LandmarkFrame into the JSON shape the existing Python
    classifier already accepts via ``LandmarkFrameModel.model_validate``."""

    def _point(p: Optional[Point]) -> Optional[dict]:
        if p is None:
            return None
        out: dict = {"x": p.x, "y": p.y, "z": p.z}
        if p.visibility is not None:
            out["visibility"] = p.visibility
        return out

    def _points(points: list[Point]) -> list[dict]:
        return [_point(p) for p in points if p is not None]  # type: ignore[list-item]

    return {
        "timestampMs": frame.timestampMs,
        "pose": {
            "nose": _point(frame.pose.nose),
            "leftShoulder": _point(frame.pose.leftShoulder),
            "rightShoulder": _point(frame.pose.rightShoulder),
            "leftElbow": _point(frame.pose.leftElbow),
            "rightElbow": _point(frame.pose.rightElbow),
            "leftWrist": _point(frame.pose.leftWrist),
            "rightWrist": _point(frame.pose.rightWrist),
        },
        "leftHand": _points(frame.leftHand),
        "rightHand": _points(frame.rightHand),
        "face": {
            "leftEyebrow": _points(frame.face.leftEyebrow),
            "rightEyebrow": _points(frame.face.rightEyebrow),
            "leftEye": _points(frame.face.leftEye),
            "rightEye": _points(frame.face.rightEye),
            "lips": _points(frame.face.lips),
        },
    }
