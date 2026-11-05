"""Face-level features.

Ports the helpers ``_mouth_open_score`` and ``_distance_to_face`` from
``sign_classifier_service/classifier.py``.
"""

from __future__ import annotations

import math

from ..schema import LandmarkFrame, Point


def _distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def mouth_open_score(frame: LandmarkFrame) -> float:
    """Port of ``_mouth_open_score``. 0 = closed, 1 = wide open."""
    lips = frame.face.lips
    if len(lips) < 4:
        return 0.0
    top = lips[3]
    bottom = lips[9] if len(lips) > 9 else lips[-1]
    corners = (lips[0], lips[min(6, len(lips) - 1)])
    mouth_width = max(_distance(corners[0], corners[1]), 1e-6)
    return min(1.0, _distance(top, bottom) / mouth_width)


def distance_to_face(frame: LandmarkFrame, hand: str) -> float:
    """Port of ``_distance_to_face``. `hand` is "left" or "right".
    Returns 99.0 when either wrist or lips are missing."""
    wrist = getattr(frame.pose, f"{hand}Wrist", None)
    lips = frame.face.lips
    if wrist is None or not lips:
        return 99.0
    lip_center = lips[len(lips) // 2]
    return _distance(wrist, lip_center)
