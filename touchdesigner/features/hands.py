"""Hand-level features.

Direct ports of the helpers in ``sign_classifier_service/classifier.py``
(``_hand_extended_count``, ``_hand_openness``, ``_palm_facing_score``) plus
new pinch-distance / pinch-state derivations.

All inputs are the SHOULDER-NORMALIZED hand landmark list (21 points) from
``tracking/landmarks.build_normalized_landmark_frame``, so all distances are
in shoulder-width units.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..schema import Point


_TIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
_BASE_INDICES: tuple[int, ...] = (2, 5, 9, 13, 17)
_EXTENSION_PAIRS: tuple[tuple[int, int], ...] = tuple(zip(_TIP_INDICES, _BASE_INDICES))


def _distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def hand_extended_count(hand: Sequence[Point]) -> float:
    """Fraction of fingers whose tip is farther from wrist than their base.
    Ported from `_hand_extended_count`. Range 0..1.
    """
    if len(hand) < 21:
        return 0.0
    wrist = hand[0]
    extended = 0
    for tip_idx, base_idx in _EXTENSION_PAIRS:
        tip = hand[tip_idx]
        base = hand[base_idx]
        if _distance(tip, wrist) > _distance(base, wrist) * 1.2:
            extended += 1
    return extended / 5.0


def hand_openness(hand: Sequence[Point]) -> float:
    """Port of `_hand_openness`. Range 0..1 (0 = fist, 1 = fully open)."""
    if len(hand) < 21:
        return 0.0
    wrist = hand[0]
    ratios: list[float] = []
    for tip_idx, base_idx in zip(_TIP_INDICES, _BASE_INDICES):
        tip = hand[tip_idx]
        base = hand[base_idx]
        base_dist = max(_distance(base, wrist), 1e-6)
        ratios.append(_distance(tip, wrist) / base_dist)
    mean_ratio = sum(ratios) / len(ratios)
    return max(0.0, min(1.0, (mean_ratio - 1.0) / 0.6))


def palm_facing_score(hand: Sequence[Point]) -> float:
    """Port of `_palm_facing_score`. 1.0 ≈ palm faces the camera."""
    if len(hand) < 18:
        return 0.0
    wrist, index_mcp, pinky_mcp = hand[0], hand[5], hand[17]
    v1 = (index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z)
    v2 = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z)
    normal = (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )
    magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if magnitude <= 1e-6:
        return 0.0
    return min(1.0, abs(normal[2]) / magnitude)


def pinch_distance(hand: Sequence[Point]) -> float:
    """Thumb-tip (4) to index-tip (8) distance in shoulder-normalized units.

    A relaxed open hand is ~0.25, a closed pinch ~0.05. Not present in the
    original project — TouchDesigner commonly wants a pinch channel, so this
    is one of the few new additions.
    """
    if len(hand) < 9:
        return float("inf")
    return _distance(hand[4], hand[8])


def pinch_state(hand: Sequence[Point], previous_closed: bool, close_t: float, open_t: float) -> bool:
    """Schmitt trigger around `pinch_distance`. Takes the prior state so the
    caller can store it. Call as::

        is_closed = pinch_state(hand, prev, close_t=0.08, open_t=0.12)
    """
    d = pinch_distance(hand)
    if previous_closed:
        # Only release once the fingers are clearly apart again.
        return d < open_t
    # Not currently pinching — fire only on a decisive close.
    return d < close_t


def finger_curls(hand: Sequence[Point]) -> tuple[str, str, str, str, str]:
    """Coarse per-finger curl classification (straight | half | full).

    Replaces fingerpose's FingerCurl estimator for our tiny rule set.
    Ordered thumb, index, middle, ring, pinky — matches config.GESTURE_RULES.

    Heuristic: for each finger, measure the angle at the PIP joint. Straight
    finger ≈ 180°, fully curled ≈ 0°. We bin:
        > 160°         → straight
        100°..160°     → half
        < 100°         → full
    For the thumb we use CMC→MCP→IP instead of PIP, with a looser straight
    threshold because thumbs don't bend as much.
    """
    if len(hand) < 21:
        return ("full", "full", "full", "full", "full")

    # Joint triples per finger: (base, middle, tip-side)
    triples: tuple[tuple[int, int, int], ...] = (
        (1, 2, 4),   # thumb: CMC → MCP → tip
        (5, 6, 8),   # index: MCP → PIP → tip
        (9, 10, 12), # middle
        (13, 14, 16),# ring
        (17, 18, 20),# pinky
    )
    straight_thresholds = (145.0, 160.0, 160.0, 160.0, 160.0)
    half_thresholds = (110.0, 100.0, 100.0, 100.0, 100.0)

    curls: list[str] = []
    for (base_i, mid_i, tip_i), straight_t, half_t in zip(triples, straight_thresholds, half_thresholds):
        angle = _joint_angle_deg(hand[base_i], hand[mid_i], hand[tip_i])
        if angle > straight_t:
            curls.append("straight")
        elif angle > half_t:
            curls.append("half")
        else:
            curls.append("full")
    return tuple(curls)  # type: ignore[return-value]


def _joint_angle_deg(a: Point, b: Point, c: Point) -> float:
    """Angle at b formed by points a-b-c, in degrees."""
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    mag = math.sqrt((ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2) * (bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2))
    if mag <= 1e-9:
        return 180.0
    cosv = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cosv))


def hand_centroid(hand: Sequence[Point]) -> tuple[float, float, float]:
    """Mean of all 21 hand landmarks. Useful as a stable position channel
    when the wrist is noisy (MediaPipe's wrist landmark is sometimes more
    jittery than the palm as a whole)."""
    if not hand:
        return 0.0, 0.0, 0.0
    sx = sum(p.x for p in hand) / len(hand)
    sy = sum(p.y for p in hand) / len(hand)
    sz = sum(p.z for p in hand) / len(hand)
    return sx, sy, sz
