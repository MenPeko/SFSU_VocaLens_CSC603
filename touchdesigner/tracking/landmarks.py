"""Landmark subset extraction + shoulder-centric normalization.

This is a direct port of the TypeScript helpers in
``src/webview/hooks/useSignLanguage.ts``:

  * FACE_SUBSET_INDICES (leftEyebrow / rightEyebrow / leftEye / rightEye / lips)
  * ARM_POSE_INDICES (the six upper-body joints we care about)
  * pickLandmarks
  * buildFaceSubset
  * normalizePoint / normalizePoints
  * buildNormalizedLandmarkFrame

Keeping byte-identical math matters: the existing trained classifier was
fit in this normalized space and refuses to serve predictions for samples
emitted under a different extractor stamp.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import mediapipe as mp

from ..schema import FaceSubset, LandmarkFrame, Point, PoseSubset


_mp_holistic = mp.solutions.holistic
_POSE_LANDMARK = _mp_holistic.PoseLandmark  # enum-like; int indices match JS POSE_LANDMARKS


# Same indices as the TS hook. Do not reorder — the classifier is index-sensitive.
FACE_SUBSET_INDICES: dict[str, tuple[int, ...]] = {
    "leftEyebrow": (70, 63, 105, 66, 107),
    "rightEyebrow": (336, 296, 334, 293, 300),
    "leftEye": (33, 160, 158, 133, 153, 144),
    "rightEye": (263, 387, 385, 362, 380, 373),
    "lips": (61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91),
}


ARM_POSE_INDICES: frozenset[int] = frozenset(
    {
        _POSE_LANDMARK.LEFT_SHOULDER.value,
        _POSE_LANDMARK.RIGHT_SHOULDER.value,
        _POSE_LANDMARK.LEFT_ELBOW.value,
        _POSE_LANDMARK.RIGHT_ELBOW.value,
        _POSE_LANDMARK.LEFT_WRIST.value,
        _POSE_LANDMARK.RIGHT_WRIST.value,
    }
)


EXPECTED_HAND_LANDMARK_COUNT = 21


def _to_point(lm) -> Point | None:
    """Convert a mediapipe NormalizedLandmark to our Point dataclass."""
    if lm is None:
        return None
    x = float(getattr(lm, "x", 0.0))
    y = float(getattr(lm, "y", 0.0))
    z = float(getattr(lm, "z", 0.0))
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    visibility = getattr(lm, "visibility", None)
    return Point(x=x, y=y, z=z, visibility=float(visibility) if visibility is not None else None)


def _points_from_landmark_list(landmark_list) -> list[Point]:
    if landmark_list is None:
        return []
    out: list[Point] = []
    for lm in landmark_list.landmark:
        p = _to_point(lm)
        if p is not None:
            out.append(p)
    return out


def pick_landmarks(points: Sequence[Point], indices: Iterable[int]) -> list[Point]:
    """TS pickLandmarks port. Silently drops missing indices."""
    out: list[Point] = []
    for index in indices:
        if 0 <= index < len(points):
            p = points[index]
            if p is not None:
                out.append(p)
    return out


def hand_landmarks_usable(points: Sequence[Point]) -> bool:
    """Mirror hasUsableHandLandmarks from useSignLanguage."""
    if len(points) < EXPECTED_HAND_LANDMARK_COUNT:
        return False
    for p in points:
        if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)):
            return False
    return True


def build_face_subset(face_points: Sequence[Point]) -> FaceSubset:
    return FaceSubset(
        leftEyebrow=pick_landmarks(face_points, FACE_SUBSET_INDICES["leftEyebrow"]),
        rightEyebrow=pick_landmarks(face_points, FACE_SUBSET_INDICES["rightEyebrow"]),
        leftEye=pick_landmarks(face_points, FACE_SUBSET_INDICES["leftEye"]),
        rightEye=pick_landmarks(face_points, FACE_SUBSET_INDICES["rightEye"]),
        lips=pick_landmarks(face_points, FACE_SUBSET_INDICES["lips"]),
    )


def _normalize_point(p: Point | None, cx: float, cy: float, cz: float, scale: float) -> Point | None:
    if p is None:
        return None
    return Point(
        x=(p.x - cx) / scale,
        y=(p.y - cy) / scale,
        z=(p.z - cz) / scale,
        visibility=p.visibility,
    )


def _normalize_points(points: Sequence[Point], cx: float, cy: float, cz: float, scale: float) -> list[Point]:
    out: list[Point] = []
    for p in points:
        np_ = _normalize_point(p, cx, cy, cz, scale)
        if np_ is not None:
            out.append(np_)
    return out


def build_pose_subset(pose_points: Sequence[Point]) -> PoseSubset:
    def _at(idx: int) -> Point | None:
        if 0 <= idx < len(pose_points):
            return pose_points[idx]
        return None

    return PoseSubset(
        nose=_at(_POSE_LANDMARK.NOSE.value),
        leftShoulder=_at(_POSE_LANDMARK.LEFT_SHOULDER.value),
        rightShoulder=_at(_POSE_LANDMARK.RIGHT_SHOULDER.value),
        leftElbow=_at(_POSE_LANDMARK.LEFT_ELBOW.value),
        rightElbow=_at(_POSE_LANDMARK.RIGHT_ELBOW.value),
        leftWrist=_at(_POSE_LANDMARK.LEFT_WRIST.value),
        rightWrist=_at(_POSE_LANDMARK.RIGHT_WRIST.value),
    )


def build_normalized_landmark_frame(
    pose_points: Sequence[Point],
    left_hand_points: Sequence[Point],
    right_hand_points: Sequence[Point],
    face_points: Sequence[Point],
    timestamp_ms: float,
) -> tuple[LandmarkFrame, tuple[float, float], float]:
    """Port of ``buildNormalizedLandmarkFrame``.

    Returns the normalized frame, the *raw* shoulder-center image coords
    (useful for TD overlays), and the raw shoulder width (useful as a
    distance-from-camera proxy).
    """
    pose_raw = build_pose_subset(pose_points)
    face_raw = build_face_subset(face_points)

    left_shoulder = pose_raw.leftShoulder
    right_shoulder = pose_raw.rightShoulder

    if left_shoulder and right_shoulder:
        center_x = (left_shoulder.x + right_shoulder.x) / 2.0
        center_y = (left_shoulder.y + right_shoulder.y) / 2.0
        center_z = (left_shoulder.z + right_shoulder.z) / 2.0
        shoulder_width = math.sqrt(
            (right_shoulder.x - left_shoulder.x) ** 2
            + (right_shoulder.y - left_shoulder.y) ** 2
            + (right_shoulder.z - left_shoulder.z) ** 2
        )
    else:
        center_x, center_y, center_z, shoulder_width = 0.5, 0.5, 0.0, 0.0

    scale = shoulder_width if shoulder_width > 1e-4 else 1.0

    normalized = LandmarkFrame(
        timestampMs=timestamp_ms,
        pose=PoseSubset(
            nose=_normalize_point(pose_raw.nose, center_x, center_y, center_z, scale),
            leftShoulder=_normalize_point(pose_raw.leftShoulder, center_x, center_y, center_z, scale),
            rightShoulder=_normalize_point(pose_raw.rightShoulder, center_x, center_y, center_z, scale),
            leftElbow=_normalize_point(pose_raw.leftElbow, center_x, center_y, center_z, scale),
            rightElbow=_normalize_point(pose_raw.rightElbow, center_x, center_y, center_z, scale),
            leftWrist=_normalize_point(pose_raw.leftWrist, center_x, center_y, center_z, scale),
            rightWrist=_normalize_point(pose_raw.rightWrist, center_x, center_y, center_z, scale),
        ),
        leftHand=_normalize_points(left_hand_points, center_x, center_y, center_z, scale),
        rightHand=_normalize_points(right_hand_points, center_x, center_y, center_z, scale),
        face=FaceSubset(
            leftEyebrow=_normalize_points(face_raw.leftEyebrow, center_x, center_y, center_z, scale),
            rightEyebrow=_normalize_points(face_raw.rightEyebrow, center_x, center_y, center_z, scale),
            leftEye=_normalize_points(face_raw.leftEye, center_x, center_y, center_z, scale),
            rightEye=_normalize_points(face_raw.rightEye, center_x, center_y, center_z, scale),
            lips=_normalize_points(face_raw.lips, center_x, center_y, center_z, scale),
        ),
    )

    return normalized, (center_x, center_y), shoulder_width


def points_from_mediapipe_results(results) -> tuple[list[Point], list[Point], list[Point], list[Point]]:
    """Pull MediaPipe's NormalizedLandmarkLists out of a Holistic result and
    convert each to our Point list. Returns (pose, leftHand, rightHand, face).
    """
    pose = _points_from_landmark_list(results.pose_landmarks)
    left = _points_from_landmark_list(results.left_hand_landmarks)
    right = _points_from_landmark_list(results.right_hand_landmarks)
    face = _points_from_landmark_list(results.face_landmarks)
    return pose, left, right, face
