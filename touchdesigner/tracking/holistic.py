"""MediaPipe Holistic wrapper.

Mirrors the option set configured in ``useSignLanguage.ts``:

    holistic.setOptions({
        selfieMode: false,
        modelComplexity: captureSettings.holisticModelComplexity,
        smoothLandmarks: true,
        refineFaceLandmarks: true,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.65,
    })

Note on selfieMode: we handle mirroring at the cv2 layer
(`WebcamCapture(flip_horizontal=True)`) so we tell MediaPipe the image is
already in "the user's view". The JS hook does the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .. import config
from .landmarks import (
    build_normalized_landmark_frame,
    hand_landmarks_usable,
    points_from_mediapipe_results,
)
from ..schema import LandmarkFrame, Point


_mp_holistic = mp.solutions.holistic


@dataclass
class HolisticFrame:
    """Output of one call to HolisticTracker.process()."""

    normalized: LandmarkFrame
    left_hand_raw: list[Point]
    right_hand_raw: list[Point]
    pose_raw: list[Point]
    face_raw: list[Point]
    shoulder_center_xy: tuple[float, float]
    shoulder_width: float
    left_hand_usable: bool
    right_hand_usable: bool


class HolisticTracker:
    def __init__(
        self,
        model_complexity: int = config.HOLISTIC_MODEL_COMPLEXITY,
        refine_face_landmarks: bool = config.HOLISTIC_REFINE_FACE_LANDMARKS,
        smooth_landmarks: bool = config.HOLISTIC_SMOOTH_LANDMARKS,
        min_detection_confidence: float = config.HOLISTIC_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.HOLISTIC_MIN_TRACKING_CONFIDENCE,
        disable_face: bool = False,
    ) -> None:
        self._holistic: Any = _mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            smooth_landmarks=bool(smooth_landmarks),
            enable_segmentation=False,
            refine_face_landmarks=bool(refine_face_landmarks) and not disable_face,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._disable_face = disable_face

    def process(self, frame_bgr: np.ndarray, timestamp_ms: float) -> HolisticFrame:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # MediaPipe is faster on a read-only array.
        rgb.flags.writeable = False
        results = self._holistic.process(rgb)

        pose, left_hand, right_hand, face = points_from_mediapipe_results(results)
        if self._disable_face:
            face = []

        normalized, shoulder_center, shoulder_width = build_normalized_landmark_frame(
            pose_points=pose,
            left_hand_points=left_hand,
            right_hand_points=right_hand,
            face_points=face,
            timestamp_ms=timestamp_ms,
        )

        return HolisticFrame(
            normalized=normalized,
            left_hand_raw=left_hand,
            right_hand_raw=right_hand,
            pose_raw=pose,
            face_raw=face,
            shoulder_center_xy=shoulder_center,
            shoulder_width=shoulder_width,
            left_hand_usable=hand_landmarks_usable(left_hand),
            right_hand_usable=hand_landmarks_usable(right_hand),
        )

    def close(self) -> None:
        try:
            self._holistic.close()
        except Exception:
            pass

    def __enter__(self) -> "HolisticTracker":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
