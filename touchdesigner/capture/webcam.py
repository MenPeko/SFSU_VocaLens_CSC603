"""cv2-backed webcam capture.

Replaces the browser's getUserMedia + HLS/WebRTC attach logic from
``src/webview/hooks/useSignLanguage.ts``. TD does not need remote stream
adapters — we want the lowest-latency local device path.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CaptureInfo:
    width: int
    height: int
    fps_requested: float
    fps_actual: float


class WebcamCapture:
    """Minimal webcam wrapper with horizontal-flip + BGR→RGB conversion.

    Mediapipe wants RGB; cv2 gives BGR. We flip horizontally because the
    original selfie flow assumed a mirror view (see useSignLanguage's
    `mirrorOverlayText`).
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        flip_horizontal: bool = True,
    ) -> None:
        self.index = index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.flip_horizontal = flip_horizontal
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> CaptureInfo:
        # AVFoundation on macOS, DSHOW on Windows, V4L2 on Linux — cv2 picks
        # the right one by default but we log what we actually got.
        backend = 0
        if sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION

        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index={self.index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.requested_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.requested_height))
        cap.set(cv2.CAP_PROP_FPS, float(self.requested_fps))
        # Minimize buffering so frames reflect "now", not 4 frames ago — the
        # TD equivalent of the CAPTURE_DRIFT_SEEK_THRESHOLD_S snap in the JS hook.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap = cap
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or self.requested_fps

        return CaptureInfo(
            width=actual_width,
            height=actual_height,
            fps_requested=self.requested_fps,
            fps_actual=actual_fps,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            return False, None
        if self.flip_horizontal:
            frame_bgr = cv2.flip(frame_bgr, 1)
        return True, frame_bgr

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def now_ms() -> float:
    """Monotonic millisecond clock — use for timestampMs stamping."""
    return time.monotonic() * 1000.0
