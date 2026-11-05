"""All tunables in one place. Values ported from:

  * src/sign/capture-config.ts
  * src/sign/capture-settings.ts
  * src/sign/core-signs.ts

Where a value was browser-specific (grid columns, JPEG compression, HLS drift)
it is dropped. TouchDesigner consumes the live feature stream, not a grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# -- Capture -----------------------------------------------------------------
# Holistic is expensive; 24 FPS on complexity=1 is the sweet spot on a 2022
# M1 laptop. Drop to 15 if complexity=2 and your CPU can't keep up.
DEFAULT_CAPTURE_FPS: float = 24.0

# Minimum camera resolution we ask cv2 for. Exact delivery depends on the
# driver; we treat lower as "sub-720p" and report it in the heartbeat OSC.
REQUESTED_CAPTURE_WIDTH: int = 1280
REQUESTED_CAPTURE_HEIGHT: int = 720


# -- Holistic ----------------------------------------------------------------
# Maps to the same options as the JS hook:
#   - modelComplexity 0 = lite (lowest latency, lowest quality)
#   - modelComplexity 1 = full (balanced)
#   - modelComplexity 2 = heavy (highest quality, ~2-3x slower)
HOLISTIC_MODEL_COMPLEXITY: int = 1
HOLISTIC_SMOOTH_LANDMARKS: bool = True
HOLISTIC_REFINE_FACE_LANDMARKS: bool = True
HOLISTIC_MIN_DETECTION_CONFIDENCE: float = 0.7
HOLISTIC_MIN_TRACKING_CONFIDENCE: float = 0.65


# -- Temporal state ----------------------------------------------------------
# Rolling window size for landmark/velocity history.
MAX_SEQUENCE_FRAMES: int = 48
# How long both hands must be absent before the session detector emits a
# `session_ended` event. Mirrors DEFAULT_HANDS_DOWN_DELAY_MS.
HANDS_DOWN_DELAY_MS: int = 1500
# Minimum elapsed wall-clock within a session before it counts as a real
# gesture rather than a flicker. Mirrors MIN_SEQUENCE_DURATION_MS.
MIN_SESSION_DURATION_MS: int = 500


# -- Pinch detection ---------------------------------------------------------
# Distances are in shoulder-width-normalized units (shoulder span = 1.0).
# A relaxed open hand has thumb-index tip distance ~0.25-0.35. A closed
# pinch is under 0.08. Use two thresholds with a gap to get clean on/off
# (Schmitt trigger) without flicker.
PINCH_CLOSE_THRESHOLD: float = 0.08
PINCH_OPEN_THRESHOLD: float = 0.12


# -- Gesture classifier (curl-based) ----------------------------------------
# Inspired by fingerpose's curl rules, collapsed into one short table so TD
# operators can tune them without reading JS source.
@dataclass(frozen=True)
class GestureRule:
    name: str
    # Required curls per finger: (thumb, index, middle, ring, pinky).
    # Each entry is one of "straight", "half", "full", or None (don't care).
    curls: tuple[str | None, str | None, str | None, str | None, str | None]


GESTURE_RULES: tuple[GestureRule, ...] = (
    GestureRule("open_palm", ("straight", "straight", "straight", "straight", "straight")),
    GestureRule("fist", ("full", "full", "full", "full", "full")),
    GestureRule("pointing", ("half", "straight", "full", "full", "full")),
    GestureRule("victory", ("half", "straight", "straight", "full", "full")),
    GestureRule("thumbs_up", ("straight", "full", "full", "full", "full")),
)


# -- OSC ---------------------------------------------------------------------
DEFAULT_OSC_HOST: str = "127.0.0.1"
DEFAULT_OSC_PORT: int = 7400
OSC_ROOT: str = "/vision"


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration resolvable from CLI flags in main.py."""

    camera_index: int = 0
    fps_target: float = DEFAULT_CAPTURE_FPS
    model_complexity: int = HOLISTIC_MODEL_COMPLEXITY
    osc_host: str = DEFAULT_OSC_HOST
    osc_port: int = DEFAULT_OSC_PORT
    osc_enabled: bool = True
    websocket_port: int | None = None
    viewer_port: int | None = None
    print_signals: bool = False
    show_preview: bool = False
    classifier_model_path: str | None = None
    flip_horizontal: bool = True  # mirror like a selfie — usually what TD wants
    log_timings: bool = False
    disable_face: bool = False  # skip face mesh for lower latency
