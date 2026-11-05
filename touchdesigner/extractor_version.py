"""Landmark extractor version lock — TouchDesigner runtime copy.

Intentionally mirrors:
  * src/sign/extractor-version.ts
  * sign_classifier_service/extractor_version.py

Bump in lockstep whenever the MediaPipe version changes OR the landmark
subset we emit changes. Used by `classifier_adapter.py` to refuse serving a
model that was trained on a different extraction regime.
"""

from __future__ import annotations

# The landmark subset, shoulder-centric normalization math, and feature
# layout we emit from Python's mediapipe.solutions.holistic are byte-for-byte
# identical to what the original JS extractor produced (see
# ``touchdesigner/tracking/landmarks.py`` — it's a direct port of
# ``src/webview/hooks/useSignLanguage.ts``). Because the shape of the frame
# written to the wire is identical, we intentionally stamp with the
# JS-compatible identifier that the currently trained classifier artifact
# expects. That makes the existing trained model (data/sign-models/active)
# usable in-process without a retrain.
#
# If you ever change the landmark subset (face indices, hand/pose selection,
# normalization regime), bump both the +schema-v<n> half here AND retrain
# the classifier so the extractor lock stays meaningful.
MEDIAPIPE_EXTRACTOR_VERSION = "mediapipe-holistic-js@0.5.1675471629+schema-v1"

# Alternate stamps the classifier adapter will try when serving older or
# newer trained artifacts. First match wins.
COMPATIBLE_EXTRACTOR_VERSIONS: tuple[str, ...] = (
    MEDIAPIPE_EXTRACTOR_VERSION,
    "mediapipe-holistic-js@0.5.1675471629+schema-v2",
    "mediapipe-holistic-py@0.10+schema-v2",
)

# Kept as a stable alias for older imports.
JS_MEDIAPIPE_EXTRACTOR_VERSION = "mediapipe-holistic-js@0.5.1675471629+schema-v2"
