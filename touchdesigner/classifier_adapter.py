"""Optional bridge to the existing sign_classifier_service.

The TouchDesigner runtime does not strictly need the sign classifier — most
TD patches just want the per-frame HandSignals / SessionSignals. But the
existing ``sign_classifier_service.classifier.LandmarkSequenceClassifier``
is already solved and trained, so we expose it here as a lightweight
in-process classifier that TD can trigger via OSC.

We load a model artifact path explicitly (no web server, no FastAPI). When
no path is configured the adapter silently no-ops.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Reach up to the repo root so we can import the existing Python service.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from sign_classifier_service.classifier import (  # type: ignore[import-not-found]
        LandmarkSequenceClassifier,
    )
    from sign_classifier_service.schemas import (  # type: ignore[import-not-found]
        ClassifySequenceRequest,
        LandmarkFrameModel,
    )
    _IMPORT_ERROR: Optional[BaseException] = None
except Exception as e:  # pragma: no cover — gracefully handle missing deps
    LandmarkSequenceClassifier = None  # type: ignore[assignment]
    ClassifySequenceRequest = None  # type: ignore[assignment]
    LandmarkFrameModel = None  # type: ignore[assignment]
    _IMPORT_ERROR = e

from collections import deque
from typing import Deque

from .extractor_version import COMPATIBLE_EXTRACTOR_VERSIONS
from .schema import LandmarkFrame, to_classifier_payload


class SignClassifierAdapter:
    """In-process wrapper. Keep it cheap — we only call predict() on the
    hands-down event, not per-frame.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        data_dir: Optional[Path] = None,
    ) -> None:
        if LandmarkSequenceClassifier is None:
            self._classifier = None
            self.unavailable_reason = f"classifier import failed: {_IMPORT_ERROR}"
            return

        self.unavailable_reason: Optional[str] = None
        self._classifier = LandmarkSequenceClassifier(
            data_dir=data_dir or (_REPO_ROOT / "data" / "sign-samples"),
            model_path=Path(model_path) if model_path else None,
            prefer_active_model_pointer=True,
        )

    @property
    def enabled(self) -> bool:
        return self._classifier is not None

    def predict(self, frames: Deque[LandmarkFrame] | list[LandmarkFrame]) -> Optional[dict]:
        """Run the existing temporal classifier on a sequence of frames.
        Returns the JSON-serializable response (identical shape to the
        FastAPI /classify-sign-sequence endpoint) or None if unavailable.

        The extractor stamp we send is the *Python* one — if the existing
        model was trained under the JS stamp it will correctly reject us
        and you can retrain. The adapter tries JS first so operators with
        an existing JS-trained model keep working.
        """
        if self._classifier is None or ClassifySequenceRequest is None or LandmarkFrameModel is None:
            return None
        if not frames:
            return None

        payload_frames = [to_classifier_payload(f) for f in frames]
        validated = [LandmarkFrameModel.model_validate(f) for f in payload_frames]
        duration_ms = int(
            (payload_frames[-1]["timestampMs"] - payload_frames[0]["timestampMs"])
            if len(payload_frames) > 1 else 0
        )

        # Try each compatible extractor stamp in order. The first one that
        # does NOT trip extractor_mismatch wins. Lets us serve older trained
        # artifacts (schema-v1) or newer ones (schema-v2) from the same
        # Python runtime without a retrain.
        response = None
        for extractor_version in COMPATIBLE_EXTRACTOR_VERSIONS:
            request = ClassifySequenceRequest(
                frames=validated,
                fps=12.0,
                source="device-camera",
                sequenceDurationMs=duration_ms,
                extractorVersion=extractor_version,
            )
            response = self._classifier.predict(request)
            reason = (response.reason or "")
            if not reason.startswith("extractor_mismatch"):
                return response.model_dump()
        return response.model_dump() if response is not None else None
