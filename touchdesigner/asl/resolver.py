"""ASL translation resolver: classifier first, Gemini fallback.

Port of ``src/services/sign-translation-resolver.ts``. Same behavior:

  * If the local classifier returns ``recognized`` with high confidence,
    use its canonical English for the recognized sign (CORE_SIGN_CANONICAL_TEXT).
  * Otherwise fall back to Gemini on the raw sign-grid image.
  * If neither produces a readable English, mark it ``no_sign``.

All failure modes degrade gracefully so one dead path doesn't block the
other. The caller receives a single ``TranslationOutcome`` either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..classifier_adapter import SignClassifierAdapter
from .gemini import SignInterpretationResult, interpret_sign_grid, translate_text


# Mirrors CORE_SIGN_CANONICAL_TEXT in src/sign/core-signs.ts.
CORE_SIGN_CANONICAL_TEXT: dict[str, str] = {
    "wait": "Wait",
    "help": "Help",
    "yes": "Yes",
    "no": "No",
    "stop": "Stop",
    "again": "Again",
    "thank_you": "Thank you",
    "please": "Please",
}


@dataclass
class TranslationOutcome:
    status: str                         # recognized | no_sign | error
    english_text: str                   # "" when no_sign
    source: str                         # classifier | gemini | none
    recognized_label: Optional[str]     # core sign id if classifier won
    confidence: float                   # 0..1 from classifier (0 for gemini)
    classifier_status: Optional[str]    # recognized | uncertain | no_sign | unavailable
    classifier_alternatives: list[dict]
    fallback_used: bool
    model: str
    gemini_raw: Optional[str]
    error: Optional[str] = None
    translated_text: Optional[str] = None
    detected_language: Optional[str] = None
    target_language: Optional[str] = None


def resolve(
    *,
    classifier: SignClassifierAdapter,
    landmark_frames,
    base64_grid: Optional[str],
    gesture_hints: Optional[list[dict]] = None,
    target_language: str = "English",
) -> TranslationOutcome:
    """Run the classifier → Gemini fallback chain and return a single
    structured outcome. Safe to call from a worker thread.
    """
    classifier_status: Optional[str] = "skipped"
    classifier_alternatives: list[dict] = []
    classifier_model = "local-knn-sequence-v1"
    classifier_error: Optional[str] = None
    fallback_used = False

    # ---- 1. local classifier ------------------------------------------------
    if classifier.enabled and landmark_frames:
        try:
            result = classifier.predict(landmark_frames)
        except Exception as e:  # defensive — adapter should not raise
            classifier_status = "unavailable"
            classifier_error = str(e)
            result = None

        if result is None:
            classifier_status = "unavailable"
        else:
            classifier_status = str(result.get("status") or "no_sign")
            classifier_alternatives = list(result.get("alternatives") or [])
            classifier_model = str(result.get("model") or classifier_model)
            label = result.get("label")
            confidence = float(result.get("confidence") or 0.0)

            if classifier_status == "recognized" and isinstance(label, str) and label in CORE_SIGN_CANONICAL_TEXT:
                english = CORE_SIGN_CANONICAL_TEXT[label]
                translated = _maybe_translate(english, target_language)
                return TranslationOutcome(
                    status="recognized",
                    english_text=english,
                    source="classifier",
                    recognized_label=label,
                    confidence=confidence,
                    classifier_status=classifier_status,
                    classifier_alternatives=classifier_alternatives,
                    fallback_used=False,
                    model=classifier_model,
                    gemini_raw=None,
                    translated_text=translated["text"],
                    detected_language=translated["detected"],
                    target_language=translated["target"],
                )

            fallback_used = True
    else:
        classifier_status = "unavailable" if not classifier.enabled else "skipped"

    # ---- 2. Gemini fallback ------------------------------------------------
    gemini_result: Optional[SignInterpretationResult] = None
    gemini_error: Optional[str] = None
    gemini_model = "gemini-unavailable"
    if base64_grid:
        try:
            gemini_result = interpret_sign_grid(
                base64_grid=base64_grid,
                gesture_hints=gesture_hints or [],
            )
        except Exception as e:
            gemini_error = str(e)

    if gemini_result is None:
        return TranslationOutcome(
            status="no_sign",
            english_text="",
            source="none",
            recognized_label=None,
            confidence=0.0,
            classifier_status=classifier_status,
            classifier_alternatives=classifier_alternatives,
            fallback_used=fallback_used,
            model=classifier_model if classifier_status != "unavailable" else gemini_model,
            gemini_raw=None,
            error=gemini_error or classifier_error,
        )

    english = gemini_result.english_text.strip()
    if not english or english == "NO_SIGN_FOUND":
        return TranslationOutcome(
            status="no_sign",
            english_text="",
            source="gemini",
            recognized_label=None,
            confidence=0.0,
            classifier_status=classifier_status,
            classifier_alternatives=classifier_alternatives,
            fallback_used=fallback_used,
            model=gemini_result.model,
            gemini_raw=gemini_result.raw_response,
        )

    translated = _maybe_translate(english, target_language)
    return TranslationOutcome(
        status="recognized",
        english_text=english,
        source="gemini",
        recognized_label=None,
        confidence=0.0,
        classifier_status=classifier_status,
        classifier_alternatives=classifier_alternatives,
        fallback_used=fallback_used,
        model=gemini_result.model,
        gemini_raw=gemini_result.raw_response,
        translated_text=translated["text"],
        detected_language=translated["detected"],
        target_language=translated["target"],
    )


def _maybe_translate(english: str, target_language: str) -> dict:
    """Run optional target-language translation. Returns a dict always
    populated so callers don't need to branch."""
    if target_language.strip().lower() in ("", "english", "en"):
        return {"text": english, "detected": "English", "target": "English"}
    try:
        tr = translate_text(english, target_language)
    except Exception:
        tr = None
    if tr is None:
        return {"text": english, "detected": "unknown", "target": target_language}
    return {"text": tr.translated_text, "detected": tr.detected_language, "target": tr.target_language}
