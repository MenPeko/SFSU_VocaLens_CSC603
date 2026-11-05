"""Python port of ``src/services/gemini-sign-interpreter.ts`` and
``src/services/gemini-translate.ts``.

Only the ASL-interpretation and translate paths are ported — the OCR
fallback-model-discovery logic from gemini-ocr.ts is omitted because it
only mattered for cold-start Gemini 1.x model retirement.

Reads ``GEMINI_API_KEY`` from the environment. If the key is missing every
call becomes a no-op that returns ``None`` — the resolver treats that as
"Gemini unavailable" and keeps the pipeline functional on classifier-only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional


TRANSIENT_STATUS_CODES: tuple[int, ...] = (429, 500, 503)
MAX_RETRIES: int = 3
RETRY_BASE_DELAY_S: float = 1.0
DEFAULT_TIMEOUT_S: float = 12.0


# ---- prompts (ported verbatim so interpretation behavior matches) --------

SIGN_INTERPRETER_SYSTEM_PROMPT = " ".join([
    "You are an expert American Sign Language interpreter.",
    "The image input is one or two chronological grids of sign-language frames.",
    "If a second image is provided, it is a skeleton-rendered view of the same sequence and should be used as structural reference only.",
    "Read the grid left-to-right and top-to-bottom as one continuous ASL sequence.",
    "Translate the meaning into concise, natural English.",
    "Prioritize ASL lexical meaning over generic gesture or command interpretation.",
    "When both hands are involved, preserve the two-handed sign rather than collapsing it into a one-hand command.",
    "Do not guess from a single static palm pose alone.",
    "Do not describe the signer, the camera, or the grid layout.",
    "If gesture hints are provided, treat them as noisy secondary hints rather than ground truth.",
    "Return plain English text only.",
    "If the signing is ambiguous, unreadable, or no clear sign is present, return exactly: NO_SIGN_FOUND",
])


def _build_interpreter_prompt(grid_cols: int, grid_rows: int, hints: list[dict], has_skeleton: bool) -> str:
    hint_text = (
        ", ".join(f"{h['name']} ({float(h.get('score', 0)):.1f})" for h in hints)
        if hints else "none"
    )
    return " ".join([
        f"This image is a {grid_cols}x{grid_rows} chronological frame grid.",
        "Interpret it as one continuous ASL phrase or short sentence.",
        "Produce the best natural English translation.",
        "Use multi-hand symmetry and movement across frames to disambiguate similar open-palm poses.",
        (
            "A second skeleton-rendered grid of the same frames is also provided to clarify handshape and motion."
            if has_skeleton else "Only the raw frame grid is provided."
        ),
        f"Auxiliary handshape hints: {hint_text}.",
    ])


def _normalize_interpretation_response(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    for marker in ("```text", "```json", "```"):
        cleaned = cleaned.replace(marker, "")
    for prefix in ("English:", "Translation:"):
        if cleaned.lstrip().lower().startswith(prefix.lower()):
            cleaned = cleaned.lstrip()[len(prefix):]
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    # Gemini sometimes wraps the answer in a JSON object; unwrap if so.
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            value = payload.get("englishText") or payload.get("translation") or ""
            return str(value).strip()
    except Exception:
        pass
    return cleaned


# ---- low-level HTTP helpers ---------------------------------------------

def _build_url(api_key: str, model: str) -> str:
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model, safe='')}"
        f":generateContent?key={urllib.parse.quote(api_key, safe='')}"
    )


def _parse_response(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    return "\n".join(str(p.get("text") or "") for p in parts).strip()


def _deduplicate_models(primary: str) -> list[str]:
    fallbacks = [primary, "gemini-2.5-flash", "gemini-2.0-flash"]
    seen: set[str] = set()
    out: list[str] = []
    for m in fallbacks:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _post(url: str, body: dict, timeout_s: float) -> tuple[int, dict | str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.getcode(), json.loads(raw)
            except json.JSONDecodeError:
                return resp.getcode(), raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


# ---- public API ---------------------------------------------------------

@dataclass
class SignInterpretationResult:
    english_text: str
    raw_response: str
    model: str


def interpret_sign_grid(
    base64_grid: str,
    gesture_hints: Optional[list[dict]] = None,
    grid_cols: int = 4,
    grid_rows: int = 4,
    mime_type: str = "image/jpeg",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Optional[SignInterpretationResult]:
    """Ask Gemini to translate a sign-grid JPEG into English.

    Returns ``None`` if ``GEMINI_API_KEY`` is unset or the grid is empty.
    Raises ``RuntimeError`` on non-transient API errors.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    if not base64_grid.strip():
        return None

    configured_model = os.environ.get("GEMINI_SIGN_MODEL", "gemini-1.5-flash")
    hints = gesture_hints or []
    prompt = _build_interpreter_prompt(grid_cols, grid_rows, hints, has_skeleton=False)
    body = {
        "system_instruction": {"parts": [{"text": SIGN_INTERPRETER_SYSTEM_PROMPT}]},
        "contents": [{
            "parts": [
                {"text": prompt},
                {"text": "Image 1 is the raw chronological RGB frame grid."},
                {"inline_data": {"mime_type": mime_type, "data": base64_grid}},
            ],
        }],
    }

    last_error = "Sign interpretation failed"
    last_status = 0
    final_model = configured_model

    for model in _deduplicate_models(configured_model):
        final_model = model
        url = _build_url(api_key, model)
        for attempt in range(MAX_RETRIES):
            status, payload = _post(url, body, timeout_s)
            last_status = status
            if 200 <= status < 300 and isinstance(payload, dict):
                raw = _parse_response(payload)
                english = _normalize_interpretation_response(raw)
                return SignInterpretationResult(english_text=english, raw_response=raw, model=model)

            if status in TRANSIENT_STATUS_CODES and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY_S * (attempt + 1))
                continue

            last_error = f"Gemini ASL interpret failed (HTTP {status})"
            if isinstance(payload, dict):
                err = ((payload.get("error") or {}).get("message")) or ""
                if err:
                    last_error = f"{last_error}: {err}"
            elif isinstance(payload, str) and payload.strip():
                last_error = f"{last_error}: {payload.strip()[:200]}"
            break

        if last_status == 404:
            # The model wasn't available for this key — try next fallback model.
            continue
        break

    raise RuntimeError(last_error)


# ---- translation (optional, for target_language != English) --------------

_TRANSLATE_PROMPT_TEMPLATE = (
    "You are a language detection and translation assistant.\n\n"
    'Given the following text, respond ONLY with a JSON object (no markdown, no code fences) in this exact format:\n'
    '{"detectedLanguage":"<language name in English>","translatedText":"<translated text>"}\n\n'
    "If the text is already in TARGET_LANGUAGE, set translatedText to the original text unchanged.\n"
    "If no text is provided or it's unreadable, respond with:\n"
    '{"detectedLanguage":"unknown","translatedText":""}\n\n'
    "Target language: TARGET_LANGUAGE\n\n"
    "Text to translate:\n\"\"\"\nTEXT_TO_TRANSLATE\n\"\"\""
)


@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    detected_language: str
    target_language: str


def translate_text(text: str, target_language: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> Optional[TranslationResult]:
    if target_language.strip().lower() in ("", "english", "en"):
        return TranslationResult(
            original_text=text,
            translated_text=text,
            detected_language="English",
            target_language=target_language or "English",
        )

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or not text.strip():
        return None

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = (
        _TRANSLATE_PROMPT_TEMPLATE
        .replace("TARGET_LANGUAGE", target_language)
        .replace("TARGET_LANGUAGE", target_language)
        .replace("TEXT_TO_TRANSLATE", text)
    )
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    url = _build_url(api_key, model)

    for attempt in range(MAX_RETRIES):
        status, payload = _post(url, body, timeout_s)
        if 200 <= status < 300 and isinstance(payload, dict):
            raw = _parse_response(payload)
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            detected = "unknown"
            translated = ""
            try:
                parsed = json.loads(cleaned) if cleaned else {}
                if isinstance(parsed, dict):
                    detected = str(parsed.get("detectedLanguage") or "unknown")
                    translated = str(parsed.get("translatedText") or "")
            except Exception:
                translated = cleaned
            return TranslationResult(
                original_text=text,
                translated_text=translated or text,
                detected_language=detected,
                target_language=target_language,
            )
        if status in TRANSIENT_STATUS_CODES and attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BASE_DELAY_S * (attempt + 1))
            continue
        break

    return None
