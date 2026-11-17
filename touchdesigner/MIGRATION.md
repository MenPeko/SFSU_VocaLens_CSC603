# Migration notes: VocaLens → TouchDesigner

## A. Summary

The original project is a dual-runtime MediaPipe system (TypeScript browser
+ Python FastAPI classifier) built around the Mentra smart-glasses and
Gemini cloud calls. For TouchDesigner we want:

- a single local process (no browser, no cloud, no Mentra)
- low latency camera-to-TD
- stable derived signals (pinch, openness, positions, gestures, sessions)

The migration keeps every piece of solved logic that is still useful and
drops only the UI/cloud/Mentra plumbing.

**Chosen approach:** Python process with `mediapipe` + `opencv-python` +
`python-osc`, talking to TouchDesigner's built-in OSC In CHOP. This is
conventional TD, and it happens to preserve ~90% of the existing Python
feature engineering verbatim.

## B. Reusable code inventory

### B.1 Reusable as-is (copied, not rewritten)

| Original location | Used in                                   | Status |
|-------------------|-------------------------------------------|--------|
| `sign_classifier_service/classifier.py` `_distance` | `touchdesigner/features/hands.py`, `face.py` | identical |
| `sign_classifier_service/classifier.py` `_hand_extended_count` | `touchdesigner/features/hands.py::hand_extended_count` | identical |
| `sign_classifier_service/classifier.py` `_hand_openness` | `touchdesigner/features/hands.py::hand_openness` | identical |
| `sign_classifier_service/classifier.py` `_palm_facing_score` | `touchdesigner/features/hands.py::palm_facing_score` | identical |
| `sign_classifier_service/classifier.py` `_mouth_open_score` | `touchdesigner/features/face.py::mouth_open_score` | identical |
| `sign_classifier_service/classifier.py` `_distance_to_face` | `touchdesigner/features/face.py::distance_to_face` | identical |
| `sign_classifier_service/classifier.py` `LandmarkSequenceClassifier.predict` + full file | `touchdesigner/classifier_adapter.py` | imported live, not copied |
| `sign_classifier_service/schemas.py` | (used via `classifier_adapter.py`) | imported live |
| `sign_classifier_service/extractor_version.py` | `touchdesigner/extractor_version.py` | mirrored (Python MediaPipe uses a different package-version half) |

### B.2 Reusable with light refactor (ported)

| Original location (TS) | Ported to | Notes |
|------------------------|-----------|-------|
| `src/webview/hooks/useSignLanguage.ts::FACE_SUBSET_INDICES` | `touchdesigner/tracking/landmarks.py::FACE_SUBSET_INDICES` | Index-identical; do not reorder |
| `src/webview/hooks/useSignLanguage.ts::ARM_POSE_INDICES` | `touchdesigner/tracking/landmarks.py::ARM_POSE_INDICES` | Uses `mp.solutions.holistic.PoseLandmark` enum |
| `src/webview/hooks/useSignLanguage.ts::hasUsableHandLandmarks` | `touchdesigner/tracking/landmarks.py::hand_landmarks_usable` | Same predicate |
| `src/webview/hooks/useSignLanguage.ts::pickLandmarks`, `buildFaceSubset` | `touchdesigner/tracking/landmarks.py::pick_landmarks`, `build_face_subset` | Same logic |
| `src/webview/hooks/useSignLanguage.ts::normalizePoint`, `normalizePoints`, `buildNormalizedLandmarkFrame` | `touchdesigner/tracking/landmarks.py::_normalize_point`, `_normalize_points`, `build_normalized_landmark_frame` | Byte-identical shoulder-centric math; preserves compatibility with existing trained classifier |
| `src/webview/hooks/useSignLanguage.ts::estimateGestureHints` (fingerpose) | `touchdesigner/features/gestures.py::classify_hand` + `touchdesigner/features/hands.py::finger_curls` | Rewritten as angle-threshold + rule table to drop the JS fingerpose dependency; same gestures (open_palm / fist / pointing / victory / thumbs_up) |
| `src/webview/hooks/useSignLanguage.ts` `captureSessionActive`+`handsDownTimeout` state | `touchdesigner/state/session.py::HandsSessionDetector` | Lifted out of React refs into a plain stateful class |
| `src/sign/sequence.ts::appendRollingWindow`, `computeSequenceDurationMs` | `touchdesigner/features/motion.py::append_rolling_window`, `compute_sequence_duration_ms` | Python uses `collections.deque(maxlen=N)` |
| `src/sign/capture-config.ts` constants | `touchdesigner/config.py` | Grid-related constants dropped (TD doesn't want JPEG grids) |
| `src/sign/capture-settings.ts::SignCaptureSettings` | `touchdesigner/config.py::HOLISTIC_MODEL_COMPLEXITY` | TD doesn't need runtime bootstrap — static config |

### B.3 Logic-only (rewritten for TouchDesigner)

| Original | Replacement | Why rewrite |
|----------|-------------|-------------|
| `useSignLanguage.ts` MediaPipe Holistic init/`onResults`/`send` | `touchdesigner/tracking/holistic.py` | JS API → Python API; removes WASM locate-file, canvas overlays, drift watchdogs |
| `useSignLanguage.ts` `navigator.mediaDevices.getUserMedia` + HLS/WebRTC attach | `touchdesigner/capture/webcam.py` | TD uses the local webcam directly |
| `useSignLanguage.ts` skeleton/grid canvas rendering | dropped (TD renders visuals) | optional debug: `--preview` |
| `src/services/sign-classifier-client.ts` HTTP client | `touchdesigner/classifier_adapter.py` in-process call | removes network hop |
| `src/api/routes.ts`, `src/api/sse-stream.ts` | `touchdesigner/transport/osc.py` (+ optional WebSocket) | OSC is TD-native |

### B.4 Not ported (intentionally)

| Directory/file | Reason |
|---|---|
| `src/webview/App.tsx`, `SignLanguageCapture.tsx`, `components/ui/**`, `frontend.tsx`, `globals.css`, `index.css`, `index.html` | React UI — TD owns the UI |
| `src/webview/hooks/useTranslationStream.ts`, `src/api/sse-stream.ts` | SSE fanout — replaced by OSC |
| `src/app/CaptionsApp.ts`, `active-session.ts`, `sign-camera-control.ts`, `managed-stream-quality.ts`, `glasses-camera-stream.ts` | Mentra AppServer / glasses lifecycle |
| `src/services/gemini-ocr.ts`, `gemini-translate.ts`, `gemini-sign-interpreter.ts`, `sign-translation-resolver.ts` | Cloud AI calls — irrelevant to a local TD runtime |
| `src/services/sign-training-store.ts` | Dataset capture — use the original app if you want to retrain |
| `public/mediapipe/**` | Browser WASM assets — Python mediapipe bundles its own models |
| `sign_classifier_service/app.py` (FastAPI), `export_dataset.py`, `train.py`, `evaluate.py`, `report_results.py` | Still useful, still work — left in place for offline training |
| `scripts/*.sh`, `ARCHITECTURE.md`, `DEVELOPMENT_ROADMAP.md`, notes, roadmap docs | Old app workflow |

## C. Signal contract (what TD gets)

See `touchdesigner/td/OSC_ADDRESSES.md` for the complete table. The short
list:

- **Hands:** `present`, `openness`, `palm_facing`, `extended_count`,
  `pinch_distance`, `pinch_state`, `wrist_(x,y,z)`, `raw_wrist_(x,y)`,
  `wrist_v(x,y)`, `gesture`, `gesture_score`.
- **Pose:** `shoulder_center_(x,y)`, `shoulder_width`.
- **Face:** `present`, `mouth_open`, `nose_(x,y)`.
- **Session:** `active`, `duration_ms`, `event/started`, `event/ended`.
- **Classifier (opt-in):** `status`, `label`, `confidence`.

## D. What stays OUTSIDE TouchDesigner

- MediaPipe inference (`mp.solutions.holistic`) — too heavy for TD's
  Python scripting environment.
- Webcam capture + any future glasses/remote stream attach.
- All feature engineering and state detection.
- The optional sign-language classifier predict call.

## E. What TouchDesigner consumes

- The OSC stream only. (Optionally the JSON WebSocket if you want the raw
  normalized landmark frames — currently the WebSocket mirrors the OSC
  signals, but you can extend `transport/websocket.py` to push the full
  `LandmarkFrame` via `schema.to_classifier_payload` if a TD patch wants
  raw skeletons.)

## F. Risks and notes

1. **Mediapipe-python vs mediapipe-js version parity.** The JS package
   and Python package tag versions independently. The landmark *indices*
   are stable across versions (both follow the standard 33-pose / 21-hand
   / 468-face schema), which is why the ports work at all. The shoulder
   normalization math is identical in both runtimes. Sample-level
   classifier compatibility is handled by the extractor version stamp:
   the adapter tries both the Python and JS stamps so an existing trained
   artifact continues to serve.
2. **Camera mirroring.** The JS hook ran with `selfieMode: false` but the
   camera source was already in selfie orientation. The Python path
   flips in `capture/webcam.py` by default; pass `--no-flip` if your
   scene expects the raw (non-mirrored) camera.
3. **Fingerpose rewrite.** The gesture classifier is now a curl-angle
   rule table rather than fingerpose's probabilistic score. Behavior is
   similar but not identical — if you ported an old TD patch that
   depended on specific fingerpose scores you may need to re-tune the
   `DEFAULT_MIN_SCORE` or add a new `GestureRule`. Adding gestures is a
   two-line change in `config.py::GESTURE_RULES`.
4. **Pinch detection is new.** Previously the project used "hand
   openness" as a proxy. `pinch_distance`/`pinch_state` are new
   additions shaped for TD interaction patterns. Tune
   `PINCH_CLOSE_THRESHOLD` / `PINCH_OPEN_THRESHOLD` in `config.py` if
   your subject-to-camera distance is unusual.
5. **Face landmarks cost ~30% of the Holistic runtime.** Pass
   `--disable-face` if you don't need `mouth_open`.
6. **No remote stream support.** The browser-side HLS/WebRTC drift
   watchdogs were not ported. If you later need to drive TD from the
   Mentra glasses, add a new adapter under `capture/` that yields
   BGR numpy frames — the rest of the pipeline stays unchanged.

## G. Things that could not be reused

- The grid-JPEG composition (`createFrameCanvas`, `redrawFrameGridCanvas`,
  `createSkeletonFrameCanvas`): only useful for shipping a single image
  to Gemini — not needed for TD.
- The HLS/WebRTC drift snap logic: specific to browser video playback.
- Fingerpose's probabilistic scoring: replaced by a simpler curl-rule
  classifier. The mapping to the same gesture names is preserved so
  downstream consumers see the same labels.
