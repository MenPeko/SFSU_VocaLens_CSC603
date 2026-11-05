# TouchDesigner vision runtime

Standalone Python process that runs the existing project's MediaPipe
tracking pipeline outside of a browser and streams derived real-time
signals to TouchDesigner over OSC (and optionally JSON WebSocket).

**Nothing in the original codebase was modified.** This folder is a
parallel module structure that reuses the Python feature engineering from
`sign_classifier_service/` and ports the TypeScript landmark/normalization
logic from `src/webview/hooks/useSignLanguage.ts`.

## Quickstart

```bash
# 1. Create a venv and install
python3 -m venv .venv-td
source .venv-td/bin/activate
pip install -r touchdesigner/requirements.txt

# 2. Run the pipeline (default OSC target is 127.0.0.1:7400)
python -m touchdesigner.main

# Sanity-check without a TD patch open:
python -m touchdesigner.main --print --no-osc
```

Then in TouchDesigner, add an **OSC In CHOP** with port `7400`. See
`touchdesigner/td/td_setup.md` and `touchdesigner/td/OSC_ADDRESSES.md` for
the full contract.

## CLI

| Flag                        | Default          | What it does                                    |
|-----------------------------|------------------|-------------------------------------------------|
| `--camera N`                | `0`              | cv2 camera index                                |
| `--fps N`                   | `24`             | Target loop rate                                |
| `--model-complexity 0/1/2`  | `1`              | MediaPipe Holistic complexity (2 = heavy)       |
| `--osc-host HOST`           | `127.0.0.1`      | Where TD is listening                           |
| `--osc-port N`              | `7400`           | OSC UDP port                                    |
| `--no-osc`                  |                  | Disable OSC (useful with `--print`)             |
| `--websocket N`             | off              | Also broadcast JSON on `ws://127.0.0.1:N`       |
| `--print`                   | off              | Compact signals log to stdout                   |
| `--preview`                 | off              | Open a cv2 debug window with live text overlay  |
| `--classifier-model PATH`   | off              | Enable the existing trained sign classifier     |
| `--no-flip`                 | off              | Do **not** horizontally mirror the webcam       |
| `--disable-face`            | off              | Skip face mesh for lower latency                |
| `--log-timings`             | off              | Per-frame timing log                            |

## What it ships

- `capture/webcam.py` — cv2 webcam wrapper, replaces
  `navigator.mediaDevices.getUserMedia` from the browser hook.
- `tracking/holistic.py` — `mp.solutions.holistic` wrapper with the same
  options (`model_complexity`, `smooth_landmarks`, thresholds) as the JS
  hook.
- `tracking/landmarks.py` — direct port of `FACE_SUBSET_INDICES`,
  `ARM_POSE_INDICES`, and `buildNormalizedLandmarkFrame` from
  `src/webview/hooks/useSignLanguage.ts` (shoulder-centric normalization).
- `features/hands.py` — `hand_openness`, `palm_facing_score`,
  `hand_extended_count` ported verbatim from
  `sign_classifier_service/classifier.py` plus new `pinch_distance`,
  `pinch_state` (Schmitt trigger), `finger_curls`, `hand_centroid`.
- `features/face.py` — `mouth_open_score`, `distance_to_face` ports.
- `features/motion.py` — `appendRollingWindow` port + new
  `WristVelocityTracker` (finite-difference + EMA smoothing).
- `features/gestures.py` — curl-rule classifier that replaces the
  fingerpose JS dep without adding Node/JS to the runtime.
- `state/session.py` — hands-up/hands-down detector ported from the
  `captureSessionActiveRef` / `handsDownTimeoutRef` logic in the hook.
- `transport/osc.py` — python-osc sender with a flat address schema (see
  `td/OSC_ADDRESSES.md`).
- `transport/websocket.py` — optional JSON fan-out.
- `classifier_adapter.py` — in-process bridge to the existing
  `sign_classifier_service.LandmarkSequenceClassifier`. Only wired when
  `--classifier-model` is passed.
- `pipeline.py` — per-frame combinator glue.
- `main.py` — loop + CLI.

## Old → new mapping

Full table in `MIGRATION.md`. Short version:

| Old | New |
|---|---|
| `src/webview/hooks/useSignLanguage.ts` (browser + holistic) | `touchdesigner/tracking/*`, `touchdesigner/features/*`, `touchdesigner/state/session.py` |
| `sign_classifier_service/classifier.py` (feature helpers) | `touchdesigner/features/hands.py`, `touchdesigner/features/face.py` |
| `sign_classifier_service/classifier.py` (predictor) | `touchdesigner/classifier_adapter.py` (imports the original) |
| `src/sign/capture-config.ts` / `capture-settings.ts` | `touchdesigner/config.py` |
| `src/sign/sequence.ts` | `touchdesigner/features/motion.py` |
| `src/sign/core-signs.ts` | still consumed via the classifier adapter |
| `src/api/routes.ts` (HTTP transport) | dropped; replaced by `touchdesigner/transport/osc.py` |
| React UI, SSE, Mentra glasses, HLS/WebRTC adapters | dropped |

## What to expect for latency

On a 2022 M1 MacBook Air with `--model-complexity 1` and face mesh
enabled, the end-to-end capture-to-OSC latency is roughly one frame of
the camera plus 15-25 ms of MediaPipe work — well inside a 60 Hz TD
frame. Pass `--disable-face` or `--model-complexity 0` to trade quality
for extra headroom.
