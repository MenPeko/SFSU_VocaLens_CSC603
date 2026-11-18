# VocaLens

VocaLens is one local web app. A single Bun server serves the page and the API. The laptop camera is the only sign-capture source, and speech plays in the browser. It has two modes:

- **Written Text** (`whiteboard`)
  - Press and hold to ask about what the laptop camera sees. The question is transcribed with Whisper, answered with Gemini, and spoken in the browser.
- **Live Sign** (`sign`)
  - The page runs MediaPipe Holistic on the laptop camera and recognizes eight ASL signs: wait, help, yes, no, stop, again, thank you, and please.
  - Recognition uses a local classifier service (`sign_classifier_service/`). When that service is unavailable or unsure, Gemini interprets the sign instead.
  - The translation is spoken in the browser.

## Requirements

- Bun
- A Gemini API key, and an OpenAI API key for spoken questions
- Optional, for the local sign classifier: Python 3.9+ with a virtualenv at `.venv/`

No Mentra account, tunnel, or glasses session is required.

## Setup

```bash
bun install
cp .env.example .env
```

Edit `.env`:

```env
PORT=3333
NODE_ENV=development
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=your_openai_api_key
VOCALENS_COMPARE_MODEL=gpt-4o
```

`PORT` defaults to `3333` when it is unset. If `GEMINI_MODEL` is not available to your key, the app discovers another accessible `generateContent` model. `.env.example` may still list `MENTRAOS_API_KEY` and `PACKAGE_NAME`. The app does not read them.

Optional settings:

| Variable | Purpose |
| --- | --- |
| `GEMINI_SIGN_MODEL` | A different Gemini model for the sign fallback. |
| `SIGN_CLASSIFIER_URL` | Classifier endpoint. Default: `http://127.0.0.1:8001/classify-sign-sequence`. |
| `SIGN_CLASSIFIER_DISABLED` | Skip the local classifier and use Gemini only. |
| `SIGN_TRAINING_SESSION` | Save captured sign samples into an evaluation session (see `sign_classifier_service/README.md`). |

Local sign classifier dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r sign_classifier_service/requirements.txt
```

## Run

```bash
bun run dev
```

Or, without hot reload:

```bash
bun src/index.ts
```

Open [http://localhost:3333](http://localhost:3333). That one process serves the page, the API, server-sent translation events, and the MediaPipe files.

`bun run dev-start` runs the tests, then starts the sign classifier in the background when `.venv` has FastAPI and uvicorn. It also prints an ngrok URL. You do not need that URL to use the app.

## Usage

Written Text:

- Open the app and stay on Written Text.
- Press and hold, then ask about what the laptop camera sees.
- The answer is spoken in the browser.

Live Sign:

- Switch to Live Sign and allow the laptop camera.
- Sign with your hands in view. The page draws the skeleton and speaks the translation.
- The skeleton overlay is smoothed with a One Euro filter (`src/sign/one-euro.ts`) to reduce jitter. Only the drawing is smoothed. The landmarks sent to the classifier are unchanged.

Developer switches in the webview's `localStorage`:

- `sign:debug-timing` = `1` logs Holistic inference times every 60 frames.
- `sign:experiment-tracking-confidence` = a value between 0 and 1 overrides Holistic's `minTrackingConfidence`. A banner shows while it is on. Captured sequences get a different extractor stamp, so the classifier rejects them and Gemini handles them. Do not record training samples in this mode.

## Tests

```bash
bun test
.venv/bin/python -m unittest discover -s sign_classifier_service -p 'test_*.py' -t .
```

## Sign classifier evaluation

The classifier is evaluated on held-out signers and sessions, with leak checks enforced in code. Start here:

- `docs/SIGN_EVAL_PLAN.md`: method, test sets, leak checks, and the keep rule
- `docs/SIGN_RESULTS.md`: experiment results (generated)
- `reports/dataset-audit.md`: what landmark data is on disk (generated)
- `reports/baseline-evaluation.md`: the current classifiers scored on held-out clips (generated)
- `sign_classifier_service/README.md`: commands

Everything committed under `docs/` and `reports/` is aggregate numbers only. Dataset clips, landmarks, signer IDs, and model weights stay local. See `docs/DATA_LICENSES.md`.

## Other docs

- `docs/ROADMAP.md`: VocaLens course roadmap
- `docs/RECORDING_CHECKLIST.md`: how to record the owner's evaluation sessions
- `docs/DATA_LICENSES.md`: dataset licenses and what may be committed
- `docs/whiteboard-reader-tts-uml.png`: architecture diagram

## Notes

- The OCR prompt is strict and returns plain text only.
- Long OCR output is trimmed to about 400 characters before speech, for stable playback.
- This is a course project for personal use and local development.
