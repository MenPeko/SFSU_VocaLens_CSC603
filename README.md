# Whiteboard Reader TTS MiniApp (VocaLens)

A Mentra mini app for Mentra Live glasses, with a React webview on the phone. It has two modes:

- **Written Text** (`whiteboard`)
  - Short-press a glasses button. The app takes a photo, reads the text with Gemini, optionally translates it, and speaks it through the glasses.
  - **VocaLens** questions run on the same page: press and hold to ask a spoken question about what the camera sees. The question is transcribed with Whisper, answered with Gemini, and read back.
- **Live Sign** (`sign`)
  - The webview runs MediaPipe Holistic on the camera feed and recognizes eight ASL signs: wait, help, yes, no, stop, again, thank you, and please.
  - Recognition uses a local classifier service (`sign_classifier_service/`). When that service is unavailable or unsure, Gemini interprets the sign instead.

## Requirements

- Mentra Live glasses connected to the MentraOS phone app
- A MiniApp registered in the Mentra console with `CAMERA` and `SPEAKER` permissions
- Bun
- A Gemini API key, and an OpenAI API key for VocaLens speech recognition
- `ngrok` for `dev-start` (`brew install ngrok`)
- Optional, for the local sign classifier: Python 3.9+ with a virtualenv at `.venv/`

## Setup

```bash
bun install
cp .env.example .env
```

Edit `.env`:

```env
PACKAGE_NAME=com.yourname.whiteboardreader
MENTRAOS_API_KEY=your_api_key_from_console
PORT=3333
NODE_ENV=development
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=your_openai_api_key
VOCALENS_COMPARE_MODEL=gpt-4o
```

If `GEMINI_MODEL` is not available to your key, the app discovers another accessible `generateContent` model.

Optional settings:

| Variable | Purpose |
| --- | --- |
| `NGROK_DOMAIN` | A stable ngrok domain, so the tunnel URL survives restarts. |
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

To run everything in the background:

```bash
bun run dev-start
bun run dev-stop
bun run dev-restart
```

`dev-start` runs `bun test` first and refuses to start if the tests fail. It then starts:

- the sign classifier on port `8001`, if FastAPI and uvicorn are installed in `.venv` (otherwise Live Sign uses Gemini only)
- the app server on port `3333`
- an ngrok tunnel to that port

It prints the tunnel URL. Use it as the Mentra app `publicUrl` and `webviewURL`. Without `NGROK_DOMAIN` the URL changes on every restart, so you would have to update the console and relaunch the app on the glasses.

## Usage

Written Text:

- Launch the app from the MentraOS phone app.
- Look at the whiteboard and short-press a glasses button.
- The app speaks the extracted text. If it finds none, it says: "I could not read text from the whiteboard. Please try again."

Live Sign:

- Switch to Live Sign in the webview and sign in front of the camera.
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
