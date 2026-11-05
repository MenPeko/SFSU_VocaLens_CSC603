# Whiteboard Reader TTS MiniApp

MVP Mentra mini app for Mentra Live glasses:

1. Press a glasses button (short press)
2. App captures a photo from camera
3. App runs OCR using Gemini
4. Extracted text is spoken through Mentra speakers

## Requirements

- Mentra Live glasses connected to MentraOS phone app
- MiniApp registered in Mentra console with:
  - `CAMERA` permission
  - `SPEAKER` permission
- Bun installed
- Gemini API key

## Setup

From `cloud/packages/apps/whiteboard-reader-tts`:

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
```

If that model is not available to your Gemini API key, the app will now automatically
discover other accessible `generateContent` models for the written text feature.

## Run

```bash
bun run dev
```

Run in background:

```bash
bun run dev-start
```

This starts both:
- the local app server on port `3333`
- a Cloudflare tunnel to that port

`dev-start` prints the current tunnel URL; use that as your Mentra app `publicUrl` and `webviewURL`.

Stop background server:

```bash
bun run dev-stop
```

Restart background server:

```bash
bun run dev-restart
```

Manual tunnel (optional; only if you don't use `dev-start`):

```bash
cloudflared tunnel --url http://localhost:3333
```

## Usage Flow

- Launch app from MentraOS phone app
- Look at whiteboard text
- Short-press glasses button
- App speaks extracted text

If no readable text is found, app says: "I could not read text from the whiteboard. Please try again."

## Notes

- OCR prompt is intentionally strict and returns plain text only.
- Long OCR outputs are trimmed to ~400 characters before speech for stable playback.
- This is designed for personal use MVP and local development only.
