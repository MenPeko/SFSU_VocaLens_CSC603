# Glasses Latency Session

## Step 1 - MediaPipe Inventory

Search command:

```sh
rg -n "new Hands|HandLandmarker|new FaceMesh|FaceLandmarker|new Pose|PoseLandmarker|new Holistic|HolisticLandmarker" src
```

Findings:

| Landmarker | File | Line | Runs on glasses path? | Runs on webcam path? | Notes |
| --- | --- | ---: | --- | --- | --- |
| `Holistic` from `@mediapipe/holistic` | `src/webview/hooks/useSignLanguage.ts` | 1611 | Yes | Yes | Initialized inside `restart()` for both `glasses-stream` and `device-camera` sources. It produces pose, face, left-hand, and right-hand landmarks from one `holistic.send(...)` call. |

No separate `Hands`, `HandLandmarker`, `FaceMesh`, `FaceLandmarker`, `Pose`, `PoseLandmarker`, or Tasks landmarker initialization was found under `src/`.

Acceptance check:

- Note file exists.
- Inventory has one initialized MediaPipe landmarker.
- The Step 1 stop condition does not apply because the initializer is `Holistic`, not only `Hands`/`HandLandmarker`.

## Step 2 - Timing Instrumentation

Added one timing wrapper around the Holistic inference call in `src/webview/hooks/useSignLanguage.ts`.

Log label:

```text
[MP_TIMING][HOLISTIC]
```

Acceptance check:

- `rg -n "MP_TIMING" src/webview` returns one match, matching the one landmarker found in Step 1.
- `bun x tsc --noEmit` passes.

Pending measurement:

- Run on glasses, sign for 10 seconds, and paste the `[MP_TIMING]` console output.

## Interruption - Managed Stream Keep-Alive Timeout

While attempting to gather glasses measurements, the app surfaced:

```text
Stream timed out waiting for keep-alive
```

This message originates from the MentraOS managed RTMP stream lifecycle, before MediaPipe timing can be measured. The webview button labeled `Connect Feed` previously only restarted the local video/MediaPipe hook, which could not recover an upstream managed stream in `error`.

Applied recovery-path change:

- `src/webview/components/SignLanguageCapture.tsx` now detects an inactive managed glasses stream and uses the sign-camera settings endpoint to stop/start the backend stream before retrying playback.

Verification:

- `bun x tsc --noEmit` passes.
- `bun test` passes: 147 tests.

Follow-up fix after timeout reproduced:

- `cloud/packages/cloud/src/services/streaming/ManagedStreamingExtension.ts` no longer starts managed-stream keep-alives immediately at stream creation time. Keep-alives now start when the glasses report `streaming`/`active`, or when Cloudflare polling verifies that the stream is live.
- The managed lifecycle sends one keep-alive immediately on activation instead of waiting for the first 15-second interval.
- `cloud/packages/cloud/src/services/streaming/StreamLifecycleController.ts` now reports whether active state changed and exposes `sendKeepAliveNow()`.

Additional verification:

- `bun test src/services/streaming/__tests__/StreamLifecycleController.test.ts` passes in `cloud/packages/cloud`.
- `bun x tsc --noEmit` passes in `cloud/packages/apps/whiteboard-reader-tts`.
- Targeted webview tests pass: `bun test src/webview/hooks/__tests__/useSignLanguage.test.ts src/webview/components/__tests__/confidence-display.test.ts`.
- Cloud package typecheck is blocked by an existing workspace resolution issue: `src/services/integrations/linear.service.ts` cannot resolve `@mentra/utils`.

Follow-up after stream still reached playback and then timed out:

- `mobile/modules/core/android/src/main/java/com/mentra/core/sgcs/MentraLive.java` now sends `keep_rtmp_stream_alive` to glasses with the wake flag, matching `start_rtmp_stream` and `stop_rtmp_stream`. The previous non-wake send could allow keep-alive commands to be dropped or delayed while the RTMP stream kept running, which fits "connected at 720p" followed by a cloud keep-alive timeout.

Verification:

- Code inspection confirms `sendRtmpKeepAlive()` now calls `sendJson(json, true)`.
- No focused Android compile was run because this checkout does not include a Gradle wrapper under `mobile/`.

Follow-up for iPhone testing:

- Latest app-server logs show the managed stream reaches `active`, then receives `status: "error"` with `message: "Stream timed out waiting for keep-alive"` about 41 seconds later.
- This confirms the webview/app server is receiving cloud's terminal managed-stream error; the missing ACK is still between cloud -> phone -> glasses -> phone -> cloud.
- `mobile/modules/core/ios/Source/sgcs/MentraLive.swift` now sends `keep_rtmp_stream_alive` with `wakeUp: true`, matching the iOS `start_rtmp_stream` and `stop_rtmp_stream` paths.
- iOS also logs `Forwarding RTMP keep-alive ACK from glasses` before emitting `keep_alive_ack` to React Native.
