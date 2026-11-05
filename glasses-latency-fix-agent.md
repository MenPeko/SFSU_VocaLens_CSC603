# Task: Fix Glasses Skeleton Tracking Lag

## Context

**Project:** GolarArt / whiteboard-reader-tts (live ASL translation on Mentra glasses)
**Stack:** React webview, Bun, MediaPipe (currently legacy `@mediapipe/hands`), FastAPI classifier service, Cloudflare dev tunnel
**Symptom:** Skeleton overlay lags behind actual motion on Mentra glasses webview. Works fine on MacBook webcam.
**Hypothesis:** Compute budget overrun on constrained webview. Running multiple landmarkers exceeds ~33ms per-frame budget at 30fps.
**Goal:** Total per-frame MediaPipe inference under 33ms on the glasses (or under 66ms if running at 15fps).
**Constraint:** Do not break the existing 8-sign classifier (`wait`, `help`, `yes`, `no`, `stop`, `again`, `thank_you`, `please`). Classifier consumes hand landmarks only via `vectorize_sequence` (344-dim features, 24-frame window).

## Operating rules

1. **Execute steps sequentially. Do not skip.**
2. **Run the acceptance check at the end of each step before proceeding.**
3. **If a step's acceptance check passes the overall goal (total inference < 33ms on glasses AND classifier still works), STOP and proceed directly to Step 7 (cleanup).**
4. **Make one change per commit. Do not batch changes.**
5. **If an acceptance check fails unexpectedly, stop and report. Do not attempt unlisted fixes.**
6. **All code edits must preserve the existing landmark output schema consumed by `vectorize_sequence` in `sign_classifier_service/classifier.py`. If the schema would change, stop and report.**

## Files in scope

Primary:
- `src/webview/hooks/useSignLanguage.ts` — MediaPipe integration
- `src/webview/components/SignLanguageCapture.tsx` — capture UI

Possibly involved (check before editing):
- Any other hook under `src/webview/hooks/` that initializes MediaPipe
- `public/` — MediaPipe WASM and model assets
- `src/services/glasses-camera-stream.ts` — glasses video delivery

Out of scope (do not modify):
- `sign_classifier_service/` — Python classifier
- `src/services/sign-translation-resolver.ts` — routing logic
- `src/sign/core-signs.ts` — label vocabulary

---

## Step 1 — Inventory what's running

**Action:**
1. Read `src/webview/hooks/useSignLanguage.ts` in full.
2. Search the codebase for other landmarker initializations:
   - `grep -rn "new Hands\|HandLandmarker\|new FaceMesh\|FaceLandmarker\|new Pose\|PoseLandmarker\|new Holistic\|HolisticLandmarker" src/`
3. Record findings in a working note (create `notes/glasses-latency-session.md`).

**Expected output in note:**
- List of every MediaPipe landmarker currently initialized
- File path and line number for each
- Whether it runs on the glasses path, the webcam path, or both

**Acceptance check:**
- Note file exists with a complete inventory.
- If only `Hands` or `HandLandmarker` is initialized anywhere, **stop and report** — the compute-budget hypothesis is wrong and a different diagnosis is needed.

---

## Step 2 — Add timing instrumentation

**Action:**
1. For each landmarker identified in Step 1, wrap its inference call with `performance.now()` timing.
2. Example pattern for the legacy Hands call:

```javascript
const __t0 = performance.now();
const results = await hands.send({ image: videoElement });
console.log(`[MP_TIMING][HANDS] ${(performance.now() - __t0).toFixed(1)}ms`);
```

3. Use a consistent log prefix `[MP_TIMING]` so logs can be grep'd from console output.
4. Label each landmarker distinctly: `[MP_TIMING][HANDS]`, `[MP_TIMING][FACE]`, `[MP_TIMING][POSE]`.

**Acceptance check:**
- `grep -rn "MP_TIMING" src/webview/` returns one match per landmarker found in Step 1.
- Code compiles (`bun run build` or equivalent succeeds).
- Report back to user: "Instrumentation added. Please run on glasses, sign for 10 seconds, and paste the `[MP_TIMING]` console output."

**Stop here and wait for user to provide measured numbers before proceeding to Step 3.**

---

## Step 3 — Disable non-hand landmarkers on the glasses path

**Precondition:** User has provided Step 2 measurements AND at least one of Face or Pose is running AND the sum of inference times exceeds 33ms.

**If precondition not met:** Skip to Step 8.

**Action:**
1. Determine how the code identifies a glasses context vs webcam context. Search for existing detection logic:
   - `grep -rn "isGlasses\|mentra\|glasses" src/webview/`
2. If no detector exists, add one using URL param. Modify the webview entrypoint to read `?mode=glasses` and expose an `isGlasses` boolean via context or prop.
3. In every file that initializes a Face or Pose landmarker, guard the initialization:

```javascript
if (!isGlasses) {
  // existing face/pose landmarker setup
}
```

4. Ensure Hands initialization is NOT guarded — it must run in both modes.
5. Ensure any rendering code that draws face/pose overlays also short-circuits when `isGlasses` is true, so you're not trying to render undefined landmarks.

**Acceptance check:**
- On glasses (with `?mode=glasses`), `[MP_TIMING][FACE]` and `[MP_TIMING][POSE]` logs do not appear.
- On MacBook webcam, all previously-running landmarkers still appear in logs.
- Classifier still returns correct labels for the 8 core signs on the glasses (manual verification by user).
- Report sum of `[MP_TIMING][HANDS]` to user. If under 33ms AND classifier works, go to Step 7. Otherwise continue to Step 4.

---

## Step 4 — Switch Hand Landmarker to lite model variant

**Action:**

**If legacy Solutions (`@mediapipe/hands`):**
In the `Hands` constructor options, set `modelComplexity: 0`:

```javascript
new Hands({
  locateFile: (file) => `/${file}`,
  // ... existing options ...
  modelComplexity: 0,  // was 1 or unset; 0 is lite
});
```

**If migrated to Tasks (`@mediapipe/tasks-vision`):**
1. Download `hand_landmarker_lite.task` from Google's model hub to `public/models/`.
2. Update the `baseOptions.modelAssetPath` to point to the lite variant.

**Acceptance check:**
- Code compiles.
- `[MP_TIMING][HANDS]` value on glasses is lower than Step 3's measurement.
- Classifier still returns correct labels for the 8 core signs.
- If classifier accuracy drops noticeably (user-reported regression on any of the 8 signs), revert this step and continue to Step 5 with the full model.
- If total inference under 33ms AND classifier works, go to Step 7.

---

## Step 5 — Downscale input frames before inference

**Action:**
1. In `useSignLanguage.ts`, create a persistent offscreen canvas for downscaling:

```javascript
const downscaleCanvas = useMemo(() => new OffscreenCanvas(480, 360), []);
const downscaleCtx = useMemo(() => downscaleCanvas.getContext('2d'), [downscaleCanvas]);

const getDownscaledFrame = useCallback((videoElement) => {
  downscaleCtx.drawImage(videoElement, 0, 0, 480, 360);
  return downscaleCanvas;
}, [downscaleCanvas, downscaleCtx]);
```

2. Replace the raw video element passed to MediaPipe with the downscaled canvas output.
3. Verify landmark coordinates are still in normalized [0, 1] space (MediaPipe normalizes by input frame dimensions, so downscaling should not change landmark values — but verify).

**Acceptance check:**
- `[MP_TIMING][HANDS]` value on glasses is lower than Step 4's measurement.
- Classifier still returns correct labels for the 8 core signs.
- Landmark JSON output schema is unchanged (compare one captured sample before and after — landmark count, coordinate ranges, handedness labels).
- If total inference under 33ms AND classifier works, go to Step 7. Otherwise continue to Step 6.

---

## Step 6 — Throttle inference to 15fps

**Action:**
1. In the video frame processing loop, process every other frame:

```javascript
const frameCounterRef = useRef(0);

function onVideoFrame() {
  frameCounterRef.current += 1;
  if (frameCounterRef.current % 2 === 0) {
    // run MediaPipe inference
    runInference();
  }
  videoElement.requestVideoFrameCallback(onVideoFrame);
}
```

2. Ensure `requestVideoFrameCallback` is used, not `requestAnimationFrame` (the former provides fresh-frame guarantees).
3. Do NOT throttle video display — only inference. Video element continues rendering at native framerate.

**Acceptance check:**
- `[MP_TIMING][HANDS]` logs appear at roughly half the frequency they did before.
- Budget is now 66ms per processed frame (since processing happens every 33ms × 2 = 66ms).
- Classifier still returns correct labels. Note: the 24-frame window now represents ~1.6s of signing instead of ~0.8s. If classifier accuracy drops on fast signs, report this — may need to adjust window or revert.
- If total inference under 66ms AND classifier works, go to Step 7.

---

## Step 7 — Verification and cleanup

**Action:**
1. Remove or gate all `[MP_TIMING]` console.logs behind a debug flag:

```javascript
const DEBUG_TIMING = process.env.NEXT_PUBLIC_DEBUG_TIMING === 'true';
if (DEBUG_TIMING) console.log(`[MP_TIMING]...`);
```

2. Run the full test suite: `bun test` (or project-specific command).
3. Manually verify all 8 core signs classify correctly on glasses.
4. Write a concise summary in `notes/glasses-latency-session.md`:
   - Initial measurements
   - Which steps were applied
   - Final measurements
   - Any known regressions or tradeoffs
5. Stage changes and write a commit message:

```
perf(webview): reduce glasses inference load

- Disable face/pose landmarkers on glasses path (compute budget)
- Switch hand landmarker to lite model variant
- Downscale input to 480x360 before inference
- Throttle inference to 15fps on glasses

Before: [X]ms/frame total inference on glasses (skeleton lag visible)
After:  [Y]ms/frame total inference on glasses (skeleton tracks motion)

Classifier accuracy on 8 core signs unchanged (verified manually).
```

6. Report to user: summary of changes, before/after numbers, any caveats. Do not push without user confirmation.

**Acceptance check:**
- Test suite passes.
- No `[MP_TIMING]` logs in production console output unless debug flag set.
- User confirms skeleton tracks motion on glasses.
- User confirms classifier still works for 8 core signs.

---

## Step 8 — Escalation (only if Steps 3–6 did not achieve the goal)

**Do not attempt fixes in this step. Gather diagnostic data and report.**

**Action:**
1. Check main thread blocking:
   - Instruct user to open Chrome DevTools on the glasses webview (if available) or use remote debugging.
   - Record a Performance profile during 10 seconds of signing.
   - Look for long tasks (>50ms) that are not MediaPipe. Report findings.
2. Check frame source:
   - `grep -rn "requestAnimationFrame\|requestVideoFrameCallback" src/webview/`
   - If `requestAnimationFrame` is used for frame capture, flag this.
3. Check React re-render patterns:
   - `grep -rn "setState\|useState" src/webview/hooks/useSignLanguage.ts`
   - If landmarks are stored in React state (not refs), flag this.
4. Check GPU delegate status (Tasks only):
   - Search browser console for any MediaPipe warnings about fallback to CPU.
5. Check Mentra SDK video delivery:
   - Look at `src/services/glasses-camera-stream.ts` for transport type.
   - Report whether transport is WebRTC, WebSocket, or HLS.

**Output:** A diagnostic report in `notes/glasses-latency-session.md` with findings from each sub-step. Stop and wait for user/next-agent instructions.

---

## Quick reference

| Step | Time budget | Primary change | Acceptance signal |
|------|------|------|------|
| 1 | 15 min | Inventory landmarkers | Complete list in note file |
| 2 | 30 min | Add timing logs | Measurements from glasses |
| 3 | 1 hr | Disable face/pose on glasses | Only `[HANDS]` logs on glasses |
| 4 | 30 min | Lite model variant | Lower `[HANDS]` time |
| 5 | 1 hr | Downscale to 480×360 | Lower `[HANDS]` time |
| 6 | 30 min | Throttle to 15fps | Logs at half frequency |
| 7 | 1 hr | Cleanup + commit | Test suite passes |
| 8 | — | Escalation diagnostics | Report only, no fixes |

**Target:** Total per-frame inference < 33ms on glasses (< 66ms at 15fps), classifier unchanged on 8 core signs.

**Stop condition:** Goal met at any step → skip to Step 7.
