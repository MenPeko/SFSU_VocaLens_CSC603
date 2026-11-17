import {useEffect, useEffectEvent, useRef, useState, type RefObject} from "react"
import {
  FACEMESH_LEFT_EYE,
  FACEMESH_LEFT_EYEBROW,
  FACEMESH_LIPS,
  FACEMESH_RIGHT_EYE,
  FACEMESH_RIGHT_EYEBROW,
  HAND_CONNECTIONS,
  Holistic,
  POSE_CONNECTIONS,
  POSE_LANDMARKS,
  type NormalizedLandmark,
  type NormalizedLandmarkList,
  type Results,
} from "@mediapipe/holistic"
import fp from "fingerpose"
import {
  DEFAULT_CAPTURE_FPS,
  DEFAULT_HANDS_DOWN_DELAY_MS,
  GRID_CELL_SIZE,
  GRID_COLUMNS,
  GRID_ROWS,
  MAX_GRID_FRAMES,
  MAX_SEQUENCE_FRAMES,
  MIN_SEQUENCE_DURATION_MS,
  gridIntervalMs,
  landmarkIntervalMs,
} from "../../sign/capture-config"
import {MEDIAPIPE_EXTRACTOR_VERSION} from "../../sign/extractor-version"
import {LandmarkListSmoother} from "../../sign/one-euro"
import {appendRollingWindow, computeSequenceDurationMs} from "../../sign/sequence"
import {
  DEFAULT_SIGN_CAPTURE_SETTINGS,
  deriveActualCaptureQuality,
  inferQualityFallback,
  type ActualCaptureQuality,
  type CaptureSettingsFallbackReason,
  type CaptureSettingsStatus,
  type RequestedCaptureQuality,
  type SignCaptureSettings,
  type SkeletonRenderMode,
} from "../../sign/capture-settings"
import type {
  GestureHint,
  SignCaptureSource,
  SignFaceSubset,
  SignLandmarkFrame,
  SignLandmarkPoint,
} from "../../sign/types"
const REMOTE_STREAM_READY_TIMEOUT_MS = 20000
const REMOTE_STREAM_MAX_ATTACH_ATTEMPTS = 6
const REMOTE_STREAM_RETRY_DELAY_MS = 1500
const WEBRTC_ICE_GATHERING_TIMEOUT_MS = 5000
const WEBRTC_STREAM_READY_TIMEOUT_MS = 8000

const FACE_SUBSET_INDICES = {
  leftEyebrow: [70, 63, 105, 66, 107],
  rightEyebrow: [336, 296, 334, 293, 300],
  leftEye: [33, 160, 158, 133, 153, 144],
  rightEye: [263, 387, 385, 362, 380, 373],
  lips: [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91],
} as const

const ARM_POSE_INDICES = new Set([
  POSE_LANDMARKS.LEFT_SHOULDER,
  POSE_LANDMARKS.RIGHT_SHOULDER,
  POSE_LANDMARKS.LEFT_ELBOW,
  POSE_LANDMARKS.RIGHT_ELBOW,
  POSE_LANDMARKS.LEFT_WRIST,
  POSE_LANDMARKS.RIGHT_WRIST,
])

interface DetectedHand {
  label: "Left" | "Right"
  landmarks: NormalizedLandmarkList
}

const EXPECTED_HAND_LANDMARK_COUNT = 21

// Shared with the offline video extractor (scripts/sign-extract/page.ts) so
// dataset clips are landmarked exactly like live captures. Changing a value
// here changes the extractor and needs a +schema-v<n> bump.
export const HOLISTIC_OPTIONS = {
  selfieMode: false,
  modelComplexity: DEFAULT_SIGN_CAPTURE_SETTINGS.holisticModelComplexity,
  smoothLandmarks: true,
  refineFaceLandmarks: true,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.65,
} as const

export interface HolisticFrameDrawingData {
  poseLandmarks: NormalizedLandmarkList
  faceLandmarks: NormalizedLandmarkList
  leftHandLandmarks: NormalizedLandmarkList
  rightHandLandmarks: NormalizedLandmarkList
  handednessLabels: string[]
}

export interface RemoteSignPlaybackSource {
  url: string
  transport?: "webrtc" | "hls" | "direct"
  mimeType?: string
}

export type SignVideoSource =
  | {
      type: "device-camera"
      requestedQuality?: RequestedCaptureQuality
    }
  | {
      type: "glasses-stream"
      sources: RemoteSignPlaybackSource[]
      requestedQuality?: RequestedCaptureQuality
      actualQuality?: ActualCaptureQuality
      qualityFallback?: boolean
    }

export interface SignSequencePayload {
  base64Grid: string
  base64SkeletonGrid?: string
  dataUrl: string
  skeletonDataUrl?: string
  mimeType: "image/jpeg"
  skeletonMimeType?: "image/png" | "image/jpeg"
  frameCount: number
  gridColumns: number
  gridRows: number
  capturedAt: number
  gestureHints: GestureHint[]
  landmarkSequence: SignLandmarkFrame[]
  frameTimestamps: number[]
  sequenceDurationMs: number
  sequenceFrameCount: number
  source: SignCaptureSource
  extractorVersion: string
  requestedQuality?: RequestedCaptureQuality
  actualQuality?: ActualCaptureQuality
  qualityFallback?: boolean
  actualInputWidth?: number
  actualInputHeight?: number
  skeletonRenderMode: SkeletonRenderMode
  captureSettingsStatus: CaptureSettingsStatus
  captureSettingsFallbackReason?: CaptureSettingsFallbackReason
  droppedFrameCount: number
}

export interface UseSignLanguageOptions {
  videoRef: RefObject<HTMLVideoElement | null>
  canvasRef: RefObject<HTMLCanvasElement | null>
  gridCanvasRef: RefObject<HTMLCanvasElement | null>
  skeletonGridCanvasRef: RefObject<HTMLCanvasElement | null>
  videoSource?: SignVideoSource
  enabled?: boolean
  showHandednessOverlay?: boolean
  captureSettings?: SignCaptureSettings
  captureSettingsStatus?: CaptureSettingsStatus
  captureSettingsFallbackReason?: CaptureSettingsFallbackReason
  captureFps?: number
  handsDownDelayMs?: number
  onHandsDown?: (payload: SignSequencePayload) => void | Promise<void>
}

export interface UseSignLanguageResult {
  cameraState: "idle" | "requesting" | "ready" | "error"
  error: string | null
  videoSource: SignVideoSource["type"]
  isSigning: boolean
  detectedHands: number
  handednessLabels: string[]
  capturedFrames: number
  sequenceFrames: number
  gestureHints: GestureHint[]
  latestSequence: SignSequencePayload | null
  requestedQuality?: RequestedCaptureQuality
  actualQuality?: ActualCaptureQuality
  qualityFallback?: boolean
  actualInputWidth?: number
  actualInputHeight?: number
  captureSettingsStatus: CaptureSettingsStatus
  captureSettingsFallbackReason?: CaptureSettingsFallbackReason
  droppedFrameCount: number
  trackingConfidenceExperiment: number | null
  restart: () => Promise<void>
  stop: () => Promise<void>
  clearSequence: () => void
}

interface SkeletonRenderOptions {
  renderMode?: SkeletonRenderMode
  showHandednessOverlay?: boolean
  mirrorOverlayText?: boolean
  maskOccludedFace?: boolean
}

export function buildVideoSourceKey(videoSource: SignVideoSource): string {
  const requestedQualitySegment = videoSource.requestedQuality ?? "default"

  if (videoSource.type === "glasses-stream") {
    return `${videoSource.type}:${requestedQualitySegment}:${videoSource.actualQuality ?? "unknown"}:${videoSource.sources
      .map((source) => `${source.transport ?? "direct"}:${source.url}:${source.mimeType ?? ""}`)
      .join("|")}`
  }

  return `${videoSource.type}:${requestedQualitySegment}`
}

export function getSkeletonRenderStyle(renderMode: SkeletonRenderMode): {
  lineCap: CanvasLineCap
  lineJoin: CanvasLineJoin
  shadowBlur: number
  shadowColor: string
} {
  return renderMode === "diagnostic"
    ? {
        lineCap: "butt",
        lineJoin: "miter",
        shadowBlur: 0,
        shadowColor: "rgba(0,0,0,0)",
      }
    : {
        lineCap: "round",
        lineJoin: "round",
        shadowBlur: 10,
        shadowColor: "rgba(30, 144, 255, 0.45)",
      }
}

function createOpenPalmGesture(): InstanceType<typeof fp.GestureDescription> {
  const gesture = new fp.GestureDescription("open_palm")

  for (const finger of fp.Finger.all) {
    gesture.addCurl(finger, fp.FingerCurl.NoCurl, 1.0)
    gesture.addCurl(finger, fp.FingerCurl.HalfCurl, 0.5)
  }

  return gesture
}

function createFistGesture(): InstanceType<typeof fp.GestureDescription> {
  const gesture = new fp.GestureDescription("fist")

  for (const finger of fp.Finger.all) {
    gesture.addCurl(finger, fp.FingerCurl.FullCurl, 1.0)
    gesture.addCurl(finger, fp.FingerCurl.HalfCurl, 0.85)
  }

  return gesture
}

function createPointGesture(): InstanceType<typeof fp.GestureDescription> {
  const gesture = new fp.GestureDescription("pointing")

  gesture.addCurl(fp.Finger.Index, fp.FingerCurl.NoCurl, 1.0)
  gesture.addCurl(fp.Finger.Index, fp.FingerCurl.HalfCurl, 0.2)
  gesture.addCurl(fp.Finger.Middle, fp.FingerCurl.FullCurl, 1.0)
  gesture.addCurl(fp.Finger.Middle, fp.FingerCurl.HalfCurl, 0.8)
  gesture.addCurl(fp.Finger.Ring, fp.FingerCurl.FullCurl, 1.0)
  gesture.addCurl(fp.Finger.Ring, fp.FingerCurl.HalfCurl, 0.8)
  gesture.addCurl(fp.Finger.Pinky, fp.FingerCurl.FullCurl, 1.0)
  gesture.addCurl(fp.Finger.Pinky, fp.FingerCurl.HalfCurl, 0.8)
  gesture.addCurl(fp.Finger.Thumb, fp.FingerCurl.HalfCurl, 0.7)
  gesture.addCurl(fp.Finger.Thumb, fp.FingerCurl.NoCurl, 0.4)

  return gesture
}

const GESTURE_ESTIMATOR = new fp.GestureEstimator([
  createOpenPalmGesture(),
  createFistGesture(),
  createPointGesture(),
  fp.Gestures.ThumbsUpGesture,
  fp.Gestures.VictoryGesture,
])

function drawFrameCover(
  context: CanvasRenderingContext2D,
  source: CanvasImageSource,
  targetX: number,
  targetY: number,
  targetWidth: number,
  targetHeight: number,
  sourceWidth: number,
  sourceHeight: number,
): void {
  const sourceAspect = sourceWidth / sourceHeight
  const targetAspect = targetWidth / targetHeight

  let cropWidth = sourceWidth
  let cropHeight = sourceHeight
  let cropX = 0
  let cropY = 0

  if (sourceAspect > targetAspect) {
    cropWidth = sourceHeight * targetAspect
    cropX = (sourceWidth - cropWidth) / 2
  } else {
    cropHeight = sourceWidth / targetAspect
    cropY = (sourceHeight - cropHeight) / 2
  }

  context.drawImage(
    source,
    cropX,
    cropY,
    cropWidth,
    cropHeight,
    targetX,
    targetY,
    targetWidth,
    targetHeight,
  )
}

function ensureCanvasSize(canvas: HTMLCanvasElement, width: number, height: number): void {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
}

async function waitForVideoReady(
  video: HTMLVideoElement,
  timeoutMs = REMOTE_STREAM_READY_TIMEOUT_MS,
  timeoutMessage = "Timed out waiting for the Mentra glasses video stream.",
): Promise<void> {
  // Require at least one decoded frame (HAVE_CURRENT_DATA) and real dimensions.
  // HAVE_METADATA only means the container header loaded, which lets the capture
  // pipeline operate on a black frame while the real video is still buffering.
  const isReady = () =>
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0 && video.videoHeight > 0

  if (isReady()) {
    return
  }

  await new Promise<void>((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      cleanup()
      reject(new Error(timeoutMessage))
    }, timeoutMs)

    const cleanup = () => {
      window.clearTimeout(timeoutId)
      video.removeEventListener("loadeddata", handleReady)
      video.removeEventListener("playing", handleReady)
      video.removeEventListener("error", handleError)
    }

    const handleReady = () => {
      if (!isReady()) {
        return
      }
      cleanup()
      resolve()
    }

    const handleError = () => {
      cleanup()
      reject(new Error("The selected video source could not be loaded."))
    }

    video.addEventListener("loadeddata", handleReady)
    video.addEventListener("playing", handleReady)
    video.addEventListener("error", handleError)
  })
}

function resetVideoElement(video: HTMLVideoElement): void {
  video.pause()
  video.srcObject = null
  video.removeAttribute("src")
  video.load()
}

async function delay(ms: number): Promise<void> {
  await new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

function normalizeRemotePlaybackSource(source: RemoteSignPlaybackSource): RemoteSignPlaybackSource {
  if (source.transport) {
    return source
  }

  if (source.mimeType === "application/vnd.apple.mpegurl" || source.url.includes(".m3u8")) {
    return {
      ...source,
      transport: "hls",
    }
  }

  return {
    ...source,
    transport: "direct",
  }
}

const REMOTE_DRIFT_CHECK_INTERVAL_MS = 1000
const HLS_DRIFT_SEEK_THRESHOLD_S = 4
const WEBRTC_DRIFT_SEEK_THRESHOLD_S = 2
const CAPTURE_DRIFT_SEEK_THRESHOLD_S = 3

const DROPPED_FRAME_STATE_INTERVAL_MS = 250
const HOLISTIC_TIMING_WINDOW_FRAMES = 60
const SIGN_TIMING_DEBUG_STORAGE_KEY = "sign:debug-timing"

export interface HolisticTimingWindow {
  count: number
  totalMs: number
  maxMs: number
}

export function createHolisticTimingWindow(): HolisticTimingWindow {
  return {count: 0, totalMs: 0, maxMs: 0}
}

// Accumulates per-inference durations and returns a summary (then resets) once
// every HOLISTIC_TIMING_WINDOW_FRAMES samples.
export function recordHolisticTiming(
  timing: HolisticTimingWindow,
  durationMs: number,
  windowFrames = HOLISTIC_TIMING_WINDOW_FRAMES,
): {averageMs: number; maxMs: number; count: number} | null {
  timing.count += 1
  timing.totalMs += durationMs
  timing.maxMs = Math.max(timing.maxMs, durationMs)

  if (timing.count < windowFrames) {
    return null
  }

  const summary = {averageMs: timing.totalMs / timing.count, maxMs: timing.maxMs, count: timing.count}
  timing.count = 0
  timing.totalMs = 0
  timing.maxMs = 0
  return summary
}

function createOverlaySmoothers() {
  return {
    pose: new LandmarkListSmoother(),
    face: new LandmarkListSmoother(),
    leftHand: new LandmarkListSmoother(),
    rightHand: new LandmarkListSmoother(),
  }
}

const TRACKING_CONFIDENCE_EXPERIMENT_STORAGE_KEY = "sign:experiment-tracking-confidence"

// Local-only experiment: overrides HOLISTIC_OPTIONS.minTrackingConfidence for
// the live hook. Payloads get a distinct extractor stamp so the classifier and
// trainer reject them instead of mixing them with schema-locked samples.
export function parseTrackingConfidenceExperiment(value: string | null): number | null {
  if (value === null || value.trim() === "") {
    return null
  }

  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 && parsed < 1 ? parsed : null
}

export function experimentExtractorVersion(trackingConfidence: number | null): string {
  return trackingConfidence === null
    ? MEDIAPIPE_EXTRACTOR_VERSION
    : `${MEDIAPIPE_EXTRACTOR_VERSION}+exp-tracking-${trackingConfidence}`
}

function readTrackingConfidenceExperiment(): number | null {
  try {
    return parseTrackingConfidenceExperiment(window.localStorage.getItem(TRACKING_CONFIDENCE_EXPERIMENT_STORAGE_KEY))
  } catch {
    return null
  }
}

function isSignTimingDebugEnabled(): boolean {
  try {
    return window.localStorage.getItem(SIGN_TIMING_DEBUG_STORAGE_KEY) === "1"
  } catch {
    return false
  }
}

function clearRemoteDriftInterval(ref: {current: number | null}): void {
  if (ref.current !== null) {
    window.clearInterval(ref.current)
    ref.current = null
  }
}

// Cloudflare Stream supports Low-Latency HLS only when the manifest URL opts in
// via `?protocol=llhls`. Standard HLS has ~10-15s latency; LL-HLS is ~2-6s. We
// rewrite URLs here rather than in the backend because the same helper works for
// both `playbackUrl` and `hlsUrl`, and for the videodelivery.net fallback path.
function maybeUpgradeToCloudflareLlHls(url: string): string {
  try {
    const parsed = new URL(url)
    const isCloudflareStream =
      /(^|\.)cloudflarestream\.com$/.test(parsed.hostname) || parsed.hostname === "videodelivery.net"
    if (!isCloudflareStream) {
      return url
    }
    if (!parsed.searchParams.has("protocol")) {
      parsed.searchParams.set("protocol", "llhls")
    }
    return parsed.toString()
  } catch {
    return url
  }
}

function parseIceServers(linkHeader: string | null): RTCIceServer[] {
  if (!linkHeader) {
    return []
  }

  const iceServers: RTCIceServer[] = []
  const entries = linkHeader.split(/,(?=\s*<)/)

  for (const entry of entries) {
    const segments = entry
      .split(";")
      .map((segment) => segment.trim())
      .filter(Boolean)
    const rawUrl = segments[0]
    if (!rawUrl?.startsWith("<") || !rawUrl.endsWith(">")) {
      continue
    }

    const params = new Map<string, string>()
    for (const segment of segments.slice(1)) {
      const [key, ...valueParts] = segment.split("=")
      if (!key || valueParts.length === 0) {
        continue
      }

      params.set(key.trim().toLowerCase(), valueParts.join("=").replace(/^"(.*)"$/, "$1"))
    }

    if (params.get("rel") !== "ice-server") {
      continue
    }

    iceServers.push({
      urls: rawUrl.slice(1, -1),
      username: params.get("username"),
      credential: params.get("credential"),
    })
  }

  return iceServers
}

async function discoverIceServersForWebRtc(sourceUrl: string): Promise<RTCIceServer[]> {
  try {
    const response = await fetch(sourceUrl, {
      method: "OPTIONS",
    })

    if (!response.ok) {
      return []
    }

    return parseIceServers(response.headers.get("Link"))
  } catch {
    return []
  }
}

async function waitForIceGatheringComplete(peerConnection: RTCPeerConnection): Promise<void> {
  if (peerConnection.iceGatheringState === "complete") {
    return
  }

  await new Promise<void>((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      cleanup()
      reject(new Error("Timed out preparing the low-latency glasses stream."))
    }, WEBRTC_ICE_GATHERING_TIMEOUT_MS)

    const cleanup = () => {
      window.clearTimeout(timeoutId)
      peerConnection.removeEventListener("icegatheringstatechange", handleChange)
    }

    const handleChange = () => {
      if (peerConnection.iceGatheringState !== "complete") {
        return
      }

      cleanup()
      resolve()
    }

    peerConnection.addEventListener("icegatheringstatechange", handleChange)
  })
}

function destroyPeerConnection(
  peerConnectionRef: {current: RTCPeerConnection | null},
  remoteSessionUrlRef: {current: string | null},
  remoteDriftIntervalRef?: {current: number | null},
): void {
  if (remoteDriftIntervalRef) {
    clearRemoteDriftInterval(remoteDriftIntervalRef)
  }

  const peerConnection = peerConnectionRef.current
  peerConnectionRef.current = null

  for (const receiver of peerConnection?.getReceivers() ?? []) {
    receiver.track?.stop()
  }

  try {
    peerConnection?.close()
  } catch {
    // Ignore close errors when the connection is already shutting down.
  }

  const sessionUrl = remoteSessionUrlRef.current
  remoteSessionUrlRef.current = null

  if (!sessionUrl) {
    return
  }

  void fetch(sessionUrl, {
    method: "DELETE",
    keepalive: true,
  }).catch(() => {
    // Session teardown is best-effort for demo playback.
  })
}

async function attachWebRtcVideoSource(
  video: HTMLVideoElement,
  sourceUrl: string,
  peerConnectionRef: {current: RTCPeerConnection | null},
  remoteSessionUrlRef: {current: string | null},
  remoteDriftIntervalRef: {current: number | null},
): Promise<void> {
  if (typeof RTCPeerConnection === "undefined") {
    throw new Error("This browser cannot open the low-latency Mentra glasses stream.")
  }

  destroyPeerConnection(peerConnectionRef, remoteSessionUrlRef, remoteDriftIntervalRef)

  const iceServers = await discoverIceServersForWebRtc(sourceUrl)
  const peerConnection = new RTCPeerConnection(iceServers.length > 0 ? {iceServers} : undefined)
  peerConnectionRef.current = peerConnection

  const remoteStream = new MediaStream()
  video.srcObject = remoteStream

  peerConnection.addTransceiver("video", {direction: "recvonly"})
  peerConnection.addTransceiver("audio", {direction: "recvonly"})
  peerConnection.ontrack = (event) => {
    if (event.streams[0]) {
      video.srcObject = event.streams[0]
      return
    }

    remoteStream.addTrack(event.track)
  }

  const offer = await peerConnection.createOffer()
  await peerConnection.setLocalDescription(offer)
  await waitForIceGatheringComplete(peerConnection)

  const offerSdp = peerConnection.localDescription?.sdp
  if (!offerSdp) {
    throw new Error("Could not prepare the low-latency glasses stream.")
  }

  const response = await fetch(sourceUrl, {
    method: "POST",
    headers: {
      Accept: "application/sdp",
      "Content-Type": "application/sdp",
    },
    body: offerSdp,
  })

  if (!response.ok) {
    throw new Error(`Glasses WebRTC playback failed (${response.status})`)
  }

  const sessionLocation = response.headers.get("Location")
  if (sessionLocation) {
    remoteSessionUrlRef.current = new URL(sessionLocation, sourceUrl).toString()
  }

  const answerSdp = await response.text()
  await peerConnection.setRemoteDescription({
    type: "answer",
    sdp: answerSdp,
  })
  await waitForVideoReady(video, WEBRTC_STREAM_READY_TIMEOUT_MS, "Timed out starting the low-latency glasses stream.")

  // WebRTC has no manifest, so use the end of the most recent buffered range as
  // the "live edge" and snap the playhead forward when drift accumulates.
  clearRemoteDriftInterval(remoteDriftIntervalRef)
  remoteDriftIntervalRef.current = window.setInterval(() => {
    if (video.buffered.length === 0) {
      return
    }
    const liveEdge = video.buffered.end(video.buffered.length - 1)
    if (liveEdge - video.currentTime > WEBRTC_DRIFT_SEEK_THRESHOLD_S) {
      video.currentTime = liveEdge
    }
  }, REMOTE_DRIFT_CHECK_INTERVAL_MS)
}

async function attachRemoteVideoSource(
  video: HTMLVideoElement,
  source: RemoteSignPlaybackSource,
  hlsRef: {current: {destroy: () => void} | null},
  peerConnectionRef: {current: RTCPeerConnection | null},
  remoteSessionUrlRef: {current: string | null},
  remoteDriftIntervalRef: {current: number | null},
): Promise<void> {
  video.crossOrigin = "anonymous"

  const normalizedSource = normalizeRemotePlaybackSource(source)
  if (normalizedSource.transport === "webrtc") {
    await attachWebRtcVideoSource(video, normalizedSource.url, peerConnectionRef, remoteSessionUrlRef, remoteDriftIntervalRef)
    return
  }

  const looksLikeHls = normalizedSource.transport === "hls"
  const sourceUrl = looksLikeHls ? maybeUpgradeToCloudflareLlHls(normalizedSource.url) : normalizedSource.url
  if (!looksLikeHls) {
    video.src = sourceUrl
    await waitForVideoReady(video)
    return
  }

  if (video.canPlayType("application/vnd.apple.mpegurl")) {
    try {
      video.src = sourceUrl
      await waitForVideoReady(video)
      return
    } catch {
      resetVideoElement(video)
    }
  }

  const HlsModule = await import("hls.js")
  const Hls = HlsModule.default

  if (!Hls.isSupported()) {
    throw new Error("This browser cannot play the Mentra glasses HLS stream.")
  }

  const hls = new Hls({
    lowLatencyMode: true,
    backBufferLength: 4,
    liveSyncDurationCount: 1,
    liveMaxLatencyDurationCount: 5,
    maxLiveSyncPlaybackRate: 1.5,
    maxBufferHole: 0.5,
    maxFragLookUpTolerance: 0.2,
  })

  hlsRef.current = hls

  await new Promise<void>((resolve, reject) => {
    const handleManifestParsed = () => {
      cleanup()
      resolve()
    }

    const handleError = (_event: string, data: {fatal?: boolean; details?: string}) => {
      if (!data?.fatal) {
        return
      }

      cleanup()
      reject(new Error(data.details ? `Glasses stream playback failed: ${data.details}` : "Glasses stream playback failed."))
    }

    const cleanup = () => {
      hls.off(Hls.Events.MANIFEST_PARSED, handleManifestParsed)
      hls.off(Hls.Events.ERROR, handleError)
    }

    hls.on(Hls.Events.MANIFEST_PARSED, handleManifestParsed)
    hls.on(Hls.Events.ERROR, handleError)
    hls.loadSource(sourceUrl)
    hls.attachMedia(video)
  })

  await waitForVideoReady(video)

  // Drift watchdog: if hls.js falls more than HLS_DRIFT_SEEK_THRESHOLD_S behind
  // the live edge, snap the playhead forward so captured frames reflect the
  // current moment rather than a stale point in the DVR window.
  clearRemoteDriftInterval(remoteDriftIntervalRef)
  remoteDriftIntervalRef.current = window.setInterval(() => {
    const liveEdge = hls.liveSyncPosition
    if (typeof liveEdge !== "number" || !Number.isFinite(liveEdge)) {
      return
    }
    if (liveEdge - video.currentTime > HLS_DRIFT_SEEK_THRESHOLD_S) {
      video.currentTime = liveEdge
    }
  }, REMOTE_DRIFT_CHECK_INTERVAL_MS)
}

async function attachRemoteVideoSourceWithRetry(
  video: HTMLVideoElement,
  sources: RemoteSignPlaybackSource[],
  hlsRef: {current: {destroy: () => void} | null},
  peerConnectionRef: {current: RTCPeerConnection | null},
  remoteSessionUrlRef: {current: string | null},
  remoteDriftIntervalRef: {current: number | null},
): Promise<void> {
  const candidateSources = sources
    .map((source) => ({
      ...normalizeRemotePlaybackSource(source),
      url: source.url.trim(),
    }))
    .filter((source) => source.url.length > 0)
    .filter(
      (source, index, values) =>
        values.findIndex((candidate) => candidate.url === source.url && candidate.transport === source.transport) === index,
    )
  let lastError: unknown = null

  for (let attempt = 1; attempt <= REMOTE_STREAM_MAX_ATTACH_ATTEMPTS; attempt += 1) {
    for (const candidateSource of candidateSources) {
      try {
        clearRemoteDriftInterval(remoteDriftIntervalRef)

        if (hlsRef.current) {
          hlsRef.current.destroy()
          hlsRef.current = null
        }

        destroyPeerConnection(peerConnectionRef, remoteSessionUrlRef, remoteDriftIntervalRef)
        resetVideoElement(video)
        await attachRemoteVideoSource(
          video,
          candidateSource,
          hlsRef,
          peerConnectionRef,
          remoteSessionUrlRef,
          remoteDriftIntervalRef,
        )
        return
      } catch (error) {
        lastError = error
      }
    }

    if (attempt === REMOTE_STREAM_MAX_ATTACH_ATTEMPTS) {
      break
    }

    await delay(REMOTE_STREAM_RETRY_DELAY_MS)
  }

  if (lastError instanceof Error) {
    throw lastError
  }

  throw new Error("The Mentra glasses stream could not be attached.")
}

function formatGestureHintName(name: string): string {
  return name.replaceAll("_", " ")
}

function hasUsableHandLandmarks(landmarks: NormalizedLandmarkList): boolean {
  return (
    landmarks.length >= EXPECTED_HAND_LANDMARK_COUNT &&
    landmarks.every(
      (landmark) =>
        Boolean(
          landmark &&
            Number.isFinite(landmark.x) &&
            Number.isFinite(landmark.y) &&
            Number.isFinite(landmark.z),
        ),
    )
  )
}

export function estimateGestureHints(hands: DetectedHand[]): GestureHint[] {
  if (hands.length === 0) {
    return []
  }

  const mergedHints = new Map<string, number>()
  const topGestureNames: string[] = []
  const topGestureScores: number[] = []

  for (const hand of hands) {
    if (!hasUsableHandLandmarks(hand.landmarks)) {
      continue
    }

    let estimate: ReturnType<typeof GESTURE_ESTIMATOR.estimate>
    try {
      estimate = GESTURE_ESTIMATOR.estimate(hand.landmarks, 7.5)
    } catch {
      continue
    }

    const rankedGestures = estimate.gestures.slice().sort((left, right) => right.score - left.score)
    if (rankedGestures.length === 0) {
      continue
    }

    topGestureNames.push(rankedGestures[0].name)
    topGestureScores.push(rankedGestures[0].score)

    for (const gesture of rankedGestures.slice(0, 2)) {
      const formattedName = `${hand.label.toLowerCase()} hand ${formatGestureHintName(gesture.name)}`
      const nextScore = Number(gesture.score.toFixed(1))
      const existingScore = mergedHints.get(formattedName) ?? 0
      mergedHints.set(formattedName, Math.max(existingScore, nextScore))
    }
  }

  if (topGestureNames.length > 1 && topGestureNames.every((name) => name === topGestureNames[0])) {
    const combinedScore = Number(
      (topGestureScores.reduce((total, score) => total + score, 0) / topGestureScores.length).toFixed(1),
    )
    mergedHints.set(`both hands ${formatGestureHintName(topGestureNames[0])}`, combinedScore)
  }

  return [...mergedHints.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5)
    .map(([name, score]) => ({
      name,
      score: Number(score.toFixed(1)),
    }))
}

export function buildDetectedHands(results: Results): DetectedHand[] {
  const hands: DetectedHand[] = []
  if (results.leftHandLandmarks?.length) {
    hands.push({label: "Left", landmarks: results.leftHandLandmarks})
  }
  if (results.rightHandLandmarks?.length) {
    hands.push({label: "Right", landmarks: results.rightHandLandmarks})
  }
  return hands
}

function buildHandednessLabels(hands: DetectedHand[]): string[] {
  return hands.map((hand) => `${hand.label.toUpperCase()} HAND`)
}

export interface LandmarkRegion {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

const HAND_REGION_PADDING = 0.25
const FACE_OCCLUSION_HIDE_FRACTION = 0.4
const OCCLUDED_FACE_ALPHA = 0.35

// Padded bounding boxes around each detected hand, in normalized coordinates.
export function computeHandRegions(
  frame: Pick<HolisticFrameDrawingData, "leftHandLandmarks" | "rightHandLandmarks">,
  padding = HAND_REGION_PADDING,
): LandmarkRegion[] {
  const regions: LandmarkRegion[] = []

  for (const hand of [frame.leftHandLandmarks, frame.rightHandLandmarks]) {
    if (!hand.length) {
      continue
    }

    const xs = hand.map((landmark) => landmark.x)
    const ys = hand.map((landmark) => landmark.y)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const padX = (maxX - minX) * padding
    const padY = (maxY - minY) * padding

    regions.push({minX: minX - padX, minY: minY - padY, maxX: maxX + padX, maxY: maxY + padY})
  }

  return regions
}

export function isInsideRegions(landmark: NormalizedLandmark, regions: LandmarkRegion[]): boolean {
  return regions.some(
    (region) =>
      landmark.x >= region.minX && landmark.x <= region.maxX && landmark.y >= region.minY && landmark.y <= region.maxY,
  )
}

// Fraction of face landmarks covered by a hand. Holistic keeps predicting the
// full face mesh under occlusion, so covered points are unreliable.
export function computeFaceOcclusion(faceLandmarks: NormalizedLandmarkList, handRegions: LandmarkRegion[]): number {
  if (!faceLandmarks.length || !handRegions.length) {
    return 0
  }

  const covered = faceLandmarks.filter((landmark) => landmark && isInsideRegions(landmark, handRegions)).length
  return covered / faceLandmarks.length
}

function drawConnections(
  context: CanvasRenderingContext2D,
  landmarks: NormalizedLandmarkList,
  connections: Array<[number, number]>,
  width: number,
  height: number,
  lineWidth: number,
  color: string,
  skip?: (landmark: NormalizedLandmark) => boolean,
): void {
  context.strokeStyle = color
  context.lineWidth = lineWidth

  for (const [startIndex, endIndex] of connections) {
    const start = landmarks[startIndex]
    const end = landmarks[endIndex]
    if (!start || !end || skip?.(start) || skip?.(end)) {
      continue
    }

    context.beginPath()
    context.moveTo(start.x * width, start.y * height)
    context.lineTo(end.x * width, end.y * height)
    context.stroke()
  }
}

function drawLandmarks(
  context: CanvasRenderingContext2D,
  landmarks: NormalizedLandmarkList,
  width: number,
  height: number,
  radius: number,
  color: string,
  skip?: (landmark: NormalizedLandmark) => boolean,
): void {
  context.fillStyle = color

  for (const landmark of landmarks) {
    if (skip?.(landmark)) {
      continue
    }
    context.beginPath()
    context.arc(landmark.x * width, landmark.y * height, radius, 0, Math.PI * 2)
    context.fill()
  }
}

function drawHandednessLabel(
  context: CanvasRenderingContext2D,
  landmarks: NormalizedLandmarkList,
  label: string,
  canvasWidth: number,
  canvasHeight: number,
  mirrorText: boolean,
): void {
  if (!landmarks.length) return

  const xs = landmarks.map((landmark) => landmark.x * canvasWidth)
  const ys = landmarks.map((landmark) => landmark.y * canvasHeight)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const labelX = Math.max(54, Math.min(canvasWidth - 54, (minX + maxX) / 2))
  const labelY = Math.max(18, minY - 16)

  context.save()
  context.font = '700 12px "Geist Mono", monospace'
  context.textAlign = "center"
  context.textBaseline = "middle"

  const textWidth = context.measureText(label).width
  const boxWidth = Math.max(76, textWidth + 18)
  const boxHeight = 22

  context.fillStyle = "rgba(0, 0, 0, 0.82)"
  context.fillRect(labelX - boxWidth / 2, labelY - boxHeight / 2, boxWidth, boxHeight)
  context.strokeStyle = "#1e90ff"
  context.lineWidth = 2
  context.strokeRect(labelX - boxWidth / 2, labelY - boxHeight / 2, boxWidth, boxHeight)
  context.fillStyle = "#ffffff"

  if (mirrorText) {
    context.translate(labelX, labelY)
    context.scale(-1, 1)
    context.fillText(label, 0, 0)
  } else {
    context.fillText(label, labelX, labelY)
  }

  context.restore()
}

function renderHolisticSkeletonToContext(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  frame: HolisticFrameDrawingData,
  options?: SkeletonRenderOptions,
): void {
  context.save()
  const renderMode = options?.renderMode ?? "presentation"
  const renderStyle = getSkeletonRenderStyle(renderMode)

  context.globalAlpha = 1
  context.setLineDash([])
  context.lineCap = renderStyle.lineCap
  context.lineJoin = renderStyle.lineJoin
  context.shadowBlur = renderStyle.shadowBlur
  context.shadowColor = renderStyle.shadowColor

  const lineWidth = Math.max(2, width / 260)
  const jointRadius = Math.max(2.5, width / 150)
  const faceRadius = Math.max(1.25, width / 320)

  const armConnections = POSE_CONNECTIONS.filter(
    ([startIndex, endIndex]) => ARM_POSE_INDICES.has(startIndex) && ARM_POSE_INDICES.has(endIndex),
  )
  drawConnections(context, frame.poseLandmarks, armConnections, width, height, lineWidth, "#1e90ff")
  drawLandmarks(
    context,
    frame.poseLandmarks.filter((_, index) => ARM_POSE_INDICES.has(index)),
    width,
    height,
    jointRadius,
    "#ffffff",
  )

  if (frame.leftHandLandmarks.length) {
    drawConnections(context, frame.leftHandLandmarks, HAND_CONNECTIONS, width, height, lineWidth, "#1e90ff")
    drawLandmarks(context, frame.leftHandLandmarks, width, height, jointRadius, "#ffffff")
  }

  if (frame.rightHandLandmarks.length) {
    drawConnections(context, frame.rightHandLandmarks, HAND_CONNECTIONS, width, height, lineWidth, "#1e90ff")
    drawLandmarks(context, frame.rightHandLandmarks, width, height, jointRadius, "#ffffff")
  }

  if (frame.poseLandmarks[POSE_LANDMARKS.LEFT_WRIST] && frame.leftHandLandmarks[0]) {
    drawConnections(
      context,
      [
        frame.poseLandmarks[POSE_LANDMARKS.LEFT_WRIST],
        frame.leftHandLandmarks[0],
      ],
      [[0, 1]],
      width,
      height,
      lineWidth,
      "#1e90ff",
    )
  }

  if (frame.poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST] && frame.rightHandLandmarks[0]) {
    drawConnections(
      context,
      [
        frame.poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST],
        frame.rightHandLandmarks[0],
      ],
      [[0, 1]],
      width,
      height,
      lineWidth,
      "#1e90ff",
    )
  }

  const faceConnectionSets = [
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_LIPS,
  ]
  const handRegions = options?.maskOccludedFace ? computeHandRegions(frame) : []
  const faceOcclusion = computeFaceOcclusion(frame.faceLandmarks, handRegions)
  if (faceOcclusion < FACE_OCCLUSION_HIDE_FRACTION) {
    const skipOccluded =
      faceOcclusion > 0 ? (landmark: NormalizedLandmark) => isInsideRegions(landmark, handRegions) : undefined
    context.globalAlpha = faceOcclusion > 0 ? OCCLUDED_FACE_ALPHA : 1
    for (const connectionSet of faceConnectionSets) {
      drawConnections(
        context,
        frame.faceLandmarks,
        connectionSet,
        width,
        height,
        Math.max(1.2, lineWidth * 0.7),
        "#ffffff",
        skipOccluded,
      )
    }
    drawLandmarks(context, frame.faceLandmarks, width, height, faceRadius, "#1e90ff", skipOccluded)
    context.globalAlpha = 1
  }

  if (options?.showHandednessOverlay) {
    if (frame.leftHandLandmarks.length) {
      drawHandednessLabel(
        context,
        frame.leftHandLandmarks,
        "LEFT HAND",
        width,
        height,
        options.mirrorOverlayText === true,
      )
    }
    if (frame.rightHandLandmarks.length) {
      drawHandednessLabel(
        context,
        frame.rightHandLandmarks,
        "RIGHT HAND",
        width,
        height,
        options.mirrorOverlayText === true,
      )
    }
  }

  context.restore()
}

function drawSkeleton(
  canvas: HTMLCanvasElement,
  frame: HolisticFrameDrawingData,
  options?: SkeletonRenderOptions,
): void {
  const context = canvas.getContext("2d")
  if (!context) {
    return
  }

  context.clearRect(0, 0, canvas.width, canvas.height)
  renderHolisticSkeletonToContext(context, canvas.width, canvas.height, frame, options)
}

function createFrameCanvas(video: HTMLVideoElement): HTMLCanvasElement {
  const frameCanvas = document.createElement("canvas")
  frameCanvas.width = GRID_CELL_SIZE
  frameCanvas.height = GRID_CELL_SIZE

  const context = frameCanvas.getContext("2d")
  if (!context) {
    return frameCanvas
  }

  context.fillStyle = "#05070b"
  context.fillRect(0, 0, frameCanvas.width, frameCanvas.height)
  drawFrameCover(context, video, 0, 0, GRID_CELL_SIZE, GRID_CELL_SIZE, video.videoWidth, video.videoHeight)

  return frameCanvas
}

function createSkeletonFrameCanvas(frame: HolisticFrameDrawingData): HTMLCanvasElement {
  const frameCanvas = document.createElement("canvas")
  frameCanvas.width = GRID_CELL_SIZE
  frameCanvas.height = GRID_CELL_SIZE

  const context = frameCanvas.getContext("2d")
  if (!context) {
    return frameCanvas
  }

  context.fillStyle = "#05070b"
  context.fillRect(0, 0, frameCanvas.width, frameCanvas.height)
  renderHolisticSkeletonToContext(context, frameCanvas.width, frameCanvas.height, frame, {
    renderMode: "diagnostic",
  })

  return frameCanvas
}

function redrawFrameGridCanvas(
  gridCanvas: HTMLCanvasElement,
  frames: HTMLCanvasElement[],
  accentColor: string,
): void {
  ensureCanvasSize(gridCanvas, GRID_COLUMNS * GRID_CELL_SIZE, GRID_ROWS * GRID_CELL_SIZE)

  const context = gridCanvas.getContext("2d")
  if (!context) {
    return
  }

  context.fillStyle = "#05070b"
  context.fillRect(0, 0, gridCanvas.width, gridCanvas.height)

  for (let index = 0; index < MAX_GRID_FRAMES; index += 1) {
    const x = (index % GRID_COLUMNS) * GRID_CELL_SIZE
    const y = Math.floor(index / GRID_COLUMNS) * GRID_CELL_SIZE
    const frame = frames[index]

    if (frame) {
      context.drawImage(frame, x, y, GRID_CELL_SIZE, GRID_CELL_SIZE)
    }

    context.strokeStyle = accentColor
    context.lineWidth = 2
    context.strokeRect(x + 1, y + 1, GRID_CELL_SIZE - 2, GRID_CELL_SIZE - 2)

    context.fillStyle = "rgba(0, 0, 0, 0.58)"
    context.fillRect(x + 8, y + 8, 30, 18)
    context.fillStyle = "#ffffff"
    context.font = "12px Geist Mono, monospace"
    context.fillText(String(index + 1).padStart(2, "0"), x + 13, y + 21)
  }
}

function toPoint(landmark: NormalizedLandmark | undefined): SignLandmarkPoint | undefined {
  if (!landmark) {
    return undefined
  }

  return {
    x: landmark.x,
    y: landmark.y,
    z: landmark.z,
    visibility: landmark.visibility,
  }
}

function pickLandmarks(landmarks: NormalizedLandmarkList, indices: readonly number[]): SignLandmarkPoint[] {
  return indices
    .map((index) => toPoint(landmarks[index]))
    .filter((point): point is SignLandmarkPoint => Boolean(point))
}

function buildFaceSubset(faceLandmarks: NormalizedLandmarkList): SignFaceSubset {
  return {
    leftEyebrow: pickLandmarks(faceLandmarks, FACE_SUBSET_INDICES.leftEyebrow),
    rightEyebrow: pickLandmarks(faceLandmarks, FACE_SUBSET_INDICES.rightEyebrow),
    leftEye: pickLandmarks(faceLandmarks, FACE_SUBSET_INDICES.leftEye),
    rightEye: pickLandmarks(faceLandmarks, FACE_SUBSET_INDICES.rightEye),
    lips: pickLandmarks(faceLandmarks, FACE_SUBSET_INDICES.lips),
  }
}

function normalizePoint(
  point: SignLandmarkPoint | undefined,
  centerX: number,
  centerY: number,
  centerZ: number,
  scale: number,
): SignLandmarkPoint | undefined {
  if (!point) {
    return undefined
  }

  return {
    x: (point.x - centerX) / scale,
    y: (point.y - centerY) / scale,
    z: (point.z - centerZ) / scale,
    visibility: point.visibility,
  }
}

function normalizePoints(
  points: SignLandmarkPoint[],
  centerX: number,
  centerY: number,
  centerZ: number,
  scale: number,
): SignLandmarkPoint[] {
  return points
    .map((point) => normalizePoint(point, centerX, centerY, centerZ, scale))
    .filter((point): point is SignLandmarkPoint => Boolean(point))
}

export function buildNormalizedLandmarkFrame(frame: HolisticFrameDrawingData, timestampMs: number): SignLandmarkFrame {
  const pose = {
    nose: toPoint(frame.poseLandmarks[POSE_LANDMARKS.NOSE]),
    leftShoulder: toPoint(frame.poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER]),
    rightShoulder: toPoint(frame.poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER]),
    leftElbow: toPoint(frame.poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW]),
    rightElbow: toPoint(frame.poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW]),
    leftWrist: toPoint(frame.poseLandmarks[POSE_LANDMARKS.LEFT_WRIST]),
    rightWrist: toPoint(frame.poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST]),
  }
  const face = buildFaceSubset(frame.faceLandmarks)
  const leftHand = frame.leftHandLandmarks.map((landmark) => ({
    x: landmark.x,
    y: landmark.y,
    z: landmark.z,
    visibility: landmark.visibility,
  }))
  const rightHand = frame.rightHandLandmarks.map((landmark) => ({
    x: landmark.x,
    y: landmark.y,
    z: landmark.z,
    visibility: landmark.visibility,
  }))

  const leftShoulder = pose.leftShoulder
  const rightShoulder = pose.rightShoulder
  const shoulderCenterX = leftShoulder && rightShoulder ? (leftShoulder.x + rightShoulder.x) / 2 : 0.5
  const shoulderCenterY = leftShoulder && rightShoulder ? (leftShoulder.y + rightShoulder.y) / 2 : 0.5
  const shoulderCenterZ = leftShoulder && rightShoulder ? (leftShoulder.z + rightShoulder.z) / 2 : 0
  const shoulderWidth =
    leftShoulder && rightShoulder
      ? Math.hypot(rightShoulder.x - leftShoulder.x, rightShoulder.y - leftShoulder.y, rightShoulder.z - leftShoulder.z)
      : 0
  const scale = shoulderWidth > 0.0001 ? shoulderWidth : 1

  return {
    timestampMs,
    pose: {
      nose: normalizePoint(pose.nose, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
      leftShoulder: normalizePoint(pose.leftShoulder, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
      rightShoulder: normalizePoint(pose.rightShoulder, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
      leftElbow: normalizePoint(pose.leftElbow, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
      rightElbow: normalizePoint(pose.rightElbow, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
      leftWrist: normalizePoint(pose.leftWrist, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
      rightWrist: normalizePoint(pose.rightWrist, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
    },
    leftHand: normalizePoints(leftHand, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
    rightHand: normalizePoints(rightHand, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
    face: {
      leftEyebrow: normalizePoints(face.leftEyebrow, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
      rightEyebrow: normalizePoints(face.rightEyebrow, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
      leftEye: normalizePoints(face.leftEye, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
      rightEye: normalizePoints(face.rightEye, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
      lips: normalizePoints(face.lips, shoulderCenterX, shoulderCenterY, shoulderCenterZ, scale),
    },
  }
}

function getCaptureSource(videoSource: SignVideoSource): SignCaptureSource {
  return videoSource.type
}

export function useSignLanguage({
  videoRef,
  canvasRef,
  gridCanvasRef,
  skeletonGridCanvasRef,
  videoSource = {type: "device-camera", requestedQuality: "720p"},
  enabled = true,
  showHandednessOverlay = false,
  captureSettings = DEFAULT_SIGN_CAPTURE_SETTINGS,
  captureSettingsStatus = "loaded",
  captureSettingsFallbackReason,
  captureFps = DEFAULT_CAPTURE_FPS,
  handsDownDelayMs = DEFAULT_HANDS_DOWN_DELAY_MS,
  onHandsDown,
}: UseSignLanguageOptions): UseSignLanguageResult {
  const [cameraState, setCameraState] = useState<UseSignLanguageResult["cameraState"]>("idle")
  const [error, setError] = useState<string | null>(null)
  const [isSigning, setIsSigning] = useState(false)
  const [detectedHands, setDetectedHands] = useState(0)
  const [handednessLabels, setHandednessLabels] = useState<string[]>([])
  const [capturedFrames, setCapturedFrames] = useState(0)
  const [sequenceFrames, setSequenceFrames] = useState(0)
  const [gestureHints, setGestureHints] = useState<GestureHint[]>([])
  const [latestSequence, setLatestSequence] = useState<SignSequencePayload | null>(null)
  const [actualInputWidth, setActualInputWidth] = useState<number | undefined>(undefined)
  const [actualInputHeight, setActualInputHeight] = useState<number | undefined>(undefined)
  const [actualQuality, setActualQuality] = useState<ActualCaptureQuality | undefined>(
    videoSource.type === "glasses-stream" ? videoSource.actualQuality : undefined,
  )
  const [qualityFallback, setQualityFallback] = useState<boolean | undefined>(
    videoSource.type === "glasses-stream" ? videoSource.qualityFallback : undefined,
  )
  const [droppedFrameCount, setDroppedFrameCount] = useState(0)

  const mountedRef = useRef(true)
  const runIdRef = useRef(0)
  const streamRef = useRef<MediaStream | null>(null)
  const hlsRef = useRef<{destroy: () => void} | null>(null)
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null)
  const remoteSessionUrlRef = useRef<string | null>(null)
  const remoteDriftIntervalRef = useRef<number | null>(null)
  const usingRemoteStreamRef = useRef(false)
  const holisticRef = useRef<Holistic | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const videoFrameCallbackRef = useRef<{video: HTMLVideoElement; handle: number} | null>(null)
  const droppedFrameIntervalRef = useRef<number | null>(null)
  const holisticTimingRef = useRef(createHolisticTimingWindow())
  const overlaySmoothersRef = useRef(createOverlaySmoothers())
  const trackingExperimentRef = useRef<number | null>(null)
  const [trackingConfidenceExperiment, setTrackingConfidenceExperiment] = useState<number | null>(null)
  const handsDownTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const framesRef = useRef<HTMLCanvasElement[]>([])
  const skeletonFramesRef = useRef<HTMLCanvasElement[]>([])
  const landmarkFramesRef = useRef<SignLandmarkFrame[]>([])
  const frameTimestampsRef = useRef<number[]>([])
  const captureSessionActiveRef = useRef(false)
  const lastLandmarkCaptureAtRef = useRef(0)
  const lastGridCaptureAtRef = useRef(0)
  const inFlightRef = useRef(false)
  const droppedFrameCountRef = useRef(0)
  const gestureSignatureRef = useRef("")
  const handednessSignatureRef = useRef("")
  const latestGestureHintsRef = useRef<GestureHint[]>([])
  const showHandednessOverlayRef = useRef(showHandednessOverlay)
  const actualInputWidthRef = useRef<number | undefined>(undefined)
  const actualInputHeightRef = useRef<number | undefined>(undefined)
  const videoSourceKey = buildVideoSourceKey(videoSource)

  const emitHandsDown = useEffectEvent(async (payload: SignSequencePayload) => {
    setLatestSequence(payload)
    await onHandsDown?.(payload)
  })

  // requestVideoFrameCallback fires once per decoded camera frame, so Holistic
  // never re-processes a stale frame; requestAnimationFrame is the fallback.
  const scheduleNextFrame = (video: HTMLVideoElement, callback: () => void) => {
    if (typeof video.requestVideoFrameCallback === "function") {
      videoFrameCallbackRef.current = {video, handle: video.requestVideoFrameCallback(callback)}
      return
    }

    animationFrameRef.current = requestAnimationFrame(callback)
  }

  const cancelScheduledFrame = () => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    const videoFrameCallback = videoFrameCallbackRef.current
    if (videoFrameCallback) {
      videoFrameCallback.video.cancelVideoFrameCallback?.(videoFrameCallback.handle)
      videoFrameCallbackRef.current = null
    }
  }

  const clearDroppedFrameInterval = () => {
    if (droppedFrameIntervalRef.current !== null) {
      window.clearInterval(droppedFrameIntervalRef.current)
      droppedFrameIntervalRef.current = null
    }
  }

  const startDroppedFrameInterval = () => {
    clearDroppedFrameInterval()
    droppedFrameIntervalRef.current = window.setInterval(() => {
      setDroppedFrameCount((current) =>
        current === droppedFrameCountRef.current ? current : droppedFrameCountRef.current,
      )
    }, DROPPED_FRAME_STATE_INTERVAL_MS)
  }

  const clearHandsDownTimer = () => {
    if (handsDownTimeoutRef.current) {
      clearTimeout(handsDownTimeoutRef.current)
      handsDownTimeoutRef.current = null
    }
  }

  const redrawGridCanvas = () => {
    const gridCanvas = gridCanvasRef.current
    if (gridCanvas) {
      redrawFrameGridCanvas(gridCanvas, framesRef.current, "rgba(30, 144, 255, 0.35)")
    }

    const skeletonGridCanvas = skeletonGridCanvasRef.current
    if (skeletonGridCanvas) {
      redrawFrameGridCanvas(skeletonGridCanvas, skeletonFramesRef.current, "rgba(255, 102, 0, 0.45)")
    }
  }

  const clearSequence = () => {
    framesRef.current = []
    skeletonFramesRef.current = []
    landmarkFramesRef.current = []
    frameTimestampsRef.current = []
    captureSessionActiveRef.current = false
    lastLandmarkCaptureAtRef.current = 0
    lastGridCaptureAtRef.current = 0
    latestGestureHintsRef.current = []
    gestureSignatureRef.current = ""
    handednessSignatureRef.current = ""
    setCapturedFrames(0)
    setSequenceFrames(0)
    setGestureHints([])
    setLatestSequence(null)

    const gridCanvas = gridCanvasRef.current
    const gridContext = gridCanvas?.getContext("2d")
    if (gridCanvas && gridContext) {
      ensureCanvasSize(gridCanvas, GRID_COLUMNS * GRID_CELL_SIZE, GRID_ROWS * GRID_CELL_SIZE)
      gridContext.clearRect(0, 0, gridCanvas.width, gridCanvas.height)
    }

    const skeletonGridCanvas = skeletonGridCanvasRef.current
    const skeletonGridContext = skeletonGridCanvas?.getContext("2d")
    if (skeletonGridCanvas && skeletonGridContext) {
      ensureCanvasSize(skeletonGridCanvas, GRID_COLUMNS * GRID_CELL_SIZE, GRID_ROWS * GRID_CELL_SIZE)
      skeletonGridContext.clearRect(0, 0, skeletonGridCanvas.width, skeletonGridCanvas.height)
    }
  }

  const buildSequencePayload = (): SignSequencePayload | null => {
    const gridCanvas = gridCanvasRef.current
    if (!gridCanvas || framesRef.current.length === 0) {
      return null
    }

    const dataUrl = gridCanvas.toDataURL("image/jpeg", 0.88)
    const skeletonGridCanvas = skeletonGridCanvasRef.current
    const skeletonDataUrl =
      skeletonGridCanvas && skeletonFramesRef.current.length > 0
        ? skeletonGridCanvas.toDataURL("image/png")
        : undefined
    const requestedQuality = videoSource.requestedQuality
    const actualQualityForPayload =
      videoSource.type === "glasses-stream"
        ? videoSource.actualQuality || actualQuality || deriveActualCaptureQuality(actualInputWidthRef.current, actualInputHeightRef.current)
        : actualQuality || deriveActualCaptureQuality(actualInputWidthRef.current, actualInputHeightRef.current)
    const qualityFallbackForPayload =
      videoSource.type === "glasses-stream"
        ? videoSource.qualityFallback ?? qualityFallback ?? inferQualityFallback(requestedQuality, actualQualityForPayload)
        : qualityFallback ?? inferQualityFallback(requestedQuality, actualQualityForPayload)

    return {
      base64Grid: dataUrl.split(",")[1] ?? "",
      base64SkeletonGrid: skeletonDataUrl?.split(",")[1] ?? undefined,
      dataUrl,
      skeletonDataUrl,
      mimeType: "image/jpeg",
      skeletonMimeType: skeletonDataUrl ? "image/png" : undefined,
      frameCount: framesRef.current.length,
      gridColumns: GRID_COLUMNS,
      gridRows: GRID_ROWS,
      capturedAt: Date.now(),
      gestureHints: latestGestureHintsRef.current,
      landmarkSequence: landmarkFramesRef.current,
      frameTimestamps: frameTimestampsRef.current,
      sequenceDurationMs: computeSequenceDurationMs(frameTimestampsRef.current),
      sequenceFrameCount: landmarkFramesRef.current.length,
      source: getCaptureSource(videoSource),
      extractorVersion: experimentExtractorVersion(trackingExperimentRef.current),
      requestedQuality,
      actualQuality: actualQualityForPayload,
      qualityFallback: qualityFallbackForPayload,
      actualInputWidth: actualInputWidthRef.current,
      actualInputHeight: actualInputHeightRef.current,
      skeletonRenderMode: "diagnostic",
      captureSettingsStatus,
      captureSettingsFallbackReason,
      droppedFrameCount: droppedFrameCountRef.current,
    }
  }

  const stop = async () => {
    runIdRef.current += 1
    clearHandsDownTimer()
    cancelScheduledFrame()
    clearDroppedFrameInterval()

    inFlightRef.current = false

    if (holisticRef.current) {
      const holistic = holisticRef.current
      holisticRef.current = null
      try {
        await holistic.close()
      } catch {
        // Ignore shutdown errors from the WASM runtime.
      }
    }

    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop()
      }
      streamRef.current = null
    }

    clearRemoteDriftInterval(remoteDriftIntervalRef)

    if (hlsRef.current) {
      hlsRef.current.destroy()
      hlsRef.current = null
    }

    destroyPeerConnection(peerConnectionRef, remoteSessionUrlRef, remoteDriftIntervalRef)
    usingRemoteStreamRef.current = false

    const video = videoRef.current
    if (video) {
      resetVideoElement(video)
    }

    const overlayCanvas = canvasRef.current
    const overlayContext = overlayCanvas?.getContext("2d")
    if (overlayCanvas && overlayContext) {
      overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height)
    }

    if (mountedRef.current) {
      setCameraState("idle")
      setError(null)
      setDetectedHands(0)
      setHandednessLabels([])
      setIsSigning(false)
      setGestureHints([])
    }
  }

  const restart = async () => {
    await stop()

    if (!enabled || captureSettingsStatus === "loading") {
      return
    }

    const remoteSources =
      videoSource.type === "glasses-stream"
        ? videoSource.sources.map((source) => ({
            ...source,
            url: source.url.trim(),
          }))
        : []
    const usingRemoteStream = videoSource.type === "glasses-stream" && remoteSources.some((source) => source.url.length > 0)

    if (!usingRemoteStream && !navigator.mediaDevices?.getUserMedia) {
      setCameraState("error")
      setError("This browser does not support camera access.")
      return
    }

    const video = videoRef.current
    const overlayCanvas = canvasRef.current
    const gridCanvas = gridCanvasRef.current
    const skeletonGridCanvas = skeletonGridCanvasRef.current

    if (!video || !overlayCanvas || !gridCanvas || !skeletonGridCanvas) {
      setCameraState("error")
      setError("Camera view is not mounted yet.")
      return
    }

    const runId = runIdRef.current + 1
    runIdRef.current = runId

    setCameraState("requesting")
    setError(null)
    droppedFrameCountRef.current = 0
    setDroppedFrameCount(0)
    actualInputWidthRef.current = undefined
    actualInputHeightRef.current = undefined
    setActualInputWidth(undefined)
    setActualInputHeight(undefined)
    setActualQuality(videoSource.type === "glasses-stream" ? videoSource.actualQuality : undefined)
    setQualityFallback(videoSource.type === "glasses-stream" ? videoSource.qualityFallback : undefined)
    clearSequence()
    redrawGridCanvas()

    usingRemoteStreamRef.current = usingRemoteStream

    try {
      video.muted = true
      video.playsInline = true

      if (usingRemoteStream) {
        await attachRemoteVideoSourceWithRetry(
          video,
          remoteSources,
          hlsRef,
          peerConnectionRef,
          remoteSessionUrlRef,
          remoteDriftIntervalRef,
        )
        streamRef.current = video.srcObject instanceof MediaStream ? video.srcObject : null
      } else {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: "user",
            width: {ideal: videoSource.requestedQuality === "1080p" ? 1920 : 1280},
            height: {ideal: videoSource.requestedQuality === "1080p" ? 1080 : 720},
            frameRate: {ideal: 30},
          },
        })

        if (runId !== runIdRef.current) {
          for (const track of stream.getTracks()) {
            track.stop()
          }
          return
        }

        streamRef.current = stream
        video.srcObject = stream
      }

      if (runId !== runIdRef.current) {
        return
      }

      await video.play()

      actualInputWidthRef.current = video.videoWidth || undefined
      actualInputHeightRef.current = video.videoHeight || undefined
      setActualInputWidth(actualInputWidthRef.current)
      setActualInputHeight(actualInputHeightRef.current)
      const deliveredQuality =
        videoSource.type === "glasses-stream"
          ? videoSource.actualQuality || deriveActualCaptureQuality(actualInputWidthRef.current, actualInputHeightRef.current)
          : deriveActualCaptureQuality(actualInputWidthRef.current, actualInputHeightRef.current)
      setActualQuality(deliveredQuality)
      setQualityFallback(
        videoSource.type === "glasses-stream"
          ? videoSource.qualityFallback ?? inferQualityFallback(videoSource.requestedQuality, deliveredQuality)
          : inferQualityFallback(videoSource.requestedQuality, deliveredQuality),
      )

      ensureCanvasSize(overlayCanvas, video.videoWidth || 1280, video.videoHeight || 720)
      redrawGridCanvas()

      const holistic = new Holistic({
        locateFile: (asset) => `/mediapipe/holistic/${asset}`,
      })

      const trackingExperiment = readTrackingConfidenceExperiment()
      trackingExperimentRef.current = trackingExperiment
      setTrackingConfidenceExperiment(trackingExperiment)
      if (trackingExperiment !== null) {
        console.warn(
          `[SIGN_EXPERIMENT] minTrackingConfidence=${trackingExperiment} (default ${HOLISTIC_OPTIONS.minTrackingConfidence}); payloads stamped ${experimentExtractorVersion(trackingExperiment)}`,
        )
      }

      holistic.setOptions({
        ...HOLISTIC_OPTIONS,
        modelComplexity: captureSettings.holisticModelComplexity,
        ...(trackingExperiment !== null ? {minTrackingConfidence: trackingExperiment} : {}),
      })

      holistic.onResults((results: Results) => {
        if (runId !== runIdRef.current || !mountedRef.current) {
          return
        }

        try {
          ensureCanvasSize(overlayCanvas, video.videoWidth || overlayCanvas.width, video.videoHeight || overlayCanvas.height)

          const hands = buildDetectedHands(results)
          const nextHandednessLabels = buildHandednessLabels(hands)
          const frame: HolisticFrameDrawingData = {
            poseLandmarks: results.poseLandmarks ?? [],
            faceLandmarks: results.faceLandmarks ?? [],
            leftHandLandmarks: results.leftHandLandmarks ?? [],
            rightHandLandmarks: results.rightHandLandmarks ?? [],
            handednessLabels: nextHandednessLabels,
          }

          const smoothedAt = performance.now()
          const smoothers = overlaySmoothersRef.current
          const displayFrame: HolisticFrameDrawingData = {
            poseLandmarks: smoothers.pose.smooth(frame.poseLandmarks, smoothedAt),
            faceLandmarks: smoothers.face.smooth(frame.faceLandmarks, smoothedAt),
            leftHandLandmarks: smoothers.leftHand.smooth(frame.leftHandLandmarks, smoothedAt),
            rightHandLandmarks: smoothers.rightHand.smooth(frame.rightHandLandmarks, smoothedAt),
            handednessLabels: frame.handednessLabels,
          }

          drawSkeleton(overlayCanvas, displayFrame, {
            renderMode: "presentation",
            showHandednessOverlay: showHandednessOverlayRef.current,
            mirrorOverlayText: videoSource.type === "device-camera",
            maskOccludedFace: true,
          })

          if (video.videoWidth > 0 && video.videoHeight > 0) {
            const nextWidth = video.videoWidth
            const nextHeight = video.videoHeight
            if (actualInputWidthRef.current !== nextWidth) {
              actualInputWidthRef.current = nextWidth
              setActualInputWidth(nextWidth)
            }
            if (actualInputHeightRef.current !== nextHeight) {
              actualInputHeightRef.current = nextHeight
              setActualInputHeight(nextHeight)
            }
            if (videoSource.type === "device-camera") {
              const nextQuality = deriveActualCaptureQuality(nextWidth, nextHeight)
              setActualQuality((current) => (current === nextQuality ? current : nextQuality))
              const nextFallback = inferQualityFallback(videoSource.requestedQuality, nextQuality)
              setQualityFallback((current) => (current === nextFallback ? current : nextFallback))
            }
          }

          const handCount = hands.length
          const hasHands = handCount > 0

          setDetectedHands((current) => (current === handCount ? current : handCount))
          setIsSigning((current) => (current === hasHands ? current : hasHands))

          const handednessSignature = nextHandednessLabels.join("|")
          if (handednessSignatureRef.current !== handednessSignature) {
            handednessSignatureRef.current = handednessSignature
            setHandednessLabels(nextHandednessLabels)
          }

          if (hasHands) {
            if (!captureSessionActiveRef.current) {
              clearSequence()
              captureSessionActiveRef.current = true
              redrawGridCanvas()
            }

            clearHandsDownTimer()

            const now = performance.now()
            const lmInterval = landmarkIntervalMs(captureFps)
            const gInterval = gridIntervalMs(captureFps)
            const validDimensions = video.videoWidth > 0 && video.videoHeight > 0

            const landmarkReady = validDimensions && now - lastLandmarkCaptureAtRef.current >= lmInterval
            const gridReady = validDimensions && now - lastGridCaptureAtRef.current >= gInterval

            if (gridReady) {
              const nextHints = estimateGestureHints(hands)
              latestGestureHintsRef.current = nextHints

              const nextSignature = nextHints.map((hint) => `${hint.name}:${hint.score.toFixed(1)}`).join("|")
              if (gestureSignatureRef.current !== nextSignature) {
                gestureSignatureRef.current = nextSignature
                setGestureHints(nextHints)
              }
            }

            if (landmarkReady || gridReady) {
              // Freshness gate: when consuming a remote glasses stream, skip
              // capture ticks whose current frame is stale beyond
              // CAPTURE_DRIFT_SEEK_THRESHOLD_S. Snap the playhead to the live
              // edge so the next RAF lands on a fresh frame, guaranteeing the
              // labeled grid never contains pre-drift content.
              if (usingRemoteStreamRef.current) {
                const hlsInstance = hlsRef.current as
                  | {destroy: () => void; liveSyncPosition?: number}
                  | null
                let liveEdge: number | undefined
                if (
                  hlsInstance &&
                  typeof hlsInstance.liveSyncPosition === "number" &&
                  Number.isFinite(hlsInstance.liveSyncPosition)
                ) {
                  liveEdge = hlsInstance.liveSyncPosition
                } else if (video.buffered.length > 0) {
                  liveEdge = video.buffered.end(video.buffered.length - 1)
                }
                if (
                  typeof liveEdge === "number" &&
                  Number.isFinite(liveEdge) &&
                  liveEdge - video.currentTime > CAPTURE_DRIFT_SEEK_THRESHOLD_S
                ) {
                  video.currentTime = liveEdge
                  return
                }
              }

              if (gridReady) {
                lastGridCaptureAtRef.current = now

                framesRef.current = appendRollingWindow(framesRef.current, createFrameCanvas(video), MAX_GRID_FRAMES)
                skeletonFramesRef.current = appendRollingWindow(
                  skeletonFramesRef.current,
                  createSkeletonFrameCanvas(frame),
                  MAX_GRID_FRAMES,
                )

                redrawGridCanvas()
                setCapturedFrames(framesRef.current.length)
              }

              if (landmarkReady) {
                lastLandmarkCaptureAtRef.current = now

                const timestampMs = Date.now()
                landmarkFramesRef.current = appendRollingWindow(
                  landmarkFramesRef.current,
                  buildNormalizedLandmarkFrame(frame, timestampMs),
                  MAX_SEQUENCE_FRAMES,
                )
                frameTimestampsRef.current = appendRollingWindow(frameTimestampsRef.current, timestampMs, MAX_SEQUENCE_FRAMES)

                setSequenceFrames(landmarkFramesRef.current.length)
              }
            }

            return
          }

          if (gestureSignatureRef.current !== "" || latestGestureHintsRef.current.length > 0) {
            gestureSignatureRef.current = ""
            latestGestureHintsRef.current = []
            setGestureHints([])
          }

          if (!captureSessionActiveRef.current || handsDownTimeoutRef.current) {
            return
          }

          handsDownTimeoutRef.current = setTimeout(() => {
            handsDownTimeoutRef.current = null

            if (!captureSessionActiveRef.current) {
              return
            }

            // Duration gate: reject sequences shorter than MIN_SEQUENCE_DURATION_MS
            // to filter accidental hand flashes. Uses wall-clock duration so the
            // behavior stays stable even if MediaPipe throughput drops below target FPS.
            const durationMs = computeSequenceDurationMs(frameTimestampsRef.current)
            if (durationMs < MIN_SEQUENCE_DURATION_MS) {
              captureSessionActiveRef.current = false
              return
            }

            captureSessionActiveRef.current = false
            const payload = buildSequencePayload()
            if (payload) {
              void emitHandsDown(payload)
            }
          }, handsDownDelayMs)
        } catch (frameError) {
          console.error("Sign tracking frame failed", frameError)
          clearHandsDownTimer()
          captureSessionActiveRef.current = false
          setCameraState("error")
          setError(frameError instanceof Error ? frameError.message : "Sign tracking frame failed.")
        }
      })

      holisticRef.current = holistic
      await holistic.initialize()

      if (runId !== runIdRef.current) {
        await holistic.close()
        return
      }

      setCameraState("ready")

      holisticTimingRef.current = createHolisticTimingWindow()
      overlaySmoothersRef.current = createOverlaySmoothers()
      const timingDebugEnabled = isSignTimingDebugEnabled()

      const tick = () => {
        if (runId !== runIdRef.current) {
          return
        }

        scheduleNextFrame(video, tick)

        const holistic = holisticRef.current
        const videoElement = videoRef.current

        if (!holistic || !videoElement) {
          return
        }

        if (inFlightRef.current) {
          droppedFrameCountRef.current += 1
          return
        }

        if (videoElement.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
          return
        }

        const holisticStartedAt = performance.now()
        inFlightRef.current = true
        void holistic
          .send({image: videoElement})
          .catch((sendError) => {
            if (!mountedRef.current || runId !== runIdRef.current) {
              return
            }

            setCameraState("error")
            setError(sendError instanceof Error ? sendError.message : "Holistic tracking failed.")
          })
          .finally(() => {
            const summary = recordHolisticTiming(holisticTimingRef.current, performance.now() - holisticStartedAt)
            if (summary && (timingDebugEnabled || showHandednessOverlayRef.current)) {
              console.log(
                `[MP_TIMING][HOLISTIC] avg ${summary.averageMs.toFixed(1)}ms max ${summary.maxMs.toFixed(1)}ms over ${summary.count} frames, ${droppedFrameCountRef.current} dropped`,
              )
            }
            inFlightRef.current = false
          })
      }

      startDroppedFrameInterval()
      tick()
    } catch (cameraError) {
      if (!mountedRef.current) {
        return
      }

      setCameraState("error")
      setError(
        cameraError instanceof Error
          ? cameraError.message
          : usingRemoteStream
            ? "Could not open the Mentra glasses video stream."
            : "Could not access the camera.",
      )
    }
  }

  useEffect(() => {
    showHandednessOverlayRef.current = showHandednessOverlay
  }, [showHandednessOverlay])

  useEffect(() => {
    if (videoSource.type !== "glasses-stream") {
      return
    }

    setActualQuality(videoSource.actualQuality)
    setQualityFallback(videoSource.qualityFallback)
  }, [videoSource])

  useEffect(() => {
    mountedRef.current = true

    if (enabled && captureSettingsStatus !== "loading") {
      void restart()
    } else {
      void stop()
    }

    return () => {
      mountedRef.current = false
      void stop()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, captureFps, handsDownDelayMs, videoSourceKey, captureSettingsStatus, captureSettings.holisticModelComplexity])

  return {
    cameraState,
    error,
    videoSource: videoSource.type,
    isSigning,
    detectedHands,
    handednessLabels,
    capturedFrames,
    sequenceFrames,
    gestureHints,
    latestSequence,
    requestedQuality: videoSource.requestedQuality,
    actualQuality,
    qualityFallback,
    actualInputWidth,
    actualInputHeight,
    captureSettingsStatus,
    captureSettingsFallbackReason,
    droppedFrameCount,
    trackingConfidenceExperiment,
    restart,
    stop,
    clearSequence,
  }
}
