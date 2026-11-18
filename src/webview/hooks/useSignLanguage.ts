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
import {
  experimentExtractorVersion,
  isTrackingConfidenceExperiment,
  parseTrackingConfidenceExperiment,
} from "../../sign/tracking-confidence-experiment"

export {experimentExtractorVersion, isTrackingConfidenceExperiment, parseTrackingConfidenceExperiment}
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

export type SignVideoSource = {
  type: "device-camera"
  requestedQuality?: RequestedCaptureQuality
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

function resetVideoElement(video: HTMLVideoElement): void {
  video.pause()
  video.srcObject = null
  video.removeAttribute("src")
  video.load()
}

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
  const [actualQuality, setActualQuality] = useState<ActualCaptureQuality | undefined>(undefined)
  const [qualityFallback, setQualityFallback] = useState<boolean | undefined>(undefined)
  const [droppedFrameCount, setDroppedFrameCount] = useState(0)

  const mountedRef = useRef(true)
  const runIdRef = useRef(0)
  const streamRef = useRef<MediaStream | null>(null)
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
      actualQuality || deriveActualCaptureQuality(actualInputWidthRef.current, actualInputHeightRef.current)
    const qualityFallbackForPayload = qualityFallback ?? inferQualityFallback(requestedQuality, actualQualityForPayload)

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

    if (!navigator.mediaDevices?.getUserMedia) {
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
    setActualQuality(undefined)
    setQualityFallback(undefined)
    clearSequence()
    redrawGridCanvas()

    try {
      video.muted = true
      video.playsInline = true

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

      if (runId !== runIdRef.current) {
        return
      }

      await video.play()

      actualInputWidthRef.current = video.videoWidth || undefined
      actualInputHeightRef.current = video.videoHeight || undefined
      setActualInputWidth(actualInputWidthRef.current)
      setActualInputHeight(actualInputHeightRef.current)
      const deliveredQuality = deriveActualCaptureQuality(actualInputWidthRef.current, actualInputHeightRef.current)
      setActualQuality(deliveredQuality)
      setQualityFallback(inferQualityFallback(videoSource.requestedQuality, deliveredQuality))

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
      setError(cameraError instanceof Error ? cameraError.message : "Could not access the camera.")
    }
  }

  useEffect(() => {
    showHandednessOverlayRef.current = showHandednessOverlay
  }, [showHandednessOverlay])

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
