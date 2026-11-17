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
  type NormalizedLandmarkList,
  type Results,
} from "@mediapipe/holistic"
import {DEFAULT_CAPTURE_FPS, landmarkIntervalMs} from "../../src/sign/capture-config"
import {MEDIAPIPE_EXTRACTOR_VERSION} from "../../src/sign/extractor-version"
import {pickLongestSequence, segmentHandSequences, type OfflineFrame} from "../../src/sign/offline-sequence"
import type {SignLandmarkFrame} from "../../src/sign/types"
import {
  HOLISTIC_OPTIONS,
  buildDetectedHands,
  buildNormalizedLandmarkFrame,
  type HolisticFrameDrawingData,
} from "../../src/webview/hooks/useSignLanguage"

export interface ClipExtraction {
  extractorVersion: string
  framesSampled: number
  sequencesFound: number
  landmarkSequence: SignLandmarkFrame[]
  frameTimestamps: number[]
  videoWidth: number
  videoHeight: number
  decoded: boolean
  leftHandFrames: number
  rightHandFrames: number
  bothHandsFrames: number
  skeletonPng?: string
  error?: string
}

export interface ExtractClipOptions {
  skeleton?: boolean
}

declare global {
  interface Window {
    extractorReady?: boolean
    extractorError?: string
    extractClip?: (videoUrl: string, options?: ExtractClipOptions) => Promise<ClipExtraction>
  }
}

const ARM_POSE_INDICES = new Set<number>([
  POSE_LANDMARKS.LEFT_SHOULDER,
  POSE_LANDMARKS.RIGHT_SHOULDER,
  POSE_LANDMARKS.LEFT_ELBOW,
  POSE_LANDMARKS.RIGHT_ELBOW,
  POSE_LANDMARKS.LEFT_WRIST,
  POSE_LANDMARKS.RIGHT_WRIST,
])

function cloneLandmarks(landmarks: NormalizedLandmarkList | undefined): NormalizedLandmarkList {
  return (landmarks ?? []).map((landmark) => ({
    x: landmark.x,
    y: landmark.y,
    z: landmark.z,
    visibility: landmark.visibility,
  }))
}

function countKeptHands(frames: SignLandmarkFrame[]) {
  let leftHandFrames = 0
  let rightHandFrames = 0
  let bothHandsFrames = 0
  for (const frame of frames) {
    const hasLeft = frame.leftHand.length > 0
    const hasRight = frame.rightHand.length > 0
    if (hasLeft) leftHandFrames += 1
    if (hasRight) rightHandFrames += 1
    if (hasLeft && hasRight) bothHandsFrames += 1
  }
  return {leftHandFrames, rightHandFrames, bothHandsFrames}
}

function drawConnections(
  context: CanvasRenderingContext2D,
  landmarks: NormalizedLandmarkList,
  connections: Array<[number, number]>,
  width: number,
  height: number,
  lineWidth: number,
  color: string,
) {
  context.strokeStyle = color
  context.lineWidth = lineWidth
  for (const [startIndex, endIndex] of connections) {
    const start = landmarks[startIndex]
    const end = landmarks[endIndex]
    if (!start || !end) continue
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
) {
  context.fillStyle = color
  for (const landmark of landmarks) {
    if (!landmark) continue
    context.beginPath()
    context.arc(landmark.x * width, landmark.y * height, radius, 0, Math.PI * 2)
    context.fill()
  }
}

function renderSkeletonPng(drawing: HolisticFrameDrawingData, sourceWidth: number, sourceHeight: number): string {
  const maxEdge = 960
  const scale = Math.min(1, maxEdge / Math.max(sourceWidth, sourceHeight, 1))
  const width = Math.max(1, Math.round((sourceWidth || 640) * scale))
  const height = Math.max(1, Math.round((sourceHeight || 480) * scale))
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext("2d")
  if (!context) return ""

  context.fillStyle = "#101418"
  context.fillRect(0, 0, width, height)
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    context.drawImage(video, 0, 0, width, height)
  }

  context.lineCap = "round"
  context.lineJoin = "round"
  context.shadowColor = "rgba(0, 0, 0, 0.85)"
  context.shadowBlur = 3
  const lineWidth = Math.max(2, width / 220)
  const jointRadius = Math.max(2.5, width / 140)
  const armConnections = POSE_CONNECTIONS.filter(
    ([startIndex, endIndex]) => ARM_POSE_INDICES.has(startIndex) && ARM_POSE_INDICES.has(endIndex),
  )
  drawConnections(context, drawing.poseLandmarks, armConnections, width, height, lineWidth, "#1e90ff")
  drawLandmarks(
    context,
    drawing.poseLandmarks.filter((_, index) => ARM_POSE_INDICES.has(index)),
    width,
    height,
    jointRadius,
    "#ffffff",
  )
  for (const [hand, wristIndex] of [
    [drawing.leftHandLandmarks, POSE_LANDMARKS.LEFT_WRIST],
    [drawing.rightHandLandmarks, POSE_LANDMARKS.RIGHT_WRIST],
  ] as const) {
    if (!hand.length) continue
    drawConnections(context, hand, HAND_CONNECTIONS, width, height, lineWidth, "#1e90ff")
    drawLandmarks(context, hand, width, height, jointRadius, "#ffffff")
    const wrist = drawing.poseLandmarks[wristIndex]
    if (wrist && hand[0]) {
      drawConnections(context, [wrist, hand[0]], [[0, 1]], width, height, lineWidth, "#1e90ff")
    }
  }
  for (const connectionSet of [
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_LIPS,
  ]) {
    drawConnections(context, drawing.faceLandmarks, connectionSet, width, height, Math.max(1.2, lineWidth * 0.65), "#ffffff")
  }
  return canvas.toDataURL("image/png")
}

function chooseSkeletonSample(
  kept: SignLandmarkFrame[],
  drawings: Map<number, HolisticFrameDrawingData>,
  framesByTimestamp: Map<number, SignLandmarkFrame>,
): {timestampMs: number; drawing: HolisticFrameDrawingData} | null {
  const keptTimestamps = kept.map((frame) => frame.timestampMs).filter((timestamp) => drawings.has(timestamp))
  const pool = keptTimestamps.length ? keptTimestamps : [...drawings.keys()]
  if (!pool.length) return null
  const both = pool.filter((timestamp) => {
    const frame = framesByTimestamp.get(timestamp)
    return Boolean(frame && frame.leftHand.length > 0 && frame.rightHand.length > 0)
  })
  const ranked = both.length ? both : pool
  const timestampMs = ranked[Math.floor(ranked.length / 2)]
  const drawing = drawings.get(timestampMs)
  return drawing ? {timestampMs, drawing} : null
}

const holistic = new Holistic({locateFile: (asset) => `/mediapipe/holistic/${asset}`})
holistic.setOptions({...HOLISTIC_OPTIONS})

let pendingResults: ((results: Results) => void) | null = null
holistic.onResults((results) => {
  pendingResults?.(results)
  pendingResults = null
})

const video = document.createElement("video")
video.muted = true
video.playsInline = true
video.preload = "auto"
document.body.appendChild(video)

function waitFor(target: HTMLMediaElement, eventName: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const onEvent = () => {
      target.removeEventListener("error", onError)
      resolve()
    }
    const onError = () => {
      target.removeEventListener(eventName, onEvent)
      reject(new Error(target.error?.message || `video error (code ${target.error?.code ?? "unknown"})`))
    }
    target.addEventListener(eventName, onEvent, {once: true})
    target.addEventListener("error", onError, {once: true})
  })
}

async function seek(timeSeconds: number): Promise<void> {
  if (Number.isFinite(video.currentTime) && Math.abs(video.currentTime - timeSeconds) < 0.0005) return
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      video.removeEventListener("seeked", onSeeked)
      video.removeEventListener("error", onError)
      reject(new Error(`seek timed out at ${timeSeconds.toFixed(3)}s`))
    }, 10_000)
    const onSeeked = () => {
      clearTimeout(timer)
      video.removeEventListener("error", onError)
      resolve()
    }
    const onError = () => {
      clearTimeout(timer)
      video.removeEventListener("seeked", onSeeked)
      reject(new Error(video.error?.message || `video error (code ${video.error?.code ?? "unknown"})`))
    }
    video.addEventListener("seeked", onSeeked, {once: true})
    video.addEventListener("error", onError, {once: true})
    video.currentTime = timeSeconds
  })
}

async function runHolistic(): Promise<Results> {
  const results = new Promise<Results>((resolve) => {
    pendingResults = resolve
  })
  await holistic.send({image: video})
  return results
}

async function extractClip(videoUrl: string, options?: ExtractClipOptions): Promise<ClipExtraction> {
  const empty: ClipExtraction = {
    extractorVersion: MEDIAPIPE_EXTRACTOR_VERSION,
    framesSampled: 0,
    sequencesFound: 0,
    landmarkSequence: [],
    frameTimestamps: [],
    videoWidth: 0,
    videoHeight: 0,
    decoded: false,
    leftHandFrames: 0,
    rightHandFrames: 0,
    bothHandsFrames: 0,
  }

  let objectUrl = ""
  let decoded = false
  try {
    // Whole-file blob so seeking never depends on HTTP range support.
    const response = await fetch(videoUrl)
    if (!response.ok) {
      return {...empty, error: `fetch failed with HTTP ${response.status}`}
    }
    objectUrl = URL.createObjectURL(await response.blob())
    const loaded = waitFor(video, "loadeddata")
    video.src = objectUrl
    video.load()
    await loaded
    decoded = video.videoWidth > 0 && video.videoHeight > 0
    if (!decoded || !Number.isFinite(video.duration) || video.duration <= 0) {
      return {
        ...empty,
        decoded,
        videoWidth: video.videoWidth || 0,
        videoHeight: video.videoHeight || 0,
        error: `video has no usable duration (${video.duration})`,
      }
    }

    holistic.reset()
    const stepSeconds = landmarkIntervalMs(DEFAULT_CAPTURE_FPS) / 1000
    const samples: OfflineFrame<SignLandmarkFrame>[] = []
    const drawings = new Map<number, HolisticFrameDrawingData>()
    const framesByTimestamp = new Map<number, SignLandmarkFrame>()
    for (let time = 0; time < video.duration; time += stepSeconds) {
      await seek(time)
      const results = await runHolistic()
      const frame: HolisticFrameDrawingData = {
        poseLandmarks: results.poseLandmarks ?? [],
        faceLandmarks: results.faceLandmarks ?? [],
        leftHandLandmarks: results.leftHandLandmarks ?? [],
        rightHandLandmarks: results.rightHandLandmarks ?? [],
        handednessLabels: [],
      }
      const timestampMs = Math.round(time * 1000)
      const normalized = buildNormalizedLandmarkFrame(frame, timestampMs)
      samples.push({
        timestampMs,
        hasHands: buildDetectedHands(results).length > 0,
        frame: normalized,
      })
      framesByTimestamp.set(timestampMs, normalized)
      if (options?.skeleton) {
        drawings.set(timestampMs, {
          poseLandmarks: cloneLandmarks(frame.poseLandmarks),
          faceLandmarks: cloneLandmarks(frame.faceLandmarks),
          leftHandLandmarks: cloneLandmarks(frame.leftHandLandmarks),
          rightHandLandmarks: cloneLandmarks(frame.rightHandLandmarks),
          handednessLabels: [],
        })
      }
    }

    const sequences = segmentHandSequences(samples)
    const best = pickLongestSequence(sequences)
    const landmarkSequence = best?.frames ?? []
    const handCounts = countKeptHands(landmarkSequence)
    let skeletonPng: string | undefined
    if (options?.skeleton) {
      const chosen = chooseSkeletonSample(landmarkSequence, drawings, framesByTimestamp)
      if (chosen) {
        try {
          await seek(chosen.timestampMs / 1000)
          skeletonPng = renderSkeletonPng(chosen.drawing, video.videoWidth, video.videoHeight) || undefined
        } catch {
          skeletonPng = undefined
        }
      }
    }
    return {
      ...empty,
      decoded,
      framesSampled: samples.length,
      sequencesFound: sequences.length,
      landmarkSequence,
      frameTimestamps: best?.frameTimestamps ?? [],
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      ...handCounts,
      ...(skeletonPng ? {skeletonPng} : {}),
    }
  } catch (error) {
    return {
      ...empty,
      decoded,
      videoWidth: video.videoWidth || 0,
      videoHeight: video.videoHeight || 0,
      error: error instanceof Error ? error.message : String(error),
    }
  } finally {
    video.removeAttribute("src")
    video.load()
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl)
    }
  }
}

window.extractClip = extractClip
holistic
  .initialize()
  .then(() => {
    window.extractorReady = true
  })
  .catch((error: unknown) => {
    window.extractorError = error instanceof Error ? error.message : String(error)
  })
