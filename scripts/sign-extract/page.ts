import {Holistic, type Results} from "@mediapipe/holistic"
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
  error?: string
}

declare global {
  interface Window {
    extractorReady?: boolean
    extractorError?: string
    extractClip?: (videoUrl: string) => Promise<ClipExtraction>
  }
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
  const seeked = waitFor(video, "seeked")
  video.currentTime = timeSeconds
  await seeked
}

async function runHolistic(): Promise<Results> {
  const results = new Promise<Results>((resolve) => {
    pendingResults = resolve
  })
  await holistic.send({image: video})
  return results
}

async function extractClip(videoUrl: string): Promise<ClipExtraction> {
  const empty: ClipExtraction = {
    extractorVersion: MEDIAPIPE_EXTRACTOR_VERSION,
    framesSampled: 0,
    sequencesFound: 0,
    landmarkSequence: [],
    frameTimestamps: [],
    videoWidth: 0,
    videoHeight: 0,
  }

  let objectUrl = ""
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
    if (!Number.isFinite(video.duration) || video.duration <= 0) {
      return {...empty, error: `video has no usable duration (${video.duration})`}
    }

    holistic.reset()
    const stepSeconds = landmarkIntervalMs(DEFAULT_CAPTURE_FPS) / 1000
    const samples: OfflineFrame<SignLandmarkFrame>[] = []
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
      samples.push({
        timestampMs,
        hasHands: buildDetectedHands(results).length > 0,
        frame: buildNormalizedLandmarkFrame(frame, timestampMs),
      })
    }

    const sequences = segmentHandSequences(samples)
    const best = pickLongestSequence(sequences)
    return {
      ...empty,
      framesSampled: samples.length,
      sequencesFound: sequences.length,
      landmarkSequence: best?.frames ?? [],
      frameTimestamps: best?.frameTimestamps ?? [],
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
    }
  } catch (error) {
    return {...empty, error: error instanceof Error ? error.message : String(error)}
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
