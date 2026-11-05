import {useEffect, useRef, useState} from "react"
import {
  DEFAULT_CAPTURE_FPS,
  MAX_GRID_FRAMES,
  MAX_SEQUENCE_FRAMES,
  effectiveGridFps,
  windowLabel,
} from "../../sign/capture-config"
import {CORE_SIGN_SAMPLE_TARGET} from "../../sign/core-signs"
import * as confidenceDisplay from "./confidence-display"
import {
  useSignLanguage,
  type RemoteSignPlaybackSource,
  type SignSequencePayload,
  type SignVideoSource,
} from "../hooks/useSignLanguage"
import type {CameraStreamState, TranslationEntry} from "../hooks/useTranslationStream"
import type {SignClassifierAlternative, SignClassifierResult} from "../../sign/types"
import type {
  ActualCaptureQuality,
  CaptureSettingsFallbackReason,
  CaptureSettingsStatus,
  RequestedCaptureQuality,
  SignCaptureSettings,
  SkeletonRenderMode,
} from "../../sign/capture-settings"

const getConfidenceDisplay =
  typeof confidenceDisplay.getConfidenceDisplay === "function"
    ? confidenceDisplay.getConfidenceDisplay
    : (confidence: number) => ({
        level: confidence >= 0.8 ? "high" : confidence >= 0.4 ? "uncertain" : "low",
        color: confidence >= 0.8 ? "#22C55E" : confidence >= 0.4 ? "#F59E0B" : "#EF4444",
        label: confidence >= 0.8 ? "High" : confidence >= 0.4 ? "Uncertain" : "Low",
        percent: `${(confidence * 100).toFixed(1)}%`,
        barWidth: `${Math.min(100, confidence * 100)}%`,
      })

const getConfidenceFeedback =
  typeof confidenceDisplay.getConfidenceFeedback === "function"
    ? confidenceDisplay.getConfidenceFeedback
    : (confidence: number) => {
        if (confidence <= 0) return null
        if (confidence < 0.4) {
          return {
            message: "Low confidence - try signing again with clearer hand positioning",
            borderColor: "#EF4444",
            backgroundColor: "#FEF2F2",
            textColor: "#991B1B",
          }
        }
        if (confidence < 0.8) {
          return {
            message: "Uncertain result - the classifier was not confident enough, routed to Gemini",
            borderColor: "#F59E0B",
            backgroundColor: "#FFFBEB",
            textColor: "#92400E",
          }
        }
        return null
      }

const getRecognizerBadgeColor =
  typeof confidenceDisplay.getRecognizerBadgeColor === "function"
    ? confidenceDisplay.getRecognizerBadgeColor
    : (recognizerType: string | undefined) => (recognizerType === "classifier" ? "#1E90FF" : "#FF6600")

const getClassifierStatusLabel =
  typeof confidenceDisplay.getClassifierStatusLabel === "function"
    ? confidenceDisplay.getClassifierStatusLabel
    : (status: string | undefined) =>
        status ? status.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) : "n/a"

const getClassifierStatusColor =
  typeof confidenceDisplay.getClassifierStatusColor === "function"
    ? confidenceDisplay.getClassifierStatusColor
    : (status: string | undefined) => {
        if (status === "recognized") return "#22C55E"
        if (status === "uncertain") return "#F59E0B"
        if (status === "no_sign") return "#EF4444"
        return "#888888"
      }

const shouldShowAlternatives =
  typeof confidenceDisplay.shouldShowAlternatives === "function"
    ? confidenceDisplay.shouldShowAlternatives
    : (result: SignTranslationDebugResult) =>
        result.operation !== "capture" &&
        Array.isArray(result.classifierAlternatives) &&
        result.classifierAlternatives.length > 0

export interface SignTranslationDebugResult {
  operation?: "translate" | "capture"
  success: boolean
  statusCode?: number
  requestDurationMs: number
  frameCount: number
  landmarkFrameCount?: number
  englishText?: string
  translatedText?: string
  targetLanguage?: string
  spokeOnGlasses?: boolean
  model?: string
  message?: string
  rawGeminiOutput?: string
  responseText?: string
  recognizerType?: "classifier" | "gemini"
  recognizedLabel?: string
  confidence?: number
  fallbackUsed?: boolean
  classifierModelVersion?: string
  classifierTrainingRunId?: string
  classifierStatus?: SignClassifierResult["status"] | "unavailable" | "skipped"
  classifierAlternatives?: SignClassifierAlternative[]
  classifierError?: string
  sampleCountForLabel?: number
  targetSampleCount?: number
  requestedQuality?: RequestedCaptureQuality
  actualQuality?: ActualCaptureQuality
  qualityFallback?: boolean
  actualInputWidth?: number
  actualInputHeight?: number
  rgbMimeType?: string
  skeletonMimeType?: string
  skeletonRenderMode?: SkeletonRenderMode
  captureSettingsStatus?: CaptureSettingsStatus
  captureSettingsFallbackReason?: CaptureSettingsFallbackReason
  droppedFrameCount?: number
  error?: string
}

interface RecognitionPathBanner {
  title: string
  detail: string
  pathBadge: string
  trainingBadge: string
  background: string
  shadow: string
  textColor: string
  badgeBackground: string
  badgeColor: string
}

function getRecognitionPathBanner(result: SignTranslationDebugResult): RecognitionPathBanner {
  if (result.operation === "capture") {
    return {
      title: "Training Capture Only",
      detail: "This sample was manually labeled and saved into the classifier training set. Gemini output is never used here.",
      pathBadge: "manual capture",
      trainingBadge: "training data",
      background: "#FFF4E6",
      shadow: "#FF6600",
      textColor: "#7C2D12",
      badgeBackground: "#FF6600",
      badgeColor: "#ffffff",
    }
  }

  if (result.recognizerType === "classifier" && !result.fallbackUsed) {
    return {
      title: "Local Classifier",
      detail: "This result came from the local sign classifier only. Inference results are for evaluation and are not written back into training data.",
      pathBadge: "local only",
      trainingBadge: "not training data",
      background: "#EAF4FF",
      shadow: "#1E90FF",
      textColor: "#0F3E79",
      badgeBackground: "#1E90FF",
      badgeColor: "#ffffff",
    }
  }

  const classifierStatus = getClassifierStatusLabel(result.classifierStatus).toLowerCase()
  const fallbackReason =
    result.classifierStatus === "unavailable"
      ? result.classifierError
        ? `The local classifier was unavailable: ${result.classifierError} Gemini handled this result instead.`
        : "The local classifier was unavailable, so Gemini handled this result instead."
      : result.fallbackUsed && result.classifierStatus && result.classifierStatus !== "skipped"
        ? `The local classifier was ${classifierStatus}, so Gemini handled this result.`
        : "Gemini handled this result instead of the local classifier."

  return {
    title: result.fallbackUsed ? "Gemini Fallback" : "Gemini Result",
    detail: `${fallbackReason} It is kept separate from classifier training data.`,
    pathBadge: result.fallbackUsed ? "fallback path" : "gemini only",
    trainingBadge: "not training data",
    background: "#FFF1EA",
    shadow: "#FF6600",
    textColor: "#7C2D12",
    badgeBackground: "#FF6600",
    badgeColor: "#ffffff",
  }
}

function isWaitingOnGlassesStream(status: CameraStreamState["status"] | undefined): boolean {
  return status === "starting" || status === "initializing" || status === "preparing"
}

interface SignLanguageCaptureProps {
  className?: string
  enabled?: boolean
  cameraStream?: CameraStreamState | null
  latestSignTranslation?: TranslationEntry | null
  showHandednessOverlay?: boolean
  captureSettings?: SignCaptureSettings
  captureSettingsStatus?: CaptureSettingsStatus
  captureSettingsFallbackReason?: CaptureSettingsFallbackReason
  debugToggleBusy?: boolean
  onToggleHandednessOverlay?: (nextValue: boolean) => void | Promise<void>
  onSequenceReady?: (payload: SignSequencePayload) => SignTranslationDebugResult | void | Promise<SignTranslationDebugResult | void>
}

function extractCloudflarePlaybackId(value: string | undefined): string | null {
  if (!value) {
    return null
  }

  try {
    const url = new URL(value)
    const firstPathSegment = url.pathname.split("/").filter(Boolean)[0]
    return firstPathSegment || null
  } catch {
    return null
  }
}

function buildGlassesPlaybackSources(cameraStream: CameraStreamState | null | undefined): RemoteSignPlaybackSource[] {
  const sources: RemoteSignPlaybackSource[] = []
  const playbackId =
    extractCloudflarePlaybackId(cameraStream?.previewUrl) ||
    extractCloudflarePlaybackId(cameraStream?.thumbnailUrl) ||
    extractCloudflarePlaybackId(cameraStream?.hlsUrl)

  if (cameraStream?.webrtcUrl) {
    sources.push({
      url: cameraStream.webrtcUrl,
      transport: "webrtc",
    })
  }

  if (cameraStream?.playbackUrl) {
    sources.push({
      url: cameraStream.playbackUrl,
      transport: cameraStream.playbackMimeType === "application/vnd.apple.mpegurl" ? "hls" : "direct",
      mimeType: cameraStream.playbackMimeType,
    })
  }

  if (cameraStream?.hlsUrl) {
    sources.push({
      url: cameraStream.hlsUrl,
      transport: "hls",
      mimeType: "application/vnd.apple.mpegurl",
    })
  }

  if (playbackId) {
    sources.push({
      url: `https://videodelivery.net/${playbackId}/manifest/video.m3u8`,
      transport: "hls",
      mimeType: "application/vnd.apple.mpegurl",
    })
  }

  return sources.filter(
    (source, index, values) =>
      values.findIndex((candidate) => candidate.url === source.url && candidate.transport === source.transport) === index,
  )
}

function getCameraStatusLabel(
  cameraState: "idle" | "requesting" | "ready" | "error",
  isSigning: boolean,
  detectedHands: number,
  sourceType: SignVideoSource["type"],
): string {
  if (cameraState === "requesting") return sourceType === "glasses-stream" ? "Connecting glasses" : "Starting camera"
  if (cameraState === "error") return sourceType === "glasses-stream" ? "Glasses stream error" : "Camera error"
  if (cameraState === "idle") return "Camera idle"
  if (isSigning) return `Signing detected (${detectedHands})`
  return sourceType === "glasses-stream" ? "Watching glasses feed" : "Watching for hands"
}

async function updateSignCameraCapture(enabled: boolean): Promise<void> {
  const response = await fetch("/api/settings/sign-camera", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({enabled}),
  })

  if (!response.ok) {
    throw new Error(`Failed to ${enabled ? "resume" : "pause"} the live sign camera.`)
  }
}

const neo = {
  card: {
    border: "4px solid #000000",
    background: "#ffffff",
    padding: "0.75rem",
    boxShadow: "6px 6px 0 #000000",
  } as const,
  infoBox: {
    border: "4px solid #000000",
    background: "#ffffff",
    padding: "0.5rem 0.75rem",
  } as const,
  badge: (active: boolean) => ({
    border: "3px solid #000000",
    background: active ? "#FF6600" : "#ffffff",
    color: active ? "#ffffff" : "#000000",
    padding: "0.25rem 0.5rem",
    fontSize: "0.62rem",
    fontWeight: 900 as const,
    letterSpacing: "0.12em",
    textTransform: "uppercase" as const,
    boxShadow: active ? "3px 3px 0 #000000" : "2px 2px 0 #000000",
  }),
  label: {
    fontSize: "0.68rem",
    fontWeight: 900 as const,
    textTransform: "uppercase" as const,
    letterSpacing: "0.14em",
    color: "#FF6600",
  } as const,
  dimText: {
    fontSize: "0.72rem",
    color: "#444444",
    fontWeight: 700 as const,
  } as const,
  statBox: {
    border: "3px solid #000000",
    background: "#ffffff",
    padding: "0.5rem 0.75rem",
    boxShadow: "3px 3px 0 #FF6600",
  } as const,
} as const

export function SignLanguageCapture({
  className = "",
  enabled = true,
  cameraStream,
  latestSignTranslation = null,
  showHandednessOverlay = false,
  captureSettings,
  captureSettingsStatus = "loading",
  captureSettingsFallbackReason,
  debugToggleBusy = false,
  onToggleHandednessOverlay,
  onSequenceReady,
}: SignLanguageCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const gridCanvasRef = useRef<HTMLCanvasElement>(null)
  const skeletonGridCanvasRef = useRef<HTMLCanvasElement>(null)
  const mountedRef = useRef(true)
  const [requestState, setRequestState] = useState<"idle" | "sending" | "success" | "error">("idle")
  const [lastApiResult, setLastApiResult] = useState<SignTranslationDebugResult | null>(null)
  const [processingCapture, setProcessingCapture] = useState(false)
  const [feedReconnectBusy, setFeedReconnectBusy] = useState(false)

  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false
    }
  }, [])

  const glassesPlaybackSources = buildGlassesPlaybackSources(cameraStream)
  const isGlassesStreamActive = cameraStream?.status === "active" && glassesPlaybackSources.length > 0
  const hasManagedGlassesSession = Boolean(cameraStream?.sessionId && cameraStream?.userId)
  const allowDeviceCameraFallback =
    !cameraStream ||
    ((!cameraStream.sessionId || !cameraStream.userId) &&
      (cameraStream.status === "idle" || cameraStream.status === "stopped"))
  const signVideoSource: SignVideoSource = isGlassesStreamActive
    ? {
        type: "glasses-stream",
        sources: glassesPlaybackSources,
        requestedQuality: cameraStream?.requestedQuality,
        actualQuality: cameraStream?.actualQuality,
        qualityFallback: cameraStream?.qualityFallback,
      }
    : {type: "device-camera", requestedQuality: "1080p"}
  const captureEnabled =
    enabled &&
    captureSettingsStatus !== "loading" &&
    !processingCapture &&
    (isGlassesStreamActive || allowDeviceCameraFallback)
  const usingDeviceCamera = signVideoSource.type === "device-camera"
  const shouldDisplayGlassesSource = isGlassesStreamActive || (!allowDeviceCameraFallback && Boolean(cameraStream))
  const shouldRequestManagedFeedRestart =
    hasManagedGlassesSession &&
    shouldDisplayGlassesSource &&
    !isGlassesStreamActive &&
    !isWaitingOnGlassesStream(cameraStream?.status)

  const handleSequenceReady = async (payload: SignSequencePayload) => {
    if (!onSequenceReady || processingCapture) {
      return
    }

    setProcessingCapture(true)
    setRequestState("sending")
    const pauseLiveFeedPromise = hasManagedGlassesSession
      ? updateSignCameraCapture(false).catch((cameraError) => {
          console.error("Failed to pause live sign camera", cameraError)
        })
      : Promise.resolve()

    try {
      const result = await onSequenceReady(payload)
      if (!mountedRef.current) {
        return
      }

      if (!result) {
        setRequestState("idle")
        return
      }

      setLastApiResult(result)
      setRequestState(result.success ? "success" : "error")
    } catch (error) {
      setLastApiResult({
        success: false,
        requestDurationMs: 0,
        frameCount: payload.frameCount,
        error: error instanceof Error ? error.message : "Sign translation failed",
      })
      setRequestState("error")
    } finally {
      await pauseLiveFeedPromise

      if (hasManagedGlassesSession && mountedRef.current && enabled) {
        try {
          await updateSignCameraCapture(true)
        } catch (cameraError) {
          console.error("Failed to resume live sign camera", cameraError)
        }
      }

      if (mountedRef.current) {
        setProcessingCapture(false)
      }
    }
  }

  const {
    cameraState,
    error,
    videoSource,
    isSigning,
    detectedHands,
    handednessLabels,
    capturedFrames,
    sequenceFrames,
    gestureHints,
    latestSequence,
    requestedQuality,
    actualQuality,
    qualityFallback,
    actualInputWidth,
    actualInputHeight,
    droppedFrameCount,
    restart,
    stop,
  } = useSignLanguage({
    enabled: captureEnabled,
    videoRef,
    canvasRef: overlayCanvasRef,
    gridCanvasRef,
    skeletonGridCanvasRef,
    videoSource: signVideoSource,
    showHandednessOverlay,
    captureSettings,
    captureSettingsStatus,
    captureSettingsFallbackReason,
    onHandsDown: handleSequenceReady,
  })

  const handleFeedControlClick = async () => {
    if (!shouldRequestManagedFeedRestart) {
      await restart()
      return
    }

    setFeedReconnectBusy(true)
    try {
      await updateSignCameraCapture(false)
      await updateSignCameraCapture(true)
    } catch (cameraError) {
      console.error("Failed to reconnect live sign camera", cameraError)
    } finally {
      if (mountedRef.current) {
        setFeedReconnectBusy(false)
      }
    }
  }

  const statusLabel =
    processingCapture
      ? "Processing sign"
      : captureSettingsStatus === "loading"
        ? "Loading capture settings"
      : !captureEnabled && isWaitingOnGlassesStream(cameraStream?.status)
      ? "Waiting for glasses stream"
      : !captureEnabled && cameraStream?.status === "error"
        ? "Glasses stream error"
        : getCameraStatusLabel(
            cameraState,
            isSigning,
            detectedHands,
            shouldDisplayGlassesSource ? "glasses-stream" : videoSource,
          )
  const sourceLabel = shouldDisplayGlassesSource ? "Mentra Glasses" : "Device Camera"
  const streamStatusLabel = processingCapture
    ? "PAUSED"
    : cameraStream?.status
      ? cameraStream.status.toUpperCase()
      : usingDeviceCamera
        ? "LOCAL"
        : "UNKNOWN"
  const previewUnavailableMessage =
    processingCapture
      ? "Live sign capture is paused while the last response is being generated. Scanning resumes automatically when the result is ready."
      : captureSettingsStatus === "loading"
      ? "Loading sign-capture runtime settings before initializing MediaPipe."
      : !captureEnabled && isWaitingOnGlassesStream(cameraStream?.status)
      ? cameraStream?.message || "Waiting for the Mentra glasses camera stream to come online."
      : !captureEnabled && cameraStream?.status === "error"
        ? cameraStream?.message || "The Mentra glasses camera stream could not be started."
        : usingDeviceCamera
          ? "Allow webcam access so the hook can track landmarks and render the sign skeleton live."
          : "Connecting to the Mentra glasses camera stream."
  const streamMessage = processingCapture
    ? "Live sign capture is paused while the current response is being generated to reduce glasses heat."
    : captureSettingsStatus === "fallback"
      ? captureSettingsFallbackReason === "timeout"
        ? "Capture settings timed out after 3 seconds, so the client started with fallback defaults."
        : "Capture settings could not be loaded, so the client started with fallback defaults."
    : cameraStream?.message
  const matchingSseTranslation =
    lastApiResult?.operation === "translate" &&
    !lastApiResult.error &&
    latestSignTranslation?.sourceType === "sign" &&
    (!latestSequence || latestSignTranslation.timestamp >= latestSequence.capturedAt)
      ? latestSignTranslation
      : null
  const displayResult = lastApiResult
    ? {
        ...lastApiResult,
        englishText: matchingSseTranslation?.originalText ?? lastApiResult.englishText,
        translatedText: matchingSseTranslation?.translatedText ?? lastApiResult.translatedText,
        targetLanguage: matchingSseTranslation?.targetLanguage ?? lastApiResult.targetLanguage,
        spokeOnGlasses: matchingSseTranslation?.spokeOnGlasses ?? lastApiResult.spokeOnGlasses,
        model: matchingSseTranslation?.sourceModel ?? lastApiResult.model,
        recognizerType: matchingSseTranslation?.recognizerType ?? lastApiResult.recognizerType,
        recognizedLabel: matchingSseTranslation?.recognizedLabel ?? lastApiResult.recognizedLabel,
        confidence: matchingSseTranslation?.confidence ?? lastApiResult.confidence,
        fallbackUsed: matchingSseTranslation?.fallbackUsed ?? lastApiResult.fallbackUsed,
        classifierModelVersion:
          matchingSseTranslation?.classifierModelVersion ?? lastApiResult.classifierModelVersion,
        classifierTrainingRunId:
          matchingSseTranslation?.classifierTrainingRunId ?? lastApiResult.classifierTrainingRunId,
        requestedQuality: lastApiResult.requestedQuality ?? latestSequence?.requestedQuality ?? requestedQuality,
        actualQuality: lastApiResult.actualQuality ?? latestSequence?.actualQuality ?? actualQuality,
        qualityFallback: lastApiResult.qualityFallback ?? latestSequence?.qualityFallback ?? qualityFallback,
        actualInputWidth: lastApiResult.actualInputWidth ?? latestSequence?.actualInputWidth ?? actualInputWidth,
        actualInputHeight: lastApiResult.actualInputHeight ?? latestSequence?.actualInputHeight ?? actualInputHeight,
        rgbMimeType: lastApiResult.rgbMimeType ?? latestSequence?.mimeType ?? "image/jpeg",
        skeletonMimeType: lastApiResult.skeletonMimeType ?? latestSequence?.skeletonMimeType,
        skeletonRenderMode: lastApiResult.skeletonRenderMode ?? latestSequence?.skeletonRenderMode ?? "diagnostic",
        captureSettingsStatus:
          lastApiResult.captureSettingsStatus ?? latestSequence?.captureSettingsStatus ?? captureSettingsStatus,
        captureSettingsFallbackReason:
          lastApiResult.captureSettingsFallbackReason ??
          latestSequence?.captureSettingsFallbackReason ??
          captureSettingsFallbackReason,
        droppedFrameCount: lastApiResult.droppedFrameCount ?? latestSequence?.droppedFrameCount ?? droppedFrameCount,
      }
    : null

  return (
    <section className={className} style={neo.card}>
      <div style={{display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.75rem"}}>
        <div>
          <p style={neo.label}>
            ASL Vision Preview
          </p>
          <p style={{...neo.dimText, marginTop: "0.25rem"}}>
            MediaPipe holistic landmarks + live skeleton overlay + temporal capture from the Mentra glasses feed.
          </p>
        </div>
        <div style={{display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "0.5rem"}}>
          {onToggleHandednessOverlay && (
            <button
              type="button"
              onClick={() => void onToggleHandednessOverlay(!showHandednessOverlay)}
              disabled={debugToggleBusy}
              style={{
                border: "3px solid #000000",
                background: showHandednessOverlay ? "#FF6600" : "#ffffff",
                color: showHandednessOverlay ? "#ffffff" : "#000000",
                padding: "0.25rem 0.5rem",
                fontSize: "0.62rem",
                fontFamily: "inherit",
                fontWeight: 900,
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                boxShadow: showHandednessOverlay ? "3px 3px 0 #000000" : "2px 2px 0 #000000",
                cursor: debugToggleBusy ? "progress" : "pointer",
                opacity: debugToggleBusy ? 0.7 : 1,
              }}
            >
              {debugToggleBusy ? "Saving..." : showHandednessOverlay ? "Debug Overlay On" : "Debug Overlay Off"}
            </button>
          )}
          <span style={neo.badge(shouldDisplayGlassesSource)}>
            {sourceLabel}
          </span>
          <span style={neo.badge(isSigning || processingCapture)}>
            {statusLabel}
          </span>
        </div>
      </div>

      <div style={{position: "relative", marginTop: "0.75rem", aspectRatio: "16/9", overflow: "hidden", border: "4px solid #000000", background: "#000000"}}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", transform: usingDeviceCamera ? "scaleX(-1)" : undefined}}
        />
        <canvas
          ref={overlayCanvasRef}
          style={{pointerEvents: "none", position: "absolute", inset: 0, width: "100%", height: "100%", transform: usingDeviceCamera ? "scaleX(-1)" : undefined}}
        />

        <div style={{pointerEvents: "none", position: "absolute", inset: "0 0 auto 0", display: "flex", alignItems: "center", justifyContent: "space-between", background: "#000000", padding: "0.5rem 0.75rem", fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.12em", color: "#ffffff", fontWeight: 900, borderBottom: "3px solid #FF6600"}}>
          <span>{shouldDisplayGlassesSource ? "Glasses live feed" : "Device preview"}</span>
          <span>{capturedFrames}/{MAX_GRID_FRAMES} grid frames</span>
        </div>

        {cameraState !== "ready" && (
          <div style={{position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "#000000", padding: "1rem", textAlign: "center"}}>
            <p style={{fontSize: "0.85rem", fontWeight: 900, color: "#FF6600"}}>
              {processingCapture
                ? "Processing captured sign..."
                : cameraState === "requesting"
                ? shouldDisplayGlassesSource
                  ? "Connecting to Mentra glasses..."
                  : "Requesting camera access..."
                : !captureEnabled && isWaitingOnGlassesStream(cameraStream?.status)
                  ? "Waiting for Mentra glasses..."
                : "Camera preview unavailable"}
            </p>
            <p style={{marginTop: "0.5rem", fontSize: "0.75rem", color: "#ffffff", fontWeight: 700}}>
              {previewUnavailableMessage}
            </p>
          </div>
        )}
      </div>

      <div style={{marginTop: "0.75rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem"}}>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Source</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>{sourceLabel}</div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Stream</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>{streamStatusLabel}</div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Hands</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>{detectedHands}</div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Grid Frames</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>{capturedFrames}</div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Sequence Frames</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>{sequenceFrames}</div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Window</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>{windowLabel(DEFAULT_CAPTURE_FPS, MAX_SEQUENCE_FRAMES)} lm / {windowLabel(effectiveGridFps(DEFAULT_CAPTURE_FPS), MAX_GRID_FRAMES)} grid</div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Requested Q</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>{requestedQuality || "n/a"}</div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Actual Q</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>{actualQuality || "unknown"}</div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Resolution</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
            {actualInputWidth && actualInputHeight ? `${actualInputWidth}x${actualInputHeight}` : "unknown"}
          </div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Dropped</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>{droppedFrameCount}</div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Settings</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
            {captureSettingsStatus.toUpperCase()}
          </div>
        </div>
        <div style={neo.statBox}>
          <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Fallback</div>
          <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
            {qualityFallback ? "Yes" : "No"}
          </div>
        </div>
      </div>

      {streamMessage && (
        <div style={{...neo.infoBox, marginTop: "0.75rem"}}>
          <p style={neo.dimText}>{streamMessage}</p>
        </div>
      )}

      {gestureHints.length > 0 && (
        <div style={{...neo.infoBox, marginTop: "0.75rem"}}>
          <p style={neo.label}>Fingerpose Hints</p>
          <div style={{marginTop: "0.5rem", display: "flex", flexWrap: "wrap", gap: "0.5rem"}}>
            {gestureHints.map((hint) => (
              <span key={hint.name} style={neo.badge(true)}>
                {hint.name} {hint.score.toFixed(1)}
              </span>
            ))}
          </div>
        </div>
      )}

      {showHandednessOverlay && handednessLabels.length > 0 && (
        <div style={{...neo.infoBox, marginTop: "0.75rem"}}>
          <p style={neo.label}>Detected Hands</p>
          <div style={{marginTop: "0.5rem", display: "flex", flexWrap: "wrap", gap: "0.5rem"}}>
            {handednessLabels.map((label) => (
              <span key={label} style={neo.badge(true)}>
                {label}
              </span>
            ))}
          </div>
        </div>
      )}

      {latestSequence && (
        <div style={{...neo.infoBox, marginTop: "0.75rem"}}>
          <div style={{display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.5rem"}}>
            <p style={neo.label}>Submitted Grids</p>
            <span style={{fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.12em", color: "#444444", fontWeight: 900}}>
              {latestSequence.frameCount}/{MAX_GRID_FRAMES} grid • {latestSequence.sequenceFrameCount}/{MAX_SEQUENCE_FRAMES} landmarks
            </span>
          </div>
          <p style={{...neo.dimText, marginTop: "0.25rem"}}>
            Posting the raw RGB grid and the skeleton-reference grid to <code style={{background: "#FF6600", color: "#ffffff", padding: "0.1rem 0.3rem", border: "2px solid #000000"}}>/api/sign-translate</code>.
          </p>
          <div style={{marginTop: "0.5rem", display: "flex", flexWrap: "wrap", gap: "0.45rem"}}>
            <span style={neo.badge(false)}>RGB {latestSequence.mimeType}</span>
            <span style={neo.badge(Boolean(latestSequence.skeletonDataUrl))}>
              Skeleton {latestSequence.skeletonMimeType || "n/a"}
            </span>
            <span style={neo.badge(Boolean(latestSequence.qualityFallback))}>
              {latestSequence.requestedQuality || "n/a"} to {latestSequence.actualQuality || "unknown"}
            </span>
            <span style={neo.badge(false)}>
              {latestSequence.actualInputWidth && latestSequence.actualInputHeight
                ? `${latestSequence.actualInputWidth}x${latestSequence.actualInputHeight}`
                : "unknown res"}
            </span>
            <span style={neo.badge(false)}>Render {latestSequence.skeletonRenderMode}</span>
            <span style={neo.badge(false)}>Dropped {latestSequence.droppedFrameCount}</span>
          </div>
          <div style={{marginTop: "0.75rem", display: "grid", gridTemplateColumns: latestSequence.skeletonDataUrl ? "1fr 1fr" : "1fr", gap: "0.75rem"}}>
            <div>
              <p style={{fontSize: "0.64rem", fontWeight: 900, textTransform: "uppercase", letterSpacing: "0.12em", color: "#444444"}}>
                Raw RGB Grid
              </p>
              <img
                src={latestSequence.dataUrl}
                alt="Chronological sign grid preview"
                style={{marginTop: "0.4rem", width: "100%", border: "4px solid #000000", objectFit: "cover", boxShadow: "4px 4px 0 #FF6600"}}
              />
            </div>
            {latestSequence.skeletonDataUrl && (
              <div>
                <p style={{fontSize: "0.64rem", fontWeight: 900, textTransform: "uppercase", letterSpacing: "0.12em", color: "#444444"}}>
                  Skeleton Grid
                </p>
                <img
                  src={latestSequence.skeletonDataUrl}
                  alt="Skeleton timeline grid preview"
                  style={{marginTop: "0.4rem", width: "100%", border: "4px solid #000000", objectFit: "cover", boxShadow: "4px 4px 0 #1E90FF"}}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {(requestState !== "idle" || displayResult) && (
        <div style={{...neo.infoBox, marginTop: "0.75rem"}}>
          <div style={{display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.5rem"}}>
            <p style={neo.label}>API Roundtrip</p>
            <span
              style={neo.badge(requestState === "sending" || requestState === "success")}
            >
              {requestState === "sending"
                ? "Sending"
                : requestState === "success"
                  ? "Success"
                  : requestState === "error"
                    ? "Error"
                    : "Idle"}
            </span>
          </div>

          {requestState === "sending" && (
            <p style={{...neo.dimText, marginTop: "0.5rem"}}>
              Uploading the temporal sign sequence, pausing the live feed, and waiting for the backend recognizer.
            </p>
          )}

          {displayResult && (
            <>
              <div style={{marginTop: "0.75rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem"}}>
                <div style={neo.statBox}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Latency</div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
                    {Math.round(displayResult.requestDurationMs)}ms
                  </div>
                </div>
                <div style={neo.statBox}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Glasses Speak</div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
                    {displayResult.operation === "capture" ? "n/a" : displayResult.spokeOnGlasses ? "Yes" : "No"}
                  </div>
                </div>
              </div>

              <div style={{marginTop: "0.5rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem"}}>
                <div style={neo.statBox}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Formats</div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.8rem", fontWeight: 900, color: "#000000"}}>
                    {displayResult.rgbMimeType || "image/jpeg"} / {displayResult.skeletonMimeType || "n/a"}
                  </div>
                </div>
                <div style={neo.statBox}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Capture Path</div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.8rem", fontWeight: 900, color: "#000000"}}>
                    {(displayResult.requestedQuality || "n/a") + " -> " + (displayResult.actualQuality || "unknown")}
                  </div>
                </div>
              </div>

              <div style={{marginTop: "0.5rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem"}}>
                <div style={neo.statBox}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Resolution</div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.8rem", fontWeight: 900, color: "#000000"}}>
                    {displayResult.actualInputWidth && displayResult.actualInputHeight
                      ? `${displayResult.actualInputWidth}x${displayResult.actualInputHeight}`
                      : "unknown"}
                  </div>
                </div>
                <div style={neo.statBox}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Diagnostics</div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.8rem", fontWeight: 900, color: "#000000"}}>
                    {displayResult.skeletonRenderMode || "diagnostic"} / {displayResult.droppedFrameCount ?? 0} dropped
                  </div>
                </div>
              </div>

              <div
                style={{
                  marginTop: "0.5rem",
                  border: "4px solid #000000",
                  background: getRecognitionPathBanner(displayResult).background,
                  padding: "0.65rem 0.75rem",
                  boxShadow: `5px 5px 0 ${getRecognitionPathBanner(displayResult).shadow}`,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    justifyContent: "space-between",
                    gap: "0.75rem",
                    flexWrap: "wrap",
                  }}
                >
                  <div style={{flex: "1 1 14rem"}}>
                    <div
                      style={{
                        fontSize: "0.65rem",
                        fontWeight: 900,
                        textTransform: "uppercase",
                        letterSpacing: "0.14em",
                        color: getRecognitionPathBanner(displayResult).textColor,
                      }}
                    >
                      Recognition Path
                    </div>
                    <div
                      style={{
                        marginTop: "0.2rem",
                        fontSize: "1rem",
                        fontWeight: 900,
                        color: "#000000",
                        textTransform: "uppercase",
                        letterSpacing: "0.05em",
                      }}
                    >
                      {getRecognitionPathBanner(displayResult).title}
                    </div>
                    <div
                      style={{
                        marginTop: "0.3rem",
                        fontSize: "0.76rem",
                        lineHeight: 1.5,
                        color: "#222222",
                        fontWeight: 700,
                      }}
                    >
                      {getRecognitionPathBanner(displayResult).detail}
                    </div>
                  </div>
                  <div style={{display: "flex", alignItems: "center", gap: "0.45rem", flexWrap: "wrap"}}>
                    <span
                      style={{
                        display: "inline-block",
                        border: "2px solid #000000",
                        background: getRecognitionPathBanner(displayResult).badgeBackground,
                        color: getRecognitionPathBanner(displayResult).badgeColor,
                        padding: "0.2rem 0.45rem",
                        fontSize: "0.58rem",
                        fontWeight: 900,
                        letterSpacing: "0.1em",
                        textTransform: "uppercase",
                      }}
                    >
                      {getRecognitionPathBanner(displayResult).pathBadge}
                    </span>
                    <span
                      style={{
                        display: "inline-block",
                        border: "2px solid #000000",
                        background: "#ffffff",
                        color: "#000000",
                        padding: "0.2rem 0.45rem",
                        fontSize: "0.58rem",
                        fontWeight: 900,
                        letterSpacing: "0.1em",
                        textTransform: "uppercase",
                      }}
                    >
                      {getRecognitionPathBanner(displayResult).trainingBadge}
                    </span>
                  </div>
                </div>
              </div>

              <div style={{marginTop: "0.5rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem"}}>
                <div style={neo.statBox}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Target</div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
                    {displayResult.targetLanguage || "English"}
                  </div>
                </div>
                <div style={neo.statBox}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>
                    {displayResult.operation === "capture" ? "Saved Label" : "Model"}
                  </div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
                    {displayResult.operation === "capture"
                      ? displayResult.recognizedLabel?.replaceAll("_", " ") || "n/a"
                      : displayResult.model || "n/a"}
                  </div>
                </div>
              </div>

              {displayResult.operation === "capture" ? (
                <div style={{marginTop: "0.5rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem"}}>
                  <div style={neo.statBox}>
                    <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Recognizer</div>
                    <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>capture</div>
                  </div>
                  <div style={neo.statBox}>
                    <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Label Progress</div>
                    <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
                      {typeof displayResult.sampleCountForLabel === "number"
                        ? `${displayResult.sampleCountForLabel}/${displayResult.targetSampleCount || CORE_SIGN_SAMPLE_TARGET}`
                        : "n/a"}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{...neo.statBox, marginTop: "0.5rem"}}>
                  <div style={{display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.5rem"}}>
                    <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>
                      Confidence
                    </div>
                    <div style={{display: "flex", alignItems: "center", gap: "0.4rem"}}>
                      {/* Recognizer path badge */}
                      <span
                        style={{
                          display: "inline-block",
                          border: "2px solid #000000",
                          background: getRecognizerBadgeColor(displayResult.recognizerType),
                          color: "#ffffff",
                          padding: "0.15rem 0.4rem",
                          fontSize: "0.58rem",
                          fontWeight: 900,
                          letterSpacing: "0.1em",
                          textTransform: "uppercase",
                        }}
                      >
                        {displayResult.recognizerType || "n/a"}
                      </span>
                      {displayResult.fallbackUsed && (
                        <span
                          style={{
                            display: "inline-block",
                            border: "2px solid #000000",
                            background: "#FFD700",
                            color: "#000000",
                            padding: "0.15rem 0.4rem",
                            fontSize: "0.58rem",
                            fontWeight: 900,
                            letterSpacing: "0.1em",
                            textTransform: "uppercase",
                          }}
                        >
                          Fallback
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Confidence bar */}
                  {typeof displayResult.confidence === "number" ? (() => {
                    const cd = getConfidenceDisplay(displayResult.confidence)
                    return (
                      <div style={{marginTop: "0.5rem"}}>
                        <div style={{display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "0.5rem"}}>
                          <span style={{fontSize: "1.1rem", fontWeight: 900, color: "#000000", fontVariantNumeric: "tabular-nums"}}>
                            {cd.percent}
                          </span>
                          <span style={{fontSize: "0.62rem", fontWeight: 900, textTransform: "uppercase", letterSpacing: "0.1em", color: cd.color}}>
                            {cd.label}
                          </span>
                        </div>
                        <div
                          style={{
                            marginTop: "0.35rem",
                            height: "10px",
                            border: "3px solid #000000",
                            background: "#f0f0f0",
                            position: "relative",
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              bottom: 0,
                              width: cd.barWidth,
                              background: cd.color,
                              transition: "width 0.3s ease-out, background 0.3s ease-out",
                            }}
                          />
                          {/* Threshold markers */}
                          <div style={{position: "absolute", top: 0, bottom: 0, left: "40%", width: "1px", background: "#00000033"}} />
                          <div style={{position: "absolute", top: 0, bottom: 0, left: "80%", width: "1px", background: "#00000033"}} />
                        </div>
                        <div style={{marginTop: "0.2rem", display: "flex", justifyContent: "space-between", fontSize: "0.55rem", color: "#999999", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em"}}>
                          <span>0%</span>
                          <span>40%</span>
                          <span>80%</span>
                          <span>100%</span>
                        </div>
                      </div>
                    )
                  })() : (
                    <div style={{marginTop: "0.5rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>n/a</div>
                  )}

                  {/* Confidence feedback */}
                  {typeof displayResult.confidence === "number" && (() => {
                    const feedback = getConfidenceFeedback(displayResult.confidence)
                    if (!feedback) return null
                    return (
                      <div
                        style={{
                          marginTop: "0.5rem",
                          padding: "0.4rem 0.6rem",
                          border: `3px solid ${feedback.borderColor}`,
                          background: feedback.backgroundColor,
                          fontSize: "0.7rem",
                          fontWeight: 800,
                          color: feedback.textColor,
                        }}
                      >
                        {feedback.message}
                      </div>
                    )
                  })()}
                </div>
              )}

              {/* Classifier alternatives */}
              {shouldShowAlternatives(displayResult) && (
                <div style={{...neo.infoBox, marginTop: "0.5rem"}}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>
                    Classifier Candidates
                  </div>
                  <div style={{marginTop: "0.4rem", display: "flex", flexDirection: "column", gap: "0.3rem"}}>
                    {displayResult.classifierAlternatives!
                      .slice(0, 4)
                      .map((alt, index) => {
                        const isTopPick = index === 0
                        const altDisplay = getConfidenceDisplay(alt.confidence)
                        return (
                          <div
                            key={alt.label}
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "0.5rem",
                              padding: "0.25rem 0.4rem",
                              border: isTopPick ? "3px solid #000000" : "2px solid #e0e0e0",
                              background: isTopPick ? "#f8f8f8" : "#ffffff",
                            }}
                          >
                            <span style={{fontSize: "0.62rem", fontWeight: 900, color: "#999999", minWidth: "1rem"}}>
                              #{index + 1}
                            </span>
                            <span style={{fontSize: "0.78rem", fontWeight: 900, color: "#000000", flex: 1, textTransform: "capitalize"}}>
                              {alt.label.replaceAll("_", " ")}
                            </span>
                            <div style={{display: "flex", alignItems: "center", gap: "0.35rem"}}>
                              <div style={{width: "40px", height: "6px", border: "2px solid #000000", background: "#f0f0f0", position: "relative", overflow: "hidden"}}>
                                <div style={{position: "absolute", top: 0, left: 0, bottom: 0, width: altDisplay.barWidth, background: altDisplay.color}} />
                              </div>
                              <span style={{fontSize: "0.65rem", fontWeight: 900, color: altDisplay.color, fontVariantNumeric: "tabular-nums", minWidth: "2.5rem", textAlign: "right"}}>
                                {altDisplay.percent}
                              </span>
                            </div>
                          </div>
                        )
                      })}
                  </div>
                </div>
              )}

              <div style={{marginTop: "0.5rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem"}}>
                <div style={neo.statBox}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Landmark Frames</div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
                    {displayResult.landmarkFrameCount || latestSequence?.sequenceFrameCount || 0}
                  </div>
                </div>
                <div style={neo.statBox}>
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Classifier Status</div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: getClassifierStatusColor(displayResult.classifierStatus)}}>
                    {getClassifierStatusLabel(displayResult.classifierStatus)}
                  </div>
                </div>
              </div>

              {displayResult.classifierStatus === "unavailable" && displayResult.classifierError && (
                <div
                  style={{
                    marginTop: "0.5rem",
                    border: "4px solid #000000",
                    borderLeftColor: "#FF6600",
                    borderLeftWidth: "8px",
                    background: "#FFF7F2",
                    padding: "0.5rem 0.75rem",
                  }}
                >
                  <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#7C2D12", fontWeight: 900}}>
                    Classifier Error
                  </div>
                  <div style={{marginTop: "0.25rem", fontSize: "0.76rem", lineHeight: 1.5, color: "#222222", fontWeight: 700}}>
                    {displayResult.classifierError}
                  </div>
                </div>
              )}

              {(displayResult.classifierModelVersion || displayResult.classifierTrainingRunId) && (
                <div style={{marginTop: "0.5rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem"}}>
                  <div style={neo.statBox}>
                    <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>
                      Classifier Version
                    </div>
                    <div style={{marginTop: "0.25rem", fontSize: "0.78rem", fontWeight: 900, color: "#000000", wordBreak: "break-word"}}>
                      {displayResult.classifierModelVersion || "n/a"}
                    </div>
                  </div>
                  <div style={neo.statBox}>
                    <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>
                      Training Run
                    </div>
                    <div style={{marginTop: "0.25rem", fontSize: "0.78rem", fontWeight: 900, color: "#000000", wordBreak: "break-word"}}>
                      {displayResult.classifierTrainingRunId || "n/a"}
                    </div>
                  </div>
                </div>
              )}

              {typeof displayResult.statusCode === "number" && (
                <div style={{marginTop: "0.5rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem"}}>
                  <div style={neo.statBox}>
                    <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>HTTP</div>
                    <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
                      {displayResult.statusCode}
                    </div>
                  </div>
                  <div style={neo.statBox}>
                    <div style={{fontSize: "0.66rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#444444", fontWeight: 900}}>Response</div>
                    <div style={{marginTop: "0.25rem", fontSize: "0.85rem", fontWeight: 900, color: "#000000"}}>
                      {displayResult.success ? "JSON" : displayResult.responseText ? "DETAILS" : "EMPTY"}
                    </div>
                  </div>
                </div>
              )}

              {displayResult.error ? (
                <div style={{marginTop: "0.75rem", border: "4px solid #000000", borderLeftColor: "#FF6600", borderLeftWidth: "8px", background: "#ffffff", padding: "0.5rem 0.75rem"}}>
                  <p style={{fontSize: "0.75rem", color: "#000000", fontWeight: 900}}>{displayResult.error}</p>
                  {displayResult.responseText && displayResult.responseText.trim() !== displayResult.error.trim() && (
                    <p style={{marginTop: "0.5rem", fontSize: "0.72rem", color: "#444444", fontWeight: 700, lineHeight: 1.5}}>
                      {displayResult.responseText}
                    </p>
                  )}
                </div>
              ) : (
                <>
                  <div style={{...neo.infoBox, marginTop: "0.75rem"}}>
                    <p style={{fontSize: "0.65rem", fontWeight: 900, textTransform: "uppercase", letterSpacing: "0.14em", color: "#444444"}}>
                      {displayResult.operation === "capture" ? "Capture Result" : "Interpreted English"}
                    </p>
                    <p style={{marginTop: "0.25rem", fontSize: "0.82rem", lineHeight: 1.6, color: "#000000", fontWeight: 700}}>
                      {displayResult.englishText || displayResult.message || "No interpretation returned."}
                    </p>
                  </div>

                  {displayResult.translatedText &&
                    displayResult.translatedText.trim() &&
                    displayResult.translatedText.trim() !== (displayResult.englishText || "").trim() && (
                      <div style={{...neo.infoBox, marginTop: "0.75rem"}}>
                        <p style={{fontSize: "0.65rem", fontWeight: 900, textTransform: "uppercase", letterSpacing: "0.14em", color: "#444444"}}>
                          Spoken Output
                        </p>
                        <p style={{marginTop: "0.25rem", fontSize: "0.82rem", lineHeight: 1.6, color: "#000000", fontWeight: 700}}>
                          {displayResult.translatedText}
                        </p>
                      </div>
                    )}

                  {displayResult.rawGeminiOutput && displayResult.rawGeminiOutput.trim() && (
                    <div style={{...neo.infoBox, marginTop: "0.75rem"}}>
                      <p style={{fontSize: "0.65rem", fontWeight: 900, textTransform: "uppercase", letterSpacing: "0.14em", color: "#444444"}}>
                        Raw Gemini Output
                      </p>
                      <pre
                        style={{
                          marginTop: "0.35rem",
                          marginBottom: 0,
                          fontSize: "0.72rem",
                          lineHeight: 1.55,
                          color: "#000000",
                          fontWeight: 700,
                          whiteSpace: "pre-wrap",
                          wordBreak: "break-word",
                          fontFamily: "inherit",
                        }}
                      >
                        {displayResult.rawGeminiOutput}
                      </pre>
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </div>
      )}

      {error && (
        <div style={{marginTop: "0.75rem", border: "4px solid #000000", borderLeftColor: "#FF6600", borderLeftWidth: "8px", background: "#ffffff", padding: "0.5rem 0.75rem", boxShadow: "4px 4px 0 #FF6600"}}>
          <p style={{fontSize: "0.75rem", color: "#000000", fontWeight: 900}}>{error}</p>
        </div>
      )}

      <div style={{marginTop: "0.75rem", display: "flex", gap: "0.5rem"}}>
        <button
          type="button"
          onClick={() => void handleFeedControlClick()}
          disabled={feedReconnectBusy}
          style={{
            border: "4px solid #000000",
            background: "#FF6600",
            color: "#ffffff",
            padding: "0.5rem 0.75rem",
            fontSize: "0.75rem",
            fontFamily: "inherit",
            fontWeight: 900,
            textTransform: "uppercase",
            letterSpacing: "0.12em",
            cursor: feedReconnectBusy ? "wait" : "pointer",
            opacity: feedReconnectBusy ? 0.78 : 1,
            boxShadow: "4px 4px 0 #000000",
          }}
        >
          {feedReconnectBusy
            ? "Connecting..."
            : shouldDisplayGlassesSource
            ? cameraState === "ready"
              ? "Reconnect Feed"
              : "Connect Feed"
            : cameraState === "ready"
              ? "Restart Camera"
              : "Enable Camera"}
        </button>
        <button
          type="button"
          onClick={() => void stop()}
          style={{
            border: "4px solid #000000",
            background: "#ffffff",
            color: "#000000",
            padding: "0.5rem 0.75rem",
            fontSize: "0.75rem",
            fontFamily: "inherit",
            fontWeight: 900,
            textTransform: "uppercase",
            letterSpacing: "0.12em",
            cursor: "pointer",
            boxShadow: "3px 3px 0 #000000",
          }}
        >
          Stop
        </button>
      </div>

      <canvas ref={gridCanvasRef} style={{display: "none"}} />
      <canvas ref={skeletonGridCanvasRef} style={{display: "none"}} />
    </section>
  )
}
