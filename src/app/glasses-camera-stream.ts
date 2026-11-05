import type {ManagedStreamStatus} from "@mentra/sdk"
import type {
  ActualCaptureQuality,
  RequestedCaptureQuality,
} from "../sign/capture-settings"

export type GlassesCameraStreamLifecycle =
  | "idle"
  | "starting"
  | "initializing"
  | "preparing"
  | "active"
  | "stopping"
  | "stopped"
  | "error"

export interface GlassesCameraStreamState {
  status: GlassesCameraStreamLifecycle
  transport?: "managed"
  sessionId?: string
  userId?: string
  streamId?: string
  playbackUrl?: string
  playbackMimeType?: string
  hlsUrl?: string
  dashUrl?: string
  webrtcUrl?: string
  previewUrl?: string
  thumbnailUrl?: string
  requestedQuality?: RequestedCaptureQuality
  actualQuality?: ActualCaptureQuality
  qualityFallback?: boolean
  actualInputWidth?: number
  actualInputHeight?: number
  message?: string
  updatedAt: number
}

let currentState: GlassesCameraStreamState = {
  status: "idle",
  updatedAt: Date.now(),
}

function shouldTreatActiveStatusAsReady(status: ManagedStreamStatus): boolean {
  if (status.status !== "active") {
    return true
  }

  const normalizedMessage = status.message?.toLowerCase() ?? ""
  return (
    Boolean(status.previewUrl) ||
    Boolean(status.thumbnailUrl) ||
    normalizedMessage.includes("stream is now live") ||
    normalizedMessage.includes("details updated")
  )
}

function publishState(nextState: GlassesCameraStreamState): GlassesCameraStreamState {
  currentState = nextState
  return currentState
}

export function getGlassesCameraStreamState(): GlassesCameraStreamState {
  return currentState
}

export function setGlassesCameraStreamState(
  next: Partial<GlassesCameraStreamState> & Pick<GlassesCameraStreamState, "status">,
): GlassesCameraStreamState {
  return publishState({
    ...currentState,
    ...next,
    updatedAt: Date.now(),
  })
}

export function resetGlassesCameraStreamState(sessionId?: string): GlassesCameraStreamState {
  if (sessionId && currentState.sessionId && currentState.sessionId !== sessionId) {
    return currentState
  }

  return publishState({
    status: "idle",
    updatedAt: Date.now(),
  })
}

export function updateGlassesCameraStreamFromManagedStatus(
  status: ManagedStreamStatus,
  sessionId?: string,
  userId?: string,
): GlassesCameraStreamState {
  if (status.status === "stopped") {
    return publishState({
      status: "stopped",
      transport: "managed",
      sessionId,
      userId,
      streamId: status.streamId,
      message: status.message,
      updatedAt: Date.now(),
    })
  }

  const nextStatus: GlassesCameraStreamLifecycle =
    status.status === "active" && !shouldTreatActiveStatusAsReady(status) && currentState.status !== "active"
      ? "preparing"
      : status.status

  return publishState({
    ...currentState,
    status: nextStatus,
    transport: "managed",
    sessionId: sessionId ?? currentState.sessionId,
    userId: userId ?? currentState.userId,
    streamId: status.streamId ?? currentState.streamId,
    playbackUrl: status.hlsUrl ?? currentState.playbackUrl,
    playbackMimeType: "application/vnd.apple.mpegurl",
    hlsUrl: status.hlsUrl ?? currentState.hlsUrl,
    dashUrl: status.dashUrl ?? currentState.dashUrl,
    webrtcUrl: status.webrtcUrl ?? currentState.webrtcUrl,
    previewUrl: status.previewUrl ?? currentState.previewUrl,
    thumbnailUrl: status.thumbnailUrl ?? currentState.thumbnailUrl,
    requestedQuality: status.requestedQuality ?? currentState.requestedQuality,
    actualQuality: status.actualQuality ?? status.quality ?? currentState.actualQuality,
    qualityFallback: status.qualityFallback ?? currentState.qualityFallback,
    message: status.message,
    updatedAt: Date.now(),
  })
}
