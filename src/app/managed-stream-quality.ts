import type {RequestedCaptureQuality} from "../sign/capture-settings"

export const PRIMARY_MANAGED_STREAM_QUALITY: RequestedCaptureQuality = "1080p"
export const FALLBACK_MANAGED_STREAM_QUALITY: RequestedCaptureQuality = "720p"

export function normalizeManagedStreamQuality(value: unknown): RequestedCaptureQuality | undefined {
  return value === "720p" || value === "1080p" ? value : undefined
}

export function doesManagedStreamQualityMatch(
  existingQuality: unknown,
  requestedQuality: RequestedCaptureQuality,
): boolean {
  return normalizeManagedStreamQuality(existingQuality) === requestedQuality
}

export function isRetryEligibleManagedStreamError(error: unknown): boolean {
  const code = typeof error === "object" && error !== null && "code" in error ? String(error.code || "") : ""
  const message = error instanceof Error ? error.message.toLowerCase() : String(error || "").toLowerCase()

  if (
    code.includes("PERMISSION") ||
    code.includes("AUTH") ||
    code.includes("UNAUTHORIZED") ||
    code.includes("FORBIDDEN")
  ) {
    return false
  }

  if (
    message.includes("permission") ||
    message.includes("not authorized") ||
    message.includes("unauthorized") ||
    message.includes("forbidden") ||
    message.includes("already in use") ||
    message.includes("stream conflict")
  ) {
    return false
  }

  return (
    message.includes("timeout") ||
    message.includes("timed out") ||
    message.includes("network") ||
    message.includes("transport") ||
    message.includes("webrtc") ||
    message.includes("socket") ||
    message.includes("quality") ||
    message.includes("resolution") ||
    message.includes("temporarily") ||
    message.includes("transient")
  )
}
