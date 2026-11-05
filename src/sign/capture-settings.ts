export type RequestedCaptureQuality = "720p" | "1080p"
export type ActualCaptureQuality = RequestedCaptureQuality | "sub-720p" | "unknown"
export type SkeletonRenderMode = "presentation" | "diagnostic"
export type CaptureSettingsStatus = "loading" | "loaded" | "fallback"
export type CaptureSettingsFallbackReason = "timeout" | "request_error"
export type HolisticModelComplexity = 0 | 1 | 2

export interface SignCaptureSettings {
  holisticModelComplexity: HolisticModelComplexity
}

export const DEFAULT_SIGN_CAPTURE_SETTINGS: SignCaptureSettings = {
  holisticModelComplexity: 2,
}

export function normalizeHolisticModelComplexity(value: unknown): HolisticModelComplexity {
  if (value === 0 || value === 1 || value === 2) {
    return value
  }

  if (typeof value === "string") {
    const trimmed = value.trim()
    if (trimmed === "0" || trimmed === "1" || trimmed === "2") {
      return Number(trimmed) as HolisticModelComplexity
    }
  }

  return DEFAULT_SIGN_CAPTURE_SETTINGS.holisticModelComplexity
}

export function normalizeSignCaptureSettings(payload: {holisticModelComplexity?: unknown} | null | undefined): SignCaptureSettings {
  return {
    holisticModelComplexity: normalizeHolisticModelComplexity(payload?.holisticModelComplexity),
  }
}

export function deriveActualCaptureQuality(
  width: number | null | undefined,
  height: number | null | undefined,
): ActualCaptureQuality {
  const resolvedWidth = Number(width) || 0
  const resolvedHeight = Number(height) || 0

  if (resolvedWidth <= 0 || resolvedHeight <= 0) {
    return "unknown"
  }

  if (resolvedWidth >= 1920 && resolvedHeight >= 1080) {
    return "1080p"
  }

  if (resolvedWidth >= 1280 && resolvedHeight >= 720) {
    return "720p"
  }

  return "sub-720p"
}

export function inferQualityFallback(
  requestedQuality: RequestedCaptureQuality | undefined,
  actualQuality: ActualCaptureQuality | undefined,
): boolean {
  if (!requestedQuality || !actualQuality || actualQuality === "unknown") {
    return false
  }

  return actualQuality !== requestedQuality
}
