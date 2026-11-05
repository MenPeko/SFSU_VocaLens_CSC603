import {
  DEFAULT_SIGN_CAPTURE_SETTINGS,
  normalizeSignCaptureSettings,
  type CaptureSettingsFallbackReason,
  type SignCaptureSettings,
} from "../sign/capture-settings"

export interface SignCaptureBootstrapResult {
  settings: SignCaptureSettings
  status: "loaded" | "fallback"
  fallbackReason?: CaptureSettingsFallbackReason
}

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

const SETTINGS_TIMEOUT_SENTINEL = "__SIGN_CAPTURE_SETTINGS_TIMEOUT__"

export async function fetchSignCaptureBootstrap(
  fetchImpl: FetchLike = fetch,
  timeoutMs = 3000,
): Promise<SignCaptureBootstrapResult> {
  let timeoutId: ReturnType<typeof setTimeout> | null = null

  try {
    const timeoutPromise = new Promise<never>((_, reject) => {
      timeoutId = setTimeout(() => reject(new Error(SETTINGS_TIMEOUT_SENTINEL)), timeoutMs)
    })

    const response = await Promise.race([fetchImpl("/api/settings/sign-capture"), timeoutPromise])
    if (!response.ok) {
      throw new Error(`Sign capture settings request failed with HTTP ${response.status}`)
    }

    const payload = (await response.json().catch(() => ({}))) as {holisticModelComplexity?: unknown}
    return {
      settings: normalizeSignCaptureSettings(payload),
      status: "loaded",
    }
  } catch (error) {
    const fallbackReason: CaptureSettingsFallbackReason =
      error instanceof Error && error.message === SETTINGS_TIMEOUT_SENTINEL ? "timeout" : "request_error"

    return {
      settings: DEFAULT_SIGN_CAPTURE_SETTINGS,
      status: "fallback",
      fallbackReason,
    }
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
  }
}
