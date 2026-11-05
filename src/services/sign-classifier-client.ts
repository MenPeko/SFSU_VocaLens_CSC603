import type {
  SignClassifierResult,
  SignLandmarkFrame,
  SignCaptureSource,
} from "../sign/types"

const DEFAULT_CLASSIFIER_URL = "http://127.0.0.1:8001/classify-sign-sequence"
const DEFAULT_CLASSIFIER_TIMEOUT_MS = 2000

function getSignClassifierTimeoutMs(): number {
  const configuredTimeout = Number(process.env.SIGN_CLASSIFIER_TIMEOUT_MS || DEFAULT_CLASSIFIER_TIMEOUT_MS)
  return Number.isFinite(configuredTimeout) && configuredTimeout > 0
    ? configuredTimeout
    : DEFAULT_CLASSIFIER_TIMEOUT_MS
}

function getErrorCauseMessage(error: Error): string | null {
  const cause = (error as Error & {cause?: unknown}).cause
  if (typeof cause === "string" && cause.trim()) {
    return cause.trim()
  }
  if (cause instanceof Error && cause.message.trim()) {
    return cause.message.trim()
  }
  if (
    typeof cause === "object" &&
    cause &&
    "message" in cause &&
    typeof cause.message === "string" &&
    cause.message.trim()
  ) {
    return cause.message.trim()
  }
  return null
}

function formatClassifierTransportError(error: unknown, requestUrl: string, timeoutMs: number): string {
  if (error instanceof Error && error.name === "AbortError") {
    return `Local sign classifier timed out after ${timeoutMs}ms at ${requestUrl}.`
  }

  if (error instanceof Error) {
    const message = error.message.trim()
    const causeMessage = getErrorCauseMessage(error)
    const combinedMessage =
      causeMessage && causeMessage !== message
        ? `${message} (${causeMessage})`
        : message

    if (
      message === "fetch failed" ||
      message === "Failed to fetch" ||
      combinedMessage.toLowerCase().includes("connect") ||
      combinedMessage.toLowerCase().includes("econnrefused")
    ) {
      return `Could not reach the local sign classifier at ${requestUrl}. ${combinedMessage}`.trim()
    }

    if (combinedMessage) {
      return `Local sign classifier request failed: ${combinedMessage}`
    }
  }

  return `Local sign classifier request failed before a response was received from ${requestUrl}.`
}

export interface ClassifySignSequenceInput {
  frames: SignLandmarkFrame[]
  fps: number
  source: SignCaptureSource
  sequenceDurationMs: number
  // Must match MEDIAPIPE_EXTRACTOR_VERSION of the capture pipeline. The
  // Python service compares this against the artifact's extractor and
  // refuses to serve on mismatch.
  extractorVersion: string
}

export function getSignClassifierUrl(): string {
  return process.env.SIGN_CLASSIFIER_URL || DEFAULT_CLASSIFIER_URL
}

export function isSignClassifierEnabled(): boolean {
  return process.env.SIGN_CLASSIFIER_DISABLED !== "true"
}

export async function classifySignSequence(input: ClassifySignSequenceInput): Promise<SignClassifierResult> {
  if (!isSignClassifierEnabled()) {
    throw new Error("Sign classifier disabled")
  }

  const requestUrl = getSignClassifierUrl()
  const timeoutMs = getSignClassifierTimeoutMs()
  const controller = new AbortController()
  const timeoutId = setTimeout(
    () => controller.abort(),
    timeoutMs,
  )

  try {
    let response: Response
    try {
      response = await fetch(requestUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(input),
        signal: controller.signal,
      })
    } catch (error) {
      throw new Error(formatClassifierTransportError(error, requestUrl, timeoutMs))
    }

    if (!response.ok) {
      const responseText = (await response.text()).trim()
      throw new Error(
        `Local sign classifier returned HTTP ${response.status}: ${responseText || response.statusText || "No response body"}`,
      )
    }

    try {
      return (await response.json()) as SignClassifierResult
    } catch (error) {
      const message =
        error instanceof Error && error.message.trim()
          ? error.message.trim()
          : "Invalid JSON response"
      throw new Error(`Local sign classifier returned invalid JSON: ${message}`)
    }
  } finally {
    clearTimeout(timeoutId)
  }
}
