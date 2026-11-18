import {MEDIAPIPE_EXTRACTOR_VERSION} from "./extractor-version"

// Local-only experiment: overrides Holistic minTrackingConfidence for the live
// hook. Payloads get a distinct extractor stamp so the classifier and trainer
// reject them instead of mixing them with schema-locked samples.
export function parseTrackingConfidenceExperiment(value: string | null): number | null {
  if (value === null || value.trim() === "") {
    return null
  }

  const parsed = Number(value)
  return isTrackingConfidenceExperiment(parsed) ? parsed : null
}

// `undefined !== null` is true, and React renders undefined as blank text.
// The banner must stay hidden unless Holistic is actually overridden.
export function isTrackingConfidenceExperiment(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value < 1
}

export function experimentExtractorVersion(trackingConfidence: number | null): string {
  return trackingConfidence === null
    ? MEDIAPIPE_EXTRACTOR_VERSION
    : `${MEDIAPIPE_EXTRACTOR_VERSION}+exp-tracking-${trackingConfidence}`
}
