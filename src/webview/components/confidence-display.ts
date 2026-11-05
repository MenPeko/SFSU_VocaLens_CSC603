import type {SignTranslationDebugResult} from "./SignLanguageCapture"

// Confidence thresholds — these match the values baked into
// sign-translation-resolver.ts (recognizedConfidence: 0.80, uncertainFloor: 0.40)
export const CONFIDENCE_HIGH = 0.80
export const CONFIDENCE_UNCERTAIN = 0.40

export type ConfidenceLevel = "high" | "uncertain" | "low"

export interface ConfidenceDisplay {
  level: ConfidenceLevel
  color: string
  label: string
  percent: string
  barWidth: string
}

export function getConfidenceLevel(confidence: number): ConfidenceLevel {
  if (confidence >= CONFIDENCE_HIGH) return "high"
  if (confidence >= CONFIDENCE_UNCERTAIN) return "uncertain"
  return "low"
}

export function getConfidenceColor(confidence: number): string {
  const level = getConfidenceLevel(confidence)
  if (level === "high") return "#22C55E"
  if (level === "uncertain") return "#F59E0B"
  return "#EF4444"
}

export function getConfidenceLabel(confidence: number): string {
  const level = getConfidenceLevel(confidence)
  if (level === "high") return "High"
  if (level === "uncertain") return "Uncertain"
  return "Low"
}

export function getConfidenceDisplay(confidence: number): ConfidenceDisplay {
  return {
    level: getConfidenceLevel(confidence),
    color: getConfidenceColor(confidence),
    label: getConfidenceLabel(confidence),
    percent: `${(confidence * 100).toFixed(1)}%`,
    barWidth: `${Math.min(100, confidence * 100)}%`,
  }
}

export interface ConfidenceFeedback {
  message: string
  borderColor: string
  backgroundColor: string
  textColor: string
}

export function getConfidenceFeedback(confidence: number): ConfidenceFeedback | null {
  if (confidence <= 0) return null

  if (confidence < CONFIDENCE_UNCERTAIN) {
    return {
      message: "Low confidence \u2014 try signing again with clearer hand positioning",
      borderColor: "#EF4444",
      backgroundColor: "#FEF2F2",
      textColor: "#991B1B",
    }
  }

  if (confidence < CONFIDENCE_HIGH) {
    return {
      message: "Uncertain result \u2014 the classifier was not confident enough, routed to Gemini",
      borderColor: "#F59E0B",
      backgroundColor: "#FFFBEB",
      textColor: "#92400E",
    }
  }

  return null
}

export function getRecognizerBadgeColor(recognizerType: string | undefined): string {
  return recognizerType === "classifier" ? "#1E90FF" : "#FF6600"
}

export function getClassifierStatusLabel(status: string | undefined): string {
  if (!status) return "n/a"
  return status.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())
}

export function getClassifierStatusColor(status: string | undefined): string {
  if (status === "recognized") return "#22C55E"
  if (status === "uncertain") return "#F59E0B"
  if (status === "no_sign") return "#EF4444"
  return "#888888"
}

export function shouldShowAlternatives(result: SignTranslationDebugResult): boolean {
  return (
    result.operation !== "capture" &&
    Array.isArray(result.classifierAlternatives) &&
    result.classifierAlternatives.length > 0
  )
}
