import {describe, expect, it} from "bun:test"
import {
  CONFIDENCE_HIGH,
  CONFIDENCE_UNCERTAIN,
  getClassifierStatusColor,
  getClassifierStatusLabel,
  getConfidenceLevel,
  getConfidenceColor,
  getConfidenceLabel,
  getConfidenceDisplay,
  getConfidenceFeedback,
  getRecognizerBadgeColor,
  shouldShowAlternatives,
} from "../confidence-display"
import type {SignTranslationDebugResult} from "../SignLanguageCapture"

// ---------------------------------------------------------------------------
// getConfidenceLevel
// ---------------------------------------------------------------------------
describe("getConfidenceLevel", () => {
  it("returns 'high' for confidence >= 0.80", () => {
    expect(getConfidenceLevel(0.80)).toBe("high")
    expect(getConfidenceLevel(0.91)).toBe("high")
    expect(getConfidenceLevel(1.0)).toBe("high")
  })

  it("returns 'uncertain' for confidence >= 0.40 and < 0.80", () => {
    expect(getConfidenceLevel(0.40)).toBe("uncertain")
    expect(getConfidenceLevel(0.63)).toBe("uncertain")
    expect(getConfidenceLevel(0.799)).toBe("uncertain")
  })

  it("returns 'low' for confidence < 0.40", () => {
    expect(getConfidenceLevel(0.39)).toBe("low")
    expect(getConfidenceLevel(0.12)).toBe("low")
    expect(getConfidenceLevel(0.0)).toBe("low")
  })

  it("handles exact boundary at 0.80", () => {
    expect(getConfidenceLevel(0.80)).toBe("high")
    expect(getConfidenceLevel(0.7999)).toBe("uncertain")
  })

  it("handles exact boundary at 0.40", () => {
    expect(getConfidenceLevel(0.40)).toBe("uncertain")
    expect(getConfidenceLevel(0.3999)).toBe("low")
  })
})

// ---------------------------------------------------------------------------
// getConfidenceColor
// ---------------------------------------------------------------------------
describe("getConfidenceColor", () => {
  it("returns green for high confidence", () => {
    expect(getConfidenceColor(0.91)).toBe("#22C55E")
  })

  it("returns yellow/amber for uncertain confidence", () => {
    expect(getConfidenceColor(0.63)).toBe("#F59E0B")
  })

  it("returns red for low confidence", () => {
    expect(getConfidenceColor(0.12)).toBe("#EF4444")
  })
})

// ---------------------------------------------------------------------------
// getConfidenceLabel
// ---------------------------------------------------------------------------
describe("getConfidenceLabel", () => {
  it("returns 'High' for high confidence", () => {
    expect(getConfidenceLabel(0.95)).toBe("High")
  })

  it("returns 'Uncertain' for uncertain confidence", () => {
    expect(getConfidenceLabel(0.55)).toBe("Uncertain")
  })

  it("returns 'Low' for low confidence", () => {
    expect(getConfidenceLabel(0.20)).toBe("Low")
  })
})

// ---------------------------------------------------------------------------
// getConfidenceDisplay
// ---------------------------------------------------------------------------
describe("getConfidenceDisplay", () => {
  it("returns a complete display object for high confidence", () => {
    const display = getConfidenceDisplay(0.91)
    expect(display.level).toBe("high")
    expect(display.color).toBe("#22C55E")
    expect(display.label).toBe("High")
    expect(display.percent).toBe("91.0%")
    expect(display.barWidth).toBe("91%")
  })

  it("formats zero confidence correctly", () => {
    const display = getConfidenceDisplay(0)
    expect(display.level).toBe("low")
    expect(display.percent).toBe("0.0%")
    expect(display.barWidth).toBe("0%")
  })

  it("caps bar width at 100%", () => {
    const display = getConfidenceDisplay(1.05)
    expect(display.barWidth).toBe("100%")
  })

  it("formats fractional confidence to one decimal place", () => {
    const display = getConfidenceDisplay(0.6789)
    expect(display.percent).toBe("67.9%")
  })
})

// ---------------------------------------------------------------------------
// getConfidenceFeedback
// ---------------------------------------------------------------------------
describe("getConfidenceFeedback", () => {
  it("returns low-confidence message below 0.40", () => {
    const feedback = getConfidenceFeedback(0.25)
    expect(feedback).not.toBeNull()
    expect(feedback!.message).toContain("Low confidence")
    expect(feedback!.message).toContain("try signing again")
    expect(feedback!.borderColor).toBe("#EF4444")
  })

  it("returns uncertain message between 0.40 and 0.80", () => {
    const feedback = getConfidenceFeedback(0.63)
    expect(feedback).not.toBeNull()
    expect(feedback!.message).toContain("Uncertain")
    expect(feedback!.message).toContain("routed to Gemini")
    expect(feedback!.borderColor).toBe("#F59E0B")
  })

  it("returns null for high confidence (no feedback needed)", () => {
    expect(getConfidenceFeedback(0.91)).toBeNull()
    expect(getConfidenceFeedback(0.80)).toBeNull()
    expect(getConfidenceFeedback(1.0)).toBeNull()
  })

  it("returns null for zero confidence (no actionable feedback)", () => {
    expect(getConfidenceFeedback(0)).toBeNull()
  })

  it("returns low-confidence feedback at confidence just above zero", () => {
    const feedback = getConfidenceFeedback(0.01)
    expect(feedback).not.toBeNull()
    expect(feedback!.message).toContain("Low confidence")
  })

  it("returns uncertain feedback at exactly 0.40", () => {
    const feedback = getConfidenceFeedback(0.40)
    expect(feedback).not.toBeNull()
    expect(feedback!.message).toContain("Uncertain")
  })

  it("returns null at exactly 0.80", () => {
    expect(getConfidenceFeedback(0.80)).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// getRecognizerBadgeColor
// ---------------------------------------------------------------------------
describe("getRecognizerBadgeColor", () => {
  it("returns blue for classifier", () => {
    expect(getRecognizerBadgeColor("classifier")).toBe("#1E90FF")
  })

  it("returns orange for gemini", () => {
    expect(getRecognizerBadgeColor("gemini")).toBe("#FF6600")
  })

  it("returns orange for undefined (fallback color)", () => {
    expect(getRecognizerBadgeColor(undefined)).toBe("#FF6600")
  })

  it("returns orange for unknown recognizer types", () => {
    expect(getRecognizerBadgeColor("unknown")).toBe("#FF6600")
  })
})

// ---------------------------------------------------------------------------
// shouldShowAlternatives
// ---------------------------------------------------------------------------
describe("shouldShowAlternatives", () => {
  const baseResult: SignTranslationDebugResult = {
    success: true,
    requestDurationMs: 350,
    frameCount: 16,
  }

  it("returns true when translate operation has alternatives", () => {
    expect(shouldShowAlternatives({
      ...baseResult,
      operation: "translate",
      classifierAlternatives: [
        {label: "wait", confidence: 0.91},
        {label: "stop", confidence: 0.04},
      ],
    })).toBe(true)
  })

  it("returns false when operation is capture (training mode)", () => {
    expect(shouldShowAlternatives({
      ...baseResult,
      operation: "capture",
      classifierAlternatives: [
        {label: "wait", confidence: 0.91},
      ],
    })).toBe(false)
  })

  it("returns false when alternatives array is empty", () => {
    expect(shouldShowAlternatives({
      ...baseResult,
      operation: "translate",
      classifierAlternatives: [],
    })).toBe(false)
  })

  it("returns false when alternatives is undefined", () => {
    expect(shouldShowAlternatives({
      ...baseResult,
      operation: "translate",
    })).toBe(false)
  })

  it("returns true when operation is undefined (defaults to translate)", () => {
    expect(shouldShowAlternatives({
      ...baseResult,
      classifierAlternatives: [
        {label: "help", confidence: 0.72},
      ],
    })).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// getClassifierStatusLabel
// ---------------------------------------------------------------------------
describe("getClassifierStatusLabel", () => {
  it("returns title-cased label for recognized", () => {
    expect(getClassifierStatusLabel("recognized")).toBe("Recognized")
  })

  it("returns title-cased label for uncertain", () => {
    expect(getClassifierStatusLabel("uncertain")).toBe("Uncertain")
  })

  it("replaces underscores and title-cases no_sign", () => {
    expect(getClassifierStatusLabel("no_sign")).toBe("No Sign")
  })

  it("returns title-cased label for unavailable", () => {
    expect(getClassifierStatusLabel("unavailable")).toBe("Unavailable")
  })

  it("returns title-cased label for skipped", () => {
    expect(getClassifierStatusLabel("skipped")).toBe("Skipped")
  })

  it("returns 'n/a' for undefined", () => {
    expect(getClassifierStatusLabel(undefined)).toBe("n/a")
  })
})

// ---------------------------------------------------------------------------
// getClassifierStatusColor
// ---------------------------------------------------------------------------
describe("getClassifierStatusColor", () => {
  it("returns green for recognized", () => {
    expect(getClassifierStatusColor("recognized")).toBe("#22C55E")
  })

  it("returns amber for uncertain", () => {
    expect(getClassifierStatusColor("uncertain")).toBe("#F59E0B")
  })

  it("returns red for no_sign", () => {
    expect(getClassifierStatusColor("no_sign")).toBe("#EF4444")
  })

  it("returns gray for unavailable", () => {
    expect(getClassifierStatusColor("unavailable")).toBe("#888888")
  })

  it("returns gray for skipped", () => {
    expect(getClassifierStatusColor("skipped")).toBe("#888888")
  })

  it("returns gray for undefined", () => {
    expect(getClassifierStatusColor(undefined)).toBe("#888888")
  })
})

// ---------------------------------------------------------------------------
// Threshold constants
// ---------------------------------------------------------------------------
describe("threshold constants", () => {
  it("exports the correct threshold values matching the resolver", () => {
    expect(CONFIDENCE_HIGH).toBe(0.80)
    expect(CONFIDENCE_UNCERTAIN).toBe(0.40)
  })
})
