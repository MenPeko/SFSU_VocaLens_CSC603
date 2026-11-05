import {describe, expect, it} from "bun:test"
import {
  doesManagedStreamQualityMatch,
  isRetryEligibleManagedStreamError,
  normalizeManagedStreamQuality,
} from "../managed-stream-quality"

describe("normalizeManagedStreamQuality", () => {
  it("accepts supported quality presets only", () => {
    expect(normalizeManagedStreamQuality("1080p")).toBe("1080p")
    expect(normalizeManagedStreamQuality("720p")).toBe("720p")
    expect(normalizeManagedStreamQuality("480p")).toBeUndefined()
  })
})

describe("doesManagedStreamQualityMatch", () => {
  it("returns true only when the existing stream quality matches the requested quality", () => {
    expect(doesManagedStreamQualityMatch("1080p", "1080p")).toBe(true)
    expect(doesManagedStreamQualityMatch("720p", "1080p")).toBe(false)
    expect(doesManagedStreamQualityMatch(undefined, "1080p")).toBe(false)
  })
})

describe("isRetryEligibleManagedStreamError", () => {
  it("retries transient startup failures", () => {
    expect(isRetryEligibleManagedStreamError(new Error("Managed stream request timeout"))).toBe(true)
    expect(isRetryEligibleManagedStreamError(new Error("Network transport failed during stream startup"))).toBe(true)
    expect(isRetryEligibleManagedStreamError(new Error("Requested quality could not be negotiated"))).toBe(true)
  })

  it("does not retry terminal permission or conflict failures", () => {
    expect(isRetryEligibleManagedStreamError(new Error("The glasses camera is already in use by another stream."))).toBe(
      false,
    )
    expect(isRetryEligibleManagedStreamError(new Error("Permission denied for managed stream"))).toBe(false)
    expect(isRetryEligibleManagedStreamError({code: "AUTH_FORBIDDEN"})).toBe(false)
  })
})
