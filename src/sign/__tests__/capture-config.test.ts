import {describe, expect, it} from "bun:test"
import {effectiveGridFps, gridIntervalMs, landmarkIntervalMs, windowLabel} from "../capture-config"

describe("effectiveGridFps", () => {
  it("caps at 12 when captureFps exceeds 12", () => {
    expect(effectiveGridFps(24)).toBe(12)
    expect(effectiveGridFps(30)).toBe(12)
  })

  it("passes through when captureFps is at or below 12", () => {
    expect(effectiveGridFps(10)).toBe(10)
    expect(effectiveGridFps(12)).toBe(12)
  })
})

describe("landmarkIntervalMs", () => {
  it("returns ~41.67ms for 24fps", () => {
    expect(landmarkIntervalMs(24)).toBeCloseTo(41.667, 1)
  })

  it("returns ~83.33ms for 12fps", () => {
    expect(landmarkIntervalMs(12)).toBeCloseTo(83.333, 1)
  })
})

describe("gridIntervalMs", () => {
  it("returns ~83.33ms for 24fps (capped at 12fps grid)", () => {
    expect(gridIntervalMs(24)).toBeCloseTo(83.333, 1)
  })

  it("returns 100ms for 10fps (uncapped)", () => {
    expect(gridIntervalMs(10)).toBe(100)
  })
})

describe("windowLabel", () => {
  it("formats the landmark window at 24fps", () => {
    expect(windowLabel(24, 48)).toBe("48 @ 24fps (~2.0s)")
  })

  it("formats the grid window at 12fps", () => {
    expect(windowLabel(12, 16)).toBe("16 @ 12fps (~1.3s)")
  })
})
