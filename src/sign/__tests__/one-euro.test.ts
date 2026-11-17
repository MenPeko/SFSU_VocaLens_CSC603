import {describe, expect, it} from "bun:test"
import {LandmarkListSmoother, OneEuroFilter} from "../one-euro"

const FRAME_MS = 1000 / 30

describe("OneEuroFilter", () => {
  it("passes a constant signal through unchanged", () => {
    const filter = new OneEuroFilter()
    for (let frame = 0; frame < 30; frame += 1) {
      expect(filter.filter(0.42, frame * FRAME_MS)).toBeCloseTo(0.42, 10)
    }
  })

  it("converges on a step input", () => {
    const filter = new OneEuroFilter()
    filter.filter(0, 0)

    let value = 0
    for (let frame = 1; frame <= 30; frame += 1) {
      value = filter.filter(1, frame * FRAME_MS)
    }

    expect(value).toBeGreaterThan(0.95)
    expect(value).toBeLessThanOrEqual(1)
  })

  it("damps small jitter around a still point", () => {
    const filter = new OneEuroFilter()
    let maxDeviation = 0
    for (let frame = 0; frame < 60; frame += 1) {
      const jitter = frame % 2 === 0 ? 0.004 : -0.004
      const value = filter.filter(0.5 + jitter, frame * FRAME_MS)
      if (frame > 10) {
        maxDeviation = Math.max(maxDeviation, Math.abs(value - 0.5))
      }
    }

    expect(maxDeviation).toBeLessThan(0.004)
  })

  it("restarts from the raw value after reset or a tracking gap", () => {
    const filter = new OneEuroFilter()
    filter.filter(0, 0)
    filter.filter(0, FRAME_MS)

    filter.reset()
    expect(filter.filter(0.8, 2 * FRAME_MS)).toBe(0.8)

    expect(filter.filter(0.1, 2 * FRAME_MS + 1000)).toBe(0.1)
  })
})

describe("LandmarkListSmoother", () => {
  it("keeps extra landmark fields and resets on an empty list", () => {
    const smoother = new LandmarkListSmoother()
    smoother.smooth([{x: 0, y: 0, z: 0, visibility: 0.9}], 0)
    const moved = smoother.smooth([{x: 1, y: 1, z: 0, visibility: 0.8}], FRAME_MS)

    expect(moved[0].x).toBeLessThan(1)
    expect(moved[0].visibility).toBe(0.8)

    expect(smoother.smooth([], 2 * FRAME_MS)).toEqual([])
    const reappeared = smoother.smooth([{x: 1, y: 1, z: 0}], 3 * FRAME_MS)
    expect(reappeared[0].x).toBe(1)
  })
})
