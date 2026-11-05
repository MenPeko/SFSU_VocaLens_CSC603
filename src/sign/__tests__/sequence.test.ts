import {describe, expect, it} from "bun:test"
import {appendRollingWindow, computeSequenceDurationMs} from "../sequence"

describe("appendRollingWindow", () => {
  it("keeps only the newest items within the max size", () => {
    const result = [1, 2, 3, 4].reduce((items, value) => appendRollingWindow(items, value, 3), [] as number[])
    expect(result).toEqual([2, 3, 4])
  })

  it("handles the larger 48-frame rolling window", () => {
    const items = Array.from({length: 50}, (_, i) => i)
    const result = items.reduce((acc, v) => appendRollingWindow(acc, v, 48), [] as number[])
    expect(result).toHaveLength(48)
    expect(result[0]).toBe(2)
    expect(result[47]).toBe(49)
  })
})

describe("computeSequenceDurationMs", () => {
  it("returns zero when fewer than two frames are present", () => {
    expect(computeSequenceDurationMs([])).toBe(0)
    expect(computeSequenceDurationMs([1000])).toBe(0)
  })

  it("computes the time span between first and last frames", () => {
    expect(computeSequenceDurationMs([1000, 1083, 1166, 1249])).toBe(249)
  })
})
