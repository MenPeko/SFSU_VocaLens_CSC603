import {describe, expect, it} from "bun:test"
import {pickLongestSequence, segmentHandSequences, type OfflineFrame} from "../offline-sequence"

function frames(pattern: string, intervalMs = 100): OfflineFrame<number>[] {
  return [...pattern].map((symbol, index) => ({timestampMs: index * intervalMs, hasHands: symbol === "h", frame: index}))
}

describe("segmentHandSequences", () => {
  it("keeps only hand-visible frames and ends at the video end", () => {
    const [sequence] = segmentHandSequences(frames("..hhhhhhhh.."))
    expect(sequence.frames).toEqual([2, 3, 4, 5, 6, 7, 8, 9])
  })

  it("bridges a hand gap shorter than the hands-down delay", () => {
    const sequences = segmentHandSequences(frames("hhhhhh..hhhhhh"), {handsDownDelayMs: 500})
    expect(sequences).toHaveLength(1)
    expect(sequences[0].frames).toHaveLength(12)
  })

  it("splits on a gap at least as long as the hands-down delay", () => {
    const sequences = segmentHandSequences(frames("hhhhhhh......hhhhhhhhh"), {handsDownDelayMs: 500})
    expect(sequences).toHaveLength(2)
    expect(pickLongestSequence(sequences)?.frames[0]).toBe(13)
  })

  it("drops sequences shorter than the minimum duration", () => {
    expect(segmentHandSequences(frames("hhh"), {minDurationMs: 500})).toEqual([])
  })

  it("keeps the last frames of a long sequence, like the live rolling window", () => {
    const [sequence] = segmentHandSequences(frames("h".repeat(60)), {maxFrames: 48})
    expect(sequence.frames).toHaveLength(48)
    expect(sequence.frames[0]).toBe(12)
  })
})
