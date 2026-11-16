import {describe, expect, it} from "bun:test"
import {normalizeForWer, wordErrorRate} from "../../../scripts/vocalens/wer"

describe("word error rate", () => {
  it("is zero when the transcript matches", () => {
    expect(wordErrorRate("What does this label say?", "what does this label say")).toBe(0)
  })

  it("counts a missing word against the reference length", () => {
    expect(wordErrorRate("the cat sat", "the cat")).toBeCloseTo(1 / 3)
  })

  it("normalizes case and punctuation before counting", () => {
    expect(normalizeForWer("Hello, world!")).toEqual(["hello", "world"])
  })
})
