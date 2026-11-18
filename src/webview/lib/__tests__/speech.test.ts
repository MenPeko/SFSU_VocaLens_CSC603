import {describe, expect, it} from "bun:test"
import {clampSpeechRate, MAX_SPEECH_RATE, MIN_SPEECH_RATE, speechLanguageCode, splitSpeakableSentences} from "../speech"

describe("speech helpers", () => {
  it("clamps the rate to the phone slider range", () => {
    expect(clampSpeechRate(0.2)).toBe(MIN_SPEECH_RATE)
    expect(clampSpeechRate(2)).toBe(MAX_SPEECH_RATE)
    expect(clampSpeechRate(1.1)).toBe(1.1)
    expect(clampSpeechRate(Number.NaN)).toBe(1)
  })

  it("maps the selected language onto a speech voice", () => {
    expect(speechLanguageCode("Spanish")).toBe("es")
    expect(speechLanguageCode("German")).toBe("de")
    expect(speechLanguageCode(undefined)).toBeUndefined()
  })

  it("splits an answer on sentence boundaries", () => {
    expect(splitSpeakableSentences("The label says milk. The date is unclear.")).toEqual([
      "The label says milk.",
      "The date is unclear.",
    ])
  })
})
