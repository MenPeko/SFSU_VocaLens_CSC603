import {readFileSync} from "node:fs"
import {describe, expect, it} from "bun:test"

const source = readFileSync(new URL("../CaptionsApp.ts", import.meta.url), "utf8")

describe("glasses written text flow", () => {
  it("still extracts text with Gemini and speaks on the glasses", () => {
    expect(source).toContain("extractTextFromImageWithGemini")
    expect(source).toContain("session.audio.speak")
    expect(source).not.toContain("VQA_PROMPT")
    expect(source).not.toContain("streamVisualAnswer")
    expect(source).not.toContain("/api/vocalens/ask")
  })
})
