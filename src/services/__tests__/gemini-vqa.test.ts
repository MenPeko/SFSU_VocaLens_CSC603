import {describe, expect, it} from "bun:test"
import {OCR_PROMPT} from "../gemini-ocr"
import {
  buildVisualQuestionBody,
  mergeGeminiStreamChunk,
  takeCompleteSentences,
  textFromGeminiSseLine,
  VQA_PROMPT,
} from "../gemini-vqa"

describe("visual question prompt", () => {
  it("asks for a short word-for-word answer and says when the image is unclear", () => {
    expect(VQA_PROMPT).toContain("short answer")
    expect(VQA_PROMPT).toContain("word for word")
    expect(VQA_PROMPT).toContain("unclear")
    expect(VQA_PROMPT).toContain("reframe")
    expect(VQA_PROMPT).not.toBe(OCR_PROMPT)
    expect(OCR_PROMPT).toContain("Extract only the readable text")
  })

  it("attaches the image to the first question and leaves follow-ups as text", () => {
    const body = JSON.parse(
      buildVisualQuestionBody("Prompt", "abc123", "image/jpeg", [
        {role: "user", text: "What does this say?"},
        {role: "model", text: "Milk."},
        {role: "user", text: "What about the date?"},
      ]),
    ) as {
      contents: Array<{role: string; parts: Array<{text?: string; inline_data?: {data: string}}>}>
    }

    expect(body.contents).toHaveLength(3)
    expect(body.contents[0].parts.some((part) => part.inline_data?.data === "abc123")).toBe(true)
    expect(body.contents[0].parts[0].text).toContain("What does this say?")
    expect(body.contents[1].parts[0].text).toBe("Milk.")
    expect(body.contents[2].parts).toHaveLength(1)
    expect(body.contents[2].parts[0].inline_data).toBeUndefined()
    expect(body.contents[2].parts[0].text).toContain("What about the date?")
  })

  it("asks for the selected language on the first question and on follow-ups", () => {
    const body = JSON.parse(
      buildVisualQuestionBody(
        "Prompt",
        "abc123",
        "image/jpeg",
        [
          {role: "user", text: "What is this?"},
          {role: "model", text: "A can."},
          {role: "user", text: "What color is it?"},
        ],
        "Spanish",
      ),
    ) as {contents: Array<{parts: Array<{text?: string}>}>}

    expect(body.contents[0].parts[0].text).toContain("Write the answer in Spanish")
    expect(body.contents[0].parts[0].text).toContain("What is this?")
    expect(body.contents[2].parts[0].text).toContain("Write the answer in Spanish")
    expect(body.contents[2].parts[0].text).toContain("What color is it?")
  })

  it("treats a repeated stream prefix as the full answer so far", () => {
    expect(mergeGeminiStreamChunk("Milk", "Milk is in the cup.")).toBe("Milk is in the cup.")
    expect(mergeGeminiStreamChunk("Milk", " is fresh.")).toBe("Milk is fresh.")
  })

  it("emits a sentence only after its ending punctuation", () => {
    expect(takeCompleteSentences("The label says milk. The date")).toEqual({
      sentences: ["The label says milk."],
      rest: "The date",
    })
  })

  it("reads text out of a Gemini event stream line", () => {
    const line = `data: ${JSON.stringify({
      candidates: [{content: {parts: [{text: "A red cup."}]}}],
    })}`
    expect(textFromGeminiSseLine(line)).toBe("A red cup.")
  })
})
