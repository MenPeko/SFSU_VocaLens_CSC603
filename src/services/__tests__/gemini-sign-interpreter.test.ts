import {describe, it, expect, beforeEach, afterEach, mock, spyOn} from "bun:test"
import {
  SIGN_INTERPRETER_SYSTEM_PROMPT,
  buildSignInterpreterPrompt,
  buildSignInterpreterRequestBody,
  normalizeSignInterpretationResponse,
  interpretSignGridWithGemini,
} from "../gemini-sign-interpreter"

describe("buildSignInterpreterPrompt", () => {
  it("includes the chronological grid description", () => {
    const prompt = buildSignInterpreterPrompt(4, 4)
    expect(prompt).toContain("4x4")
    expect(prompt).toContain("continuous ASL phrase")
  })

  it("includes fingerpose hints when present", () => {
    const prompt = buildSignInterpreterPrompt(4, 4, [{name: "open_palm", score: 8.2}])
    expect(prompt).toContain("open_palm (8.2)")
  })

  it("mentions the skeleton reference grid when present", () => {
    const prompt = buildSignInterpreterPrompt(4, 4, [], true)
    expect(prompt).toContain("skeleton-rendered grid")
  })
})

describe("buildSignInterpreterRequestBody", () => {
  it("creates a request body with system instruction and inline image data", () => {
    const body = JSON.parse(buildSignInterpreterRequestBody("interpret this", "ZmFrZQ==", "image/jpeg"))

    expect(body.system_instruction.parts[0].text).toBe(SIGN_INTERPRETER_SYSTEM_PROMPT)
    expect(body.contents[0].parts[0].text).toBe("interpret this")
    expect(body.contents[0].parts[2].inline_data.data).toBe("ZmFrZQ==")
  })

  it("adds the skeleton reference image when provided", () => {
    const body = JSON.parse(
      buildSignInterpreterRequestBody("interpret this", "ZmFrZQ==", "image/jpeg", "c2tlbGV0b24=", "image/png"),
    )

    expect(body.contents[0].parts[4].inline_data.data).toBe("c2tlbGV0b24=")
    expect(body.contents[0].parts[4].inline_data.mime_type).toBe("image/png")
  })

  it("defaults the skeleton MIME type to the RGB MIME type when absent", () => {
    const body = JSON.parse(
      buildSignInterpreterRequestBody("interpret this", "ZmFrZQ==", "image/jpeg", "c2tlbGV0b24="),
    )

    expect(body.contents[0].parts[4].inline_data.mime_type).toBe("image/jpeg")
  })
})

describe("normalizeSignInterpretationResponse", () => {
  it("keeps plain English text", () => {
    expect(normalizeSignInterpretationResponse("Hello there")).toBe("Hello there")
  })

  it("removes markdown fences and prefixes", () => {
    expect(normalizeSignInterpretationResponse("```text\nEnglish: Nice to meet you\n```")).toBe("Nice to meet you")
  })

  it("extracts englishText from JSON if returned", () => {
    expect(normalizeSignInterpretationResponse('{"englishText":"Thank you"}')).toBe("Thank you")
  })
})

describe("interpretSignGridWithGemini", () => {
  const originalEnv = {...process.env}

  beforeEach(() => {
    process.env.GEMINI_API_KEY = "test-api-key"
    process.env.GEMINI_SIGN_MODEL = "gemini-1.5-flash"
  })

  afterEach(() => {
    process.env = {...originalEnv}
    mock.restore()
  })

  it("throws when GEMINI_API_KEY is not set", async () => {
    delete process.env.GEMINI_API_KEY

    await expect(interpretSignGridWithGemini({base64Grid: "ZmFrZQ=="})).rejects.toThrow("GEMINI_API_KEY is not set")
  })

  it("returns normalized English text on success", async () => {
    const mockResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: "English: I need help"}]}}],
      }),
      {status: 200},
    )

    spyOn(globalThis, "fetch").mockResolvedValue(mockResponse)

    const result = await interpretSignGridWithGemini({
      base64Grid: "ZmFrZQ==",
      gestureHints: [{name: "pointing", score: 8.9}],
    })

    expect(result.englishText).toBe("I need help")
    expect(result.model).toBe("gemini-1.5-flash")
  })
})
