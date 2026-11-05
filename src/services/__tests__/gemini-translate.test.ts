import {describe, it, expect, beforeEach, afterEach, mock, spyOn} from "bun:test"
import {clearGeminiModelDiscoveryCache} from "../gemini-ocr"
import {
  buildTranslationPrompt,
  parseTranslationResponse,
  translateText,
} from "../gemini-translate"

// ============================================
// buildTranslationPrompt
// ============================================

describe("buildTranslationPrompt", () => {
  it("includes target language", () => {
    const prompt = buildTranslationPrompt("Hola mundo", "English")
    expect(prompt).toContain("Target language: English")
  })

  it("includes text to translate", () => {
    const prompt = buildTranslationPrompt("Bonjour le monde", "English")
    expect(prompt).toContain("Bonjour le monde")
  })

  it("instructs JSON-only response", () => {
    const prompt = buildTranslationPrompt("test", "Spanish")
    expect(prompt).toContain("JSON")
    expect(prompt).toContain("detectedLanguage")
    expect(prompt).toContain("translatedText")
  })
})

// ============================================
// parseTranslationResponse
// ============================================

describe("parseTranslationResponse", () => {
  it("parses valid JSON response", () => {
    const result = parseTranslationResponse('{"detectedLanguage":"Spanish","translatedText":"Hello world"}')
    expect(result.detectedLanguage).toBe("Spanish")
    expect(result.translatedText).toBe("Hello world")
  })

  it("handles JSON wrapped in code fences", () => {
    const result = parseTranslationResponse('```json\n{"detectedLanguage":"French","translatedText":"Hello"}\n```')
    expect(result.detectedLanguage).toBe("French")
    expect(result.translatedText).toBe("Hello")
  })

  it("returns unknown for empty input", () => {
    const result = parseTranslationResponse("")
    expect(result.detectedLanguage).toBe("unknown")
    expect(result.translatedText).toBe("")
  })

  it("returns raw text as translation when JSON parsing fails", () => {
    const result = parseTranslationResponse("Just some plain text")
    expect(result.detectedLanguage).toBe("unknown")
    expect(result.translatedText).toBe("Just some plain text")
  })

  it("handles missing fields in JSON", () => {
    const result = parseTranslationResponse("{}")
    expect(result.detectedLanguage).toBe("unknown")
    expect(result.translatedText).toBe("")
  })

  it("handles JSON with extra whitespace", () => {
    const result = parseTranslationResponse('  {"detectedLanguage":"German","translatedText":"Hallo"}  ')
    expect(result.detectedLanguage).toBe("German")
    expect(result.translatedText).toBe("Hallo")
  })
})

// ============================================
// translateText
// ============================================

describe("translateText", () => {
  const originalEnv = {...process.env}

  beforeEach(() => {
    process.env.GEMINI_API_KEY = "test-api-key"
    process.env.GEMINI_MODEL = "gemini-2.5-flash"
    clearGeminiModelDiscoveryCache()
  })

  afterEach(() => {
    process.env = {...originalEnv}
    clearGeminiModelDiscoveryCache()
    mock.restore()
  })

  it("throws when GEMINI_API_KEY is not set", async () => {
    delete process.env.GEMINI_API_KEY
    await expect(translateText("hello", "Spanish")).rejects.toThrow("GEMINI_API_KEY is not set")
  })

  it("returns empty result for empty text", async () => {
    const result = await translateText("", "Spanish")
    expect(result.translatedText).toBe("")
    expect(result.detectedLanguage).toBe("unknown")
  })

  it("returns translation on success", async () => {
    const mockResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: '{"detectedLanguage":"Spanish","translatedText":"Hello world"}'}]}}],
      }),
      {status: 200},
    )

    spyOn(globalThis, "fetch").mockResolvedValue(mockResponse)

    const result = await translateText("Hola mundo", "English")
    expect(result.originalText).toBe("Hola mundo")
    expect(result.translatedText).toBe("Hello world")
    expect(result.detectedLanguage).toBe("Spanish")
    expect(result.targetLanguage).toBe("English")
  })

  it("retries on 503 and succeeds", async () => {
    const failResponse = new Response("overloaded", {status: 503, statusText: "Service Unavailable"})
    const successResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: '{"detectedLanguage":"French","translatedText":"Hello"}'}]}}],
      }),
      {status: 200},
    )

    const fetchSpy = spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(failResponse)
      .mockResolvedValueOnce(successResponse)

    const result = await translateText("Bonjour", "English")
    expect(result.translatedText).toBe("Hello")
    expect(fetchSpy).toHaveBeenCalledTimes(2)
  })

  it("throws after all retries and model fallbacks exhausted", async () => {
    const fail = () => new Response("overloaded", {status: 503, statusText: "Service Unavailable"})

    // 3 retries x 2 models (gemini-2.5-flash, gemini-2.0-flash) = 6 total fetch calls
    const spy = spyOn(globalThis, "fetch")
    for (let i = 0; i < 6; i++) {
      spy.mockResolvedValueOnce(fail())
    }

    await expect(translateText("test", "English")).rejects.toThrow("503 Service Unavailable")
  }, 15000)

  it("falls back to next model on 503 after retries", async () => {
    const fail = () => new Response("overloaded", {status: 503, statusText: "Service Unavailable"})
    const successResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: '{"detectedLanguage":"Spanish","translatedText":"Hello"}'}]}}],
      }),
      {status: 200},
    )

    // First model: 3 failures, second model: success
    const spy = spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(fail())
      .mockResolvedValueOnce(fail())
      .mockResolvedValueOnce(fail())
      .mockResolvedValueOnce(successResponse)

    const result = await translateText("Hola", "English")
    expect(result.translatedText).toBe("Hello")
    expect(spy).toHaveBeenCalledTimes(4)
  })

  it("discovers an accessible model when configured translation fallbacks are not found", async () => {
    process.env.GEMINI_MODEL = "nonexistent-model"

    const notFound = () => new Response("model not found", {status: 404, statusText: "Not Found"})
    const discoveryResponse = new Response(
      JSON.stringify({
        models: [
          {name: "models/gemini-1.5-flash", supportedGenerationMethods: ["generateContent"]},
        ],
      }),
      {status: 200},
    )
    const successResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: '{"detectedLanguage":"Spanish","translatedText":"Hello"}'}]}}],
      }),
      {status: 200},
    )

    const fetchSpy = spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(notFound())
      .mockResolvedValueOnce(notFound())
      .mockResolvedValueOnce(notFound())
      .mockResolvedValueOnce(discoveryResponse)
      .mockResolvedValueOnce(successResponse)

    const result = await translateText("Hola", "English")
    expect(result.translatedText).toBe("Hello")
    expect(fetchSpy).toHaveBeenCalledTimes(5)
    expect((fetchSpy.mock.calls[3][0] as string)).toContain("/v1beta/models?key=test-api-key")
    expect((fetchSpy.mock.calls[4][0] as string)).toContain("gemini-1.5-flash")
  })

  it("falls back to original text when translation is empty", async () => {
    const mockResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: '{"detectedLanguage":"unknown","translatedText":""}'}]}}],
      }),
      {status: 200},
    )

    spyOn(globalThis, "fetch").mockResolvedValue(mockResponse)

    const result = await translateText("Some text", "English")
    expect(result.translatedText).toBe("Some text")
  })

  it("sends target language in the prompt", async () => {
    const mockResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: '{"detectedLanguage":"English","translatedText":"Hola"}'}]}}],
      }),
      {status: 200},
    )

    const fetchSpy = spyOn(globalThis, "fetch").mockResolvedValue(mockResponse)

    await translateText("Hello", "Spanish")

    const body = JSON.parse((fetchSpy.mock.calls[0] as [string, RequestInit])[1].body as string)
    const promptText = body.contents[0].parts[0].text
    expect(promptText).toContain("Target language: Spanish")
    expect(promptText).toContain("Hello")
  })
})
