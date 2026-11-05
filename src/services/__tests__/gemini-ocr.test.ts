import {describe, it, expect, beforeEach, afterEach, mock, spyOn} from "bun:test"
import {
  buildGeminiModelsListUrl,
  buildGeminiErrorMessage,
  buildGeminiRequestBody,
  buildGeminiUrl,
  clearGeminiModelDiscoveryCache,
  discoverAdditionalGenerateContentModels,
  extractGenerateContentModelNames,
  parseGeminiResponse,
  deduplicateModels,
  extractTextFromImageWithGemini,
  OCR_PROMPT,
} from "../gemini-ocr"

// ============================================
// buildGeminiErrorMessage
// ============================================

describe("buildGeminiErrorMessage", () => {
  it("detects expired API key", () => {
    const msg = buildGeminiErrorMessage(401, "Unauthorized", '{"error": "API key expired"}')
    expect(msg).toBe("Gemini API key expired. Please renew your key.")
  })

  it("detects invalid API key (api_key_invalid)", () => {
    const msg = buildGeminiErrorMessage(400, "Bad Request", '{"error": "API_KEY_INVALID"}')
    expect(msg).toBe("Gemini API key is invalid.")
  })

  it("detects invalid API key (invalid api key)", () => {
    const msg = buildGeminiErrorMessage(403, "Forbidden", "Invalid API key provided")
    expect(msg).toBe("Gemini API key is invalid.")
  })

  it("detects permission denied", () => {
    const msg = buildGeminiErrorMessage(403, "Forbidden", '{"error": "PERMISSION_DENIED"}')
    expect(msg).toBe("Gemini API key does not have permission for this model.")
  })

  it("detects model not found via status 404", () => {
    const msg = buildGeminiErrorMessage(404, "Not Found", "model does not exist")
    expect(msg).toBe("Gemini model not found for this API key.")
  })

  it("detects model not found via body text", () => {
    const msg = buildGeminiErrorMessage(400, "Bad Request", "Model not found in registry")
    expect(msg).toBe("Gemini model not found for this API key.")
  })

  it("returns generic error for unknown status", () => {
    const msg = buildGeminiErrorMessage(503, "Service Unavailable", "server overloaded")
    expect(msg).toBe("Gemini OCR request failed: 503 Service Unavailable")
  })

  it("is case insensitive for error body matching", () => {
    const msg = buildGeminiErrorMessage(401, "Unauthorized", "API KEY EXPIRED")
    expect(msg).toBe("Gemini API key expired. Please renew your key.")
  })
})

// ============================================
// buildGeminiUrl
// ============================================

describe("buildGeminiUrl", () => {
  it("builds correct URL with model and key", () => {
    const url = buildGeminiUrl("test-key", "gemini-2.5-flash")
    expect(url).toBe(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=test-key",
    )
  })

  it("encodes special characters in model name", () => {
    const url = buildGeminiUrl("key", "model/with spaces")
    expect(url).toContain("model%2Fwith%20spaces")
  })

  it("encodes special characters in API key", () => {
    const url = buildGeminiUrl("key&with=special", "gemini-2.5-flash")
    expect(url).toContain("key%26with%3Dspecial")
  })
})

describe("buildGeminiModelsListUrl", () => {
  it("builds correct URL for model discovery", () => {
    const url = buildGeminiModelsListUrl("test-key")
    expect(url).toBe("https://generativelanguage.googleapis.com/v1beta/models?key=test-key")
  })
})

describe("extractGenerateContentModelNames", () => {
  it("returns normalized names for models that support generateContent", () => {
    const names = extractGenerateContentModelNames({
      models: [
        {name: "models/gemini-2.5-flash", supportedGenerationMethods: ["generateContent"]},
        {name: "models/text-embedding-004", supportedGenerationMethods: ["embedContent"]},
        {name: "gemini-1.5-flash", supportedActions: ["generateContent"]},
      ],
    })

    expect(names).toEqual(["gemini-2.5-flash", "gemini-1.5-flash"])
  })
})

// ============================================
// buildGeminiRequestBody
// ============================================

describe("buildGeminiRequestBody", () => {
  it("creates valid JSON with base64 image", () => {
    const buffer = Buffer.from("fake image data")
    const body = JSON.parse(buildGeminiRequestBody("test prompt", buffer, "image/jpeg"))

    expect(body.contents).toHaveLength(1)
    expect(body.contents[0].parts).toHaveLength(2)
    expect(body.contents[0].parts[0].text).toBe("test prompt")
    expect(body.contents[0].parts[1].inline_data.mime_type).toBe("image/jpeg")
    expect(body.contents[0].parts[1].inline_data.data).toBe(buffer.toString("base64"))
  })

  it("defaults mime_type to image/jpeg when empty", () => {
    const buffer = Buffer.from("data")
    const body = JSON.parse(buildGeminiRequestBody("prompt", buffer, ""))

    expect(body.contents[0].parts[1].inline_data.mime_type).toBe("image/jpeg")
  })

  it("preserves custom mime type", () => {
    const buffer = Buffer.from("data")
    const body = JSON.parse(buildGeminiRequestBody("prompt", buffer, "image/png"))

    expect(body.contents[0].parts[1].inline_data.mime_type).toBe("image/png")
  })
})

// ============================================
// parseGeminiResponse
// ============================================

describe("parseGeminiResponse", () => {
  it("extracts text from valid response", () => {
    const payload = {
      candidates: [
        {
          content: {
            parts: [{text: "Hello world"}],
          },
        },
      ],
    }
    expect(parseGeminiResponse(payload)).toBe("Hello world")
  })

  it("joins multiple parts with newlines", () => {
    const payload = {
      candidates: [
        {
          content: {
            parts: [{text: "Line 1"}, {text: "Line 2"}],
          },
        },
      ],
    }
    expect(parseGeminiResponse(payload)).toBe("Line 1\nLine 2")
  })

  it("trims whitespace", () => {
    const payload = {
      candidates: [
        {
          content: {
            parts: [{text: "  trimmed  "}],
          },
        },
      ],
    }
    expect(parseGeminiResponse(payload)).toBe("trimmed")
  })

  it("returns empty string for missing candidates", () => {
    expect(parseGeminiResponse({})).toBe("")
    expect(parseGeminiResponse({candidates: []})).toBe("")
  })

  it("returns empty string for missing content", () => {
    expect(parseGeminiResponse({candidates: [{}]})).toBe("")
    expect(parseGeminiResponse({candidates: [{content: {}}]})).toBe("")
  })

  it("handles parts with no text field", () => {
    const payload = {
      candidates: [
        {
          content: {
            parts: [{}, {text: "valid"}],
          },
        },
      ],
    }
    expect(parseGeminiResponse(payload)).toBe("valid")
  })

  it("handles null/undefined input", () => {
    expect(parseGeminiResponse(null)).toBe("")
    expect(parseGeminiResponse(undefined)).toBe("")
  })
})

// ============================================
// deduplicateModels
// ============================================

describe("deduplicateModels", () => {
  it("returns unique models in order", () => {
    const models = deduplicateModels("gemini-2.5-flash")
    expect(models).toEqual(["gemini-2.5-flash", "gemini-2.0-flash"])
  })

  it("puts configured model first", () => {
    const models = deduplicateModels("gemini-custom")
    expect(models).toEqual(["gemini-custom", "gemini-2.5-flash", "gemini-2.0-flash"])
  })

  it("deduplicates when configured model matches a fallback", () => {
    const models = deduplicateModels("gemini-2.0-flash")
    expect(models).toEqual(["gemini-2.0-flash", "gemini-2.5-flash"])
  })
})

// ============================================
// OCR_PROMPT
// ============================================

describe("OCR_PROMPT", () => {
  it("includes OCR instruction", () => {
    expect(OCR_PROMPT).toContain("OCR assistant")
  })

  it("includes NO_TEXT_FOUND sentinel", () => {
    expect(OCR_PROMPT).toContain("NO_TEXT_FOUND")
  })

  it("instructs plain text only output", () => {
    expect(OCR_PROMPT).toContain("plain text only")
  })
})

// ============================================
// extractTextFromImageWithGemini (integration-style)
// ============================================

describe("extractTextFromImageWithGemini", () => {
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

    await expect(extractTextFromImageWithGemini(Buffer.from("img"), "image/jpeg")).rejects.toThrow(
      "GEMINI_API_KEY is not set",
    )
  })

  it("returns extracted text on success", async () => {
    const mockResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: "Board says: Hello"}]}}],
      }),
      {status: 200},
    )

    const fetchSpy = spyOn(globalThis, "fetch").mockResolvedValue(mockResponse)

    const result = await extractTextFromImageWithGemini(Buffer.from("img"), "image/jpeg")
    expect(result.text).toBe("Board says: Hello")
    expect(fetchSpy).toHaveBeenCalledTimes(1)
  })

  it("retries on 503 and succeeds", async () => {
    const failResponse = new Response("overloaded", {status: 503, statusText: "Service Unavailable"})
    const successResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: "Recovered text"}]}}],
      }),
      {status: 200},
    )

    const fetchSpy = spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(failResponse)
      .mockResolvedValueOnce(successResponse)

    const result = await extractTextFromImageWithGemini(Buffer.from("img"), "image/jpeg")
    expect(result.text).toBe("Recovered text")
    expect(fetchSpy).toHaveBeenCalledTimes(2)
  })

  it("retries on 429 rate limit", async () => {
    const rateLimitResponse = new Response("rate limited", {status: 429, statusText: "Too Many Requests"})
    const successResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: "OK"}]}}],
      }),
      {status: 200},
    )

    const fetchSpy = spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(rateLimitResponse)
      .mockResolvedValueOnce(successResponse)

    const result = await extractTextFromImageWithGemini(Buffer.from("img"), "image/jpeg")
    expect(result.text).toBe("OK")
    expect(fetchSpy).toHaveBeenCalledTimes(2)
  })

  it("throws after all retries and model fallbacks exhausted on 503", async () => {
    const failResponse = () => new Response("overloaded", {status: 503, statusText: "Service Unavailable"})

    // 3 retries x 2 models (gemini-2.5-flash, gemini-2.0-flash) = 6 calls
    const spy = spyOn(globalThis, "fetch")
    for (let i = 0; i < 6; i++) {
      spy.mockResolvedValueOnce(failResponse())
    }

    await expect(extractTextFromImageWithGemini(Buffer.from("img"), "image/jpeg")).rejects.toThrow(
      "Gemini OCR request failed: 503 Service Unavailable",
    )
  }, 15000)

  it("falls back to next model on 503 after retries exhausted", async () => {
    const failResponse = () => new Response("overloaded", {status: 503, statusText: "Service Unavailable"})
    const successResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: "Fallback after 503"}]}}],
      }),
      {status: 200},
    )

    // First model: 3 failures, second model: success
    const spy = spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(failResponse())
      .mockResolvedValueOnce(failResponse())
      .mockResolvedValueOnce(failResponse())
      .mockResolvedValueOnce(successResponse)

    const result = await extractTextFromImageWithGemini(Buffer.from("img"), "image/jpeg")
    expect(result.text).toBe("Fallback after 503")
    expect(spy).toHaveBeenCalledTimes(4)
  })

  it("falls back to next model on 404", async () => {
    process.env.GEMINI_MODEL = "nonexistent-model"

    const notFoundResponse = new Response("model not found", {status: 404, statusText: "Not Found"})
    const successResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: "Fallback worked"}]}}],
      }),
      {status: 200},
    )

    const fetchSpy = spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(notFoundResponse)
      .mockResolvedValueOnce(successResponse)

    const result = await extractTextFromImageWithGemini(Buffer.from("img"), "image/jpeg")
    expect(result.text).toBe("Fallback worked")

    // First call should use nonexistent-model, second should use gemini-2.5-flash
    const firstCallUrl = fetchSpy.mock.calls[0][0] as string
    const secondCallUrl = fetchSpy.mock.calls[1][0] as string
    expect(firstCallUrl).toContain("nonexistent-model")
    expect(secondCallUrl).toContain("gemini-2.5-flash")
  })

  it("does not retry on 401 auth error", async () => {
    const authError = new Response("API_KEY_INVALID", {status: 401, statusText: "Unauthorized"})

    const fetchSpy = spyOn(globalThis, "fetch").mockResolvedValueOnce(authError)

    await expect(extractTextFromImageWithGemini(Buffer.from("img"), "image/jpeg")).rejects.toThrow(
      "Gemini API key is invalid.",
    )
    expect(fetchSpy).toHaveBeenCalledTimes(1)
  })

  it("discovers an accessible model when all built-in fallbacks are not found", async () => {
    process.env.GEMINI_MODEL = "nonexistent-model"

    const notFoundResponse = () => new Response("model not found", {status: 404, statusText: "Not Found"})
    const discoveryResponse = new Response(
      JSON.stringify({
        models: [
          {name: "models/gemini-1.5-pro", supportedGenerationMethods: ["generateContent"]},
        ],
      }),
      {status: 200},
    )
    const successResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: "Discovered fallback worked"}]}}],
      }),
      {status: 200},
    )

    const fetchSpy = spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(notFoundResponse())
      .mockResolvedValueOnce(notFoundResponse())
      .mockResolvedValueOnce(notFoundResponse())
      .mockResolvedValueOnce(discoveryResponse)
      .mockResolvedValueOnce(successResponse)

    const result = await extractTextFromImageWithGemini(Buffer.from("img"), "image/jpeg")
    expect(result.text).toBe("Discovered fallback worked")
    expect(fetchSpy).toHaveBeenCalledTimes(5)

    const discoveryCallUrl = fetchSpy.mock.calls[3][0] as string
    const discoveredModelUrl = fetchSpy.mock.calls[4][0] as string
    expect(discoveryCallUrl).toContain("/v1beta/models?key=test-api-key")
    expect(discoveredModelUrl).toContain("gemini-1.5-pro")
  })

  it("sends correct request format to Gemini", async () => {
    const successResponse = new Response(
      JSON.stringify({
        candidates: [{content: {parts: [{text: "test"}]}}],
      }),
      {status: 200},
    )

    const fetchSpy = spyOn(globalThis, "fetch").mockResolvedValue(successResponse)

    const imageBuffer = Buffer.from("test image data")
    await extractTextFromImageWithGemini(imageBuffer, "image/png")

    const [url, options] = fetchSpy.mock.calls[0] as [string, RequestInit]
    expect(url).toContain("gemini-2.5-flash")
    expect(url).toContain("key=test-api-key")
    expect(options.method).toBe("POST")

    const body = JSON.parse(options.body as string)
    expect(body.contents[0].parts[0].text).toContain("OCR assistant")
    expect(body.contents[0].parts[1].inline_data.mime_type).toBe("image/png")
    expect(body.contents[0].parts[1].inline_data.data).toBe(imageBuffer.toString("base64"))
  })
})

describe("discoverAdditionalGenerateContentModels", () => {
  beforeEach(() => {
    clearGeminiModelDiscoveryCache()
  })

  afterEach(() => {
    clearGeminiModelDiscoveryCache()
    mock.restore()
  })

  it("returns available generateContent models that have not already been tried", async () => {
    spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          models: [
            {name: "models/gemini-1.5-flash", supportedGenerationMethods: ["generateContent"]},
            {name: "models/gemini-1.5-pro", supportedGenerationMethods: ["generateContent"]},
            {name: "models/text-embedding-004", supportedGenerationMethods: ["embedContent"]},
          ],
        }),
        {status: 200},
      ),
    )

    const models = await discoverAdditionalGenerateContentModels(
      "test-api-key",
      ["gemini-2.5-flash", "gemini-2.0-flash"],
      "gemini-2.5-flash",
    )

    expect(models).toEqual(["gemini-1.5-flash", "gemini-1.5-pro"])
  })
})
