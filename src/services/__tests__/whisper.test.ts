import {describe, expect, it} from "bun:test"
import {formatGeminiTranscriptionError, selectTranscriptionProvider, transcribeAudio, transcribeWithGemini} from "../whisper"

describe("whisper provider selection", () => {
  it("uses the Whisper API when a key is available", () => {
    expect(selectTranscriptionProvider({hasApiKey: true, apiFailed: false})).toBe("openai")
  })

  it("falls back to local faster-whisper when the API fails or the key is missing", () => {
    expect(selectTranscriptionProvider({hasApiKey: true, apiFailed: true})).toBe("faster-whisper")
    expect(selectTranscriptionProvider({hasApiKey: false, apiFailed: false})).toBe("faster-whisper")
  })

  it("uses the configured Gemini key when the Whisper key is missing", () => {
    expect(selectTranscriptionProvider({hasApiKey: false, apiFailed: false, hasGeminiKey: true})).toBe("gemini")
  })

  it("tries local faster-whisper after the API call fails", async () => {
    const result = await transcribeAudio(Buffer.from("audio"), "audio/webm", {
      apiKey: "test-key",
      geminiApiKey: "",
      transcribeWithOpenAi: async () => {
        throw new Error("Whisper API request failed: 503")
      },
      transcribeLocally: async () => "what does this label say",
    })

    expect(result).toEqual({text: "what does this label say", provider: "faster-whisper"})
  })

  it("reports both failures when the API and the local fallback fail", async () => {
    await expect(
      transcribeAudio(Buffer.from("audio"), "audio/webm", {
        apiKey: "test-key",
        geminiApiKey: "",
        transcribeWithOpenAi: async () => {
          throw new Error("Whisper API request failed: 401")
        },
        transcribeLocally: async () => {
          throw new Error("Local faster-whisper failed: not installed")
        },
      }),
    ).rejects.toThrow("Speech transcription failed")
  })

  it("transcribes with Gemini when the Whisper key is not set", async () => {
    const result = await transcribeAudio(Buffer.from("audio"), "audio/webm;codecs=opus", {
      apiKey: "",
      geminiApiKey: "gemini-key",
      transcribeWithGemini: async (_audio, mimeType, apiKey) => {
        expect(mimeType).toBe("audio/webm;codecs=opus")
        expect(apiKey).toBe("gemini-key")
        return "what is on the sign"
      },
      transcribeLocally: async () => {
        throw new Error("local fallback should not run")
      },
    })

    expect(result).toEqual({text: "what is on the sign", provider: "gemini"})
  })

  it("tells the user when the local recording is silent", async () => {
    await expect(
      transcribeAudio(Buffer.from("audio"), "audio/webm", {
        apiKey: "test-key",
        geminiApiKey: "gemini-key",
        transcribeWithOpenAi: async () => {
          throw new Error("Whisper API request failed: 429 Too Many Requests")
        },
        transcribeWithGemini: async () => {
          throw new Error("Gemini transcription request failed: 403 Forbidden")
        },
        transcribeLocally: async () => {
          throw new Error("Local faster-whisper failed: The microphone recording was silent. Hold the button and speak the question.")
        },
      }),
    ).rejects.toThrow("The microphone recording was silent")
  })

  it("retries Gemini transcription when a model returns 403", async () => {
    const originalFetch = globalThis.fetch
    const models: string[] = []
    globalThis.fetch = (async (input: string | URL | Request) => {
      const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url)
      models.push(url.pathname)
      if (models.length === 1) {
        return new Response(JSON.stringify({error: {status: "PERMISSION_DENIED", message: "Permission denied"}}), {
          status: 403,
          statusText: "Forbidden",
        })
      }
      return new Response(JSON.stringify({candidates: [{content: {parts: [{text: "what is on the page"}]}}]}), {
        status: 200,
        statusText: "OK",
      })
    }) as typeof fetch

    try {
      const text = await transcribeWithGemini(Buffer.from("audio"), "audio/webm", "test-key", "gemini-2.5-flash")
      expect(text).toBe("what is on the page")
      expect(models[0]).toContain("gemini-2.5-flash")
      expect(models[1]).toContain("gemini-2.0-flash")
    } finally {
      globalThis.fetch = originalFetch
    }
  })

  it("keeps Gemini key material out of transcription errors", () => {
    const message = formatGeminiTranscriptionError(
      403,
      "Forbidden",
      JSON.stringify({error: {status: "PERMISSION_DENIED", message: "blocked key=AIzaSecretValue"}}),
    )
    expect(message).not.toContain("AIzaSecretValue")
    expect(message).toContain("permission")
  })
})
