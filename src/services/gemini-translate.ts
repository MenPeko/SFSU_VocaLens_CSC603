import {
  buildGeminiUrl,
  buildGeminiErrorMessage,
  deduplicateModels,
  discoverAdditionalGenerateContentModels,
  TRANSIENT_STATUS_CODES,
  MAX_RETRIES,
  RETRY_BASE_DELAY_MS,
} from "./gemini-ocr"

export interface TranslationResult {
  originalText: string
  translatedText: string
  detectedLanguage: string
  targetLanguage: string
}

export const TRANSLATION_PROMPT_TEMPLATE = `You are a language detection and translation assistant.

Given the following text, respond ONLY with a JSON object (no markdown, no code fences) in this exact format:
{"detectedLanguage":"<language name in English>","translatedText":"<translated text>"}

If the text is already in TARGET_LANGUAGE, set translatedText to the original text unchanged.
If no text is provided or it's unreadable, respond with:
{"detectedLanguage":"unknown","translatedText":""}

Target language: TARGET_LANGUAGE

Text to translate:
"""
TEXT_TO_TRANSLATE
"""`

export function buildTranslationPrompt(text: string, targetLanguage: string): string {
  return TRANSLATION_PROMPT_TEMPLATE.replace("TARGET_LANGUAGE", targetLanguage)
    .replace("TARGET_LANGUAGE", targetLanguage)
    .replace("TEXT_TO_TRANSLATE", text)
}

export function parseTranslationResponse(responseText: string): {detectedLanguage: string; translatedText: string} {
  if (!responseText) return {detectedLanguage: "unknown", translatedText: ""}

  // Strip markdown code fences if present
  const cleaned = responseText.replace(/```json\s*/g, "").replace(/```\s*/g, "").trim()

  try {
    const parsed = JSON.parse(cleaned)
    return {
      detectedLanguage: parsed.detectedLanguage || "unknown",
      translatedText: parsed.translatedText || "",
    }
  } catch {
    // If JSON parse fails, return the raw text as translation
    return {detectedLanguage: "unknown", translatedText: responseText.trim()}
  }
}

export async function translateText(text: string, targetLanguage: string): Promise<TranslationResult> {
  const apiKey = process.env.GEMINI_API_KEY
  const model = process.env.GEMINI_MODEL || "gemini-2.5-flash"

  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not set")
  }

  if (!text.trim()) {
    return {originalText: text, translatedText: "", detectedLanguage: "unknown", targetLanguage}
  }

  const prompt = buildTranslationPrompt(text, targetLanguage)
  let modelsToTry = deduplicateModels(model)
  let attemptedModelDiscovery = false

  let response: Response | null = null
  let lastError = "Translation request failed"
  let responseStatus: number | null = null

  for (let modelIndex = 0; modelIndex < modelsToTry.length; modelIndex += 1) {
    const currentModel = modelsToTry[modelIndex]
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      response = await fetch(buildGeminiUrl(apiKey, currentModel), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          contents: [{parts: [{text: prompt}]}],
        }),
      })
      responseStatus = response.status

      if (response.ok) break

      const isTransient = TRANSIENT_STATUS_CODES.includes(response.status)
      if (isTransient && attempt < MAX_RETRIES - 1) {
        await new Promise((r) => setTimeout(r, RETRY_BASE_DELAY_MS * (attempt + 1)))
        continue
      }

      const errorBody = await response.text()
      lastError = buildGeminiErrorMessage(response.status, response.statusText, errorBody)
      break
    }

    if (response?.ok) break

    const shouldTryNextModel = response && (response.status === 404 || TRANSIENT_STATUS_CODES.includes(response.status))
    if (!shouldTryNextModel) {
      throw new Error(lastError)
    }

    if (response?.status === 404 && modelIndex === modelsToTry.length - 1 && !attemptedModelDiscovery) {
      attemptedModelDiscovery = true

      try {
        const discoveredModels = await discoverAdditionalGenerateContentModels(apiKey, modelsToTry, model)
        if (discoveredModels.length > 0) {
          modelsToTry = [...modelsToTry, ...discoveredModels]
        }
      } catch {
        // Fall through to the original error if discovery is unavailable.
      }
    }
  }

  if (!response || !response.ok) {
    if (responseStatus === 404 && attemptedModelDiscovery) {
      throw new Error(
        `${lastError} Tried configured fallbacks and Gemini API model discovery, but no accessible generateContent model was found.`,
      )
    }

    throw new Error(lastError)
  }

  const payload = (await response.json()) as {
    candidates?: Array<{content?: {parts?: Array<{text?: string}>}}>
  }

  const rawResponse = payload.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("").trim() || ""
  const {detectedLanguage, translatedText} = parseTranslationResponse(rawResponse)

  return {
    originalText: text,
    translatedText: translatedText || text,
    detectedLanguage,
    targetLanguage,
  }
}
