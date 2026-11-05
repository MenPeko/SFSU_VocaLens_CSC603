export interface GeminiOcrResult {
  text: string
}

interface GeminiModelListResponse {
  models?: Array<{
    name?: string
    supportedGenerationMethods?: string[]
    supportedActions?: string[]
    supported_actions?: string[]
  }>
}

const DISCOVERY_PREFERRED_MODELS = [
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "gemini-2.0-flash",
  "gemini-1.5-flash",
  "gemini-1.5-pro",
]
const MODEL_DISCOVERY_CACHE_TTL_MS = 5 * 60 * 1000

let modelDiscoveryCache:
  | {
      apiKey: string
      expiresAt: number
      models: string[]
    }
  | null = null

export function buildGeminiErrorMessage(status: number, statusText: string, errorBody: string): string {
  const normalized = errorBody.toLowerCase()

  if (normalized.includes("api key expired")) {
    return "Gemini API key expired. Please renew your key."
  }

  if (normalized.includes("api_key_invalid") || normalized.includes("invalid api key")) {
    return "Gemini API key is invalid."
  }

  if (normalized.includes("permission_denied")) {
    return "Gemini API key does not have permission for this model."
  }

  if (status === 404 || normalized.includes("not found")) {
    return "Gemini model not found for this API key."
  }

  return `Gemini OCR request failed: ${status} ${statusText}`
}

export function buildGeminiRequestBody(prompt: string, imageBuffer: Buffer, mimeType: string): string {
  return JSON.stringify({
    contents: [
      {
        parts: [
          {text: prompt},
          {
            inline_data: {
              mime_type: mimeType || "image/jpeg",
              data: imageBuffer.toString("base64"),
            },
          },
        ],
      },
    ],
  })
}

export function buildGeminiUrl(apiKey: string, model: string): string {
  return `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`
}

export function buildGeminiModelsListUrl(apiKey: string): string {
  return `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(apiKey)}`
}

export function normalizeGeminiModelName(name: string | undefined): string | null {
  if (!name) {
    return null
  }

  const trimmed = name.trim()
  if (!trimmed) {
    return null
  }

  return trimmed.startsWith("models/") ? trimmed.slice("models/".length) : trimmed
}

export function extractGenerateContentModelNames(payload: unknown): string[] {
  const typed = payload as GeminiModelListResponse
  const names = (typed.models || [])
    .filter((model) => {
      const supportedMethods =
        model.supportedGenerationMethods || model.supportedActions || model.supported_actions || []
      return supportedMethods.includes("generateContent")
    })
    .map((model) => normalizeGeminiModelName(model.name))
    .filter((name): name is string => Boolean(name))

  return Array.from(new Set(names))
}

export function parseGeminiResponse(payload: unknown): string {
  if (!payload) return ""
  const typed = payload as {
    candidates?: Array<{
      content?: {
        parts?: Array<{text?: string}>
      }
    }>
  }
  return typed.candidates?.[0]?.content?.parts?.map((part) => part.text || "").join("\n").trim() || ""
}

export function deduplicateModels(configuredModel: string): string[] {
  const fallbackModels = [configuredModel, "gemini-2.5-flash", "gemini-2.0-flash"]
  return Array.from(new Set(fallbackModels))
}

export function clearGeminiModelDiscoveryCache(): void {
  modelDiscoveryCache = null
}

async function listAvailableGenerateContentModels(apiKey: string): Promise<string[]> {
  if (
    modelDiscoveryCache &&
    modelDiscoveryCache.apiKey === apiKey &&
    modelDiscoveryCache.expiresAt > Date.now()
  ) {
    return modelDiscoveryCache.models
  }

  const response = await fetch(buildGeminiModelsListUrl(apiKey))
  if (!response.ok) {
    throw new Error(`Gemini model discovery failed: ${response.status} ${response.statusText}`)
  }

  const payload = (await response.json()) as GeminiModelListResponse
  const models = extractGenerateContentModelNames(payload)

  modelDiscoveryCache = {
    apiKey,
    expiresAt: Date.now() + MODEL_DISCOVERY_CACHE_TTL_MS,
    models,
  }

  return models
}

export async function discoverAdditionalGenerateContentModels(
  apiKey: string,
  alreadyTriedModels: string[],
  configuredModel: string,
): Promise<string[]> {
  const availableModels = await listAvailableGenerateContentModels(apiKey)
  if (availableModels.length === 0) {
    return []
  }

  const availableSet = new Set(availableModels)
  const triedSet = new Set(alreadyTriedModels.map((model) => normalizeGeminiModelName(model)).filter(Boolean))
  const orderedCandidates = [configuredModel, ...DISCOVERY_PREFERRED_MODELS, ...availableModels]
    .map((model) => normalizeGeminiModelName(model))
    .filter((model): model is string => Boolean(model))

  return Array.from(new Set(orderedCandidates)).filter(
    (model) => availableSet.has(model) && !triedSet.has(model),
  )
}

export const OCR_PROMPT = [
  "You are an OCR assistant.",
  "Extract only the readable text from this whiteboard image.",
  "Return plain text only.",
  "If nothing is readable, return exactly: NO_TEXT_FOUND",
].join(" ")

export const TRANSIENT_STATUS_CODES = [429, 500, 503]
export const MAX_RETRIES = 3
export const RETRY_BASE_DELAY_MS = 1000

async function requestOcrWithModel(
  apiKey: string,
  model: string,
  prompt: string,
  imageBuffer: Buffer,
  mimeType: string,
): Promise<Response> {
  return fetch(buildGeminiUrl(apiKey, model), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: buildGeminiRequestBody(prompt, imageBuffer, mimeType),
  })
}

export async function extractTextFromImageWithGemini(imageBuffer: Buffer, mimeType: string): Promise<GeminiOcrResult> {
  const apiKey = process.env.GEMINI_API_KEY
  const configuredModel = process.env.GEMINI_MODEL || "gemini-2.5-flash"

  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not set")
  }

  let modelsToTry = deduplicateModels(configuredModel)
  let attemptedModelDiscovery = false

  let response: Response | null = null
  let lastError = "Gemini OCR request failed"
  let responseStatus: number | null = null

  for (let modelIndex = 0; modelIndex < modelsToTry.length; modelIndex += 1) {
    const model = modelsToTry[modelIndex]
    // Retry up to 3 times for transient errors (429, 500, 503)
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      response = await requestOcrWithModel(apiKey, model, OCR_PROMPT, imageBuffer, mimeType)
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

    // Try next model on 404 or transient errors (503, 429, 500) after retries exhausted
    const shouldTryNextModel = response && (response.status === 404 || TRANSIENT_STATUS_CODES.includes(response.status))
    if (!shouldTryNextModel) {
      throw new Error(lastError)
    }

    if (response?.status === 404 && modelIndex === modelsToTry.length - 1 && !attemptedModelDiscovery) {
      attemptedModelDiscovery = true

      try {
        const discoveredModels = await discoverAdditionalGenerateContentModels(apiKey, modelsToTry, configuredModel)
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

  const payload = await response.json()
  const text = parseGeminiResponse(payload)

  return {text}
}
