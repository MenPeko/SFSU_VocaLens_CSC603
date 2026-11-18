import {
  MAX_RETRIES,
  RETRY_BASE_DELAY_MS,
  TRANSIENT_STATUS_CODES,
  buildGeminiErrorMessage,
  deduplicateModels,
  discoverAdditionalGenerateContentModels,
  parseGeminiResponse,
} from "./gemini-ocr"
import type {VocalensTurn} from "./vocalens-session"

export const VQA_PROMPT = [
  "You are a visual assistant for a blind or low-vision person.",
  "Give a short answer in one or two sentences.",
  "Read visible text word for word. Do not paraphrase it.",
  "If the image is blurry, too dark, or the question cannot be answered, say the image is unclear and suggest how to reframe the shot.",
  "Do not guess.",
].join(" ")

export function answerLanguageInstruction(targetLanguage: string): string {
  const language = targetLanguage.trim() || "English"
  return `Write the answer in ${language}, even when the question is in another language. Quote visible text exactly as printed.`
}

interface GeminiContentPart {
  text?: string
  inline_data?: {
    mime_type: string
    data: string
  }
}

interface GeminiContent {
  role: "user" | "model"
  parts: GeminiContentPart[]
}

export function buildVisualQuestionContents(
  prompt: string,
  imageBase64: string,
  mimeType: string,
  turns: VocalensTurn[],
  targetLanguage = "English",
): {contents: GeminiContent[]} {
  const languageLine = answerLanguageInstruction(targetLanguage)
  let attachedImage = false
  const contents = turns.map((turn) => {
    if (turn.role === "model") {
      return {role: "model" as const, parts: [{text: turn.text}]}
    }

    if (!attachedImage) {
      attachedImage = true
      return {
        role: "user" as const,
        parts: [
          {text: `${prompt} ${languageLine}\n\nQuestion: ${turn.text}`},
          {
            inline_data: {
              mime_type: mimeType || "image/jpeg",
              data: imageBase64,
            },
          },
        ],
      }
    }

    return {role: "user" as const, parts: [{text: `${languageLine}\n\nQuestion: ${turn.text}`}]}
  })

  return {contents}
}

export function buildVisualQuestionBody(
  prompt: string,
  imageBase64: string,
  mimeType: string,
  turns: VocalensTurn[],
  targetLanguage = "English",
): string {
  return JSON.stringify(buildVisualQuestionContents(prompt, imageBase64, mimeType, turns, targetLanguage))
}

export function buildGeminiStreamUrl(apiKey: string, model: string): string {
  return `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`
}

export function mergeGeminiStreamChunk(accumulated: string, chunk: string): string {
  if (!chunk) return accumulated
  if (!accumulated) return chunk
  if (chunk.startsWith(accumulated)) return chunk
  return accumulated + chunk
}

export function takeCompleteSentences(buffer: string): {sentences: string[]; rest: string} {
  const sentences: string[] = []
  let rest = buffer
  while (rest.length > 0) {
    const match = rest.match(/^[\s\S]*?[.!?](?=\s|$)/)
    if (!match) break
    const sentence = match[0].trim()
    if (sentence) sentences.push(sentence)
    rest = rest.slice(match[0].length).replace(/^\s+/, "")
  }
  return {sentences, rest}
}

export function textFromGeminiSseLine(line: string): string {
  const trimmed = line.trim()
  if (!trimmed.startsWith("data:")) return ""
  const payload = trimmed.slice("data:".length).trim()
  if (!payload || payload === "[DONE]") return ""

  try {
    return parseGeminiResponse(JSON.parse(payload))
  } catch {
    return ""
  }
}

export async function* readGeminiSseDeltas(response: Response): AsyncGenerator<string> {
  const reader = response.body?.getReader()
  if (!reader) return

  const decoder = new TextDecoder()
  let buffer = ""
  while (true) {
    const {done, value} = await reader.read()
    if (done) break
    buffer += decoder.decode(value, {stream: true})
    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""
    for (const line of lines) {
      const text = textFromGeminiSseLine(line)
      if (text) yield text
    }
  }

  const tail = textFromGeminiSseLine(buffer)
  if (tail) yield tail
}

async function openVisualAnswerResponse(
  image: Buffer,
  mimeType: string,
  turns: VocalensTurn[],
  targetLanguage = "English",
): Promise<Response> {
  const apiKey = process.env.GEMINI_API_KEY
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not set")
  }

  const configuredModel = process.env.GEMINI_MODEL || "gemini-2.5-flash"
  const language = targetLanguage.trim() || "English"
  const body = buildVisualQuestionBody(VQA_PROMPT, image.toString("base64"), mimeType, turns, language)
  let modelsToTry = deduplicateModels(configuredModel)
  let attemptedModelDiscovery = false
  let lastError = "Gemini visual question request failed"

  for (let modelIndex = 0; modelIndex < modelsToTry.length; modelIndex += 1) {
    const model = modelsToTry[modelIndex]
    let response: Response | null = null

    for (let attempt = 0; attempt < MAX_RETRIES; attempt += 1) {
      response = await fetch(buildGeminiStreamUrl(apiKey, model), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body,
      })

      if (response.ok) return response

      const isTransient = TRANSIENT_STATUS_CODES.includes(response.status)
      if (isTransient && attempt < MAX_RETRIES - 1) {
        await new Promise((resolve) => setTimeout(resolve, RETRY_BASE_DELAY_MS * (attempt + 1)))
        continue
      }

      const errorBody = await response.text()
      lastError = buildGeminiErrorMessage(response.status, response.statusText, errorBody)
      break
    }

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
        // Keep the original error when model discovery is unavailable.
      }
    }
  }

  throw new Error(lastError)
}

export async function* streamVisualAnswer(
  image: Buffer,
  mimeType: string,
  turns: VocalensTurn[],
  targetLanguage = "English",
): AsyncGenerator<string> {
  const response = await openVisualAnswerResponse(image, mimeType, turns, targetLanguage)
  yield* readGeminiSseDeltas(response)
}

export async function answerVisualQuestion(
  image: Buffer,
  mimeType: string,
  question: string,
  priorTurns: VocalensTurn[] = [],
  targetLanguage = "English",
): Promise<string> {
  const turns: VocalensTurn[] = [...priorTurns, {role: "user", text: question}]
  let full = ""
  for await (const chunk of streamVisualAnswer(image, mimeType, turns, targetLanguage)) {
    full = mergeGeminiStreamChunk(full, chunk)
  }
  return full.trim()
}
