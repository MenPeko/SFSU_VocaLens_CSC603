import {
  buildGeminiErrorMessage,
  buildGeminiUrl,
  deduplicateModels,
  parseGeminiResponse,
  TRANSIENT_STATUS_CODES,
  MAX_RETRIES,
  RETRY_BASE_DELAY_MS,
} from "./gemini-ocr"

export interface SignGestureHint {
  name: string
  score: number
}

export interface SignInterpreterInput {
  base64Grid: string
  base64SkeletonGrid?: string
  mimeType?: string
  skeletonMimeType?: string
  gridColumns?: number
  gridRows?: number
  gestureHints?: SignGestureHint[]
}

export interface SignInterpretationResult {
  englishText: string
  rawResponse: string
  model: string
}

export const SIGN_INTERPRETER_SYSTEM_PROMPT = [
  "You are an expert American Sign Language interpreter.",
  "The image input is one or two chronological grids of sign-language frames.",
  "If a second image is provided, it is a skeleton-rendered view of the same sequence and should be used as structural reference only.",
  "Read the grid left-to-right and top-to-bottom as one continuous ASL sequence.",
  "Translate the meaning into concise, natural English.",
  "Prioritize ASL lexical meaning over generic gesture or command interpretation.",
  "When both hands are involved, preserve the two-handed sign rather than collapsing it into a one-hand command.",
  "Do not guess from a single static palm pose alone.",
  "Do not describe the signer, the camera, or the grid layout.",
  "If gesture hints are provided, treat them as noisy secondary hints rather than ground truth.",
  "Return plain English text only.",
  "If the signing is ambiguous, unreadable, or no clear sign is present, return exactly: NO_SIGN_FOUND",
].join(" ")

export function buildSignInterpreterPrompt(
  gridColumns: number,
  gridRows: number,
  gestureHints: SignGestureHint[] = [],
  hasSkeletonReference = false,
): string {
  const hints =
    gestureHints.length > 0
      ? gestureHints.map((hint) => `${hint.name} (${hint.score.toFixed(1)})`).join(", ")
      : "none"

  return [
    `This image is a ${gridColumns}x${gridRows} chronological frame grid.`,
    "Interpret it as one continuous ASL phrase or short sentence.",
    "Produce the best natural English translation.",
    "Use multi-hand symmetry and movement across frames to disambiguate similar open-palm poses.",
    hasSkeletonReference
      ? "A second skeleton-rendered grid of the same frames is also provided to clarify handshape and motion."
      : "Only the raw frame grid is provided.",
    `Auxiliary handshape hints: ${hints}.`,
  ].join(" ")
}

export function buildSignInterpreterRequestBody(
  prompt: string,
  base64Grid: string,
  mimeType = "image/jpeg",
  base64SkeletonGrid?: string,
  skeletonMimeType?: string,
): string {
  const parts: Array<
    | {text: string}
    | {
        inline_data: {
          mime_type: string
          data: string
        }
      }
  > = [
    {text: prompt},
    {text: "Image 1 is the raw chronological RGB frame grid."},
    {
      inline_data: {
        mime_type: mimeType || "image/jpeg",
        data: base64Grid,
      },
    },
  ]

  if (base64SkeletonGrid?.trim()) {
    parts.push({text: "Image 2 is a skeleton-rendered grid of the same sequence for structural reference."})
    parts.push({
      inline_data: {
        mime_type: skeletonMimeType || mimeType || "image/jpeg",
        data: base64SkeletonGrid.trim(),
      },
    })
  }

  return JSON.stringify({
    system_instruction: {
      parts: [{text: SIGN_INTERPRETER_SYSTEM_PROMPT}],
    },
    contents: [
      {
        parts,
      },
    ],
  })
}

export function normalizeSignInterpretationResponse(responseText: string): string {
  if (!responseText) return ""

  const cleaned = responseText
    .replace(/```text\s*/gi, "")
    .replace(/```json\s*/gi, "")
    .replace(/```\s*/g, "")
    .replace(/^English:\s*/i, "")
    .replace(/^Translation:\s*/i, "")
    .trim()

  if (!cleaned) return ""

  try {
    const parsed = JSON.parse(cleaned) as {englishText?: string; translation?: string}
    return (parsed.englishText || parsed.translation || "").trim()
  } catch {
    return cleaned
  }
}

async function requestSignInterpretationWithModel(
  apiKey: string,
  model: string,
  prompt: string,
  base64Grid: string,
  mimeType: string,
  base64SkeletonGrid?: string,
  skeletonMimeType?: string,
): Promise<Response> {
  return fetch(buildGeminiUrl(apiKey, model), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: buildSignInterpreterRequestBody(prompt, base64Grid, mimeType, base64SkeletonGrid, skeletonMimeType),
  })
}

export async function interpretSignGridWithGemini({
  base64Grid,
  base64SkeletonGrid,
  mimeType = "image/jpeg",
  skeletonMimeType,
  gridColumns = 4,
  gridRows = 4,
  gestureHints = [],
}: SignInterpreterInput): Promise<SignInterpretationResult> {
  const apiKey = process.env.GEMINI_API_KEY
  const configuredModel = process.env.GEMINI_SIGN_MODEL || "gemini-1.5-flash"

  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not set")
  }

  if (!base64Grid.trim()) {
    throw new Error("Sign grid image is empty")
  }

  const prompt = buildSignInterpreterPrompt(gridColumns, gridRows, gestureHints, Boolean(base64SkeletonGrid?.trim()))
  const modelsToTry = deduplicateModels(configuredModel)

  let response: Response | null = null
  let resolvedModel = configuredModel
  let lastError = "Sign interpretation request failed"

  for (const model of modelsToTry) {
    resolvedModel = model

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      response = await requestSignInterpretationWithModel(
        apiKey,
        model,
        prompt,
        base64Grid,
        mimeType,
        base64SkeletonGrid,
        skeletonMimeType,
      )

      if (response.ok) break

      const isTransient = TRANSIENT_STATUS_CODES.includes(response.status)
      if (isTransient && attempt < MAX_RETRIES - 1) {
        await new Promise((resolve) => setTimeout(resolve, RETRY_BASE_DELAY_MS * (attempt + 1)))
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
  }

  if (!response || !response.ok) {
    throw new Error(lastError)
  }

  const payload = await response.json()
  const rawResponse = parseGeminiResponse(payload)
  const englishText = normalizeSignInterpretationResponse(rawResponse)

  return {
    englishText,
    rawResponse,
    model: resolvedModel,
  }
}
