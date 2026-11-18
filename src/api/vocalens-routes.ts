import {sseManager} from "./sse-stream"
import {mergeGeminiStreamChunk, streamVisualAnswer, takeCompleteSentences} from "../services/gemini-vqa"
import type {VocalensTurn} from "../services/vocalens-session"
import {
  appendVocalensTurn,
  clearVocalensSession,
  dropIdleVocalensSessions,
  getVocalensSession,
  replaceVocalensSession,
} from "../services/vocalens-session"
import {appendTimingRecord, buildTimingRecord, type VocalensTimingRecord} from "../services/vocalens-timings"
import {transcribeAudio, type TranscriptionResult} from "../services/whisper"

const MAX_AUDIO_BYTES = 8_000_000
const MAX_FRAME_BYTES = 8_000_000
const SESSION_ID_PATTERN = /^[A-Za-z0-9_-]{8,80}$/

export interface VocalensAskDeps {
  transcribe: (audio: Buffer, mimeType: string) => Promise<TranscriptionResult>
  streamAnswer: (image: Buffer, mimeType: string, turns: VocalensTurn[], targetLanguage: string) => AsyncIterable<string>
  now: () => number
}

const defaultDeps: VocalensAskDeps = {
  transcribe: (audio, mimeType) => transcribeAudio(audio, mimeType),
  streamAnswer: (image, mimeType, turns, targetLanguage) => streamVisualAnswer(image, mimeType, turns, targetLanguage),
  now: () => Date.now(),
}

export function resolveAskFrame(input: {
  followUp: boolean
  uploaded: Buffer | null
  stored: Buffer | null
}): Buffer | {error: string} {
  if (input.followUp) {
    if (!input.stored || input.stored.length === 0) {
      return {error: "Take a photo before asking a follow-up."}
    }
    return input.stored
  }

  if (!input.uploaded || input.uploaded.length === 0) {
    return {error: "A camera frame is required for a new question."}
  }

  return input.uploaded
}

function jsonError(message: string, status: number): Response {
  return Response.json({error: message}, {status})
}

async function readFileField(value: FormDataEntryValue | null): Promise<{bytes: Buffer; mimeType: string} | null> {
  if (!(value instanceof File) || value.size === 0) return null
  return {
    bytes: Buffer.from(await value.arrayBuffer()),
    mimeType: value.type || "application/octet-stream",
  }
}

function writeEvent(controller: ReadableStreamDefaultController<Uint8Array>, encoder: TextEncoder, event: unknown): void {
  controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`))
}

export async function handleVocalensAsk(req: Request, deps: VocalensAskDeps = defaultDeps): Promise<Response> {
  let form: FormData
  try {
    form = await req.formData()
  } catch {
    return jsonError("The question upload could not be read.", 400)
  }

  const sessionId = String(form.get("sessionId") || "")
  if (!SESSION_ID_PATTERN.test(sessionId)) {
    return jsonError("A valid session id is required.", 400)
  }

  const followUp = String(form.get("followUp") || "false") === "true"
  const captureField = Number(form.get("captureAt"))
  const captureAt = Number.isFinite(captureField) ? captureField : null
  const audio = await readFileField(form.get("audio"))
  const uploadedFrame = await readFileField(form.get("frame"))

  if (!audio) {
    return jsonError("A spoken question is required.", 400)
  }
  if (audio.bytes.length > MAX_AUDIO_BYTES) {
    return jsonError("The recording is too long. Ask a shorter question.", 400)
  }
  if (uploadedFrame && uploadedFrame.bytes.length > MAX_FRAME_BYTES) {
    return jsonError("The camera frame is too large.", 400)
  }

  dropIdleVocalensSessions(deps.now())
  const existing = getVocalensSession(sessionId, deps.now())
  const frame = resolveAskFrame({
    followUp,
    uploaded: uploadedFrame?.bytes ?? null,
    stored: existing?.image ?? null,
  })
  if ("error" in frame) {
    return jsonError(frame.error, 400)
  }

  if (!followUp) {
    replaceVocalensSession(sessionId, frame, uploadedFrame?.mimeType || "image/jpeg", captureAt, deps.now())
  }

  const session = getVocalensSession(sessionId, deps.now())
  if (!session) {
    return jsonError("The photo for this question is no longer available. Take a new photo.", 400)
  }

  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const transcriptionStartedAt = deps.now()
      try {
        const {getTargetLanguage} = await import("./routes")
        const storedLanguage = getTargetLanguage()
        writeEvent(controller, encoder, {type: "frame_stored", followUp})
        const transcript = await deps.transcribe(audio.bytes, audio.mimeType)
        const question = transcript.text.trim()
        const transcriptionEndedAt = deps.now()
        if (!question) {
          writeEvent(controller, encoder, {
            type: "error",
            message: "I could not hear a question. Please try again.",
          })
          return
        }

        writeEvent(controller, encoder, {type: "transcript", text: question})
        const turns: VocalensTurn[] = [...session.turns, {role: "user", text: question}]
        const visionStartedAt = deps.now()
        let accumulated = ""
        let emittedSentences = 0
        for await (const chunk of deps.streamAnswer(session.image, session.mimeType, turns, storedLanguage)) {
          accumulated = mergeGeminiStreamChunk(accumulated, chunk)
          const spoken = takeCompleteSentences(accumulated)
          while (emittedSentences < spoken.sentences.length) {
            writeEvent(controller, encoder, {
              type: "sentence",
              text: spoken.sentences[emittedSentences],
              targetLanguage: storedLanguage,
            })
            emittedSentences += 1
          }
        }
        const remainder = takeCompleteSentences(accumulated).rest.trim()
        if (remainder) {
          writeEvent(controller, encoder, {type: "sentence", text: remainder, targetLanguage: storedLanguage})
        }

        const answer = accumulated.trim()
        const visionEndedAt = deps.now()
        if (!answer) {
          writeEvent(controller, encoder, {
            type: "error",
            message: "I could not answer that question. Please try again.",
          })
          return
        }

        appendVocalensTurn(sessionId, {role: "user", text: question}, deps.now())
        appendVocalensTurn(sessionId, {role: "model", text: answer}, deps.now())

        const timings = {
          captureAt: session.captureAt,
          transcriptionStartedAt,
          transcriptionEndedAt,
          visionStartedAt,
          visionEndedAt,
        }

        sseManager.broadcast({
          type: "translation_result",
          originalText: question,
          translatedText: answer,
          detectedLanguage: "English",
          targetLanguage: storedLanguage,
          timestamp: deps.now(),
          sourceType: "vocalens",
          sourceLabel: "Spoken question",
        })

        writeEvent(controller, encoder, {
          type: "done",
          question,
          answer,
          targetLanguage: storedLanguage,
          provider: transcript.provider,
          timings,
        })
      } catch (error) {
        const message = error instanceof Error ? error.message : "The question could not be answered."
        writeEvent(controller, encoder, {type: "error", message})
      } finally {
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson",
      "Cache-Control": "no-cache",
    },
  })
}

export async function handleVocalensClear(req: Request): Promise<Response> {
  let body: {sessionId?: string}
  try {
    body = (await req.json()) as {sessionId?: string}
  } catch {
    return jsonError("A session id is required.", 400)
  }

  const sessionId = body.sessionId || ""
  if (!SESSION_ID_PATTERN.test(sessionId)) {
    return jsonError("A valid session id is required.", 400)
  }

  clearVocalensSession(sessionId)
  return Response.json({cleared: true})
}

export async function handleVocalensTimings(req: Request): Promise<Response> {
  let body: Partial<VocalensTimingRecord> & Record<string, unknown>
  try {
    body = (await req.json()) as Partial<VocalensTimingRecord> & Record<string, unknown>
  } catch {
    return jsonError("The timing record could not be read.", 400)
  }

  const record = buildTimingRecord(body)
  if (!SESSION_ID_PATTERN.test(record.sessionId) || !record.question || !record.answer) {
    return jsonError("A session id, question, and answer are required.", 400)
  }

  const saved = await appendTimingRecord(record)
  return Response.json({saved: true, record: saved})
}

export const vocalensRoutes = {
  "/api/vocalens/ask": {
    POST: (req: Request) => handleVocalensAsk(req),
  },
  "/api/vocalens/session/clear": {
    POST: (req: Request) => handleVocalensClear(req),
  },
  "/api/vocalens/timings": {
    POST: (req: Request) => handleVocalensTimings(req),
  },
}
