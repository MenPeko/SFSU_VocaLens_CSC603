import {mkdtemp, rm, writeFile} from "node:fs/promises"
import {tmpdir} from "node:os"
import {join} from "node:path"
import {buildGeminiErrorMessage, buildGeminiUrl, parseGeminiResponse} from "./gemini-ocr"

export type TranscriptionProvider = "openai" | "gemini" | "faster-whisper"

export interface TranscriptionResult {
  text: string
  provider: TranscriptionProvider
}

export function selectTranscriptionProvider(input: {
  hasApiKey: boolean
  apiFailed: boolean
  hasGeminiKey?: boolean
  geminiFailed?: boolean
}): TranscriptionProvider {
  if (input.hasApiKey && !input.apiFailed) return "openai"
  if (input.hasGeminiKey && !input.geminiFailed) return "gemini"
  return "faster-whisper"
}

const TRANSCRIPTION_PROMPT = "Transcribe the spoken question. Return only the words that were spoken."
const GEMINI_TRANSCRIPTION_FALLBACKS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]

export function formatGeminiTranscriptionError(status: number, statusText: string, errorBody: string): string {
  const base = buildGeminiErrorMessage(status, statusText, errorBody).replace(
    "Gemini OCR request failed",
    "Gemini transcription request failed",
  )
  let detail = ""
  try {
    const payload = JSON.parse(errorBody) as {error?: {message?: string; status?: string}}
    detail = payload.error?.message?.trim() || ""
  } catch {
    detail = ""
  }
  if (!detail || /api[_ ]?key|AIza|sk-/i.test(detail)) return base
  if (base.toLowerCase().includes(detail.toLowerCase())) return base
  return `${base} ${detail}`.slice(0, 280)
}

function geminiTranscriptionModels(model: string): string[] {
  return [...new Set([model, ...GEMINI_TRANSCRIPTION_FALLBACKS].map((candidate) => candidate.trim()).filter(Boolean))]
}

export function geminiAudioMimeType(mimeType: string): string {
  const base = mimeType.split(";")[0]?.trim().toLowerCase() || ""
  if (base === "audio/mp4" || base === "audio/m4a" || base === "audio/x-m4a" || base === "audio/aac") return "audio/mp4"
  if (base === "audio/mpeg" || base === "audio/mp3") return "audio/mp3"
  if (base === "audio/wav" || base === "audio/x-wav" || base === "audio/wave") return "audio/wav"
  if (base === "audio/ogg" || base === "audio/opus") return "audio/ogg"
  return "audio/webm"
}

export async function transcribeWithGemini(
  audio: Buffer,
  mimeType: string,
  apiKey: string,
  model = process.env.GEMINI_MODEL || "gemini-2.5-flash",
): Promise<string> {
  const body = JSON.stringify({
    contents: [
      {
        role: "user",
        parts: [
          {text: TRANSCRIPTION_PROMPT},
          {
            inline_data: {
              mime_type: geminiAudioMimeType(mimeType),
              data: audio.toString("base64"),
            },
          },
        ],
      },
    ],
  })
  let lastError = "Gemini transcription request failed"

  for (const candidate of geminiTranscriptionModels(model)) {
    const response = await fetch(buildGeminiUrl(apiKey, candidate), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body,
    })

    if (response.ok) {
      const text = parseGeminiResponse(await response.json()).trim()
      if (text) return text
      lastError = "Gemini returned an empty transcript"
      continue
    }

    lastError = formatGeminiTranscriptionError(response.status, response.statusText, await response.text())
    if (response.status !== 403 && response.status !== 404) break
  }

  throw new Error(lastError)
}

function extensionForMime(mimeType: string): string {
  const normalized = mimeType.toLowerCase()
  if (normalized.includes("wav")) return "wav"
  if (normalized.includes("mp4") || normalized.includes("m4a") || normalized.includes("aac")) return "m4a"
  if (normalized.includes("mpeg") || normalized.includes("mp3")) return "mp3"
  if (normalized.includes("ogg")) return "ogg"
  return "webm"
}

export async function transcribeWithOpenAi(audio: Buffer, mimeType: string, apiKey: string): Promise<string> {
  const form = new FormData()
  const extension = extensionForMime(mimeType)
  form.append("file", new Blob([new Uint8Array(audio)], {type: mimeType || "audio/webm"}), `question.${extension}`)
  form.append("model", "whisper-1")

  const response = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: {Authorization: `Bearer ${apiKey}`},
    body: form,
  })

  if (!response.ok) {
    throw new Error(`Whisper API request failed: ${response.status} ${response.statusText}`)
  }

  const payload = (await response.json()) as {text?: string}
  return (payload.text || "").trim()
}

export function localTranscribeScriptPath(): string {
  return join(import.meta.dir, "../../scripts/vocalens/transcribe_local.py")
}

function pythonCandidates(): string[] {
  const candidates = [process.env.PYTHON_BIN, "python3", "/usr/bin/python3"].filter(
    (candidate): candidate is string => Boolean(candidate && candidate.trim()),
  )
  return [...new Set(candidates.map((candidate) => candidate.trim()))]
}

let resolvedFasterWhisperPython: Promise<string> | null = null

export function resolveFasterWhisperPython(): Promise<string> {
  if (!resolvedFasterWhisperPython) {
    resolvedFasterWhisperPython = (async () => {
      for (const candidate of pythonCandidates()) {
        const probe = Bun.spawn([candidate, "-c", "import faster_whisper"], {
          stdout: "ignore",
          stderr: "ignore",
        })
        if ((await probe.exited) === 0) return candidate
      }
      return pythonCandidates()[0] || "python3"
    })()
  }
  return resolvedFasterWhisperPython
}

export async function transcribeWithFasterWhisper(audio: Buffer, mimeType: string): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "vocalens-audio-"))
  const audioPath = join(directory, `question.${extensionForMime(mimeType)}`)

  try {
    await writeFile(audioPath, audio)
    const python = await resolveFasterWhisperPython()
    const proc = Bun.spawn([python, localTranscribeScriptPath(), audioPath], {
      stdout: "pipe",
      stderr: "pipe",
    })
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ])

    if (exitCode !== 0) {
      const detail =
        stderr
          .trim()
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line && !line.includes("[ctranslate2]"))
          .join(" ") || `exit ${exitCode}`
      throw new Error(`Local faster-whisper failed: ${detail}`)
    }

    return stdout.trim()
  } finally {
    await rm(directory, {recursive: true, force: true})
  }
}

export async function transcribeAudio(
  audio: Buffer,
  mimeType: string,
  options?: {
    apiKey?: string
    geminiApiKey?: string
    geminiModel?: string
    transcribeWithOpenAi?: (audio: Buffer, mimeType: string, apiKey: string) => Promise<string>
    transcribeWithGemini?: (audio: Buffer, mimeType: string, apiKey: string) => Promise<string>
    transcribeLocally?: (audio: Buffer, mimeType: string) => Promise<string>
  },
): Promise<TranscriptionResult> {
  const apiKey = (options?.apiKey ?? process.env.OPENAI_API_KEY ?? "").trim()
  const geminiKey = (options?.geminiApiKey ?? process.env.GEMINI_API_KEY ?? "").trim()
  const geminiModel = (options?.geminiModel ?? process.env.GEMINI_MODEL ?? "gemini-2.5-flash").trim()
  const callOpenAi = options?.transcribeWithOpenAi ?? transcribeWithOpenAi
  const callGemini =
    options?.transcribeWithGemini ?? ((audioBuffer, mime, key) => transcribeWithGemini(audioBuffer, mime, key, geminiModel))
  const callLocal = options?.transcribeLocally ?? transcribeWithFasterWhisper
  const failures: string[] = []

  if (apiKey) {
    try {
      const text = await callOpenAi(audio, mimeType, apiKey)
      if (!text) throw new Error("Whisper API returned an empty transcript")
      return {text, provider: "openai"}
    } catch (error) {
      failures.push(error instanceof Error ? error.message : String(error))
    }
  } else if (!geminiKey) {
    failures.push("Whisper API key is not set")
  }

  if (geminiKey) {
    try {
      const text = await callGemini(audio, mimeType, geminiKey)
      if (!text) throw new Error("Gemini returned an empty transcript")
      return {text, provider: "gemini"}
    } catch (error) {
      failures.push(error instanceof Error ? error.message : String(error))
    }
  }

  try {
    const text = await callLocal(audio, mimeType)
    if (!text) throw new Error("Local faster-whisper returned an empty transcript")
    return {text, provider: "faster-whisper"}
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    if (message.includes("microphone recording was silent") || message.includes("Recording is too short")) {
      throw new Error(message.replace(/^Local faster-whisper failed:\s*/, ""))
    }
    failures.push(message)
  }

  throw new Error(`Speech transcription failed. ${failures.join(". ")}.`)
}
