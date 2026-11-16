import {appendFile, mkdir} from "node:fs/promises"
import {dirname, join} from "node:path"

export interface VocalensTimingRecord {
  sessionId: string
  question: string
  answer: string
  captureAt: number | null
  transcriptionStartedAt: number
  transcriptionEndedAt: number
  visionStartedAt: number
  visionEndedAt: number
  speechStartedAt: number | null
}

export function defaultTimingLogPath(): string {
  return join(process.cwd(), "data/vocalens/timings.jsonl")
}

function optionalTimestamp(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

function requiredTimestamp(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0
}

export function buildTimingRecord(input: Partial<VocalensTimingRecord> & Record<string, unknown>): VocalensTimingRecord {
  return {
    sessionId: typeof input.sessionId === "string" ? input.sessionId : "",
    question: typeof input.question === "string" ? input.question : "",
    answer: typeof input.answer === "string" ? input.answer : "",
    captureAt: optionalTimestamp(input.captureAt),
    transcriptionStartedAt: requiredTimestamp(input.transcriptionStartedAt),
    transcriptionEndedAt: requiredTimestamp(input.transcriptionEndedAt),
    visionStartedAt: requiredTimestamp(input.visionStartedAt),
    visionEndedAt: requiredTimestamp(input.visionEndedAt),
    speechStartedAt: optionalTimestamp(input.speechStartedAt),
  }
}

export async function appendTimingRecord(input: Partial<VocalensTimingRecord> & Record<string, unknown>, path = defaultTimingLogPath()): Promise<VocalensTimingRecord> {
  const record = buildTimingRecord(input)
  await mkdir(dirname(path), {recursive: true})
  await appendFile(path, `${JSON.stringify(record)}\n`, "utf8")
  return record
}
