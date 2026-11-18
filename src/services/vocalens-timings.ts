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

type TimingRecordInput = Partial<VocalensTimingRecord> & Record<string, unknown>

export function buildTimingRecord(input: TimingRecordInput | VocalensTimingRecord): VocalensTimingRecord {
  const source = input as TimingRecordInput
  return {
    sessionId: typeof source.sessionId === "string" ? source.sessionId : "",
    question: typeof source.question === "string" ? source.question : "",
    answer: typeof source.answer === "string" ? source.answer : "",
    captureAt: optionalTimestamp(source.captureAt),
    transcriptionStartedAt: requiredTimestamp(source.transcriptionStartedAt),
    transcriptionEndedAt: requiredTimestamp(source.transcriptionEndedAt),
    visionStartedAt: requiredTimestamp(source.visionStartedAt),
    visionEndedAt: requiredTimestamp(source.visionEndedAt),
    speechStartedAt: optionalTimestamp(source.speechStartedAt),
  }
}

export async function appendTimingRecord(
  input: TimingRecordInput | VocalensTimingRecord,
  path = defaultTimingLogPath(),
): Promise<VocalensTimingRecord> {
  const record = buildTimingRecord(input)
  await mkdir(dirname(path), {recursive: true})
  await appendFile(path, `${JSON.stringify(record)}\n`, "utf8")
  return record
}
