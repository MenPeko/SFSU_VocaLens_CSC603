import {appendFile, mkdir, readFile} from "node:fs/promises"
import {join} from "node:path"
import {answerVisualQuestion} from "../../src/services/gemini-vqa"

export const EMPTY_VIZWIZ_MESSAGE =
  "No local VizWiz sample in data/vizwiz/. Add questions.jsonl and images on this machine. Nothing was graded."

export interface VizwizQuestion {
  id: string
  image: string
  question: string
}

export interface VizwizOutputRow {
  id: string
  question: string
  answer: string
  correct: null
  unclearFlag: null
  notes: null
  errorGroup: ""
}

function parseJsonLine(line: string, message: string): unknown {
  try {
    return JSON.parse(line)
  } catch {
    throw new Error(message)
  }
}

export function parseVizwizQuestions(raw: string): VizwizQuestion[] {
  const questions: VizwizQuestion[] = []
  for (const line of raw.split("\n")) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const item = parseJsonLine(trimmed, "A VizWiz question line is not valid JSON. Nothing was graded.") as {
      id?: string
      image?: string
      question?: string
    }
    if (!item.image || !item.question) {
      throw new Error("Each VizWiz line needs an image file name and a question.")
    }
    questions.push({
      id: item.id?.trim() || item.image,
      image: item.image,
      question: item.question,
    })
  }
  return questions
}

export function cachedVizwizIds(raw: string): Set<string> {
  const ids = new Set<string>()
  for (const line of raw.split("\n")) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const row = parseJsonLine(trimmed, "A cached VizWiz line is not valid JSON. Nothing new was asked.") as {id?: string}
    if (typeof row.id === "string" && row.id.trim()) ids.add(row.id.trim())
  }
  return ids
}

export function questionsToAsk(questions: VizwizQuestion[], cachedIds: Set<string>): VizwizQuestion[] {
  return questions.filter((question) => !cachedIds.has(question.id))
}

export function buildVizwizOutputRow(question: VizwizQuestion, answer: string): VizwizOutputRow {
  return {
    id: question.id,
    question: question.question,
    answer,
    correct: null,
    unclearFlag: null,
    notes: null,
    errorGroup: "",
  }
}

export async function runVizwiz(input: {
  questionsRaw: string
  cachedRaw: string
  readImage: (image: string) => Promise<Buffer>
  answer: (image: Buffer, mimeType: string, question: string) => Promise<string>
  writeRow: (row: VizwizOutputRow) => Promise<void>
}): Promise<{wrote: number; skipped: number}> {
  const questions = parseVizwizQuestions(input.questionsRaw)
  const pending = questionsToAsk(questions, cachedVizwizIds(input.cachedRaw))
  for (const question of pending) {
    const image = await input.readImage(question.image)
    const answer = await input.answer(image, "image/jpeg", question.question)
    await input.writeRow(buildVizwizOutputRow(question, answer))
  }
  return {wrote: pending.length, skipped: questions.length - pending.length}
}

async function main(): Promise<void> {
  const root = join(process.cwd(), "data/vizwiz")
  const questionsPath = join(root, "questions.jsonl")
  let questionsRaw = ""
  try {
    questionsRaw = await readFile(questionsPath, "utf8")
  } catch {
    console.error(EMPTY_VIZWIZ_MESSAGE)
    process.exit(1)
  }

  if (parseVizwizQuestions(questionsRaw).length === 0) {
    console.error(EMPTY_VIZWIZ_MESSAGE)
    process.exit(1)
  }

  const outputDir = join(process.cwd(), "data/vocalens")
  await mkdir(outputDir, {recursive: true})
  const outputPath = join(outputDir, "vizwiz-outputs.jsonl")
  let cachedRaw = ""
  try {
    cachedRaw = await readFile(outputPath, "utf8")
  } catch {
    cachedRaw = ""
  }

  const result = await runVizwiz({
    questionsRaw,
    cachedRaw,
    readImage: (image) => readFile(join(root, image)),
    answer: (image, mimeType, question) => answerVisualQuestion(image, mimeType, question),
    writeRow: async (row) => {
      await appendFile(outputPath, `${JSON.stringify(row)}\n`, "utf8")
    },
  })

  console.log(
    `Wrote ${result.wrote} answers to ${outputPath}. Skipped ${result.skipped} cached questions. Grade fields were left empty.`,
  )
}

if (import.meta.main) {
  void main()
}
