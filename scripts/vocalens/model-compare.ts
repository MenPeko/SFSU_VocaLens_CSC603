import {appendFile, mkdir, readFile} from "node:fs/promises"
import {join} from "node:path"
import {answerVisualQuestion} from "../../src/services/gemini-vqa"
import {answerWithCompareModel, compareModelName} from "./compare-vision"
import {parseVizwizQuestions} from "./vizwiz-run"

const EMPTY_COMPARE_MESSAGE =
  "No local comparison sample in data/vizwiz/compare-30.jsonl. Add that file on this machine. Nothing was compared."

export interface ModelCompareRow {
  id: string
  question: string
  geminiAnswer: string
  geminiLatencyMs: number
  otherModel: string
  otherAnswer: string
  otherLatencyMs: number
  geminiCorrect: null
  otherCorrect: null
  preferred: null
  notes: null
}

export function buildCompareRow(input: {
  id: string
  question: string
  geminiAnswer: string
  geminiLatencyMs: number
  otherModel: string
  otherAnswer: string
  otherLatencyMs: number
}): ModelCompareRow {
  return {
    id: input.id,
    question: input.question,
    geminiAnswer: input.geminiAnswer,
    geminiLatencyMs: input.geminiLatencyMs,
    otherModel: input.otherModel,
    otherAnswer: input.otherAnswer,
    otherLatencyMs: input.otherLatencyMs,
    geminiCorrect: null,
    otherCorrect: null,
    preferred: null,
    notes: null,
  }
}

async function timedAnswer(answer: () => Promise<string>): Promise<{answer: string; latencyMs: number}> {
  const started = Date.now()
  const text = await answer()
  return {answer: text, latencyMs: Date.now() - started}
}

export async function runModelCompare(input: {
  questionsRaw: string
  otherModel: string
  readImage: (image: string) => Promise<Buffer>
  answerWithGemini: (image: Buffer, mimeType: string, question: string) => Promise<string>
  answerWithOther: (image: Buffer, mimeType: string, question: string) => Promise<string>
  writeRow: (row: ModelCompareRow) => Promise<void>
}): Promise<number> {
  const questions = parseVizwizQuestions(input.questionsRaw)
  for (const question of questions) {
    const image = await input.readImage(question.image)
    const gemini = await timedAnswer(() => input.answerWithGemini(image, "image/jpeg", question.question))
    const other = await timedAnswer(() => input.answerWithOther(image, "image/jpeg", question.question))
    await input.writeRow(
      buildCompareRow({
        id: question.id,
        question: question.question,
        geminiAnswer: gemini.answer,
        geminiLatencyMs: gemini.latencyMs,
        otherModel: input.otherModel,
        otherAnswer: other.answer,
        otherLatencyMs: other.latencyMs,
      }),
    )
  }
  return questions.length
}

async function main(): Promise<void> {
  const root = join(process.cwd(), "data/vizwiz")
  const questionsPath = join(root, "compare-30.jsonl")
  let questionsRaw = ""
  try {
    questionsRaw = await readFile(questionsPath, "utf8")
  } catch {
    console.error(EMPTY_COMPARE_MESSAGE)
    process.exit(1)
  }

  if (parseVizwizQuestions(questionsRaw).length === 0) {
    console.error(EMPTY_COMPARE_MESSAGE)
    process.exit(1)
  }

  let otherModel = ""
  try {
    otherModel = compareModelName()
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error))
    process.exit(1)
  }

  if (!process.env.GEMINI_API_KEY?.trim()) {
    console.error("GEMINI_API_KEY is not set. Nothing was compared.")
    process.exit(1)
  }
  if (!process.env.OPENAI_API_KEY?.trim()) {
    console.error("OPENAI_API_KEY is not set. Nothing was compared.")
    process.exit(1)
  }

  const outputDir = join(process.cwd(), "data/vocalens")
  await mkdir(outputDir, {recursive: true})
  const outputPath = join(outputDir, "model-compare.jsonl")

  const wrote = await runModelCompare({
    questionsRaw,
    otherModel,
    readImage: (image) => readFile(join(root, image)),
    answerWithGemini: (image, mimeType, question) => answerVisualQuestion(image, mimeType, question),
    answerWithOther: (image, mimeType, question) => answerWithCompareModel(image, mimeType, question),
    writeRow: async (row) => {
      await appendFile(outputPath, `${JSON.stringify(row)}\n`, "utf8")
    },
  })

  console.log(`Wrote ${wrote} comparisons to ${outputPath}. Judgment fields were left empty.`)
}

if (import.meta.main) {
  void main()
}
