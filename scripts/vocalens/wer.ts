import {readFile} from "node:fs/promises"

export function normalizeForWer(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9'\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
}

export function wordErrorDistance(reference: string[], hypothesis: string[]): number {
  const rows = reference.length + 1
  const cols = hypothesis.length + 1
  const distances = Array.from({length: rows}, () => new Array<number>(cols).fill(0))

  for (let row = 0; row < rows; row += 1) distances[row][0] = row
  for (let column = 0; column < cols; column += 1) distances[0][column] = column

  for (let row = 1; row < rows; row += 1) {
    for (let column = 1; column < cols; column += 1) {
      const substitution = reference[row - 1] === hypothesis[column - 1] ? 0 : 1
      distances[row][column] = Math.min(
        distances[row - 1][column] + 1,
        distances[row][column - 1] + 1,
        distances[row - 1][column - 1] + substitution,
      )
    }
  }

  return distances[reference.length][hypothesis.length]
}

export function wordErrorRate(reference: string, hypothesis: string): number {
  const referenceWords = normalizeForWer(reference)
  const hypothesisWords = normalizeForWer(hypothesis)
  if (referenceWords.length === 0) return hypothesisWords.length === 0 ? 0 : 1
  return wordErrorDistance(referenceWords, hypothesisWords) / referenceWords.length
}

async function main(): Promise<void> {
  const inputPath = process.argv[2]
  if (!inputPath) {
    console.error("Usage: bun scripts/vocalens/wer.ts pairs.jsonl")
    console.error('Each line is {"reference":"...","hypothesis":"..."}. This command only prints the rate.')
    process.exit(1)
  }

  let raw = ""
  try {
    raw = await readFile(inputPath, "utf8")
  } catch {
    console.error(`Could not read ${inputPath}. No word error rate was recorded.`)
    process.exit(1)
  }

  const lines = raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
  if (lines.length === 0) {
    console.error("The pairs file is empty. No word error rate was recorded.")
    process.exit(1)
  }

  let totalDistance = 0
  let totalWords = 0
  for (const line of lines) {
    const pair = JSON.parse(line) as {reference?: string; hypothesis?: string}
    const reference = pair.reference || ""
    const hypothesis = pair.hypothesis || ""
    const referenceWords = normalizeForWer(reference)
    totalDistance += wordErrorDistance(referenceWords, normalizeForWer(hypothesis))
    totalWords += referenceWords.length
    console.log(`${wordErrorRate(reference, hypothesis).toFixed(4)}\t${reference}`)
  }

  const overall = totalWords === 0 ? 0 : totalDistance / totalWords
  console.log(`overall\t${overall.toFixed(4)}\t${lines.length} pairs`)
}

if (import.meta.main) {
  void main()
}
