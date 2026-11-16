import {readFile} from "node:fs/promises"
import {join} from "node:path"

export const ERROR_GROUPS = [
  "invented_text",
  "wrong_object",
  "answered_unreadable",
  "refused_visible",
  "ok",
] as const

export type ErrorGroup = (typeof ERROR_GROUPS)[number]

const EMPTY_SUMMARY = "Every errorGroup is empty. No score was computed."
const MISSING_SUMMARY = "No VizWiz outputs in data/vocalens/vizwiz-outputs.jsonl. No score was computed."

function isErrorGroup(value: string): value is ErrorGroup {
  return (ERROR_GROUPS as readonly string[]).includes(value)
}

export function renderErrorGroupSummary(raw: string | null): {exitCode: number; lines: string[]} {
  if (raw === null) {
    return {exitCode: 0, lines: [MISSING_SUMMARY]}
  }

  const counts = Object.fromEntries(ERROR_GROUPS.map((group) => [group, 0])) as Record<ErrorGroup, number>
  const unknown = new Map<string, number>()
  let filled = 0

  for (const line of raw.split("\n")) {
    const trimmed = line.trim()
    if (!trimmed) continue
    let row: {errorGroup?: unknown}
    try {
      row = JSON.parse(trimmed) as {errorGroup?: unknown}
    } catch {
      return {exitCode: 1, lines: ["A VizWiz output line is not valid JSON. No score was computed."]}
    }

    const group = typeof row.errorGroup === "string" ? row.errorGroup.trim() : ""
    if (!group) continue
    if (!isErrorGroup(group)) {
      unknown.set(group, (unknown.get(group) || 0) + 1)
      continue
    }
    counts[group] += 1
    filled += 1
  }

  if (filled === 0 && unknown.size === 0) {
    return {exitCode: 0, lines: [EMPTY_SUMMARY]}
  }

  const lines = ERROR_GROUPS.filter((group) => counts[group] > 0).map((group) => `${group}\t${counts[group]}`)
  for (const label of [...unknown.keys()].sort()) {
    lines.push(`unrecognized\t${label}\t${unknown.get(label)}`)
  }
  if (filled === 0) lines.push("No score was computed.")
  return {exitCode: 0, lines}
}

async function main(): Promise<void> {
  const outputPath = join(process.cwd(), "data/vocalens/vizwiz-outputs.jsonl")
  let raw: string | null = null
  try {
    raw = await readFile(outputPath, "utf8")
  } catch {
    raw = null
  }

  const rendered = renderErrorGroupSummary(raw)
  for (const line of rendered.lines) console.log(line)
  if (rendered.exitCode !== 0) process.exit(rendered.exitCode)
}

if (import.meta.main) {
  void main()
}
