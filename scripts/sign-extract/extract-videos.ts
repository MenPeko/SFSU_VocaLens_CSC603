// Runs dataset videos through the app's own MediaPipe extractor in headless
// Chromium and writes landmark records for the held-out evaluation.
//
//   bun scripts/sign-extract/extract-videos.ts [--input CSV] [--output JSONL] [--limit N]
//       [--channel chrome | --executable PATH_TO_CHROMIUM]
//
// --executable (or SIGN_EXTRACT_CHROMIUM) points at an already installed
// Chromium when `playwright install` cannot download one.
//
// Input is the filtered CSV from `sign_classifier_service.eval.asl_citizen filter`.
// Output defaults to data/landmarks/asl-citizen/<extractorVersion>/clips.jsonl
// (gitignored). Already extracted video IDs are skipped, so a stopped run resumes.

import {appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync} from "node:fs"
import {dirname, extname, isAbsolute, join, resolve} from "node:path"
import {chromium} from "playwright"
import {MEDIAPIPE_EXTRACTOR_VERSION} from "../../src/sign/extractor-version"
import type {ClipExtraction} from "./page"

const APP_ROOT = resolve(import.meta.dir, "../..")
const DEFAULT_INPUT = join(APP_ROOT, "data/sign-eval/asl-citizen-filtered.csv")
const DEFAULT_OUTPUT = join(APP_ROOT, "data/landmarks/asl-citizen", MEDIAPIPE_EXTRACTOR_VERSION, "clips.jsonl")
const FAILURES_PATH = join(APP_ROOT, "data/sign-eval/asl-citizen-extract-failures.jsonl")
const SUMMARY_PATH = join(APP_ROOT, "data/sign-eval/asl-citizen-extract-summary.json")
const MEDIAPIPE_DIR = join(APP_ROOT, "public/mediapipe")
const SKELETON_DIR = join(APP_ROOT, "data/sign-reports")
const VIDEO_TYPES: Record<string, string> = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}

interface ClipReport {
  label: string
  sourceGloss: string
  videoFile: string
  decoded: boolean
  kept: number
  leftHandFrames: number
  rightHandFrames: number
  bothHandsFrames: number
  error?: string
}

function writeSkeletonPng(dataUrl: string, filePath: string) {
  const encoded = dataUrl.slice(dataUrl.indexOf(",") + 1)
  mkdirSync(dirname(filePath), {recursive: true})
  writeFileSync(filePath, Buffer.from(encoded, "base64"))
}

function bothHandsInMostFrames(report: ClipReport): string {
  if (report.kept === 0) return "n/a"
  return report.bothHandsFrames * 2 > report.kept ? "yes" : "no"
}

interface FilteredRow {
  participantId: string
  videoFile: string
  sourceGloss: string
  label: string
  videoPath: string
}

function parseArgs(argv: string[]) {
  const options = {
    input: DEFAULT_INPUT,
    output: DEFAULT_OUTPUT,
    limit: Infinity,
    channel: "",
    executable: process.env.SIGN_EXTRACT_CHROMIUM ?? "",
  }
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index + 1]
    switch (argv[index]) {
      case "--input":
        options.input = resolve(value)
        index += 1
        break
      case "--output":
        options.output = resolve(value)
        index += 1
        break
      case "--limit":
        options.limit = Number(value)
        index += 1
        break
      case "--channel":
        options.channel = value
        index += 1
        break
      case "--executable":
        options.executable = resolve(value)
        index += 1
        break
      default:
        throw new Error(`Unknown argument ${argv[index]}`)
    }
  }
  return options
}

function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ""
  let quoted = false
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        field += '"'
        index += 1
      } else if (char === '"') {
        quoted = false
      } else {
        field += char
      }
    } else if (char === '"') {
      quoted = true
    } else if (char === ",") {
      row.push(field)
      field = ""
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[index + 1] === "\n") index += 1
      row.push(field)
      rows.push(row)
      row = []
      field = ""
    } else {
      field += char
    }
  }
  if (field || row.length) {
    row.push(field)
    rows.push(row)
  }
  const [header, ...body] = rows.filter((line) => line.some((cell) => cell !== ""))
  return body.map((cells) => Object.fromEntries(header.map((name, index) => [name, cells[index] ?? ""])))
}

function readDoneIds(outputPath: string): Set<string> {
  if (!existsSync(outputPath)) return new Set()
  const done = new Set<string>()
  for (const line of readFileSync(outputPath, "utf8").split("\n")) {
    if (line.trim()) done.add(JSON.parse(line).sourceVideoId)
  }
  return done
}

async function buildPageBundle(): Promise<string> {
  const result = await Bun.build({entrypoints: [join(import.meta.dir, "page.ts")], target: "browser"})
  if (!result.success) {
    throw new Error(`Could not bundle the extractor page:\n${result.logs.join("\n")}`)
  }
  return await result.outputs[0].text()
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  if (!existsSync(options.input)) {
    throw new Error(`${options.input} does not exist. Run sign_classifier_service.eval.asl_citizen filter first.`)
  }
  const rows = parseCsv(readFileSync(options.input, "utf8")) as unknown as FilteredRow[]
  const done = readDoneIds(options.output)
  const pending = rows.filter((row) => row.videoPath && !done.has(row.videoFile)).slice(0, options.limit)
  const missingVideo = rows.filter((row) => !row.videoPath).length
  console.log(`${rows.length} clips listed, ${done.size} already extracted, ${pending.length} to extract now.`)

  const bundle = await buildPageBundle()
  const server = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    fetch(request) {
      const {pathname} = new URL(request.url)
      if (pathname === "/") {
        return new Response('<!doctype html><html><body><script src="/page.js"></script></body></html>', {
          headers: {"content-type": "text/html"},
        })
      }
      if (pathname === "/page.js") {
        return new Response(bundle, {headers: {"content-type": "text/javascript"}})
      }
      if (pathname.startsWith("/mediapipe/")) {
        const filePath = resolve(MEDIAPIPE_DIR, `.${pathname.slice("/mediapipe".length)}`)
        if (!filePath.startsWith(MEDIAPIPE_DIR) || !existsSync(filePath)) return new Response("not found", {status: 404})
        return new Response(Bun.file(filePath))
      }
      // Only rows from the filtered CSV are served, by position.
      const match = pathname.match(/^\/video\/(\d+)$/)
      const row = match ? pending[Number(match[1])] : undefined
      if (row) {
        const filePath = isAbsolute(row.videoPath) ? row.videoPath : join(APP_ROOT, row.videoPath)
        const contentType = VIDEO_TYPES[extname(filePath).toLowerCase()] ?? "application/octet-stream"
        return new Response(Bun.file(filePath), {headers: {"content-type": contentType}})
      }
      return new Response("not found", {status: 404})
    },
  })

  const browser = await chromium.launch({
    headless: true,
    args: ["--enable-webgl", "--ignore-gpu-blocklist", "--use-gl=angle"],
    ...(options.channel ? {channel: options.channel} : {}),
    ...(options.executable ? {executablePath: options.executable} : {}),
  })
  const counts = {extracted: 0, noSequence: 0, failed: 0}
  const clipReports: ClipReport[] = []
  let skeletonPath = ""
  try {
    const page = await browser.newPage()
    await page.goto(`http://127.0.0.1:${server.port}/`)
    await page.waitForFunction(() => window.extractorReady || window.extractorError, null, {timeout: 120_000})
    const startupError = await page.evaluate(() => window.extractorError)
    if (startupError) throw new Error(`Holistic failed to initialize: ${startupError}`)

    mkdirSync(dirname(options.output), {recursive: true})
    mkdirSync(dirname(FAILURES_PATH), {recursive: true})

    for (const [index, row] of pending.entries()) {
      const result = (await page.evaluate(
        ({url, skeleton}) => window.extractClip!(url, skeleton ? {skeleton: true} : undefined),
        {url: `/video/${index}`, skeleton: skeletonPath === ""},
      )) as ClipExtraction
      if (result.extractorVersion !== MEDIAPIPE_EXTRACTOR_VERSION) {
        throw new Error(`Page reported extractor ${result.extractorVersion}, expected ${MEDIAPIPE_EXTRACTOR_VERSION}`)
      }
      if (result.skeletonPng && !skeletonPath) {
        skeletonPath = join(SKELETON_DIR, `asl-citizen-smoke-${row.label}-skeleton.png`)
        writeSkeletonPng(result.skeletonPng, skeletonPath)
      }
      const report: ClipReport = {
        label: row.label,
        sourceGloss: row.sourceGloss,
        videoFile: row.videoFile,
        decoded: result.decoded,
        kept: result.landmarkSequence.length,
        leftHandFrames: result.leftHandFrames,
        rightHandFrames: result.rightHandFrames,
        bothHandsFrames: result.bothHandsFrames,
        error: result.error,
      }
      clipReports.push(report)
      console.log(
        `${report.label} ${report.videoFile} decoded=${report.decoded ? "yes" : "no"} kept=${report.kept} bothHands=${report.bothHandsFrames}/${report.kept} most=${bothHandsInMostFrames(report)}`,
      )
      if (result.error || result.landmarkSequence.length === 0) {
        const reason = result.error ?? "no hand-visible sequence"
        if (result.error) counts.failed += 1
        else counts.noSequence += 1
        appendFileSync(FAILURES_PATH, `${JSON.stringify({sourceVideoId: row.videoFile, reason})}\n`)
      } else {
        counts.extracted += 1
        const record = {
          dataset: "asl-citizen",
          label: row.label,
          sourceGloss: row.sourceGloss,
          signerId: row.participantId,
          sourceVideoId: row.videoFile,
          source: "asl-citizen-video",
          extractorVersion: result.extractorVersion,
          extractionMode: "offline-video",
          framesSampled: result.framesSampled,
          sequencesFound: result.sequencesFound,
          landmarkSequence: result.landmarkSequence,
          frameTimestamps: result.frameTimestamps,
        }
        appendFileSync(options.output, `${JSON.stringify(record)}\n`)
      }
      if ((index + 1) % 25 === 0 || index + 1 === pending.length) {
        console.log(`${index + 1}/${pending.length}  extracted=${counts.extracted} noSequence=${counts.noSequence} failed=${counts.failed}`)
      }
    }
  } finally {
    await browser.close()
    server.stop(true)
  }

  writeFileSync(
    SUMMARY_PATH,
    `${JSON.stringify(
      {
        extractorVersion: MEDIAPIPE_EXTRACTOR_VERSION,
        listed: rows.length,
        missingVideo,
        previouslyExtracted: done.size,
        ...counts,
        finishedAt: new Date().toISOString(),
      },
      null,
      2,
    )}\n`,
  )
  if (counts.failed > 0) {
    console.log(
      `${counts.failed} videos failed to decode or load. Reasons are in ${FAILURES_PATH}. ` +
        "Playwright's bundled Chromium cannot decode H.264; rerun with --channel chrome if Google Chrome is installed.",
    )
  }
  console.log("clip report:")
  for (const report of clipReports) {
    console.log(
      `${report.label}\t${report.sourceGloss}\t${report.videoFile}\tdecoded=${report.decoded ? "yes" : "no"}\tkept=${report.kept}\tleft=${report.leftHandFrames}\tright=${report.rightHandFrames}\tboth=${report.bothHandsFrames}\tmostBothHands=${bothHandsInMostFrames(report)}${report.error ? `\terror=${report.error}` : ""}`,
    )
  }
  if (skeletonPath) console.log(`skeleton ${skeletonPath}`)
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error)
  process.exit(1)
})
