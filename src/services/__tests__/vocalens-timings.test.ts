import {mkdtemp, readFile} from "node:fs/promises"
import {tmpdir} from "node:os"
import {join} from "node:path"
import {describe, expect, it} from "bun:test"
import {appendTimingRecord, buildTimingRecord} from "../vocalens-timings"

describe("vocalens timings", () => {
  it("keeps stage timestamps and drops image or audio fields", () => {
    const record = buildTimingRecord({
      sessionId: "session-one",
      question: "What does this say?",
      answer: "Milk.",
      captureAt: 10,
      transcriptionStartedAt: 20,
      transcriptionEndedAt: 30,
      visionStartedAt: 40,
      visionEndedAt: 50,
      speechStartedAt: 60,
      image: "secret-photo",
      audio: "secret-audio",
    })

    expect(record).toEqual({
      sessionId: "session-one",
      question: "What does this say?",
      answer: "Milk.",
      captureAt: 10,
      transcriptionStartedAt: 20,
      transcriptionEndedAt: 30,
      visionStartedAt: 40,
      visionEndedAt: 50,
      speechStartedAt: 60,
    })
    expect("image" in record).toBe(false)
    expect("audio" in record).toBe(false)
  })

  it("appends one json line per question", async () => {
    const directory = await mkdtemp(join(tmpdir(), "vocalens-timings-"))
    const path = join(directory, "timings.jsonl")
    await appendTimingRecord(
      {
        sessionId: "session-one",
        question: "What is this?",
        answer: "A cup.",
        captureAt: 1,
        transcriptionStartedAt: 2,
        transcriptionEndedAt: 3,
        visionStartedAt: 4,
        visionEndedAt: 5,
        speechStartedAt: 6,
      },
      path,
    )

    const lines = (await readFile(path, "utf8")).trim().split("\n")
    expect(lines).toHaveLength(1)
    expect(JSON.parse(lines[0]).answer).toBe("A cup.")
  })
})
