import {readFileSync} from "node:fs"
import {describe, expect, it} from "bun:test"
import {compareModelName, textFromOpenAiVisionResponse} from "../../../scripts/vocalens/compare-vision"
import {buildCompareRow, runModelCompare} from "../../../scripts/vocalens/model-compare"
import {ERROR_GROUPS, renderErrorGroupSummary} from "../../../scripts/vocalens/vizwiz-summary"
import {runVizwiz, type VizwizOutputRow} from "../../../scripts/vocalens/vizwiz-run"

const askRoute = readFileSync(new URL("../../../src/api/vocalens-routes.ts", import.meta.url), "utf8")

describe("vizwiz answer cache", () => {
  it("does not call the model again for ids already in the output file", async () => {
    const asked: string[] = []
    const written: VizwizOutputRow[] = []
    const questionsRaw = [
      JSON.stringify({id: "already", image: "already.jpg", question: "What is this?"}),
      JSON.stringify({id: "new-one", image: "new.jpg", question: "What color is it?"}),
    ].join("\n")
    const cachedRaw = `${JSON.stringify({
      id: "already",
      question: "What is this?",
      answer: "A cup.",
      correct: null,
      unclearFlag: null,
      notes: null,
      errorGroup: "",
    })}\n`

    const result = await runVizwiz({
      questionsRaw,
      cachedRaw,
      readImage: async (image) => Buffer.from(image),
      answer: async (_image, _mimeType, question) => {
        asked.push(question)
        return "Blue."
      },
      writeRow: async (row) => {
        written.push(row)
      },
    })

    expect(result).toEqual({wrote: 1, skipped: 1})
    expect(asked).toEqual(["What color is it?"])
    expect(written).toEqual([
      {
        id: "new-one",
        question: "What color is it?",
        answer: "Blue.",
        correct: null,
        unclearFlag: null,
        notes: null,
        errorGroup: "",
      },
    ])
  })

  it("skips a cached image name when the question has no separate id", async () => {
    const asked: string[] = []
    const result = await runVizwiz({
      questionsRaw: JSON.stringify({image: "label.jpg", question: "Read this."}),
      cachedRaw: JSON.stringify({id: "label.jpg", answer: "Milk.", errorGroup: ""}),
      readImage: async () => {
        throw new Error("cached image was read")
      },
      answer: async () => {
        asked.push("called")
        return "nope"
      },
      writeRow: async () => {
        throw new Error("cached row was written again")
      },
    })

    expect(result).toEqual({wrote: 0, skipped: 1})
    expect(asked).toEqual([])
  })

  it("does not call the model when a cached line is not valid JSON", async () => {
    const asked: string[] = []
    await expect(
      runVizwiz({
        questionsRaw: JSON.stringify({id: "q1", image: "a.jpg", question: "What?"}),
        cachedRaw: "{",
        readImage: async () => Buffer.from("img"),
        answer: async () => {
          asked.push("called")
          return "answer"
        },
        writeRow: async () => {},
      }),
    ).rejects.toThrow("Nothing new was asked")
    expect(asked).toEqual([])
  })
})

describe("vizwiz error group summary", () => {
  it("counts only the filled allowed groups", () => {
    const raw = [
      {errorGroup: "invented_text", correct: true},
      {errorGroup: "invented_text"},
      {errorGroup: "wrong_object"},
      {errorGroup: "answered_unreadable"},
      {errorGroup: "refused_visible"},
      {errorGroup: "ok"},
      {errorGroup: "ok"},
      {errorGroup: ""},
      {correct: false},
    ]
      .map((row) => JSON.stringify(row))
      .join("\n")

    const rendered = renderErrorGroupSummary(raw)
    expect(rendered.exitCode).toBe(0)
    expect(rendered.lines).toEqual([
      "invented_text\t2",
      "wrong_object\t1",
      "answered_unreadable\t1",
      "refused_visible\t1",
      "ok\t2",
    ])
    expect(rendered.lines.join("\n")).not.toMatch(/%|accuracy|score/i)
  })

  it("exits quietly when every error group is empty", () => {
    const rendered = renderErrorGroupSummary(
      [JSON.stringify({id: "a", correct: true, errorGroup: ""}), JSON.stringify({id: "b", notes: null})].join("\n"),
    )

    expect(rendered.exitCode).toBe(0)
    expect(rendered.lines).toEqual(["Every errorGroup is empty. No score was computed."])
    expect(ERROR_GROUPS).toEqual(["invented_text", "wrong_object", "answered_unreadable", "refused_visible", "ok"])
  })

  it("does not turn an unknown label into a score", () => {
    const rendered = renderErrorGroupSummary(`${JSON.stringify({errorGroup: "maybe"})}\n`)
    expect(rendered.exitCode).toBe(0)
    expect(rendered.lines).toEqual(["unrecognized\tmaybe\t1", "No score was computed."])
  })
})

describe("hosted vision comparison", () => {
  it("stores both answers and latencies and leaves judgments empty", async () => {
    const written: unknown[] = []
    const wrote = await runModelCompare({
      questionsRaw: JSON.stringify({id: "q1", image: "a.jpg", question: "What is this?"}),
      otherModel: "gpt-4o",
      readImage: async () => Buffer.from("img"),
      answerWithGemini: async () => "A cup.",
      answerWithOther: async () => "A mug.",
      writeRow: async (row) => {
        written.push(row)
      },
    })

    expect(wrote).toBe(1)
    const row = written[0] as ReturnType<typeof buildCompareRow>
    expect(row.geminiAnswer).toBe("A cup.")
    expect(row.otherAnswer).toBe("A mug.")
    expect(row.otherModel).toBe("gpt-4o")
    expect(row.geminiLatencyMs).toBeGreaterThanOrEqual(0)
    expect(row.otherLatencyMs).toBeGreaterThanOrEqual(0)
    expect(row.geminiCorrect).toBeNull()
    expect(row.otherCorrect).toBeNull()
    expect(row.preferred).toBeNull()
    expect(row.notes).toBeNull()
    expect(row).not.toHaveProperty("score")
  })

  it("takes the other model from the environment and does not accept Gemini", () => {
    expect(compareModelName({VOCALENS_COMPARE_MODEL: "gpt-4o"} as NodeJS.ProcessEnv)).toBe("gpt-4o")
    expect(() => compareModelName({} as NodeJS.ProcessEnv)).toThrow(/VOCALENS_COMPARE_MODEL/)
    expect(() => compareModelName({VOCALENS_COMPARE_MODEL: "gemini-2.5-flash"} as NodeJS.ProcessEnv)).toThrow(/Gemini/)
  })

  it("reads text from a hosted vision response without filling a grade", () => {
    expect(textFromOpenAiVisionResponse({choices: [{message: {content: "A red cup."}}]})).toBe("A red cup.")
    expect(textFromOpenAiVisionResponse({choices: [{message: {content: [{text: "Milk."}]}}]})).toBe("Milk.")
  })

  it("keeps the live ask route on Gemini", () => {
    expect(askRoute).toContain("streamVisualAnswer")
    expect(askRoute).not.toContain("compare-vision")
    expect(askRoute).not.toContain("VOCALENS_COMPARE_MODEL")
    expect(askRoute).not.toContain("answerWithCompareModel")
  })
})
