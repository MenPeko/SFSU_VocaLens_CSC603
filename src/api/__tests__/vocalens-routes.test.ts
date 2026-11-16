import {beforeEach, describe, expect, it} from "bun:test"
import {handleVocalensAsk, resolveAskFrame} from "../vocalens-routes"
import {clearAllVocalensSessions, getVocalensSession} from "../../services/vocalens-session"
import type {VocalensTurn} from "../../services/vocalens-session"

const sessionId = "session-one"

function askRequest(followUp: boolean, frame: Buffer | null, audioText = "audio"): Request {
  const form = new FormData()
  form.set("sessionId", sessionId)
  form.set("followUp", followUp ? "true" : "false")
  form.set("captureAt", "1000")
  form.set("audio", new File([new TextEncoder().encode(audioText)], "question.webm", {type: "audio/webm"}))
  if (frame) {
    form.set("frame", new File([new Uint8Array(frame)], "frame.jpg", {type: "image/jpeg"}))
  }
  return new Request("http://localhost/api/vocalens/ask", {method: "POST", body: form})
}

async function readEvents(response: Response): Promise<Array<Record<string, unknown>>> {
  const text = await response.text()
  return text
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>)
}

describe("vocalens ask route", () => {
  beforeEach(() => {
    clearAllVocalensSessions()
  })

  it("reuses the stored frame for a follow-up even when another image is uploaded", () => {
    const stored = Buffer.from("stored-frame")
    const uploaded = Buffer.from("new-frame")
    const frame = resolveAskFrame({followUp: true, uploaded, stored})
    expect(frame).toBe(stored)
  })

  it("answers a new question and a follow-up from the first photo", async () => {
    const seenImages: string[] = []
    const seenTurns: VocalensTurn[][] = []
    const deps = {
      now: () => 5_000,
      transcribe: async () => ({text: "what is this", provider: "openai" as const}),
      streamAnswer: async function* (image: Buffer, _mimeType: string, turns: VocalensTurn[]) {
        seenImages.push(image.toString())
        seenTurns.push(turns.map((turn) => ({...turn})))
        yield "A red cup."
      },
    }

    const first = await handleVocalensAsk(askRequest(false, Buffer.from("stored-frame")), deps)
    const firstEvents = await readEvents(first)
    expect(firstEvents.some((event) => event.type === "sentence" && event.text === "A red cup.")).toBe(true)
    expect(firstEvents.at(-1)?.type).toBe("done")

    deps.transcribe = async () => ({text: "what color is it", provider: "openai"})
    deps.streamAnswer = async function* (image: Buffer, _mimeType: string, turns: VocalensTurn[]) {
      seenImages.push(image.toString())
      seenTurns.push(turns.map((turn) => ({...turn})))
      yield "It is red."
    }

    const followUp = await handleVocalensAsk(askRequest(true, Buffer.from("should-not-be-used")), deps)
    const followUpEvents = await readEvents(followUp)

    expect(seenImages).toEqual(["stored-frame", "stored-frame"])
    expect(seenTurns[1][0]).toEqual({role: "user", text: "what is this"})
    expect(seenTurns[1][1]).toEqual({role: "model", text: "A red cup."})
    expect(seenTurns[1][2]).toEqual({role: "user", text: "what color is it"})
    expect(followUpEvents.at(-1)?.answer).toBe("It is red.")
    expect(getVocalensSession(sessionId, 5_000)?.image.toString()).toBe("stored-frame")
  })
})
