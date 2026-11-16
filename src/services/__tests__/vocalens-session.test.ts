import {beforeEach, describe, expect, it} from "bun:test"
import {
  appendVocalensTurn,
  clearAllVocalensSessions,
  getVocalensSession,
  replaceVocalensSession,
  VOCALENS_SESSION_IDLE_MS,
} from "../vocalens-session"

describe("vocalens session", () => {
  beforeEach(() => {
    clearAllVocalensSessions()
  })

  it("replaces the stored photo and drops earlier turns", () => {
    replaceVocalensSession("session-one", Buffer.from("first-photo"), "image/jpeg", 10, 1_000)
    appendVocalensTurn("session-one", {role: "user", text: "What is this?"}, 1_000)
    appendVocalensTurn("session-one", {role: "model", text: "A cup."}, 1_000)

    replaceVocalensSession("session-one", Buffer.from("second-photo"), "image/jpeg", 20, 2_000)
    const session = getVocalensSession("session-one", 2_000)

    expect(session?.image.toString()).toBe("second-photo")
    expect(session?.turns).toEqual([])
    expect(session?.captureAt).toBe(20)
  })

  it("drops a session after it has been idle", () => {
    replaceVocalensSession("session-one", Buffer.from("photo"), "image/jpeg", 10, 1_000)
    expect(getVocalensSession("session-one", 1_000 + VOCALENS_SESSION_IDLE_MS + 1)).toBeNull()
  })
})
