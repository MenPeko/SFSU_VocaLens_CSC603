import {afterAll, afterEach, describe, expect, it} from "bun:test"
import {existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from "node:fs"
import {join} from "node:path"
import {tmpdir} from "node:os"
import {MEDIAPIPE_EXTRACTOR_VERSION} from "../../sign/extractor-version"

const tempDataDir = mkdtempSync(join(tmpdir(), "vocalens-sign-store-"))
process.env.SIGN_TRAINING_DATA_DIR = tempDataDir

const {saveSignTrainingSample, SignTrainingSampleValidationError} = await import("../sign-training-store")

afterAll(() => {
  rmSync(tempDataDir, {recursive: true, force: true})
  delete process.env.SIGN_TRAINING_DATA_DIR
})

function createSample(overrides: Partial<Parameters<typeof saveSignTrainingSample>[0]> = {}) {
  return {
    label: "wait" as const,
    labelSource: "manual_capture" as const,
    landmarkSequence: [],
    frameTimestamps: [],
    sequenceDurationMs: 0,
    sequenceFrameCount: 0,
    source: "device-camera" as const,
    extractorVersion: MEDIAPIPE_EXTRACTOR_VERSION,
    capturedAt: Date.now(),
    frameCount: 1,
    gridColumns: 4,
    gridRows: 4,
    mimeType: "image/jpeg" as const,
    skeletonMimeType: "image/png" as const,
    gestureHints: [],
    base64Grid: Buffer.from("rgb-bytes").toString("base64"),
    base64SkeletonGrid: Buffer.from("png-bytes").toString("base64"),
    ...overrides,
  }
}

describe("saveSignTrainingSample", () => {
  it("writes new skeleton captures as PNG when skeletonMimeType is image/png", async () => {
    const {sampleId} = await saveSignTrainingSample(createSample())
    const sampleBasePath = join(tempDataDir, "wait", sampleId)

    expect(existsSync(`${sampleBasePath}.json`)).toBe(true)
    expect(existsSync(`${sampleBasePath}-rgb.jpg`)).toBe(true)
    expect(existsSync(`${sampleBasePath}-skeleton.png`)).toBe(true)
  })

  it("falls back to JPG skeleton captures when the skeleton MIME type is absent", async () => {
    const {sampleId} = await saveSignTrainingSample(
      createSample({
        skeletonMimeType: undefined,
      }),
    )
    const sampleBasePath = join(tempDataDir, "wait", sampleId)

    expect(existsSync(`${sampleBasePath}-skeleton.jpg`)).toBe(true)
  })
})

describe("saveSignTrainingSample with SIGN_TRAINING_SESSION", () => {
  afterEach(() => {
    delete process.env.SIGN_TRAINING_SESSION
  })

  it("writes into the session folder and stamps the session fields", async () => {
    const sessionId = "2026-11-10_owner_desk-daylight"
    const sessionDir = join(tempDataDir, "sessions", sessionId)
    mkdirSync(sessionDir, {recursive: true})
    writeFileSync(
      join(sessionDir, "session.json"),
      JSON.stringify({
        sessionId,
        role: "test",
        signer: "owner",
        recordedOn: "2026-11-10",
        lighting: "desk-daylight",
        camera: "iphone13",
        extractorVersion: MEDIAPIPE_EXTRACTOR_VERSION,
        createdAt: "2026-11-10T00:00:00Z",
      }),
    )
    process.env.SIGN_TRAINING_SESSION = sessionId

    const {sampleId, stats} = await saveSignTrainingSample(createSample({label: "yes"}))
    const saved = JSON.parse(readFileSync(join(sessionDir, "yes", `${sampleId}.json`), "utf8"))

    expect(saved.sessionId).toBe(sessionId)
    expect(saved.sessionRole).toBe("test")
    expect(saved.signer).toBe("owner")
    expect(stats.labels.yes).toBe(1)
  })

  function writeSession(sessionId: string, role: string) {
    const sessionDir = join(tempDataDir, "sessions", sessionId)
    mkdirSync(sessionDir, {recursive: true})
    writeFileSync(
      join(sessionDir, "session.json"),
      JSON.stringify({sessionId, role, signer: "owner", recordedOn: "2026-11-12", lighting: "office", camera: "iphone13"}),
    )
    return sessionDir
  }

  it("records into a validation session", async () => {
    const sessionId = "2026-11-12_owner_office"
    const sessionDir = writeSession(sessionId, "validation")
    process.env.SIGN_TRAINING_SESSION = sessionId

    const {sampleId} = await saveSignTrainingSample(createSample({label: "help"}))
    const saved = JSON.parse(readFileSync(join(sessionDir, "help", `${sampleId}.json`), "utf8"))

    expect(saved.sessionRole).toBe("validation")
  })

  it("refuses a session.json with an unknown role", async () => {
    const sessionId = "2026-11-12_owner_holdout"
    writeSession(sessionId, "holdout")
    process.env.SIGN_TRAINING_SESSION = sessionId

    await expect(saveSignTrainingSample(createSample())).rejects.toBeInstanceOf(SignTrainingSampleValidationError)
  })

  it("refuses to save when the session has no session.json", async () => {
    process.env.SIGN_TRAINING_SESSION = "2026-11-11_owner_missing"

    await expect(saveSignTrainingSample(createSample())).rejects.toBeInstanceOf(SignTrainingSampleValidationError)
  })

  it("rejects session ids that could escape the samples folder", async () => {
    process.env.SIGN_TRAINING_SESSION = "../outside"

    await expect(saveSignTrainingSample(createSample())).rejects.toBeInstanceOf(SignTrainingSampleValidationError)
  })
})
