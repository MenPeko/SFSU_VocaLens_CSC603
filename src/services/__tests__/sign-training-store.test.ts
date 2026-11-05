import {afterAll, describe, expect, it} from "bun:test"
import {existsSync, mkdtempSync, rmSync} from "node:fs"
import {join} from "node:path"
import {tmpdir} from "node:os"
import {MEDIAPIPE_EXTRACTOR_VERSION} from "../../sign/extractor-version"

const tempDataDir = mkdtempSync(join(tmpdir(), "whiteboard-reader-tts-sign-store-"))
process.env.SIGN_TRAINING_DATA_DIR = tempDataDir

const {saveSignTrainingSample} = await import("../sign-training-store")

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
