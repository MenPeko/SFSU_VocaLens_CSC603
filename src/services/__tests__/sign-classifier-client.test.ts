import {afterEach, beforeEach, describe, expect, it, mock, spyOn} from "bun:test"
import {MEDIAPIPE_EXTRACTOR_VERSION} from "../../sign/extractor-version"
import {classifySignSequence} from "../sign-classifier-client"
import type {SignClassifierResult} from "../../sign/types"

describe("classifySignSequence", () => {
  const originalEnv = {...process.env}
  const input = {
    frames: [],
    fps: 12,
    source: "device-camera" as const,
    sequenceDurationMs: 0,
    extractorVersion: MEDIAPIPE_EXTRACTOR_VERSION,
  }

  beforeEach(() => {
    process.env = {...originalEnv}
    delete process.env.SIGN_CLASSIFIER_DISABLED
    delete process.env.SIGN_CLASSIFIER_TIMEOUT_MS
    delete process.env.SIGN_CLASSIFIER_URL
  })

  afterEach(() => {
    process.env = {...originalEnv}
    mock.restore()
  })

  it("formats timeout errors with the local classifier url", async () => {
    const abortError = new Error("The operation was aborted.")
    abortError.name = "AbortError"

    spyOn(globalThis, "fetch").mockRejectedValue(abortError)

    await expect(classifySignSequence(input)).rejects.toThrow(
      "Local sign classifier timed out after 2000ms at http://127.0.0.1:8001/classify-sign-sequence.",
    )
  })

  it("formats connection failures as unreachable local classifier errors", async () => {
    spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"))

    await expect(classifySignSequence(input)).rejects.toThrow(
      "Could not reach the local sign classifier at http://127.0.0.1:8001/classify-sign-sequence.",
    )
  })

  it("passes through successful classifier responses", async () => {
    const payload: SignClassifierResult = {
      status: "recognized",
      label: "wait",
      confidence: 0.92,
      alternatives: [{label: "wait", confidence: 0.92}],
      model: "temporal-landmark-softmax-v1",
      modelVersion: "core-sign-test-model",
      trainingRunId: "core-sign-test-run",
    }

    spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: {"Content-Type": "application/json"},
      }),
    )

    await expect(classifySignSequence(input)).resolves.toEqual(payload)
  })
})
