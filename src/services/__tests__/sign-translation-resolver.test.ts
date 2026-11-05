import {describe, expect, it, mock} from "bun:test"
import {DEFAULT_CAPTURE_FPS} from "../../sign/capture-config"
import {MEDIAPIPE_EXTRACTOR_VERSION} from "../../sign/extractor-version"
import {resolveSignInterpretation} from "../sign-translation-resolver"
import type {SignInterpreterInput} from "../gemini-sign-interpreter"
import type {SignClassifierResult, SignLandmarkFrame} from "../../sign/types"

function createFrame(timestampMs: number): SignLandmarkFrame {
  return {
    timestampMs,
    pose: {
      leftShoulder: {x: -0.5, y: 0, z: 0},
      rightShoulder: {x: 0.5, y: 0, z: 0},
      leftElbow: {x: -0.4, y: 0.2, z: 0},
      rightElbow: {x: 0.4, y: 0.2, z: 0},
      leftWrist: {x: -0.35, y: 0.4, z: 0},
      rightWrist: {x: 0.35, y: 0.4, z: 0},
    },
    leftHand: [],
    rightHand: [],
    face: {
      leftEyebrow: [],
      rightEyebrow: [],
      leftEye: [],
      rightEye: [],
      lips: [],
    },
  }
}

describe("resolveSignInterpretation", () => {
  const baseInput = {
    base64Grid: "ZmFrZQ==",
    base64SkeletonGrid: "c2tlbGV0b24=",
    mimeType: "image/jpeg",
    skeletonMimeType: "image/png",
    landmarkSequence: [createFrame(1000), createFrame(1083), createFrame(1166), createFrame(1249), createFrame(1332), createFrame(1415)],
    frameTimestamps: [1000, 1083, 1166, 1249, 1332, 1415],
    sequenceDurationMs: 415,
    sequenceFrameCount: 6,
    source: "device-camera" as const,
    gestureHints: [],
    extractorVersion: MEDIAPIPE_EXTRACTOR_VERSION,
  }

  it("uses the classifier when confidence is high", async () => {
    const classify = mock(async (): Promise<SignClassifierResult> => ({
      status: "recognized",
      label: "wait",
      confidence: 0.91,
      alternatives: [{label: "wait", confidence: 0.91}],
      model: "temporal-landmark-softmax-v1",
      modelVersion: "core-sign-20260409T120000Z",
      trainingRunId: "core-sign-20260409T120000Z",
    }))

    const interpret = mock(async () => {
      throw new Error("Gemini should not be called")
    })

    const result = await resolveSignInterpretation(baseInput, {classify, interpret})

    expect(result.recognizerType).toBe("classifier")
    expect(result.englishText).toBe("Wait")
    expect(result.fallbackUsed).toBe(false)
    expect(result.classifierModelVersion).toBe("core-sign-20260409T120000Z")
    expect(result.classifierTrainingRunId).toBe("core-sign-20260409T120000Z")
  })

  it("falls back to Gemini when the classifier is uncertain", async () => {
    const classify = mock(async (): Promise<SignClassifierResult> => ({
      status: "uncertain",
      label: "stop",
      confidence: 0.63,
      alternatives: [{label: "stop", confidence: 0.63}],
      model: "temporal-landmark-softmax-v1",
      modelVersion: "core-sign-20260409T121500Z",
      trainingRunId: "core-sign-20260409T121500Z",
      reason: "Low confidence margin",
    }))

    let interpretInput: SignInterpreterInput | undefined
    const interpret = mock(async (input: SignInterpreterInput) => {
      interpretInput = input
      return {
      englishText: "Please wait",
      rawResponse: "Please wait",
      model: "gemini-1.5-flash",
      }
    })

    const result = await resolveSignInterpretation(baseInput, {classify, interpret})

    expect(result.recognizerType).toBe("gemini")
    expect(result.englishText).toBe("Please wait")
    expect(result.fallbackUsed).toBe(true)
    expect(result.classifierModelVersion).toBe("core-sign-20260409T121500Z")
    expect(result.classifierTrainingRunId).toBe("core-sign-20260409T121500Z")
    expect(classify).toHaveBeenCalledTimes(1)
    expect(interpret).toHaveBeenCalledTimes(1)
    expect(interpretInput?.skeletonMimeType).toBe("image/png")
  })

  it("returns no_sign when Gemini also cannot interpret the sequence", async () => {
    const classify = mock(async (): Promise<SignClassifierResult> => ({
      status: "no_sign",
      confidence: 0.12,
      alternatives: [],
      model: "local-knn-sequence-v1",
      reason: "Below threshold",
    }))

    const interpret = mock(async () => ({
      englishText: "NO_SIGN_FOUND",
      rawResponse: "NO_SIGN_FOUND",
      model: "gemini-1.5-flash",
    }))

    const result = await resolveSignInterpretation(baseInput, {classify, interpret})

    expect(result.status).toBe("no_sign")
    expect(result.recognizerType).toBe("gemini")
    expect(result.fallbackUsed).toBe(true)
  })

  it("surfaces the classifier transport error when the local service is unavailable", async () => {
    const classify = mock(async (): Promise<SignClassifierResult> => {
      throw new Error("Local sign classifier timed out after 2000ms at http://127.0.0.1:8001/classify-sign-sequence.")
    })

    const interpret = mock(async () => ({
      englishText: "Thank you",
      rawResponse: "Thank you",
      model: "gemini-1.5-flash",
    }))

    const result = await resolveSignInterpretation(baseInput, {classify, interpret})

    expect(result.recognizerType).toBe("gemini")
    expect(result.fallbackUsed).toBe(true)
    expect(result.classifierStatus).toBe("unavailable")
    expect(result.classifierError).toContain("timed out after 2000ms")
  })

  it("computes fps=24 from 24fps-spaced timestamps, not 25", async () => {
    // 25 timestamps at ~41.67ms intervals → 1000ms total duration, 24 intervals
    const timestamps = Array.from({length: 25}, (_, i) => 1000 + Math.round(i * (1000 / 24)))
    const input = {
      ...baseInput,
      landmarkSequence: timestamps.map((t) => createFrame(t)),
      frameTimestamps: timestamps,
      sequenceDurationMs: timestamps[timestamps.length - 1] - timestamps[0],
      sequenceFrameCount: timestamps.length,
    }

    const classify = mock(async (_request: {fps: number}): Promise<SignClassifierResult> => ({
      status: "recognized",
      label: "wait",
      confidence: 0.95,
      alternatives: [{label: "wait", confidence: 0.95}],
      model: "temporal-landmark-softmax-v1",
    }))

    const interpret = mock(async () => {
      throw new Error("Gemini should not be called")
    })

    await resolveSignInterpretation(input, {classify, interpret})

    expect(classify.mock.calls.length).toBe(1)
    const classifyCall = classify.mock.calls[0]![0]
    expect(classifyCall.fps).toBe(24)
  })

  it("falls back to DEFAULT_CAPTURE_FPS when timestamps are sparse", async () => {
    const input = {
      ...baseInput,
      landmarkSequence: [createFrame(1000)],
      frameTimestamps: [1000],
      sequenceDurationMs: 0,
      sequenceFrameCount: 1,
    }

    const classify = mock(async (_request: {fps: number}): Promise<SignClassifierResult> => ({
      status: "uncertain",
      confidence: 0.3,
      alternatives: [],
      model: "temporal-landmark-softmax-v1",
    }))

    const interpret = mock(async () => ({
      englishText: "Hello",
      rawResponse: "Hello",
      model: "gemini-1.5-flash",
    }))

    await resolveSignInterpretation(input, {classify, interpret})

    expect(classify.mock.calls.length).toBe(1)
    const classifyCall = classify.mock.calls[0]![0]
    expect(classifyCall.fps).toBe(DEFAULT_CAPTURE_FPS)
  })
})
