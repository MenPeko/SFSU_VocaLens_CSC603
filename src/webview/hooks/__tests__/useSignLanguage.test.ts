import {describe, expect, it} from "bun:test"
import {
  buildNormalizedLandmarkFrame,
  buildVideoSourceKey,
  estimateGestureHints,
  getSkeletonRenderStyle,
  type HolisticFrameDrawingData,
} from "../useSignLanguage"

describe("buildNormalizedLandmarkFrame", () => {
  it("normalizes coordinates around the shoulder center and width", () => {
    const frame: HolisticFrameDrawingData = {
      poseLandmarks: Array.from({length: 33}, () => ({x: 0, y: 0, z: 0})),
      faceLandmarks: Array.from({length: 400}, () => ({x: 0.5, y: 0.5, z: 0})),
      leftHandLandmarks: [{x: 0.4, y: 0.4, z: 0}, {x: 0.35, y: 0.35, z: 0}],
      rightHandLandmarks: [{x: 0.6, y: 0.4, z: 0}, {x: 0.65, y: 0.35, z: 0}],
      handednessLabels: ["LEFT HAND", "RIGHT HAND"],
    }

    frame.poseLandmarks[11] = {x: 0.4, y: 0.5, z: 0}
    frame.poseLandmarks[12] = {x: 0.6, y: 0.5, z: 0}
    frame.poseLandmarks[13] = {x: 0.35, y: 0.65, z: 0}
    frame.poseLandmarks[14] = {x: 0.65, y: 0.65, z: 0}
    frame.poseLandmarks[15] = {x: 0.3, y: 0.8, z: 0}
    frame.poseLandmarks[16] = {x: 0.7, y: 0.8, z: 0}

    const normalized = buildNormalizedLandmarkFrame(frame, 1234)

    expect(normalized.pose.leftShoulder?.x).toBeCloseTo(-0.5, 4)
    expect(normalized.pose.rightShoulder?.x).toBeCloseTo(0.5, 4)
    expect(normalized.leftHand[0]?.x).toBeCloseTo(-0.5, 4)
    expect(normalized.rightHand[0]?.x).toBeCloseTo(0.5, 4)
    expect(normalized.timestampMs).toBe(1234)
  })
})

describe("estimateGestureHints", () => {
  it("skips partial hand landmark sets instead of throwing", () => {
    const partialHands = [{label: "Left", landmarks: [{x: 0.4, y: 0.4, z: 0}]}] as Parameters<typeof estimateGestureHints>[0]

    expect(() => estimateGestureHints(partialHands)).not.toThrow()
    expect(estimateGestureHints(partialHands)).toEqual([])
  })
})

describe("buildVideoSourceKey", () => {
  it("includes requested and actual quality so quality changes restart the pipeline", () => {
    expect(
      buildVideoSourceKey({
        type: "device-camera",
        requestedQuality: "1080p",
      }),
    ).toContain("1080p")

    expect(
      buildVideoSourceKey({
        type: "glasses-stream",
        requestedQuality: "1080p",
        actualQuality: "720p",
        qualityFallback: true,
        sources: [{url: "https://example.com/live.m3u8", transport: "hls"}],
      }),
    ).toContain("720p")
  })
})

describe("getSkeletonRenderStyle", () => {
  it("returns crisp diagnostic styling for saved skeleton frames", () => {
    expect(getSkeletonRenderStyle("diagnostic")).toEqual({
      lineCap: "butt",
      lineJoin: "miter",
      shadowBlur: 0,
      shadowColor: "rgba(0,0,0,0)",
    })
  })
})
