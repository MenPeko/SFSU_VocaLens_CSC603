import {describe, expect, it} from "bun:test"
import {
  buildNormalizedLandmarkFrame,
  buildVideoSourceKey,
  computeFaceOcclusion,
  computeHandRegions,
  createHolisticTimingWindow,
  estimateGestureHints,
  experimentExtractorVersion,
  parseTrackingConfidenceExperiment,
  recordHolisticTiming,
  getSkeletonRenderStyle,
  type HolisticFrameDrawingData,
} from "../useSignLanguage"
import {MEDIAPIPE_EXTRACTOR_VERSION} from "../../../sign/extractor-version"

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

describe("face occlusion", () => {
  const hand = [
    {x: 0.4, y: 0.4, z: 0},
    {x: 0.5, y: 0.5, z: 0},
  ]

  it("pads each hand's bounding box", () => {
    const [region] = computeHandRegions({leftHandLandmarks: hand, rightHandLandmarks: []}, 0.5)

    expect(region.minX).toBeCloseTo(0.35, 6)
    expect(region.maxX).toBeCloseTo(0.55, 6)
  })

  it("reports the fraction of face points under a hand", () => {
    const regions = computeHandRegions({leftHandLandmarks: hand, rightHandLandmarks: []}, 0)
    const face = [
      {x: 0.45, y: 0.45, z: 0},
      {x: 0.9, y: 0.9, z: 0},
    ]

    expect(computeFaceOcclusion(face, regions)).toBe(0.5)
    expect(computeFaceOcclusion(face, [])).toBe(0)
  })
})

describe("tracking confidence experiment", () => {
  it("accepts only values strictly between 0 and 1", () => {
    expect(parseTrackingConfidenceExperiment(null)).toBeNull()
    expect(parseTrackingConfidenceExperiment("")).toBeNull()
    expect(parseTrackingConfidenceExperiment("abc")).toBeNull()
    expect(parseTrackingConfidenceExperiment("0")).toBeNull()
    expect(parseTrackingConfidenceExperiment("1")).toBeNull()
    expect(parseTrackingConfidenceExperiment("0.35")).toBe(0.35)
  })

  it("stamps a distinct extractor version only while the experiment is on", () => {
    expect(experimentExtractorVersion(null)).toBe(MEDIAPIPE_EXTRACTOR_VERSION)
    expect(experimentExtractorVersion(0.35)).toBe(`${MEDIAPIPE_EXTRACTOR_VERSION}+exp-tracking-0.35`)
  })
})

describe("recordHolisticTiming", () => {
  it("summarizes once per window and then resets", () => {
    const timing = createHolisticTimingWindow()

    expect(recordHolisticTiming(timing, 10, 3)).toBeNull()
    expect(recordHolisticTiming(timing, 20, 3)).toBeNull()
    const summary = recordHolisticTiming(timing, 30, 3)

    expect(summary?.averageMs).toBeCloseTo(20, 6)
    expect(summary?.maxMs).toBe(30)
    expect(summary?.count).toBe(3)
    expect(timing).toEqual(createHolisticTimingWindow())
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
