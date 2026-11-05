declare module "fingerpose" {
  export interface GestureMatch {
    name: string
    score: number
  }

  export interface GestureEstimate {
    poseData: Array<[string, string, string]>
    gestures: GestureMatch[]
  }

  export class GestureDescription {
    constructor(name: string)
    addCurl(finger: number, curl: number, confidence?: number): void
    addDirection(finger: number, direction: number, confidence?: number): void
    matchAgainst(curls: number[], directions: number[]): number
  }

  export class GestureEstimator {
    constructor(gestures: GestureDescription[], options?: Record<string, number>)
    estimate(landmarks: Array<{x: number; y: number; z: number}>, minScore: number): GestureEstimate
  }

  export const Finger: {
    Thumb: number
    Index: number
    Middle: number
    Ring: number
    Pinky: number
    all: number[]
  }

  export const FingerCurl: {
    NoCurl: number
    HalfCurl: number
    FullCurl: number
  }

  export const FingerDirection: {
    VerticalUp: number
    VerticalDown: number
    HorizontalLeft: number
    HorizontalRight: number
    DiagonalUpRight: number
    DiagonalUpLeft: number
    DiagonalDownRight: number
    DiagonalDownLeft: number
  }

  export const Gestures: {
    VictoryGesture: GestureDescription
    ThumbsUpGesture: GestureDescription
  }

  const fingerpose: {
    GestureEstimator: typeof GestureEstimator
    GestureDescription: typeof GestureDescription
    Finger: typeof Finger
    FingerCurl: typeof FingerCurl
    FingerDirection: typeof FingerDirection
    Gestures: typeof Gestures
  }

  export default fingerpose
}
