import {describe, expect, it} from "bun:test"
import {fitFrameSize, JPEG_QUALITY, MAX_FRAME_EDGE} from "../resize-frame"

describe("fitFrameSize", () => {
  it("keeps a frame that is already within the longest edge", () => {
    expect(fitFrameSize(640, 480)).toEqual({width: 640, height: 480})
  })

  it("shrinks the longest edge to 1280 and keeps the aspect ratio", () => {
    expect(fitFrameSize(4032, 3024)).toEqual({width: MAX_FRAME_EDGE, height: 960})
  })

  it("shrinks a portrait frame by its height", () => {
    expect(fitFrameSize(1000, 2000)).toEqual({width: 640, height: MAX_FRAME_EDGE})
  })

  it("returns an empty size for a missing frame", () => {
    expect(fitFrameSize(0, 100)).toEqual({width: 0, height: 0})
  })

  it("uses the jpeg quality the upload path expects", () => {
    expect(JPEG_QUALITY).toBe(0.8)
  })
})
