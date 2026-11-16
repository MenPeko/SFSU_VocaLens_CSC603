import {describe, expect, it} from "bun:test"
import {pickAudioInputId} from "../audio-input"

const devices = [
  {deviceId: "default", label: "Default - ThinkPad USB-C Dock Gen2 USB Audio"},
  {deviceId: "dock", label: "ThinkPad USB-C Dock Gen2 USB Audio"},
  {deviceId: "mac", label: "MacBook Pro Microphone (Built-in)"},
  {deviceId: "zoom", label: "ZoomAudioDevice (Virtual)"},
]

describe("microphone selection", () => {
  it("prefers the built-in microphone over a silent dock default", () => {
    expect(pickAudioInputId(devices, "")).toBe("mac")
  })

  it("keeps an explicit microphone choice", () => {
    expect(pickAudioInputId(devices, "dock")).toBe("dock")
  })

  it("uses another real input when there is no built-in microphone", () => {
    expect(
      pickAudioInputId(
        [
          {deviceId: "default", label: "Default - USB Audio"},
          {deviceId: "headset", label: "Headset Microphone"},
        ],
        "",
      ),
    ).toBe("headset")
  })
})
