import {describe, expect, it} from "bun:test"
import {DEFAULT_SIGN_CAPTURE_SETTINGS} from "../../sign/capture-settings"
import {fetchSignCaptureBootstrap} from "../sign-capture-bootstrap"

describe("fetchSignCaptureBootstrap", () => {
  it("returns loaded settings when the bootstrap request succeeds", async () => {
    const result = await fetchSignCaptureBootstrap(
      async () =>
        new Response(JSON.stringify({holisticModelComplexity: 1}), {
          status: 200,
          headers: {"Content-Type": "application/json"},
        }),
      50,
    )

    expect(result.status).toBe("loaded")
    expect(result.settings.holisticModelComplexity).toBe(1)
  })

  it("falls back after a timeout", async () => {
    const result = await fetchSignCaptureBootstrap(async () => new Promise<Response>(() => {}), 5)

    expect(result.status).toBe("fallback")
    expect(result.fallbackReason).toBe("timeout")
    expect(result.settings).toEqual(DEFAULT_SIGN_CAPTURE_SETTINGS)
  })

  it("falls back after a request error", async () => {
    const result = await fetchSignCaptureBootstrap(async () => {
      throw new Error("boom")
    }, 50)

    expect(result.status).toBe("fallback")
    expect(result.fallbackReason).toBe("request_error")
    expect(result.settings).toEqual(DEFAULT_SIGN_CAPTURE_SETTINGS)
  })
})
