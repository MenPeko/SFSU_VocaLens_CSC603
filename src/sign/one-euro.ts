/**
 * One Euro filter (Casiez et al., CHI 2012): an adaptive low-pass filter that
 * smooths heavily when a signal is still and lightly when it moves fast, which
 * removes landmark jitter without adding visible lag during signing.
 */

export interface OneEuroOptions {
  /** Cutoff frequency (Hz) used when the signal is still. Lower = smoother at rest. */
  minCutoff: number
  /** Speed coefficient. Higher = less lag during fast motion. */
  beta: number
  /** Cutoff frequency (Hz) for the derivative estimate. */
  derivativeCutoff: number
}

// Tuned for MediaPipe normalized [0, 1] image coordinates at ~30 fps.
export const DEFAULT_ONE_EURO_OPTIONS: OneEuroOptions = {
  minCutoff: 1.0,
  beta: 10,
  derivativeCutoff: 1.0,
}

// A gap longer than this means tracking was lost; restart instead of easing
// from a stale position.
const MAX_SAMPLE_GAP_MS = 250

function smoothingFactor(cutoffHz: number, elapsedSeconds: number): number {
  const tau = 1 / (2 * Math.PI * cutoffHz)
  return 1 / (1 + tau / elapsedSeconds)
}

export class OneEuroFilter {
  private previousValue: number | null = null
  private previousDerivative = 0
  private previousTimestampMs: number | null = null

  constructor(private readonly options: OneEuroOptions = DEFAULT_ONE_EURO_OPTIONS) {}

  filter(value: number, timestampMs: number): number {
    if (
      this.previousValue === null ||
      this.previousTimestampMs === null ||
      timestampMs - this.previousTimestampMs > MAX_SAMPLE_GAP_MS
    ) {
      this.previousValue = value
      this.previousDerivative = 0
      this.previousTimestampMs = timestampMs
      return value
    }

    const elapsedSeconds = (timestampMs - this.previousTimestampMs) / 1000
    if (elapsedSeconds <= 0) {
      return this.previousValue
    }

    const derivative = (value - this.previousValue) / elapsedSeconds
    const derivativeAlpha = smoothingFactor(this.options.derivativeCutoff, elapsedSeconds)
    const smoothedDerivative = derivativeAlpha * derivative + (1 - derivativeAlpha) * this.previousDerivative

    const cutoff = this.options.minCutoff + this.options.beta * Math.abs(smoothedDerivative)
    const alpha = smoothingFactor(cutoff, elapsedSeconds)
    const smoothedValue = alpha * value + (1 - alpha) * this.previousValue

    this.previousValue = smoothedValue
    this.previousDerivative = smoothedDerivative
    this.previousTimestampMs = timestampMs
    return smoothedValue
  }

  reset(): void {
    this.previousValue = null
    this.previousDerivative = 0
    this.previousTimestampMs = null
  }
}

interface LandmarkLike {
  x: number
  y: number
  z: number
  visibility?: number
}

/** Smooths a fixed-length landmark list (one hand, the pose, or the face mesh). */
export class LandmarkListSmoother {
  private filters: Array<{x: OneEuroFilter; y: OneEuroFilter; z: OneEuroFilter}> = []

  constructor(private readonly options: OneEuroOptions = DEFAULT_ONE_EURO_OPTIONS) {}

  smooth<T extends LandmarkLike>(landmarks: readonly T[], timestampMs: number): T[] {
    if (landmarks.length === 0) {
      this.reset()
      return []
    }

    if (this.filters.length !== landmarks.length) {
      this.filters = landmarks.map(() => ({
        x: new OneEuroFilter(this.options),
        y: new OneEuroFilter(this.options),
        z: new OneEuroFilter(this.options),
      }))
    }

    return landmarks.map((landmark, index) => {
      const filter = this.filters[index]
      return {
        ...landmark,
        x: filter.x.filter(landmark.x, timestampMs),
        y: filter.y.filter(landmark.y, timestampMs),
        z: filter.z.filter(landmark.z, timestampMs),
      }
    })
  }

  reset(): void {
    this.filters = []
  }
}
