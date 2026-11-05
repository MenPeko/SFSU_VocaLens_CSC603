export const GRID_COLUMNS = 4
export const GRID_ROWS = 4
export const MAX_GRID_FRAMES = GRID_COLUMNS * GRID_ROWS
export const MAX_SEQUENCE_FRAMES = 48
export const GRID_CELL_SIZE = 160
export const DEFAULT_CAPTURE_FPS = 24
export const DEFAULT_HANDS_DOWN_DELAY_MS = 1500
export const MIN_SEQUENCE_DURATION_MS = 500

export function effectiveGridFps(captureFps: number): number {
  return Math.min(captureFps, 12)
}

export function landmarkIntervalMs(captureFps: number): number {
  return 1000 / captureFps
}

export function gridIntervalMs(captureFps: number): number {
  return 1000 / effectiveGridFps(captureFps)
}

export function windowLabel(fps: number, maxFrames: number): string {
  const durationSec = (maxFrames / fps).toFixed(1)
  return `${maxFrames} @ ${fps}fps (~${durationSec}s)`
}
