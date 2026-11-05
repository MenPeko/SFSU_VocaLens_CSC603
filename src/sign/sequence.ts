export function appendRollingWindow<T>(items: T[], value: T, maxSize: number): T[] {
  return [...items, value].slice(-maxSize)
}

export function computeSequenceDurationMs(frameTimestamps: number[]): number {
  if (frameTimestamps.length < 2) {
    return 0
  }

  return Math.max(0, frameTimestamps[frameTimestamps.length - 1] - frameTimestamps[0])
}
