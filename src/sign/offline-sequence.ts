import {DEFAULT_HANDS_DOWN_DELAY_MS, MAX_SEQUENCE_FRAMES, MIN_SEQUENCE_DURATION_MS} from "./capture-config"
import {appendRollingWindow, computeSequenceDurationMs} from "./sequence"

export interface OfflineFrame<T> {
  timestampMs: number
  hasHands: boolean
  frame: T
}

export interface OfflineSequence<T> {
  frames: T[]
  frameTimestamps: number[]
}

// Replays the live capture rule in useSignLanguage.ts over frames sampled at
// the capture rate: frames are kept only while a hand is visible, in a rolling
// window of MAX_SEQUENCE_FRAMES; a sequence ends once no hand has been seen for
// DEFAULT_HANDS_DOWN_DELAY_MS, and is discarded if shorter than
// MIN_SEQUENCE_DURATION_MS. The end of the video counts as hands down.
export function segmentHandSequences<T>(
  samples: OfflineFrame<T>[],
  {
    maxFrames = MAX_SEQUENCE_FRAMES,
    handsDownDelayMs = DEFAULT_HANDS_DOWN_DELAY_MS,
    minDurationMs = MIN_SEQUENCE_DURATION_MS,
  } = {},
): OfflineSequence<T>[] {
  const sequences: OfflineSequence<T>[] = []
  let frames: T[] = []
  let timestamps: number[] = []
  let active = false
  let handsDownSince: number | null = null

  const emit = () => {
    if (computeSequenceDurationMs(timestamps) >= minDurationMs) {
      sequences.push({frames, frameTimestamps: timestamps})
    }
    active = false
    handsDownSince = null
  }

  for (const sample of samples) {
    if (sample.hasHands) {
      if (!active) {
        frames = []
        timestamps = []
        active = true
      }
      handsDownSince = null
      frames = appendRollingWindow(frames, sample.frame, maxFrames)
      timestamps = appendRollingWindow(timestamps, sample.timestampMs, maxFrames)
      continue
    }

    if (!active) {
      continue
    }
    if (handsDownSince === null) {
      handsDownSince = sample.timestampMs
    }
    if (sample.timestampMs - handsDownSince >= handsDownDelayMs) {
      emit()
    }
  }

  if (active) {
    emit()
  }
  return sequences
}

export function pickLongestSequence<T>(sequences: OfflineSequence<T>[]): OfflineSequence<T> | null {
  let best: OfflineSequence<T> | null = null
  for (const sequence of sequences) {
    if (!best || sequence.frames.length > best.frames.length) {
      best = sequence
    }
  }
  return best
}
