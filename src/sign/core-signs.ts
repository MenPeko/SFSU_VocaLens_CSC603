export const CORE_SIGN_LABELS = [
  "wait",
  "help",
  "yes",
  "no",
  "stop",
  "again",
  "thank_you",
  "please",
] as const

export type CoreSignLabel = (typeof CORE_SIGN_LABELS)[number]

export const CORE_SIGN_SAMPLE_TARGET = 500

export const CORE_SIGN_CANONICAL_TEXT: Record<CoreSignLabel, string> = {
  wait: "Wait",
  help: "Help",
  yes: "Yes",
  no: "No",
  stop: "Stop",
  again: "Again",
  thank_you: "Thank you",
  please: "Please",
}

export function isCoreSignLabel(value: string): value is CoreSignLabel {
  return CORE_SIGN_LABELS.includes(value as CoreSignLabel)
}
