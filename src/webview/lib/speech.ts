export const MIN_SPEECH_RATE = 0.75
export const MAX_SPEECH_RATE = 1.5
export const DEFAULT_SPEECH_RATE = 1
export const SPEECH_RATE_STORAGE_KEY = "vocalens_speech_rate"

const SPEECH_LANGUAGE_CODES: Record<string, string> = {
  English: "en",
  Spanish: "es",
  French: "fr",
  German: "de",
  Italian: "it",
  Portuguese: "pt",
  Chinese: "zh-CN",
  Japanese: "ja",
  Korean: "ko",
  Arabic: "ar",
  Hindi: "hi",
  Russian: "ru",
  Dutch: "nl",
  Turkish: "tr",
  Vietnamese: "vi",
  Thai: "th",
  Polish: "pl",
  Swedish: "sv",
  Indonesian: "id",
  Tagalog: "fil",
}

export function speechLanguageCode(language: string | undefined): string | undefined {
  if (!language) return undefined
  return SPEECH_LANGUAGE_CODES[language.trim()] || undefined
}

export function clampSpeechRate(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_SPEECH_RATE
  return Math.min(MAX_SPEECH_RATE, Math.max(MIN_SPEECH_RATE, value))
}

export function readSpeechRate(): number {
  try {
    const stored = localStorage.getItem(SPEECH_RATE_STORAGE_KEY)
    if (!stored) return DEFAULT_SPEECH_RATE
    return clampSpeechRate(Number(stored))
  } catch {
    return DEFAULT_SPEECH_RATE
  }
}

export function writeSpeechRate(rate: number): number {
  const clamped = clampSpeechRate(rate)
  try {
    localStorage.setItem(SPEECH_RATE_STORAGE_KEY, String(clamped))
  } catch {
    // The slider still applies for this page view when storage is unavailable.
  }
  return clamped
}

export function splitSpeakableSentences(text: string): string[] {
  const trimmed = text.trim()
  if (!trimmed) return []

  const sentences: string[] = []
  let rest = trimmed
  while (rest.length > 0) {
    const match = rest.match(/^[\s\S]*?[.!?](?=\s|$)/)
    if (!match) {
      sentences.push(rest.trim())
      break
    }
    const sentence = match[0].trim()
    if (sentence) sentences.push(sentence)
    rest = rest.slice(match[0].length).trim()
  }

  return sentences
}

let speechQueue: Promise<void> = Promise.resolve()
let speechGeneration = 0

export function cancelSpeech(): void {
  speechGeneration += 1
  speechQueue = Promise.resolve()
  if (typeof window !== "undefined" && window.speechSynthesis) {
    window.speechSynthesis.cancel()
  }
}

export function speakText(text: string, rate = readSpeechRate(), language?: string): Promise<void> {
  const lang = speechLanguageCode(language)
  const chunks = splitSpeakableSentences(text)
  if (chunks.length === 0) return Promise.resolve()

  const generation = speechGeneration
  const spoken = speechQueue.then(() => {
    if (generation !== speechGeneration) return undefined
    return speakChunks(chunks, clampSpeechRate(rate), generation, lang)
  })
  speechQueue = spoken.then(
    () => undefined,
    () => undefined,
  )
  return spoken
}

function speakChunks(chunks: string[], rate: number, generation: number, lang?: string): Promise<void> {
  if (typeof window === "undefined" || !window.speechSynthesis) {
    return Promise.resolve()
  }

  return new Promise((resolve) => {
    let index = 0
    const next = () => {
      if (generation !== speechGeneration || index >= chunks.length) {
        resolve()
        return
      }

      const utterance = new SpeechSynthesisUtterance(chunks[index])
      utterance.rate = rate
      if (lang) utterance.lang = lang
      const advance = () => {
        if (generation !== speechGeneration) {
          resolve()
          return
        }
        index += 1
        next()
      }
      utterance.onend = advance
      utterance.onerror = advance
      window.speechSynthesis.speak(utterance)
    }

    next()
  })
}
