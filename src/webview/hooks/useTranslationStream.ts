import {useState, useEffect, useRef, useCallback} from "react"

export interface TranslationEntry {
  id: string
  originalText: string
  translatedText: string
  detectedLanguage: string
  targetLanguage: string
  timestamp: number
  sourceType?: "text" | "sign" | "vocalens"
  sourceLabel?: string
  sourceModel?: string
  frameCount?: number
  sequenceFrameCount?: number
  recognizerType?: "classifier" | "gemini"
  recognizedLabel?: string
  confidence?: number
  fallbackUsed?: boolean
  classifierModelVersion?: string
  classifierTrainingRunId?: string
}

export type AppStatus = "ready" | "capturing" | "extracting" | "translating" | "interpreting" | "disconnected"

const HEARTBEAT_TIMEOUT_MS = 45000
const HEARTBEAT_CHECK_INTERVAL_MS = 10000
const INITIAL_RETRY_DELAY_MS = 1000
const MAX_RETRY_DELAY_MS = 30000
const MAX_RECONNECT_ATTEMPTS = 20

export function useTranslationStream() {
  const [translations, setTranslations] = useState<TranslationEntry[]>([])
  const [status, setStatus] = useState<AppStatus>("ready")
  const [statusMessage, setStatusMessage] = useState("Connecting...")
  const [connected, setConnected] = useState(false)
  const [ocrText, setOcrText] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const eventSourceRef = useRef<EventSource | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const heartbeatCheckRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const lastActivityRef = useRef<number>(Date.now())
  const mountedRef = useRef(true)
  const attemptRef = useRef(0)

  const cleanup = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (heartbeatCheckRef.current) {
      clearInterval(heartbeatCheckRef.current)
      heartbeatCheckRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    cleanup()

    const es = new EventSource("/api/stream")
    eventSourceRef.current = es

    es.onopen = () => {
      if (!mountedRef.current) return
      setConnected(true)
      setError(null)
      attemptRef.current = 0
      lastActivityRef.current = Date.now()

      heartbeatCheckRef.current = setInterval(() => {
        if (Date.now() - lastActivityRef.current > HEARTBEAT_TIMEOUT_MS) {
          es.close()
          scheduleReconnect()
        }
      }, HEARTBEAT_CHECK_INTERVAL_MS)
    }

    es.onmessage = (event) => {
      if (!mountedRef.current) return
      lastActivityRef.current = Date.now()

      try {
        const data = JSON.parse(event.data)

        switch (data.type) {
          case "status":
            setStatus(data.status)
            setStatusMessage(data.message || "")
            if (data.status === "ready") setOcrText(null)
            break

          case "ocr_result":
            setOcrText(data.text)
            break

          case "translation_result":
            setTranslations((prev) => [
              {
                id: `tr_${data.timestamp}_${Math.random().toString(36).slice(2, 7)}`,
                originalText: data.originalText,
                translatedText: data.translatedText,
                detectedLanguage: data.detectedLanguage,
                targetLanguage: data.targetLanguage,
                timestamp: data.timestamp,
                sourceType: data.sourceType || "text",
                sourceLabel: data.sourceLabel,
                sourceModel: data.sourceModel,
                frameCount: data.frameCount,
                sequenceFrameCount: data.sequenceFrameCount,
                recognizerType: data.recognizerType,
                recognizedLabel: data.recognizedLabel,
                confidence: data.confidence,
                fallbackUsed: data.fallbackUsed,
                classifierModelVersion: data.classifierModelVersion,
                classifierTrainingRunId: data.classifierTrainingRunId,
              },
              ...prev,
            ].slice(0, 50))
            break

          case "error":
            setError(data.message)
            break

          case "heartbeat":
          case "connected":
            break
        }
      } catch {
        // Ignore malformed messages
      }
    }

    es.onerror = () => {
      if (!mountedRef.current) return
      setConnected(false)
      es.close()
      scheduleReconnect()
    }
  }, [cleanup])

  const scheduleReconnect = useCallback(() => {
    if (!mountedRef.current) return
    if (attemptRef.current >= MAX_RECONNECT_ATTEMPTS) {
      setError("Connection lost. Refresh the page to reconnect.")
      return
    }

    const delay = Math.min(INITIAL_RETRY_DELAY_MS * Math.pow(2, attemptRef.current), MAX_RETRY_DELAY_MS)
    attemptRef.current++

    reconnectTimeoutRef.current = setTimeout(() => {
      if (mountedRef.current) connect()
    }, delay)
  }, [connect])

  useEffect(() => {
    mountedRef.current = true

    connect()
    return () => {
      mountedRef.current = false
      cleanup()
    }
  }, [connect, cleanup])

  const removeTranslation = useCallback((id: string) => {
    setTranslations((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const clearTranslations = useCallback(() => {
    setTranslations([])
  }, [])

  return {
    translations,
    status,
    statusMessage,
    connected,
    ocrText,
    error,
    removeTranslation,
    clearTranslations,
  }
}
