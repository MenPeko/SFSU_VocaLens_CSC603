import {useState, useEffect, useRef} from "react"
import {useTranslationStream, type TranslationEntry, type AppStatus} from "./hooks/useTranslationStream"
import {SignLanguageCapture, type SignTranslationDebugResult} from "./components/SignLanguageCapture"
import type {SignSequencePayload} from "./hooks/useSignLanguage"
import {CORE_SIGN_LABELS, CORE_SIGN_SAMPLE_TARGET, type CoreSignLabel} from "../sign/core-signs"
import type {SignTrainingStats} from "../sign/types"
import {
  DEFAULT_SIGN_CAPTURE_SETTINGS,
  type CaptureSettingsFallbackReason,
  type CaptureSettingsStatus,
  type SignCaptureSettings,
} from "../sign/capture-settings"
import {fetchSignCaptureBootstrap} from "./sign-capture-bootstrap"
import "./index.css"

const LANGUAGES = [
  "English",
  "Spanish",
  "French",
  "German",
  "Italian",
  "Portuguese",
  "Chinese",
  "Japanese",
  "Korean",
  "Arabic",
  "Hindi",
  "Russian",
  "Dutch",
  "Turkish",
  "Vietnamese",
  "Thai",
  "Polish",
  "Swedish",
  "Indonesian",
  "Tagalog",
]

const STATUS_CONFIG: Record<AppStatus, {label: string; color: string; pulse: boolean}> = {
  ready: {label: "READY", color: "#FF6600", pulse: false},
  capturing: {label: "CAPTURING", color: "#000000", pulse: true},
  extracting: {label: "EXTRACTING", color: "#FF6600", pulse: true},
  translating: {label: "TRANSLATING", color: "#FF6600", pulse: true},
  interpreting: {label: "INTERPRETING", color: "#FF6600", pulse: true},
  disconnected: {label: "OFFLINE", color: "#000000", pulse: false},
}

type AppMode = "whiteboard" | "sign"

const APP_MODE_CONFIG: Record<
  AppMode,
  {
    navTitle: string
    navSubtitle: string
    panelTitle: string
    panelDescription: string
    emptyTitle: string
    emptyHint: string
    footerCountLabel: string
    chipLabel: string
  }
> = {
  whiteboard: {
    navTitle: "Written Text",
    navSubtitle: "OCR to Speech",
    panelTitle: "Written Text To Speech",
    panelDescription:
      "Use the glasses button to capture whiteboards, notes, or printed text. OCR and translation results are spoken back on the glasses.",
    emptyTitle: "PRESS GLASSES BUTTON",
    emptyHint: "to capture written text and speak the translation",
    footerCountLabel: "TEXT TRANSLATIONS",
    chipLabel: "GLASSES BUTTON",
  },
  sign: {
    navTitle: "Live Sign",
    navSubtitle: "ASL to Speech",
    panelTitle: "Live Sign Translation To Speech",
    panelDescription:
      "Track hands live from the glasses camera, render the sign skeleton overlay, and convert the captured ASL sequence into spoken translated text.",
    emptyTitle: "START SIGNING",
    emptyHint: "with your hands visible to the glasses camera",
    footerCountLabel: "SIGN TRANSLATIONS",
    chipLabel: "LIVE CAMERA",
  },
}

function getModeButtonStyle(active: boolean) {
  return {
    width: "100%",
    border: "4px solid #000000",
    background: active ? "#FF6600" : "#ffffff",
    color: active ? "#ffffff" : "#000000",
    padding: "0.65rem 0.7rem",
    textAlign: "left" as const,
    boxShadow: active ? "5px 5px 0 #000000" : "4px 4px 0 #000000",
    cursor: "pointer",
    fontFamily: "inherit",
  }
}

function StatusBadge({status}: {status: AppStatus}) {
  const config = STATUS_CONFIG[status]
  return (
    <span
      className={config.pulse ? "cli-pulse" : ""}
      style={{
        color: config.color,
        fontSize: "0.75rem",
        letterSpacing: "0.1em",
        fontWeight: 900,
        display: "flex",
        alignItems: "center",
        gap: "8px",
        textTransform: "uppercase",
      }}
    >
      <span
        style={{
          width: 10,
          height: 10,
          background: config.color,
          display: "inline-block",
          border: "2px solid #000000",
        }}
      />
      {config.label}
    </span>
  )
}

const SWIPE_THRESHOLD = 100

function TranslationCard({entry, onRemove}: {entry: TranslationEntry; onRemove: (id: string) => void}) {
  const originalText = typeof entry.originalText === "string" ? entry.originalText : ""
  const translatedText = typeof entry.translatedText === "string" ? entry.translatedText : ""
  const detectedLanguage = typeof entry.detectedLanguage === "string" ? entry.detectedLanguage : "unknown"
  const targetLanguage = typeof entry.targetLanguage === "string" ? entry.targetLanguage : "English"
  const isSignSource = entry.sourceType === "sign"
  const sourceLabel = entry.sourceLabel || (isSignSource ? "ASL GRID" : "WHITEBOARD OCR")
  const recognizerLabel =
    entry.recognizerType === "classifier"
      ? entry.recognizedLabel
        ? `LOCAL ${entry.recognizedLabel.replaceAll("_", " ").toUpperCase()}`
        : "LOCAL CLASSIFIER"
      : entry.recognizerType === "gemini"
        ? entry.fallbackUsed
          ? "GEMINI FALLBACK"
          : "GEMINI"
        : null
  const isSameLanguage =
    (!isSignSource && detectedLanguage.toLowerCase() === targetLanguage.toLowerCase()) ||
    detectedLanguage === "unknown"
  const shouldShowSignSourceText =
    isSignSource &&
    originalText.trim().length > 0 &&
    originalText.trim() !== translatedText.trim()

  const cardRef = useRef<HTMLDivElement>(null)
  const touchStartX = useRef(0)
  const touchCurrentX = useRef(0)
  const isSwiping = useRef(false)

  const handleTouchStart = (e: React.TouchEvent) => {
    touchStartX.current = e.touches[0].clientX
    touchCurrentX.current = e.touches[0].clientX
    isSwiping.current = false
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    touchCurrentX.current = e.touches[0].clientX
    const deltaX = touchCurrentX.current - touchStartX.current

    if (deltaX < -10) {
      isSwiping.current = true
      if (cardRef.current) {
        const progress = Math.min(Math.abs(deltaX) / SWIPE_THRESHOLD, 1)
        cardRef.current.style.transform = `translateX(${deltaX}px)`
        cardRef.current.style.opacity = `${1 - progress * 0.5}`
      }
    }
  }

  const handleTouchEnd = () => {
    const deltaX = touchCurrentX.current - touchStartX.current

    if (deltaX < -SWIPE_THRESHOLD) {
      if (cardRef.current) {
        cardRef.current.classList.add("swiping-out")
        setTimeout(() => onRemove(entry.id), 250)
      }
    } else if (cardRef.current) {
      cardRef.current.style.transform = ""
      cardRef.current.style.opacity = ""
    }
    isSwiping.current = false
  }

  return (
    <div
      ref={cardRef}
      className="translation-card"
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      style={{
        background: "#ffffff",
        borderLeft: "4px solid #FF6600",
        border: "4px solid #000000",
        borderLeftColor: "#FF6600",
        padding: "0.75rem 1rem",
        marginBottom: "0.75rem",
        boxShadow: "6px 6px 0 #000000",
      }}
    >
      <div style={{display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem"}}>
        <span style={{fontSize: "0.65rem", color: "#FF6600", letterSpacing: "0.12em", fontWeight: 900}}>
          {sourceLabel}
        </span>
        <span style={{fontSize: "0.6rem", color: "#444444", letterSpacing: "0.08em", fontWeight: 700}}>
          {entry.frameCount ? `${entry.frameCount} GRID` : entry.sourceModel || ""}
        </span>
      </div>
      {isSignSource && recognizerLabel && (
        <div style={{marginBottom: "0.45rem", display: "flex", justifyContent: "space-between", gap: "0.5rem"}}>
          <span style={{fontSize: "0.62rem", color: "#444444", letterSpacing: "0.08em", fontWeight: 900}}>
            {recognizerLabel}
          </span>
          <span style={{fontSize: "0.62rem", color: "#444444", letterSpacing: "0.08em", fontWeight: 900}}>
            {typeof entry.confidence === "number" ? `${entry.confidence.toFixed(2)} CONF` : entry.sourceModel || ""}
          </span>
        </div>
      )}

      {!isSameLanguage && !isSignSource && (
        <div style={{marginBottom: "0.5rem"}}>
          <span style={{fontSize: "0.7rem", color: "#444444", letterSpacing: "0.05em", fontWeight: 700}}>
            [{detectedLanguage.toUpperCase()}]
          </span>
          <p style={{fontSize: "0.85rem", color: "#444444", marginTop: 4}}>
            {originalText}
          </p>
        </div>
      )}
      {shouldShowSignSourceText && (
        <div style={{marginBottom: "0.5rem"}}>
          <span style={{fontSize: "0.68rem", color: "#444444", letterSpacing: "0.06em", fontWeight: 700}}>
            [INTERPRETED ENGLISH]
          </span>
          <p style={{fontSize: "0.82rem", color: "#444444", marginTop: 4}}>
            {originalText}
          </p>
        </div>
      )}
      <div>
        {isSignSource ? (
          <span style={{fontSize: "0.7rem", color: "#FF6600", letterSpacing: "0.05em", fontWeight: 700}}>
            [ASL TO {targetLanguage.toUpperCase()}]
          </span>
        ) : !isSameLanguage ? (
          <span style={{fontSize: "0.7rem", color: "#FF6600", letterSpacing: "0.05em", fontWeight: 700}}>
            [{targetLanguage.toUpperCase()}]
          </span>
        ) : null}
        <p style={{fontSize: "0.9rem", color: "#000000", fontWeight: 900, marginTop: 4}}>
          {translatedText}
        </p>
      </div>
      <div style={{display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "0.4rem", borderTop: "2px solid #000000", paddingTop: "0.4rem"}}>
        <span style={{fontSize: "0.6rem", color: "#FF6600", fontWeight: 700}}>
          {entry.spokeOnGlasses ? "SPOKEN ON GLASSES" : ""}
        </span>
        <p style={{fontSize: "0.6rem", color: "#444444", textAlign: "right", margin: 0, fontWeight: 700}}>
          {new Date(entry.timestamp).toLocaleTimeString()}
        </p>
      </div>
    </div>
  )
}

const PRIVACY_KEY = "glosstrans_privacy_accepted"
const APP_MODE_KEY = "glosstrans_active_mode"
const SIGN_TRAINING_MODE_KEY = "glosstrans_sign_training_mode"
const SIGN_TRAINING_LABEL_KEY = "glosstrans_sign_training_label"

function PrivacyGate({onAccept}: {onAccept: () => void}) {
  return (
    <div
      style={{
        maxWidth: 480,
        margin: "0 auto",
        padding: "1.5rem",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        background: "#ffffff",
      }}
    >
      <h1
        style={{
          fontSize: "1.5rem",
          fontWeight: 900,
          color: "#000000",
          letterSpacing: "0.08em",
          margin: "0 0 0.25rem 0",
          textTransform: "uppercase",
        }}
      >
        GLOSSTRANS
      </h1>
      <span style={{fontSize: "0.7rem", color: "#FF6600", letterSpacing: "0.1em", marginBottom: "1.5rem", fontWeight: 900}}>
        PRIVACY POLICY
      </span>

      <div
        style={{
          flex: 1,
          maxHeight: "60vh",
          overflowY: "auto",
          background: "#ffffff",
          border: "4px solid #000000",
          padding: "1rem",
          marginBottom: "1rem",
          fontSize: "0.75rem",
          lineHeight: 1.7,
          color: "#000000",
          boxShadow: "6px 6px 0 #FF6600",
        }}
      >
        <p style={{color: "#FF6600", fontWeight: 900, marginTop: 0}}>DATA WE PROCESS</p>
        <p>When you use GlossTrans, the following data is processed:</p>
        <ul style={{paddingLeft: "1.2rem", margin: "0.5rem 0"}}>
          <li>Photos captured through your smart glasses</li>
          <li>Text extracted from those photos via OCR</li>
          <li>Translations of extracted text</li>
          <li>Your selected target language preference</li>
        </ul>

        <p style={{color: "#FF6600", fontWeight: 900}}>THIRD-PARTY SERVICES</p>
        <p>
          Photos and text are sent to the <strong>Google Gemini API</strong> for
          text extraction and translation. Google's privacy policy applies to
          data processed by their services.
        </p>

        <p style={{color: "#FF6600", fontWeight: 900}}>DATA RETENTION</p>
        <p>
          GlossTrans does <strong>not</strong> store your photos, text, or translations
          permanently. All data is held in temporary memory and cleared when the
          session ends or the app restarts. No data is written to a database.
        </p>

        <p style={{color: "#FF6600", fontWeight: 900}}>DATA SHARING</p>
        <p>
          We do not sell, trade, or rent your personal information. Data is shared
          only with Google Gemini API for the purpose of providing the translation service.
        </p>

        <p style={{color: "#FF6600", fontWeight: 900}}>YOUR RIGHTS (CCPA/CPRA)</p>
        <p>
          As a California resident, you have the right to know what data is collected,
          request deletion, and opt out of data sales (we do not sell data).
        </p>

        <p style={{color: "#444444", fontSize: "0.65rem", marginBottom: 0, fontWeight: 700}}>
          Full policy: support@glosstrans.com | Effective: April 8, 2026
        </p>
      </div>

      <button
        onClick={onAccept}
        style={{
          width: "100%",
          padding: "0.85rem",
          background: "#FF6600",
          color: "#ffffff",
          border: "4px solid #000000",
          fontSize: "0.8rem",
          fontFamily: "inherit",
          fontWeight: 900,
          letterSpacing: "0.12em",
          cursor: "pointer",
          boxShadow: "6px 6px 0 #000000",
          textTransform: "uppercase",
        }}
      >
        I ACCEPT
      </button>
      <p style={{fontSize: "0.6rem", color: "#444444", textAlign: "center", marginTop: "0.75rem", fontWeight: 700}}>
        By accepting, you consent to the collection and use of your data as described above.
      </p>
    </div>
  )
}

export function App() {
  const {translations, status, statusMessage, connected, ocrText, error, cameraStream, removeTranslation} =
    useTranslationStream()
  const [targetLanguage, setTargetLanguage] = useState("English")
  const [showHandednessOverlay, setShowHandednessOverlay] = useState(false)
  const [signDebugSaving, setSignDebugSaving] = useState(false)
  const [signCaptureSettings, setSignCaptureSettings] = useState<SignCaptureSettings>(DEFAULT_SIGN_CAPTURE_SETTINGS)
  const [captureSettingsStatus, setCaptureSettingsStatus] = useState<CaptureSettingsStatus>("loading")
  const [captureSettingsFallbackReason, setCaptureSettingsFallbackReason] = useState<
    CaptureSettingsFallbackReason | undefined
  >(undefined)
  const [privacyGateBootId, setPrivacyGateBootId] = useState<string | null>(null)
  const [privacyGateReady, setPrivacyGateReady] = useState(false)
  const [trainingMode, setTrainingMode] = useState(() => {
    try {
      return localStorage.getItem(SIGN_TRAINING_MODE_KEY) === "true"
    } catch {
      return false
    }
  })
  const [trainingLabel, setTrainingLabel] = useState<CoreSignLabel>(() => {
    try {
      const value = localStorage.getItem(SIGN_TRAINING_LABEL_KEY)
      return CORE_SIGN_LABELS.includes(value as CoreSignLabel) ? (value as CoreSignLabel) : "wait"
    } catch {
      return "wait"
    }
  })
  const [trainingStats, setTrainingStats] = useState<SignTrainingStats | null>(null)
  const [activeMode, setActiveMode] = useState<AppMode>(() => {
    try {
      return localStorage.getItem(APP_MODE_KEY) === "sign" ? "sign" : "whiteboard"
    } catch {
      return "whiteboard"
    }
  })
  const [privacyAccepted, setPrivacyAccepted] = useState(false)

  useEffect(() => {
    fetch("/api/settings/privacy-gate")
      .then((r) => r.json())
      .then((data: {bootId?: string}) => {
        const bootId = data.bootId?.trim()
        setPrivacyGateBootId(bootId || null)

        try {
          setPrivacyAccepted(Boolean(bootId) && localStorage.getItem(PRIVACY_KEY) === bootId)
        } catch {
          setPrivacyAccepted(false)
        }
      })
      .catch(() => {
        setPrivacyAccepted(false)
      })
      .finally(() => {
        setPrivacyGateReady(true)
      })

    fetch("/api/settings/language")
      .then((r) => r.json())
      .then((data: {targetLanguage?: string}) => {
        if (data.targetLanguage) setTargetLanguage(data.targetLanguage)
      })
      .catch(() => {})

    fetch("/api/settings/sign-debug")
      .then((r) => r.json())
      .then((data: {showHandednessOverlay?: boolean}) => {
        if (typeof data.showHandednessOverlay === "boolean") {
          setShowHandednessOverlay(data.showHandednessOverlay)
        }
      })
      .catch(() => {})

    void fetchSignCaptureBootstrap()
      .then((result) => {
        setSignCaptureSettings(result.settings)
        setCaptureSettingsStatus(result.status)
        setCaptureSettingsFallbackReason(result.fallbackReason)
      })
      .catch(() => {
        setSignCaptureSettings(DEFAULT_SIGN_CAPTURE_SETTINGS)
        setCaptureSettingsStatus("fallback")
        setCaptureSettingsFallbackReason("request_error")
      })

    fetch("/api/sign-training-samples/stats")
      .then((r) => r.json())
      .then((data: SignTrainingStats) => {
        setTrainingStats(data)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!privacyAccepted) {
      return
    }

    fetch("/api/settings/sign-camera", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({enabled: activeMode === "sign"}),
    }).catch(() => {})
  }, [activeMode, privacyAccepted])

  const handleLanguageChange = (lang: string) => {
    setTargetLanguage(lang)
    fetch("/api/settings/language", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({targetLanguage: lang}),
    }).catch(() => {})
  }

  const handleAcceptPrivacy = () => {
    try {
      if (privacyGateBootId) {
        localStorage.setItem(PRIVACY_KEY, privacyGateBootId)
      }
    } catch {
      // localStorage may be unavailable
    }
    setPrivacyAccepted(true)
  }

  const handleModeChange = (mode: AppMode) => {
    setActiveMode(mode)
    try {
      localStorage.setItem(APP_MODE_KEY, mode)
    } catch {
      // localStorage may be unavailable
    }
  }

  const handleToggleHandednessOverlay = async (nextValue: boolean) => {
    setSignDebugSaving(true)

    try {
      const response = await fetch("/api/settings/sign-debug", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({showHandednessOverlay: nextValue}),
      })

      const payload = (await response.json().catch(() => null)) as {showHandednessOverlay?: boolean} | null

      if (!response.ok || typeof payload?.showHandednessOverlay !== "boolean") {
        throw new Error("Failed to update sign debug settings.")
      }

      setShowHandednessOverlay(payload.showHandednessOverlay)
    } catch (toggleError) {
      console.error("Failed to update sign debug settings", toggleError)
    } finally {
      setSignDebugSaving(false)
    }
  }

  const refreshTrainingStats = async () => {
    const response = await fetch("/api/sign-training-samples/stats")
    if (!response.ok) {
      throw new Error("Failed to load sign training stats")
    }
    const payload = (await response.json()) as SignTrainingStats
    setTrainingStats(payload)
    return payload
  }

  const handleTrainingModeChange = (nextValue: boolean) => {
    setTrainingMode(nextValue)
    try {
      localStorage.setItem(SIGN_TRAINING_MODE_KEY, nextValue ? "true" : "false")
    } catch {
      // localStorage may be unavailable
    }
  }

  const handleTrainingLabelChange = (nextLabel: CoreSignLabel) => {
    setTrainingLabel(nextLabel)
    try {
      localStorage.setItem(SIGN_TRAINING_LABEL_KEY, nextLabel)
    } catch {
      // localStorage may be unavailable
    }
  }

  const handleSignSequenceReady = async (payload: SignSequencePayload): Promise<SignTranslationDebugResult> => {
    const requestStartedAt = performance.now()
    const captureDebugFields = {
      requestedQuality: payload.requestedQuality,
      actualQuality: payload.actualQuality,
      qualityFallback: payload.qualityFallback,
      actualInputWidth: payload.actualInputWidth,
      actualInputHeight: payload.actualInputHeight,
      rgbMimeType: payload.mimeType,
      skeletonMimeType: payload.skeletonMimeType,
      skeletonRenderMode: payload.skeletonRenderMode,
      captureSettingsStatus: payload.captureSettingsStatus,
      captureSettingsFallbackReason: payload.captureSettingsFallbackReason,
      droppedFrameCount: payload.droppedFrameCount,
    }

    try {
      if (trainingMode) {
        const response = await fetch("/api/sign-training-samples", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            label: trainingLabel,
            base64Grid: payload.base64Grid,
            base64SkeletonGrid: payload.base64SkeletonGrid,
            landmarkSequence: payload.landmarkSequence,
            frameTimestamps: payload.frameTimestamps,
            sequenceDurationMs: payload.sequenceDurationMs,
            sequenceFrameCount: payload.sequenceFrameCount,
            source: payload.source,
            mimeType: payload.mimeType,
            skeletonMimeType: payload.skeletonMimeType,
            frameCount: payload.frameCount,
            gridColumns: payload.gridColumns,
            gridRows: payload.gridRows,
            gestureHints: payload.gestureHints,
            extractorVersion: payload.extractorVersion,
          }),
        })

        const responseText = await response.text()
        const responsePayload = (() => {
          if (!responseText.trim()) {
            return null
          }

          try {
            return JSON.parse(responseText) as {
              success?: boolean
              sampleId?: string
              label?: string
              stats?: SignTrainingStats
              error?: string
            }
          } catch {
            return null
          }
        })()

        if (!response.ok || !responsePayload?.success) {
          const errorMessage =
            responsePayload?.error ||
            responseText.trim() ||
            `Sample capture failed (HTTP ${response.status})`

          return {
            operation: "capture",
            success: false,
            statusCode: response.status,
            requestDurationMs: performance.now() - requestStartedAt,
            frameCount: payload.frameCount,
            landmarkFrameCount: payload.sequenceFrameCount,
            ...captureDebugFields,
            responseText: responseText.trim() || undefined,
            error: errorMessage,
          }
        }

        const nextStats = responsePayload.stats || (await refreshTrainingStats())
        setTrainingStats(nextStats)

        return {
          operation: "capture",
          success: true,
          statusCode: response.status,
          requestDurationMs: performance.now() - requestStartedAt,
          frameCount: payload.frameCount,
          landmarkFrameCount: payload.sequenceFrameCount,
          message: `Saved ${trainingLabel.replaceAll("_", " ")} sample.`,
          recognizedLabel: responsePayload.label || trainingLabel,
          sampleCountForLabel: nextStats.labels[trainingLabel],
          targetSampleCount: nextStats.sampleTarget,
          ...captureDebugFields,
          responseText: responseText.trim() || undefined,
        }
      }

      const response = await fetch("/api/sign-translate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          base64Grid: payload.base64Grid,
          base64SkeletonGrid: payload.base64SkeletonGrid,
          landmarkSequence: payload.landmarkSequence,
          frameTimestamps: payload.frameTimestamps,
          sequenceDurationMs: payload.sequenceDurationMs,
          sequenceFrameCount: payload.sequenceFrameCount,
          source: payload.source,
          mimeType: payload.mimeType,
          skeletonMimeType: payload.skeletonMimeType,
          frameCount: payload.frameCount,
          gridColumns: payload.gridColumns,
          gridRows: payload.gridRows,
          gestureHints: payload.gestureHints,
          extractorVersion: payload.extractorVersion,
          speakOnGlasses: true,
        }),
      })

      const responseText = await response.text()
      const responsePayload = (() => {
        if (!responseText.trim()) {
          return null
        }

        try {
          return JSON.parse(responseText) as {
            englishText?: string
            translatedText?: string
            targetLanguage?: string
            spokeOnGlasses?: boolean
            model?: string
            message?: string
            rawGeminiOutput?: string
            recognizerType?: "classifier" | "gemini"
            recognizedLabel?: string
            confidence?: number
            fallbackUsed?: boolean
            classifierModelVersion?: string
            classifierTrainingRunId?: string
            classifierStatus?: string
            classifierAlternatives?: SignTranslationDebugResult["classifierAlternatives"]
            classifierError?: string
            error?: string
          }
        } catch {
          return null
        }
      })() as
        | {
            englishText?: string
            translatedText?: string
            targetLanguage?: string
            spokeOnGlasses?: boolean
            model?: string
            message?: string
            rawGeminiOutput?: string
            recognizerType?: "classifier" | "gemini"
            recognizedLabel?: string
            confidence?: number
            fallbackUsed?: boolean
            classifierModelVersion?: string
            classifierTrainingRunId?: string
            classifierStatus?: string
            classifierAlternatives?: SignTranslationDebugResult["classifierAlternatives"]
            classifierError?: string
            error?: string
          }
        | null

      if (!response.ok) {
        const detailedError =
          responsePayload?.error ||
          responsePayload?.message ||
          responseText.trim() ||
          `Sign translation failed (HTTP ${response.status})`

        return {
          operation: "translate",
          success: false,
          statusCode: response.status,
          requestDurationMs: performance.now() - requestStartedAt,
          frameCount: payload.frameCount,
          landmarkFrameCount: payload.sequenceFrameCount,
          ...captureDebugFields,
          model: responsePayload?.model,
          recognizerType: responsePayload?.recognizerType,
          recognizedLabel: responsePayload?.recognizedLabel,
          confidence: responsePayload?.confidence,
          fallbackUsed: responsePayload?.fallbackUsed,
          classifierModelVersion: responsePayload?.classifierModelVersion,
          classifierTrainingRunId: responsePayload?.classifierTrainingRunId,
          classifierStatus: responsePayload?.classifierStatus as SignTranslationDebugResult["classifierStatus"],
          classifierAlternatives: responsePayload?.classifierAlternatives,
          classifierError: responsePayload?.classifierError,
          rawGeminiOutput: responsePayload?.rawGeminiOutput,
          responseText: responseText.trim() || undefined,
          error: detailedError,
        }
      }

      return {
        operation: "translate",
        success: true,
        statusCode: response.status,
        requestDurationMs: performance.now() - requestStartedAt,
        frameCount: payload.frameCount,
        landmarkFrameCount: payload.sequenceFrameCount,
        ...captureDebugFields,
        englishText: responsePayload?.englishText,
        translatedText: responsePayload?.translatedText,
        targetLanguage: responsePayload?.targetLanguage,
        spokeOnGlasses: responsePayload?.spokeOnGlasses,
        model: responsePayload?.model,
        recognizerType: responsePayload?.recognizerType,
        recognizedLabel: responsePayload?.recognizedLabel,
        confidence: responsePayload?.confidence,
        fallbackUsed: responsePayload?.fallbackUsed,
        classifierModelVersion: responsePayload?.classifierModelVersion,
        classifierTrainingRunId: responsePayload?.classifierTrainingRunId,
        classifierStatus: responsePayload?.classifierStatus as SignTranslationDebugResult["classifierStatus"],
        classifierAlternatives: responsePayload?.classifierAlternatives,
        classifierError: responsePayload?.classifierError,
        message: responsePayload?.message,
        rawGeminiOutput: responsePayload?.rawGeminiOutput,
        responseText: responseText.trim() || undefined,
      }
    } catch (signError) {
      console.error("Sign translation request failed", signError)

      return {
        operation: trainingMode ? "capture" : "translate",
        success: false,
        requestDurationMs: performance.now() - requestStartedAt,
        frameCount: payload.frameCount,
        landmarkFrameCount: payload.sequenceFrameCount,
        ...captureDebugFields,
        error: signError instanceof Error ? signError.message : "Sign translation request failed",
      }
    }
  }

  if (!privacyGateReady) {
    return null
  }

  if (!privacyAccepted) {
    return <PrivacyGate onAccept={handleAcceptPrivacy} />
  }

  const activeModeConfig = APP_MODE_CONFIG[activeMode]
  const latestSignTranslation = translations.find((entry) => entry.sourceType === "sign") || null
  const visibleTranslations = translations.filter((entry) =>
    activeMode === "sign" ? entry.sourceType === "sign" : entry.sourceType !== "sign",
  )
  const hiddenTranslationCount = translations.length - visibleTranslations.length

  return (
    <div
      style={{
        width: "100%",
        maxWidth: activeMode === "sign" ? 960 : 480,
        margin: "0 auto",
        padding: "1rem",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "#ffffff",
      }}
    >
      {/* Header */}
      <div style={{marginBottom: "1rem"}}>
        <div style={{textAlign: "center", marginBottom: "0.75rem"}}>
          <h1
            style={{
              fontSize: "2.4rem",
              fontWeight: 900,
              color: "#000000",
              letterSpacing: "0.06em",
              margin: "0 0 0.15rem 0",
              textTransform: "uppercase",
              fontFamily: "'Impact', 'Arial Black', 'Helvetica Neue', sans-serif",
              lineHeight: 1,
            }}
          >
            GlossTrans
          </h1>
          <span style={{fontSize: "0.65rem", color: "#FF6600", letterSpacing: "0.08em", fontWeight: 900}}>
            TEXT TO SPEECH + ASL TO SPEECH
          </span>
        </div>
        <div style={{display: "flex", justifyContent: "center", marginBottom: "0.75rem"}}>
          <StatusBadge status={status} />
        </div>

        <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginBottom: "0.75rem"}}>
          {(["whiteboard", "sign"] as const).map((mode) => {
            const config = APP_MODE_CONFIG[mode]
            const active = activeMode === mode

            return (
              <button key={mode} type="button" onClick={() => handleModeChange(mode)} style={getModeButtonStyle(active)}>
                <div style={{fontSize: "0.76rem", fontWeight: 900, letterSpacing: "0.08em", textTransform: "uppercase"}}>
                  {config.navTitle}
                </div>
                <div
                  style={{
                    fontSize: "0.62rem",
                    fontWeight: 700,
                    letterSpacing: "0.06em",
                    marginTop: "0.2rem",
                    opacity: active ? 0.95 : 0.75,
                  }}
                >
                  {config.navSubtitle}
                </div>
              </button>
            )
          })}
        </div>

        {/* Language selector */}
        <div style={{marginBottom: "0.25rem"}}>
          <span style={{fontSize: "0.7rem", color: "#444444", letterSpacing: "0.05em", fontWeight: 700, display: "block", marginBottom: "0.35rem"}}>
            Language for translation & speech of choice:
          </span>
        </div>
        <div style={{display: "flex", alignItems: "center", gap: "0.5rem"}}>
          <select
            value={targetLanguage}
            onChange={(e) => handleLanguageChange(e.target.value)}
            style={{
              flex: 1,
              background: "#ffffff",
              color: "#000000",
              border: "4px solid #000000",
              padding: "0.5rem 0.6rem",
              fontSize: "0.8rem",
              fontFamily: "inherit",
              letterSpacing: "0.05em",
              fontWeight: 700,
              outline: "none",
              cursor: "pointer",
              boxShadow: "4px 4px 0 #FF6600",
            }}
          >
            {LANGUAGES.map((lang) => (
              <option key={lang} value={lang} style={{background: "#ffffff", color: "#000000"}}>
                {lang}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Connection warning */}
      {!connected && (
        <div
          style={{
            background: "#FF6600",
            border: "4px solid #000000",
            padding: "0.6rem 0.75rem",
            marginBottom: "0.75rem",
            boxShadow: "4px 4px 0 #000000",
          }}
        >
          <p style={{fontSize: "0.8rem", color: "#ffffff", fontWeight: 900, margin: 0}}>
            CONNECTING TO SERVER...
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          style={{
            background: "#ffffff",
            border: "4px solid #000000",
            borderLeftColor: "#FF6600",
            borderLeftWidth: "8px",
            padding: "0.6rem 0.75rem",
            marginBottom: "0.75rem",
            boxShadow: "4px 4px 0 #FF6600",
          }}
        >
          <p style={{fontSize: "0.8rem", color: "#000000", margin: 0, fontWeight: 900}}>{error}</p>
        </div>
      )}

      <div
        style={{
          background: "#ffffff",
          border: "4px solid #000000",
          padding: "0.75rem",
          marginBottom: "0.75rem",
          boxShadow: "6px 6px 0 #000000",
        }}
      >
        <div style={{display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.75rem"}}>
          <div>
            <p
              style={{
                fontSize: "0.68rem",
                color: "#FF6600",
                letterSpacing: "0.14em",
                fontWeight: 900,
                textTransform: "uppercase",
                margin: 0,
              }}
            >
              {activeModeConfig.panelTitle}
            </p>
            <p style={{fontSize: "0.78rem", color: "#000000", margin: "0.35rem 0 0 0", fontWeight: 700, lineHeight: 1.5}}>
              {activeModeConfig.panelDescription}
            </p>
          </div>
          <span
            style={{
              border: "3px solid #000000",
              background: "#ffffff",
              color: "#000000",
              padding: "0.3rem 0.45rem",
              fontSize: "0.62rem",
              fontWeight: 900,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              boxShadow: "3px 3px 0 #FF6600",
              whiteSpace: "nowrap",
            }}
          >
            {activeModeConfig.chipLabel}
          </span>
        </div>
        {hiddenTranslationCount > 0 && (
          <p style={{fontSize: "0.68rem", color: "#444444", margin: "0.65rem 0 0 0", fontWeight: 700}}>
            {hiddenTranslationCount} translation{hiddenTranslationCount === 1 ? "" : "s"} from the other mode hidden.
          </p>
        )}
      </div>

      {/* Live status / OCR preview */}
      {activeMode === "whiteboard" && status !== "ready" && status !== "disconnected" && (
        <div
          className="cli-glow"
          style={{
            background: "#ffffff",
            border: "4px solid #FF6600",
            padding: "0.7rem 0.75rem",
            marginBottom: "0.75rem",
            boxShadow: "6px 6px 0 #000000",
          }}
        >
          <p
            className="cli-pulse"
            style={{fontSize: "0.75rem", color: "#FF6600", letterSpacing: "0.05em", margin: 0, fontWeight: 900}}
          >
            {statusMessage}
          </p>
          {ocrText && (
            <div style={{marginTop: "0.5rem", borderTop: "2px solid #000000", paddingTop: "0.5rem"}}>
              <span style={{fontSize: "0.65rem", color: "#444444", letterSpacing: "0.05em", fontWeight: 900}}>
                EXTRACTED:
              </span>
              <p style={{fontSize: "0.85rem", color: "#000000", marginTop: 4, fontWeight: 700}}>{ocrText}</p>
            </div>
          )}
        </div>
      )}

      {activeMode === "sign" && (
        <div
          style={{
            background: "#ffffff",
            border: "4px solid #000000",
            padding: "0.75rem",
            marginBottom: "0.75rem",
            boxShadow: "6px 6px 0 #FF6600",
          }}
        >
          <div style={{display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap"}}>
            <div>
              <p style={{fontSize: "0.68rem", color: "#FF6600", letterSpacing: "0.14em", fontWeight: 900, textTransform: "uppercase", margin: 0}}>
                Dataset Capture
              </p>
              <p style={{fontSize: "0.75rem", color: "#000000", margin: "0.35rem 0 0 0", fontWeight: 700, lineHeight: 1.5}}>
                Toggle labeled capture mode to save app-specific landmark sequences for the local classifier.
              </p>
            </div>
            <button
              type="button"
              onClick={() => handleTrainingModeChange(!trainingMode)}
              style={{
                border: "4px solid #000000",
                background: trainingMode ? "#FF6600" : "#ffffff",
                color: trainingMode ? "#ffffff" : "#000000",
                padding: "0.45rem 0.65rem",
                fontFamily: "inherit",
                fontWeight: 900,
                fontSize: "0.7rem",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                boxShadow: trainingMode ? "4px 4px 0 #000000" : "3px 3px 0 #000000",
                cursor: "pointer",
              }}
            >
              {trainingMode ? "Label Capture On" : "Translation Mode"}
            </button>
          </div>

          <div style={{display: "grid", gridTemplateColumns: "1fr auto", gap: "0.75rem", marginTop: "0.75rem", alignItems: "end"}}>
            <div>
              <span style={{fontSize: "0.65rem", color: "#444444", letterSpacing: "0.06em", fontWeight: 900, display: "block", marginBottom: "0.35rem"}}>
                CAPTURE LABEL
              </span>
              <select
                value={trainingLabel}
                onChange={(event) => handleTrainingLabelChange(event.target.value as CoreSignLabel)}
                disabled={!trainingMode}
                style={{
                  width: "100%",
                  background: "#ffffff",
                  color: "#000000",
                  border: "4px solid #000000",
                  padding: "0.5rem 0.6rem",
                  fontSize: "0.8rem",
                  fontFamily: "inherit",
                  letterSpacing: "0.05em",
                  fontWeight: 700,
                  outline: "none",
                  cursor: trainingMode ? "pointer" : "not-allowed",
                  boxShadow: "4px 4px 0 #000000",
                  opacity: trainingMode ? 1 : 0.7,
                }}
              >
                {CORE_SIGN_LABELS.map((label) => (
                  <option key={label} value={label}>
                    {label.replaceAll("_", " ").toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
            <div
              style={{
                border: "4px solid #000000",
                background: "#ffffff",
                padding: "0.5rem 0.75rem",
                minWidth: 170,
                boxShadow: "4px 4px 0 #FF6600",
              }}
            >
              <div style={{fontSize: "0.65rem", color: "#444444", letterSpacing: "0.06em", fontWeight: 900}}>PROGRESS</div>
              <div style={{marginTop: "0.25rem", fontSize: "0.88rem", color: "#000000", fontWeight: 900}}>
                {(trainingStats?.labels[trainingLabel] || 0)}/{trainingStats?.sampleTarget || CORE_SIGN_SAMPLE_TARGET}
              </div>
            </div>
          </div>

          {trainingStats && (
            <div style={{display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))", gap: "0.45rem", marginTop: "0.75rem"}}>
              {CORE_SIGN_LABELS.map((label) => (
                <div
                  key={label}
                  style={{
                    border: "3px solid #000000",
                    background: label === trainingLabel ? "#FF6600" : "#ffffff",
                    color: label === trainingLabel ? "#ffffff" : "#000000",
                    padding: "0.35rem 0.5rem",
                    boxShadow: label === trainingLabel ? "3px 3px 0 #000000" : "2px 2px 0 #000000",
                  }}
                >
                  <div style={{fontSize: "0.58rem", fontWeight: 900, letterSpacing: "0.08em", textTransform: "uppercase"}}>
                    {label.replaceAll("_", " ")}
                  </div>
                  <div style={{marginTop: "0.2rem", fontSize: "0.82rem", fontWeight: 900}}>
                    {trainingStats.labels[label]}/{trainingStats.sampleTarget}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeMode === "sign" && (
        <div style={{marginBottom: "0.75rem"}}>
          <SignLanguageCapture
            enabled
            cameraStream={cameraStream}
            latestSignTranslation={latestSignTranslation}
            showHandednessOverlay={showHandednessOverlay}
            captureSettings={signCaptureSettings}
            captureSettingsStatus={captureSettingsStatus}
            captureSettingsFallbackReason={captureSettingsFallbackReason}
            debugToggleBusy={signDebugSaving}
            onToggleHandednessOverlay={handleToggleHandednessOverlay}
            onSequenceReady={handleSignSequenceReady}
          />
        </div>
      )}

      {/* Translation history */}
      <div style={{flex: 1, overflowY: "auto"}}>
        {visibleTranslations.length === 0 && status === "ready" && connected && (
          <div style={{textAlign: "center", marginTop: "3rem"}}>
            <p style={{fontSize: "0.85rem", color: "#000000", letterSpacing: "0.05em", fontWeight: 900}}>
              {activeModeConfig.emptyTitle}
            </p>
            <p style={{fontSize: "0.75rem", color: "#444444", marginTop: "0.25rem", fontWeight: 700}}>
              {activeModeConfig.emptyHint}
            </p>
          </div>
        )}
        {visibleTranslations.map((entry) => (
          <TranslationCard key={entry.id} entry={entry} onRemove={removeTranslation} />
        ))}
      </div>

      {/* Footer */}
      <div
        style={{
          borderTop: "4px solid #000000",
          paddingTop: "0.6rem",
          marginTop: "0.6rem",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span style={{fontSize: "0.65rem", color: "#FF6600", letterSpacing: "0.08em", fontWeight: 900}}>
          MENTRA OS
        </span>
        <span style={{fontSize: "0.65rem", color: "#000000", fontWeight: 900}}>
          {visibleTranslations.length > 0 ? `${visibleTranslations.length} ${activeModeConfig.footerCountLabel}` : ""}
        </span>
      </div>
    </div>
  )
}

export default App
