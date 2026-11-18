import {streamRoute} from "./sse-stream"
import {translateText} from "../services/gemini-translate"
import {sseManager} from "./sse-stream"
import type {SignGestureHint} from "../services/gemini-sign-interpreter"
import {resolveSignInterpretation} from "../services/sign-translation-resolver"
import {
  getSignTrainingStats,
  saveSignTrainingSample,
  SignTrainingSampleValidationError,
} from "../services/sign-training-store"
import {isCoreSignLabel} from "../sign/core-signs"
import type {SignLandmarkFrame, SignCaptureSource} from "../sign/types"
import {normalizeSignCaptureSettings} from "../sign/capture-settings"
import {vocalensRoutes} from "./vocalens-routes"

// In-memory language preference (per-app, not per-user for simplicity)
let targetLanguage = "English"
let showHandednessOverlay = process.env.SIGN_DEBUG_OVERLAY === "true"
const privacyGateBootId = `server-boot-${Date.now()}`

export function getTargetLanguage(): string {
  return targetLanguage
}

export function getSignDebugSettings(): {showHandednessOverlay: boolean} {
  return {showHandednessOverlay}
}

export function getPrivacyGateSettings(): {bootId: string} {
  return {bootId: privacyGateBootId}
}

export function getSignCaptureSettings() {
  return normalizeSignCaptureSettings({
    holisticModelComplexity: process.env.SIGN_CAPTURE_HOLISTIC_MODEL_COMPLEXITY,
  })
}

interface SignTranslateRequestBody {
  base64Grid?: string
  base64SkeletonGrid?: string
  mimeType?: string
  skeletonMimeType?: string
  frameCount?: number
  landmarkSequence?: SignLandmarkFrame[]
  frameTimestamps?: number[]
  sequenceDurationMs?: number
  sequenceFrameCount?: number
  source?: SignCaptureSource
  gridColumns?: number
  gridRows?: number
  gestureHints?: SignGestureHint[]
  extractorVersion?: string
}

function requireExtractorVersion(value: unknown): string | Response {
  if (typeof value !== "string" || !value.trim()) {
    return Response.json(
      {
        error:
          "extractorVersion is required — the capture client must stamp MEDIAPIPE_EXTRACTOR_VERSION on every request.",
      },
      {status: 400},
    )
  }
  return value.trim()
}

interface SignTrainingSampleRequestBody extends SignTranslateRequestBody {
  label?: string
}

export const routes = {
  // SSE stream for live translation updates
  ...streamRoute,
  ...vocalensRoutes,

  "/api/settings/language": {
    async GET() {
      return Response.json({targetLanguage})
    },
    async POST(req: Request) {
      const body = (await req.json()) as {targetLanguage?: string}
      if (body.targetLanguage) {
        targetLanguage = body.targetLanguage
      }
      return Response.json({targetLanguage})
    },
  },

  "/api/settings/sign-debug": {
    async GET() {
      return Response.json(getSignDebugSettings())
    },
    async POST(req: Request) {
      const body = (await req.json().catch(() => ({}))) as {showHandednessOverlay?: boolean}
      if (typeof body.showHandednessOverlay === "boolean") {
        showHandednessOverlay = body.showHandednessOverlay
      }
      return Response.json(getSignDebugSettings())
    },
  },

  "/api/settings/sign-capture": {
    async GET() {
      return Response.json(getSignCaptureSettings())
    },
  },

  "/api/settings/privacy-gate": {
    async GET() {
      return Response.json(getPrivacyGateSettings())
    },
  },

  "/api/captions/status": {
    async GET() {
      return Response.json({active: true, captionsEnabled: true})
    },
  },

  "/api/sign-translate": {
    async POST(req: Request) {
      let body: SignTranslateRequestBody

      try {
        body = (await req.json()) as SignTranslateRequestBody
      } catch {
        return Response.json({error: "Invalid JSON body"}, {status: 400})
      }

      const base64Grid = body.base64Grid?.trim()
      if (!base64Grid) {
        return Response.json({error: "base64Grid is required"}, {status: 400})
      }

      const extractorVersionOrError = requireExtractorVersion(body.extractorVersion)
      if (extractorVersionOrError instanceof Response) return extractorVersionOrError
      const extractorVersion = extractorVersionOrError

      const targetLang = getTargetLanguage()

      try {
        sseManager.broadcast({
          type: "status",
          status: "interpreting",
          message: "Classifying sign sequence...",
        })

        const interpretation = await resolveSignInterpretation({
          base64Grid,
          base64SkeletonGrid: body.base64SkeletonGrid,
          mimeType: body.mimeType || "image/jpeg",
          skeletonMimeType: body.skeletonMimeType || body.mimeType || "image/jpeg",
          landmarkSequence: body.landmarkSequence || [],
          frameTimestamps: body.frameTimestamps || [],
          sequenceDurationMs: body.sequenceDurationMs || 0,
          sequenceFrameCount: body.sequenceFrameCount || body.landmarkSequence?.length || 0,
          source: body.source || "device-camera",
          gridColumns: body.gridColumns || 4,
          gridRows: body.gridRows || 4,
          gestureHints: body.gestureHints || [],
          extractorVersion,
        })

        const englishText = interpretation.englishText.trim()
        if (interpretation.status === "no_sign" || !englishText) {
          const noSignMessage = "I could not interpret the sign sequence. Please try again."
          sseManager.broadcast({type: "status", status: "ready", message: noSignMessage})

          return Response.json({
            success: true,
            englishText: "",
            translatedText: "",
            targetLanguage: targetLang,
            spokeOnGlasses: false,
            message: noSignMessage,
            model: interpretation.model,
            recognizerType: interpretation.recognizerType,
            recognizedLabel: interpretation.recognizedLabel,
            confidence: interpretation.confidence,
            fallbackUsed: interpretation.fallbackUsed,
            classifierModelVersion: interpretation.classifierModelVersion,
            classifierTrainingRunId: interpretation.classifierTrainingRunId,
            classifierStatus: interpretation.classifierStatus,
            classifierAlternatives: interpretation.classifierAlternatives,
            classifierError: interpretation.classifierError,
            rawGeminiOutput: interpretation.rawGeminiOutput,
          })
        }

        const translation =
          targetLang.toLowerCase() === "english"
            ? {
                originalText: englishText,
                translatedText: englishText,
                detectedLanguage: "English",
                targetLanguage: targetLang,
              }
            : await translateText(englishText, targetLang)

        const translatedText = (translation.translatedText || englishText).trim()

        sseManager.broadcast({
          type: "translation_result",
          originalText: englishText,
          translatedText,
          detectedLanguage: "ASL",
          targetLanguage: targetLang,
          timestamp: Date.now(),
          sourceType: "sign",
          sourceLabel: interpretation.recognizerType === "classifier" ? "ASL Core Sign" : "ASL Grid",
          sourceModel: interpretation.model,
          frameCount: body.frameCount || 0,
          sequenceFrameCount: body.sequenceFrameCount || body.landmarkSequence?.length || 0,
          spokeOnGlasses: false,
          recognizerType: interpretation.recognizerType,
          recognizedLabel: interpretation.recognizedLabel,
          confidence: interpretation.confidence,
          fallbackUsed: interpretation.fallbackUsed,
          classifierModelVersion: interpretation.classifierModelVersion,
          classifierTrainingRunId: interpretation.classifierTrainingRunId,
        })

        sseManager.broadcast({
          type: "status",
          status: "ready",
          message: "Ready. Sign sequence interpreted.",
        })

        return Response.json({
          success: true,
          englishText,
          translatedText,
          targetLanguage: targetLang,
          spokeOnGlasses: false,
          model: interpretation.model,
          recognizerType: interpretation.recognizerType,
          recognizedLabel: interpretation.recognizedLabel,
          confidence: interpretation.confidence,
          fallbackUsed: interpretation.fallbackUsed,
          classifierModelVersion: interpretation.classifierModelVersion,
          classifierTrainingRunId: interpretation.classifierTrainingRunId,
          classifierStatus: interpretation.classifierStatus,
          classifierAlternatives: interpretation.classifierAlternatives,
          classifierError: interpretation.classifierError,
          rawGeminiOutput: interpretation.rawGeminiOutput,
        })
      } catch (error) {
        const message = error instanceof Error ? error.message : "Sign translation failed"

        sseManager.broadcast({type: "error", message})
        sseManager.broadcast({
          type: "status",
          status: "ready",
          message: "Ready. Sign translation failed.",
        })

        return Response.json({error: message}, {status: 500})
      }
    },
  },

  "/api/sign-training-samples/stats": {
    async GET() {
      return Response.json(await getSignTrainingStats())
    },
  },

  "/api/sign-training-samples": {
    async POST(req: Request) {
      let body: SignTrainingSampleRequestBody

      try {
        body = (await req.json()) as SignTrainingSampleRequestBody
      } catch {
        return Response.json({error: "Invalid JSON body"}, {status: 400})
      }

      if (!body.label || !isCoreSignLabel(body.label)) {
        return Response.json({error: "label must be one of the supported core signs"}, {status: 400})
      }

      if (!body.landmarkSequence?.length) {
        return Response.json({error: "landmarkSequence is required"}, {status: 400})
      }

      const extractorVersionOrError = requireExtractorVersion(body.extractorVersion)
      if (extractorVersionOrError instanceof Response) return extractorVersionOrError
      const extractorVersion = extractorVersionOrError

      let saved: Awaited<ReturnType<typeof saveSignTrainingSample>>
      try {
        saved = await saveSignTrainingSample({
          label: body.label,
          labelSource: "manual_capture",
          landmarkSequence: body.landmarkSequence,
          frameTimestamps: body.frameTimestamps || [],
          sequenceDurationMs: body.sequenceDurationMs || 0,
          sequenceFrameCount: body.sequenceFrameCount || body.landmarkSequence.length,
          source: body.source || "device-camera",
          capturedAt: Date.now(),
          frameCount: body.frameCount || 0,
          gridColumns: body.gridColumns || 4,
          gridRows: body.gridRows || 4,
          mimeType: body.mimeType === "image/jpeg" ? "image/jpeg" : "image/jpeg",
          skeletonMimeType:
            body.skeletonMimeType === "image/png" || body.skeletonMimeType === "image/jpeg"
              ? body.skeletonMimeType
              : body.mimeType === "image/jpeg"
                ? "image/jpeg"
                : undefined,
          gestureHints: body.gestureHints || [],
          base64Grid: body.base64Grid?.trim() || undefined,
          base64SkeletonGrid: body.base64SkeletonGrid?.trim() || undefined,
          extractorVersion,
        })
      } catch (error) {
        if (error instanceof SignTrainingSampleValidationError) {
          return Response.json({error: error.message}, {status: 400})
        }
        throw error
      }
      const {sampleId, stats} = saved

      return Response.json({
        success: true,
        sampleId,
        label: body.label,
        stats,
      })
    },
  },
}
