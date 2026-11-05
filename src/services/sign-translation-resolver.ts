import {DEFAULT_CAPTURE_FPS} from "../sign/capture-config"
import {CORE_SIGN_CANONICAL_TEXT} from "../sign/core-signs"
import type {SignSequenceCapture, SignClassifierAlternative, SignClassifierResult} from "../sign/types"
import {classifySignSequence} from "./sign-classifier-client"
import {interpretSignGridWithGemini, type SignGestureHint} from "./gemini-sign-interpreter"

export interface ResolveSignInterpretationInput extends SignSequenceCapture {
  base64Grid: string
  base64SkeletonGrid?: string
  mimeType?: string
  skeletonMimeType?: string
  gridColumns?: number
  gridRows?: number
  gestureHints?: SignGestureHint[]
}

export interface ResolvedSignInterpretation {
  status: "recognized" | "no_sign"
  englishText: string
  recognizerType: "classifier" | "gemini"
  recognizedLabel?: string
  confidence?: number
  fallbackUsed: boolean
  model: string
  classifierModelVersion?: string
  classifierTrainingRunId?: string
  rawGeminiOutput?: string
  classifierStatus?: SignClassifierResult["status"] | "unavailable" | "skipped"
  classifierAlternatives?: SignClassifierAlternative[]
  classifierError?: string
}

interface ResolveSignInterpretationDeps {
  classify?: typeof classifySignSequence
  interpret?: typeof interpretSignGridWithGemini
}

export async function resolveSignInterpretation(
  input: ResolveSignInterpretationInput,
  deps: ResolveSignInterpretationDeps = {},
): Promise<ResolvedSignInterpretation> {
  const classify = deps.classify || classifySignSequence
  const interpret = deps.interpret || interpretSignGridWithGemini

  let fallbackUsed = false
  let classifierStatus: ResolvedSignInterpretation["classifierStatus"] = "skipped"
  let classifierAlternatives: SignClassifierAlternative[] | undefined
  let classifierModelVersion: string | undefined
  let classifierTrainingRunId: string | undefined
  let classifierError: string | undefined

  if (input.landmarkSequence.length > 0) {
    try {
      const classification = await classify({
        frames: input.landmarkSequence,
        fps: input.sequenceDurationMs > 0 && input.frameTimestamps.length > 1
          ? Math.max(1, Math.round(((input.frameTimestamps.length - 1) * 1000) / input.sequenceDurationMs))
          : DEFAULT_CAPTURE_FPS,
        source: input.source,
        sequenceDurationMs: input.sequenceDurationMs,
        extractorVersion: input.extractorVersion,
      })

      classifierStatus = classification.status
      classifierAlternatives = classification.alternatives
      classifierModelVersion = classification.modelVersion
      classifierTrainingRunId = classification.trainingRunId

      if (classification.status === "recognized" && classification.label) {
        return {
          status: "recognized",
          englishText: CORE_SIGN_CANONICAL_TEXT[classification.label],
          recognizerType: "classifier",
          recognizedLabel: classification.label,
          confidence: classification.confidence,
          fallbackUsed: false,
          model: classification.model,
          classifierModelVersion: classification.modelVersion,
          classifierTrainingRunId: classification.trainingRunId,
          classifierStatus,
          classifierAlternatives,
        }
      }

      fallbackUsed = true
    } catch (error) {
      classifierStatus = "unavailable"
      classifierError = error instanceof Error ? error.message : "Local sign classifier request failed"
      fallbackUsed = true
    }
  }

  const interpretation = await interpret({
    base64Grid: input.base64Grid,
    base64SkeletonGrid: input.base64SkeletonGrid,
    mimeType: input.mimeType || "image/jpeg",
    skeletonMimeType: input.skeletonMimeType || input.mimeType || "image/jpeg",
    gridColumns: input.gridColumns || 4,
    gridRows: input.gridRows || 4,
    gestureHints: input.gestureHints || [],
  })

  const englishText = interpretation.englishText.trim()
  if (!englishText || englishText === "NO_SIGN_FOUND") {
    return {
      status: "no_sign",
      englishText: "",
      recognizerType: "gemini",
      fallbackUsed,
      model: interpretation.model,
      classifierModelVersion,
      classifierTrainingRunId,
      rawGeminiOutput: interpretation.rawResponse,
      classifierStatus,
      classifierAlternatives,
      classifierError,
    }
  }

  return {
    status: "recognized",
    englishText,
    recognizerType: "gemini",
    fallbackUsed,
    model: interpretation.model,
    classifierModelVersion,
    classifierTrainingRunId,
    rawGeminiOutput: interpretation.rawResponse,
    classifierStatus,
    classifierAlternatives,
    classifierError,
  }
}
