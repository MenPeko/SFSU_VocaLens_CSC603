import type {CoreSignLabel} from "./core-signs"
import type {RequestedCaptureQuality, ActualCaptureQuality, CaptureSettingsStatus, CaptureSettingsFallbackReason, SkeletonRenderMode} from "./capture-settings"

export interface GestureHint {
  name: string
  score: number
}

export type SignCaptureSource = "device-camera" | "glasses-stream"

export interface SignLandmarkPoint {
  x: number
  y: number
  z: number
  visibility?: number
}

export interface SignPoseSubset {
  nose?: SignLandmarkPoint
  leftShoulder?: SignLandmarkPoint
  rightShoulder?: SignLandmarkPoint
  leftElbow?: SignLandmarkPoint
  rightElbow?: SignLandmarkPoint
  leftWrist?: SignLandmarkPoint
  rightWrist?: SignLandmarkPoint
}

export interface SignFaceSubset {
  leftEyebrow: SignLandmarkPoint[]
  rightEyebrow: SignLandmarkPoint[]
  leftEye: SignLandmarkPoint[]
  rightEye: SignLandmarkPoint[]
  lips: SignLandmarkPoint[]
}

export interface SignLandmarkFrame {
  timestampMs: number
  pose: SignPoseSubset
  leftHand: SignLandmarkPoint[]
  rightHand: SignLandmarkPoint[]
  face: SignFaceSubset
}

export interface SignSequenceCapture {
  landmarkSequence: SignLandmarkFrame[]
  frameTimestamps: number[]
  sequenceDurationMs: number
  sequenceFrameCount: number
  source: SignCaptureSource
  // Extractor version stamp — required so downstream code can detect when a
  // MediaPipe bump or landmark schema change has silently invalidated a
  // sample. See src/sign/extractor-version.ts for the format.
  extractorVersion: string
}

export interface SignClassifierAlternative {
  label: CoreSignLabel
  confidence: number
}

export type SignTrainingLabelSource = "manual_capture"

export type SignSessionRole = "train" | "validation" | "test"

export const SIGN_SESSION_ROLES: readonly SignSessionRole[] = ["train", "validation", "test"]

// Mirrors data/sign-samples/sessions/<sessionId>/session.json, written by
// sign_classifier_service/eval/sessions.py. See docs/SIGN_EVAL_PLAN.md.
export interface SignRecordingSession {
  sessionId: string
  role: SignSessionRole
  signer: string
  recordedOn: string
  lighting: string
  camera: string
  extractorVersion: string
  createdAt: string
}

export interface SignClassifierResult {
  status: "recognized" | "uncertain" | "no_sign"
  label?: CoreSignLabel
  confidence: number
  alternatives: SignClassifierAlternative[]
  model: string
  modelVersion?: string
  trainingRunId?: string
  reason?: string
  // Extractor version the loaded model was trained on. Clients can compare
  // this to their own MEDIAPIPE_EXTRACTOR_VERSION to surface drift in the UI.
  extractorVersion?: string
}

export interface SignTrainingSample extends SignSequenceCapture {
  label: CoreSignLabel
  // Manual labels are the only samples eligible for classifier training.
  // Gemini-derived interpretations must never be mixed into this dataset.
  labelSource: SignTrainingLabelSource
  capturedAt: number
  frameCount: number
  gridColumns: number
  gridRows: number
  mimeType: "image/jpeg"
  skeletonMimeType?: "image/png" | "image/jpeg"
  gestureHints: GestureHint[]
  base64Grid?: string
  base64SkeletonGrid?: string
  requestedQuality?: RequestedCaptureQuality
  actualQuality?: ActualCaptureQuality
  qualityFallback?: boolean
  actualInputWidth?: number
  actualInputHeight?: number
  skeletonRenderMode?: SkeletonRenderMode
  captureSettingsStatus?: CaptureSettingsStatus
  captureSettingsFallbackReason?: CaptureSettingsFallbackReason
  droppedFrameCount?: number
  // Stamped by the store from SIGN_TRAINING_SESSION, never by the client.
  sessionId?: string
  signer?: string
  sessionRole?: SignSessionRole
}

export interface SignTrainingStats {
  totalSamples: number
  sampleTarget: number
  labels: Record<CoreSignLabel, number>
}
