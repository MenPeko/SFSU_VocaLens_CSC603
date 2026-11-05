from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


CoreSignLabel = Literal["wait", "help", "yes", "no", "stop", "again", "thank_you", "please"]
SignSource = Literal["device-camera", "glasses-stream"]


class PointModel(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = None


class PoseSubsetModel(BaseModel):
    nose: Optional[PointModel] = None
    leftShoulder: Optional[PointModel] = None
    rightShoulder: Optional[PointModel] = None
    leftElbow: Optional[PointModel] = None
    rightElbow: Optional[PointModel] = None
    leftWrist: Optional[PointModel] = None
    rightWrist: Optional[PointModel] = None


class FaceSubsetModel(BaseModel):
    leftEyebrow: list[PointModel] = Field(default_factory=list)
    rightEyebrow: list[PointModel] = Field(default_factory=list)
    leftEye: list[PointModel] = Field(default_factory=list)
    rightEye: list[PointModel] = Field(default_factory=list)
    lips: list[PointModel] = Field(default_factory=list)


class LandmarkFrameModel(BaseModel):
    timestampMs: float
    pose: PoseSubsetModel
    leftHand: list[PointModel] = Field(default_factory=list)
    rightHand: list[PointModel] = Field(default_factory=list)
    face: FaceSubsetModel


class ClassifySequenceRequest(BaseModel):
    frames: list[LandmarkFrameModel]
    fps: float = 12.0
    source: SignSource
    sequenceDurationMs: int = 0
    # Required — must match MEDIAPIPE_EXTRACTOR_VERSION used by the capture
    # client. The service compares this against the loaded model's extractor
    # and refuses to serve on mismatch (returns no_sign with a clear reason).
    extractorVersion: str


class ClassifierAlternative(BaseModel):
    label: CoreSignLabel
    confidence: float


class ClassifySequenceResponse(BaseModel):
    status: Literal["recognized", "uncertain", "no_sign"]
    label: Optional[CoreSignLabel] = None
    confidence: float = 0.0
    alternatives: list[ClassifierAlternative] = Field(default_factory=list)
    model: str = "local-knn-sequence-v1"
    modelVersion: Optional[str] = None
    trainingRunId: Optional[str] = None
    reason: Optional[str] = None
    # Extractor version of the loaded model artifact (None for the legacy
    # heuristic path that has no artifact behind it).
    extractorVersion: Optional[str] = None
