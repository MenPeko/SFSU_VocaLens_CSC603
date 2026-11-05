from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .schemas import (
    ClassifierAlternative,
    ClassifySequenceRequest,
    ClassifySequenceResponse,
    CoreSignLabel,
    LandmarkFrameModel,
)

CORE_SIGN_LABELS: tuple[CoreSignLabel, ...] = (
    "wait",
    "help",
    "yes",
    "no",
    "stop",
    "again",
    "thank_you",
    "please",
)
RECOGNIZED_CONFIDENCE = 0.80
RECOGNIZED_MARGIN = 0.15
UNCERTAIN_FLOOR = 0.40
MIN_SEQUENCE_FRAMES = 6
LEGACY_MODEL_FAMILY = "local-knn-sequence-v1"
TRAINED_MODEL_FAMILY = "temporal-landmark-softmax-v1"
APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_ROOT / "data" / "sign-models"
ACTIVE_MODEL_POINTER_PATH = MODEL_DIR / "active-model.json"
LEGACY_MODEL_PATH = MODEL_DIR / "core-sign-latest.json"
DEFAULT_MODEL_PATH = LEGACY_MODEL_PATH


@dataclass
class SampleVector:
    label: CoreSignLabel
    vector: list[float]
    source: str
    sample_path: Path


@dataclass
class TrainingExample:
    label: CoreSignLabel
    frames: list[LandmarkFrameModel]
    source: str
    sample_path: Path
    vector: list[float]
    # Extractor version that produced this sample's landmark layout. Carried
    # through so that `train.build_artifact` can refuse to train on a mixed
    # set (see M1 task 2 in DEVELOPMENT_ROADMAP.md).
    extractor_version: str


@dataclass
class TrainedSequenceModel:
    model_family: str
    model_version: str
    training_run_id: str
    labels: list[CoreSignLabel]
    feature_mean: list[float]
    feature_std: list[float]
    weights: list[list[float]]
    bias: list[float]
    # Extractor version this model was trained on — must match the
    # extractorVersion on every classify request or the service refuses to
    # serve the prediction.
    extractor_version: str = ""
    recognized_confidence: float = RECOGNIZED_CONFIDENCE
    recognized_margin: float = RECOGNIZED_MARGIN
    uncertain_floor: float = UNCERTAIN_FLOOR
    metrics: dict[str, Any] | None = None
    trained_at: str | None = None
    artifact_path: Path | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any], artifact_path: Path) -> Optional["TrainedSequenceModel"]:
        labels = payload.get("labels")
        feature_mean = payload.get("featureMean")
        feature_std = payload.get("featureStd")
        weights = payload.get("weights")
        bias = payload.get("bias")

        if not isinstance(labels, list) or not labels:
            return None
        if not isinstance(feature_mean, list) or not isinstance(feature_std, list):
            return None
        if not isinstance(weights, list) or not isinstance(bias, list):
            return None
        if len(weights) != len(labels) or len(bias) != len(labels):
            return None
        feature_count = len(feature_mean)
        if feature_count == 0 or len(feature_std) != feature_count:
            return None
        if any(not isinstance(row, list) or len(row) != feature_count for row in weights):
            return None

        typed_labels: list[CoreSignLabel] = []
        for label in labels:
            if label not in CORE_SIGN_LABELS:
                return None
            typed_labels.append(label)

        thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}

        extractor_value = payload.get("extractor")
        if not isinstance(extractor_value, str) or not extractor_value.strip():
            # Reject artifacts that do not record their extractor version.
            # This is exactly the silent-killer case M1 task 2 is guarding
            # against — an unlabeled artifact is unsafe to serve because we
            # cannot tell whether the inference-time landmarks agree with it.
            return None

        return cls(
            model_family=str(payload.get("modelFamily") or TRAINED_MODEL_FAMILY),
            model_version=str(payload.get("modelVersion") or artifact_path.stem),
            training_run_id=str(payload.get("trainingRunId") or "unknown-run"),
            labels=typed_labels,
            feature_mean=[float(value) for value in feature_mean],
            feature_std=[float(value) for value in feature_std],
            weights=[[float(value) for value in row] for row in weights],
            bias=[float(value) for value in bias],
            extractor_version=extractor_value.strip(),
            recognized_confidence=float(thresholds.get("recognizedConfidence", RECOGNIZED_CONFIDENCE)),
            recognized_margin=float(thresholds.get("recognizedMargin", RECOGNIZED_MARGIN)),
            uncertain_floor=float(thresholds.get("uncertainFloor", UNCERTAIN_FLOOR)),
            metrics=payload.get("metrics") if isinstance(payload.get("metrics"), dict) else None,
            trained_at=str(payload.get("trainedAt")) if payload.get("trainedAt") else None,
            artifact_path=artifact_path,
        )

    def predict(self, vector: list[float]) -> list[tuple[CoreSignLabel, float]]:
        if len(vector) != len(self.feature_mean):
            return []

        standardized = [
            (value - mean) / std if abs(std) > 1e-6 else value - mean
            for value, mean, std in zip(vector, self.feature_mean, self.feature_std)
        ]
        logits: list[float] = []

        for class_index, label in enumerate(self.labels):
            del label
            logit = self.bias[class_index]
            weights = self.weights[class_index]
            logit += sum(weight * value for weight, value in zip(weights, standardized))
            logits.append(logit)

        max_logit = max(logits)
        exp_values = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exp_values)
        if total <= 1e-6:
            return []

        ranked = sorted(
            ((label, exp_values[index] / total) for index, label in enumerate(self.labels)),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(label, round(confidence, 6)) for label, confidence in ranked]


@dataclass
class ActiveModelPointer:
    artifact_path: Path
    promoted_at: str
    training_run_id: str | None = None
    notes: str = ""
    pointer_path: Path = ACTIVE_MODEL_POINTER_PATH


def resolve_app_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (APP_ROOT / path).resolve()


def get_env_model_path() -> Optional[Path]:
    configured = os.environ.get("SIGN_CLASSIFIER_MODEL_PATH")
    if not configured:
        return None
    return resolve_app_path(configured)


def get_default_model_path() -> Path:
    return get_env_model_path() or DEFAULT_MODEL_PATH


def load_active_model_pointer(pointer_path: Path = ACTIVE_MODEL_POINTER_PATH) -> Optional[ActiveModelPointer]:
    if not pointer_path.exists():
        return None

    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    artifact_path = payload.get("artifactPath")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return None

    return ActiveModelPointer(
        artifact_path=resolve_app_path(artifact_path),
        promoted_at=str(payload.get("promotedAt") or ""),
        training_run_id=str(payload.get("trainingRunId")) if payload.get("trainingRunId") else None,
        notes=str(payload.get("notes") or ""),
        pointer_path=pointer_path,
    )


def _distance(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def _tuple_distance(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    return math.dist(p1, p2)


def _safe_point(point) -> tuple[float, float, float]:
    if point is None:
        return (0.0, 0.0, 0.0)
    return (point.x, point.y, point.z)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _hand_extended_count(hand) -> float:
    if len(hand) < 21:
        return 0.0

    wrist = hand[0]
    extension_pairs = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]
    extended = 0

    for tip_index, base_index in extension_pairs:
        tip = hand[tip_index]
        base = hand[base_index]
        if _distance(tip, wrist) > _distance(base, wrist) * 1.2:
            extended += 1

    return extended / 5.0


def _hand_openness(hand) -> float:
    if len(hand) < 21:
        return 0.0

    wrist = hand[0]
    tip_indices = [4, 8, 12, 16, 20]
    base_indices = [2, 5, 9, 13, 17]
    ratios: list[float] = []
    for tip_index, base_index in zip(tip_indices, base_indices):
        tip = hand[tip_index]
        base = hand[base_index]
        base_distance = max(_distance(base, wrist), 1e-6)
        ratios.append(_distance(tip, wrist) / base_distance)

    return max(0.0, min(1.0, (_mean(ratios) - 1.0) / 0.6))


def _palm_facing_score(hand) -> float:
    if len(hand) < 18:
        return 0.0

    wrist = hand[0]
    index_mcp = hand[5]
    pinky_mcp = hand[17]
    v1 = (index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z)
    v2 = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z)
    normal = (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )
    magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if magnitude <= 1e-6:
        return 0.0
    return min(1.0, abs(normal[2]) / magnitude)


def _mouth_open_score(frame: LandmarkFrameModel) -> float:
    lips = frame.face.lips
    if len(lips) < 4:
        return 0.0
    top = lips[3]
    bottom = lips[9] if len(lips) > 9 else lips[-1]
    corners = [lips[0], lips[min(6, len(lips) - 1)]]
    mouth_width = max(_distance(corners[0], corners[1]), 1e-6)
    return min(1.0, _distance(top, bottom) / mouth_width)


def _distance_to_face(frame: LandmarkFrameModel, hand: str) -> float:
    wrist = getattr(frame.pose, f"{hand}Wrist", None)
    lips = frame.face.lips
    if wrist is None or not lips:
        return 99.0
    lip_center = lips[len(lips) // 2]
    return _distance(wrist, lip_center)


def _resample_frames(frames: list[LandmarkFrameModel], target_size: int) -> list[LandmarkFrameModel]:
    if not frames:
        return []
    if len(frames) == target_size:
        return frames
    return [frames[min(len(frames) - 1, round(index * (len(frames) - 1) / max(target_size - 1, 1)))] for index in range(target_size)]


def vectorize_sequence(frames: list[LandmarkFrameModel]) -> list[float]:
    if not frames:
        return [0.0] * (24 * 14 + 8)

    resampled = _resample_frames(frames, 24)
    per_frame: list[float] = []
    left_wrist_positions: list[tuple[float, float, float]] = []
    right_wrist_positions: list[tuple[float, float, float]] = []
    hand_presence: list[int] = []

    for frame in resampled:
        left_extended = _hand_extended_count(frame.leftHand)
        right_extended = _hand_extended_count(frame.rightHand)
        left_open = _hand_openness(frame.leftHand)
        right_open = _hand_openness(frame.rightHand)
        left_facing = _palm_facing_score(frame.leftHand)
        right_facing = _palm_facing_score(frame.rightHand)
        left_wrist = _safe_point(frame.pose.leftWrist)
        right_wrist = _safe_point(frame.pose.rightWrist)
        left_elbow = _safe_point(frame.pose.leftElbow)
        right_elbow = _safe_point(frame.pose.rightElbow)
        left_shoulder = _safe_point(frame.pose.leftShoulder)
        right_shoulder = _safe_point(frame.pose.rightShoulder)

        per_frame.extend(
            [
                1.0 if frame.leftHand else 0.0,
                1.0 if frame.rightHand else 0.0,
                left_extended,
                right_extended,
                left_open,
                right_open,
                left_facing,
                right_facing,
                left_wrist[0],
                left_wrist[1],
                right_wrist[0],
                right_wrist[1],
                _mouth_open_score(frame),
                _tuple_distance(left_shoulder, left_elbow) + _tuple_distance(right_shoulder, right_elbow),
            ]
        )
        left_wrist_positions.append(left_wrist)
        right_wrist_positions.append(right_wrist)
        hand_presence.append(int(bool(frame.leftHand)) + int(bool(frame.rightHand)))

    def average_motion(points: list[tuple[float, float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        deltas = [
            math.dist(points[index - 1], points[index])
            for index in range(1, len(points))
            if any(abs(value) > 1e-6 for value in points[index - 1]) and any(abs(value) > 1e-6 for value in points[index])
        ]
        return _mean(deltas)

    summary = [
        _mean(hand_presence) / 2.0,
        average_motion(left_wrist_positions),
        average_motion(right_wrist_positions),
        _mean(_distance_to_face(frame, "left") for frame in resampled),
        _mean(_distance_to_face(frame, "right") for frame in resampled),
        _mean(_hand_openness(frame.leftHand) for frame in resampled),
        _mean(_hand_openness(frame.rightHand) for frame in resampled),
        _mean(_mouth_open_score(frame) for frame in resampled),
    ]
    return per_frame + summary


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude <= 1e-6:
        return [0.0 for _ in vector]
    return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    normalized_left = _normalize(left)
    normalized_right = _normalize(right)
    similarity = sum(l_value * r_value for l_value, r_value in zip(normalized_left, normalized_right))
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def heuristic_scores(frames: list[LandmarkFrameModel]) -> dict[CoreSignLabel, float]:
    if not frames:
        return {}

    resampled = _resample_frames(frames, min(24, len(frames)))
    both_hands_ratio = _mean(1.0 if frame.leftHand and frame.rightHand else 0.0 for frame in resampled)
    left_open = _mean(_hand_openness(frame.leftHand) for frame in resampled)
    right_open = _mean(_hand_openness(frame.rightHand) for frame in resampled)
    open_both = min(left_open, right_open)
    palm_both = min(_mean(_palm_facing_score(frame.leftHand) for frame in resampled), _mean(_palm_facing_score(frame.rightHand) for frame in resampled))

    left_motion = _mean(
        math.dist(_safe_point(resampled[index - 1].pose.leftWrist), _safe_point(resampled[index].pose.leftWrist))
        for index in range(1, len(resampled))
    )
    right_motion = _mean(
        math.dist(_safe_point(resampled[index - 1].pose.rightWrist), _safe_point(resampled[index].pose.rightWrist))
        for index in range(1, len(resampled))
    )
    stationary_score = max(0.0, 1.0 - ((left_motion + right_motion) / 2.0) * 2.5)

    one_hand_ratio = _mean(1.0 if bool(frame.leftHand) ^ bool(frame.rightHand) else 0.0 for frame in resampled)
    dominant_open = max(left_open, right_open)
    dominant_facing = max(
        _mean(_palm_facing_score(frame.leftHand) for frame in resampled),
        _mean(_palm_facing_score(frame.rightHand) for frame in resampled),
    )
    fist_motion = max(
        _mean(1.0 - _hand_openness(frame.leftHand) for frame in resampled),
        _mean(1.0 - _hand_openness(frame.rightHand) for frame in resampled),
    ) * max(left_motion, right_motion) * 1.8

    scores: dict[CoreSignLabel, float] = {}
    wait_score = min(0.92, 0.45 + both_hands_ratio * 0.18 + open_both * 0.18 + palm_both * 0.15 + stationary_score * 0.12)
    if both_hands_ratio > 0.55 and open_both > 0.6:
        scores["wait"] = wait_score

    stop_score = min(0.9, 0.42 + one_hand_ratio * 0.18 + dominant_open * 0.17 + dominant_facing * 0.15 + stationary_score * 0.1)
    if one_hand_ratio > 0.45 and dominant_open > 0.65:
        scores["stop"] = stop_score

    help_score = min(
        0.78,
        0.35
        + max(
            0.0,
            1.0 - _mean(_distance_to_face(frame, "left") for frame in resampled) * 2.0,
            1.0 - _mean(_distance_to_face(frame, "right") for frame in resampled) * 2.0,
        )
        * 0.28
        + max(0.0, 1.0 - dominant_open) * 0.15,
    )
    if help_score > 0.55:
        scores["help"] = help_score

    yes_score = min(0.75, 0.30 + fist_motion)
    if yes_score > 0.55:
        scores["yes"] = yes_score

    return scores


def load_training_examples(data_dir: Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    if not data_dir.exists():
        return examples

    for label in CORE_SIGN_LABELS:
        label_dir = data_dir / label
        if not label_dir.exists():
            continue

        for sample_path in sorted(label_dir.glob("*.json")):
            try:
                sample_data = json.loads(sample_path.read_text(encoding="utf-8"))
                label_source = sample_data.get("labelSource")
                recognizer_type = sample_data.get("recognizerType")
                fallback_used = sample_data.get("fallbackUsed")

                if isinstance(label_source, str) and label_source.strip() and label_source != "manual_capture":
                    print(
                        f"WARNING: skipping {sample_path} — "
                        f"labelSource={label_source!r} is not eligible for classifier training"
                    )
                    continue

                if recognizer_type == "gemini" or fallback_used is True or "rawGeminiOutput" in sample_data:
                    print(
                        f"WARNING: skipping {sample_path} — sample carries Gemini-derived interpretation metadata"
                    )
                    continue

                frames = [LandmarkFrameModel.model_validate(frame) for frame in sample_data.get("landmarkSequence", [])]
                if not frames:
                    continue
                extractor_value = sample_data.get("extractorVersion")
                if not isinstance(extractor_value, str) or not extractor_value.strip():
                    # Skip unversioned samples rather than silently mixing
                    # them into a training run. Print so the operator can see
                    # drops.
                    print(f"WARNING: skipping {sample_path} — missing extractorVersion")
                    continue
                examples.append(
                    TrainingExample(
                        label=label,
                        frames=frames,
                        source=str(sample_data.get("source", "device-camera")),
                        sample_path=sample_path,
                        vector=vectorize_sequence(frames),
                        extractor_version=extractor_value.strip(),
                    )
                )
            except Exception:
                continue

    return examples


class LandmarkSequenceClassifier:
    def __init__(
        self,
        data_dir: Path,
        model_path: Optional[Path] = None,
        active_model_pointer_path: Optional[Path] = None,
        prefer_active_model_pointer: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.model_path = resolve_app_path(model_path) if model_path else None
        self.active_model_pointer_path = (
            resolve_app_path(active_model_pointer_path) if active_model_pointer_path else ACTIVE_MODEL_POINTER_PATH
        )
        self.prefer_active_model_pointer = prefer_active_model_pointer
        self.samples: list[SampleVector] = []
        self.trained_model: Optional[TrainedSequenceModel] = None
        self.active_model_pointer: Optional[ActiveModelPointer] = None
        self.loaded_model_path: Optional[Path] = None
        self.reload_runtime_assets(force=True)

    def reload_samples(self) -> None:
        self.samples = [
            SampleVector(
                label=example.label,
                vector=example.vector,
                source=example.source,
                sample_path=example.sample_path,
            )
            for example in load_training_examples(self.data_dir)
        ]

    def _candidate_model_paths(self) -> list[Path]:
        candidates: list[Path] = []
        self.active_model_pointer = None

        if self.prefer_active_model_pointer:
            self.active_model_pointer = load_active_model_pointer(self.active_model_pointer_path)
            if self.active_model_pointer:
                candidates.append(self.active_model_pointer.artifact_path)

        if self.model_path:
            candidates.append(self.model_path)

        env_model_path = get_env_model_path()
        if env_model_path:
            candidates.append(env_model_path)

        candidates.append(LEGACY_MODEL_PATH)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            resolved_path = resolve_app_path(path)
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            deduped.append(resolved_path)

        return deduped

    def reload_trained_model(self, force: bool = False) -> None:
        del force
        self.trained_model = None
        self.loaded_model_path = None

        for artifact_path in self._candidate_model_paths():
            if not artifact_path.exists():
                continue

            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            model = TrainedSequenceModel.from_payload(payload, artifact_path)
            if not model:
                continue

            self.trained_model = model
            self.loaded_model_path = artifact_path
            break

    def reload_runtime_assets(self, force: bool = False) -> None:
        self.reload_samples()
        self.reload_trained_model(force=force)

    def runtime_summary(self) -> dict[str, Any]:
        active_model = self.trained_model.model_family if self.trained_model else LEGACY_MODEL_FAMILY
        return {
            "activeModel": active_model,
            "sampleCount": len(self.samples),
            "modelPath": str(self.loaded_model_path or self.model_path or get_env_model_path() or LEGACY_MODEL_PATH),
            "modelLoaded": bool(self.trained_model),
            "modelVersion": self.trained_model.model_version if self.trained_model else None,
            "trainingRunId": self.trained_model.training_run_id if self.trained_model else None,
        }

    def _rank_legacy_scores(self, request_vector: list[float], frames: list[LandmarkFrameModel]) -> list[tuple[CoreSignLabel, float]]:
        score_map: dict[CoreSignLabel, float] = defaultdict(float)

        for label, score in heuristic_scores(frames).items():
            score_map[label] = max(score_map[label], score)

        for sample in self.samples:
            similarity = cosine_similarity(request_vector, sample.vector)
            score_map[sample.label] = max(score_map[sample.label], similarity)

        return sorted(score_map.items(), key=lambda item: item[1], reverse=True)

    def _build_ranked_response(
        self,
        ranked: list[tuple[CoreSignLabel, float]],
        model: str,
        model_version: str | None,
        training_run_id: str | None,
        recognized_confidence: float,
        recognized_margin: float,
        uncertain_floor: float,
        extractor_version: str | None,
    ) -> ClassifySequenceResponse:
        if not ranked:
            return ClassifySequenceResponse(
                status="no_sign",
                confidence=0.0,
                model=model,
                modelVersion=model_version,
                trainingRunId=training_run_id,
                reason="No local classifier signal",
                extractorVersion=extractor_version,
            )

        alternatives = [
            ClassifierAlternative(label=label, confidence=round(confidence, 4))
            for label, confidence in ranked[:3]
        ]
        top_label, top_confidence = ranked[0]
        second_confidence = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_confidence - second_confidence

        if top_confidence >= recognized_confidence and margin >= recognized_margin:
            return ClassifySequenceResponse(
                status="recognized",
                label=top_label,
                confidence=round(top_confidence, 4),
                alternatives=alternatives,
                model=model,
                modelVersion=model_version,
                trainingRunId=training_run_id,
                extractorVersion=extractor_version,
            )

        if top_confidence >= uncertain_floor:
            return ClassifySequenceResponse(
                status="uncertain",
                label=top_label,
                confidence=round(top_confidence, 4),
                alternatives=alternatives,
                model=model,
                modelVersion=model_version,
                trainingRunId=training_run_id,
                reason=f"Low confidence margin ({margin:.3f})",
                extractorVersion=extractor_version,
            )

        return ClassifySequenceResponse(
            status="no_sign",
            confidence=round(top_confidence, 4),
            alternatives=alternatives,
            model=model,
            modelVersion=model_version,
            trainingRunId=training_run_id,
            reason="Below no-sign threshold",
            extractorVersion=extractor_version,
        )

    def predict(self, request: ClassifySequenceRequest) -> ClassifySequenceResponse:
        self.reload_runtime_assets()

        if len(request.frames) < MIN_SEQUENCE_FRAMES:
            return ClassifySequenceResponse(
                status="no_sign",
                confidence=0.0,
                reason="Not enough frames",
                extractorVersion=self.trained_model.extractor_version if self.trained_model else None,
            )

        request_vector = vectorize_sequence(request.frames)

        if self.trained_model:
            # Extractor lock: refuse to serve predictions when the request's
            # landmark extractor disagrees with what the model was trained on.
            # The resolver will see no_sign + reason=extractor_mismatch and
            # fall back to Gemini — noisy but safe.
            if request.extractorVersion != self.trained_model.extractor_version:
                return ClassifySequenceResponse(
                    status="no_sign",
                    confidence=0.0,
                    model=self.trained_model.model_family,
                    modelVersion=self.trained_model.model_version,
                    trainingRunId=self.trained_model.training_run_id,
                    reason=(
                        f"extractor_mismatch: model={self.trained_model.extractor_version} "
                        f"request={request.extractorVersion}"
                    ),
                    extractorVersion=self.trained_model.extractor_version,
                )

            ranked = self.trained_model.predict(request_vector)
            return self._build_ranked_response(
                ranked,
                model=self.trained_model.model_family,
                model_version=self.trained_model.model_version,
                training_run_id=self.trained_model.training_run_id,
                recognized_confidence=self.trained_model.recognized_confidence,
                recognized_margin=self.trained_model.recognized_margin,
                uncertain_floor=self.trained_model.uncertain_floor,
                extractor_version=self.trained_model.extractor_version,
            )

        ranked = self._rank_legacy_scores(request_vector, request.frames)
        return self._build_ranked_response(
            ranked,
            model=LEGACY_MODEL_FAMILY,
            model_version=None,
            training_run_id=None,
            recognized_confidence=RECOGNIZED_CONFIDENCE,
            recognized_margin=RECOGNIZED_MARGIN,
            uncertain_floor=UNCERTAIN_FLOOR,
            extractor_version=None,
        )
