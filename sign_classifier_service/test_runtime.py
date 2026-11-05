from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from .classifier import LandmarkSequenceClassifier, load_training_examples
from .extractor_version import MEDIAPIPE_EXTRACTOR_VERSION
from .schemas import ClassifySequenceRequest, PointModel
from .train import build_artifact, resolve_single_extractor_version, stratified_split, write_artifacts


def make_open_hand(offset_x: float, offset_y: float) -> list[dict[str, float]]:
    wrist = {"x": offset_x, "y": offset_y, "z": 0.0}
    points: list[dict[str, float]] = [wrist]
    for index in range(1, 21):
        row = (index - 1) // 4
        col = (index - 1) % 4
        points.append(
            {
                "x": offset_x + (col - 1.5) * 0.03,
                "y": offset_y - 0.05 - row * 0.06,
                "z": 0.0,
            }
        )
    return points


def make_frame(timestamp_ms: int, label: str, jitter: float) -> dict:
    left_hand = make_open_hand(-0.35, 0.45 - jitter) if label == "wait" else []
    right_hand = make_open_hand(0.35, 0.45 - jitter)
    pose = {
        "leftShoulder": {"x": -0.5, "y": 0.0, "z": 0.0},
        "rightShoulder": {"x": 0.5, "y": 0.0, "z": 0.0},
        "leftElbow": {"x": -0.42, "y": 0.18, "z": 0.0},
        "rightElbow": {"x": 0.42, "y": 0.18, "z": 0.0},
        "leftWrist": {"x": -0.35, "y": 0.4 - jitter, "z": 0.0},
        "rightWrist": {"x": 0.35, "y": 0.4 - jitter, "z": 0.0},
    }
    if label == "stop":
        pose["leftWrist"] = {"x": -0.2, "y": 0.05, "z": 0.0}
        pose["leftElbow"] = {"x": -0.3, "y": 0.08, "z": 0.0}

    return {
        "timestampMs": timestamp_ms,
        "pose": pose,
        "leftHand": left_hand,
        "rightHand": right_hand,
        "face": {
            "leftEyebrow": [],
            "rightEyebrow": [],
            "leftEye": [],
            "rightEye": [],
            "lips": [
                PointModel(x=-0.02, y=0.05, z=0.0).model_dump(),
                PointModel(x=0.02, y=0.05, z=0.0).model_dump(),
                PointModel(x=0.0, y=0.045, z=0.0).model_dump(),
                PointModel(x=0.0, y=0.055, z=0.0).model_dump(),
            ],
        },
    }


def write_sample(
    base_dir: Path,
    label: str,
    sample_index: int,
    extractor_version: str = MEDIAPIPE_EXTRACTOR_VERSION,
    extra_fields: dict | None = None,
) -> None:
    label_dir = base_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    frames = [make_frame(1000 + frame_index * 83, label, jitter=sample_index * 0.005) for frame_index in range(6)]
    payload = {
        "label": label,
        "labelSource": "manual_capture",
        "source": "device-camera",
        "sequenceDurationMs": 415,
        "sequenceFrameCount": len(frames),
        "frameTimestamps": [frame["timestampMs"] for frame in frames],
        "landmarkSequence": frames,
        "extractorVersion": extractor_version,
    }
    if extra_fields:
        payload.update(extra_fields)
    (label_dir / f"sample-{sample_index}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


class SignClassifierRuntimeTests(unittest.TestCase):
    def test_loader_skips_samples_marked_as_gemini_or_non_manual(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            samples_dir = Path(temp_dir) / "samples"

            write_sample(samples_dir, "wait", 0)
            write_sample(samples_dir, "wait", 1, extra_fields={"labelSource": "gemini_fallback"})
            write_sample(
                samples_dir,
                "wait",
                2,
                extra_fields={
                    "recognizerType": "gemini",
                    "fallbackUsed": True,
                    "rawGeminiOutput": "WAIT",
                },
            )

            examples = load_training_examples(samples_dir)

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].label, "wait")
            self.assertEqual(examples[0].sample_path.name, "sample-0.json")

    def test_trained_artifact_is_loaded_by_runtime_classifier_via_active_model_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            samples_dir = base_dir / "samples"
            model_dir = base_dir / "models"

            for sample_index in range(4):
                write_sample(samples_dir, "wait", sample_index)
                write_sample(samples_dir, "stop", sample_index)

            examples = load_training_examples(samples_dir)
            split = stratified_split(examples, validation_ratio=0.25, seed=42)
            artifact, _ = build_artifact(
                train_examples=split.train,
                validation_examples=split.validation,
                epochs=120,
                learning_rate=0.15,
                weight_decay=0.0005,
                model_version="unit-test-model",
                training_run_id="unit-test-run",
            )
            paths = write_artifacts(artifact, model_dir)

            classifier = LandmarkSequenceClassifier(
                samples_dir,
                active_model_pointer_path=Path(paths["activeModelPath"]),
            )
            wait_example = next(example for example in examples if example.label == "wait")
            result = classifier.predict(
                ClassifySequenceRequest(
                    frames=wait_example.frames,
                    fps=12.0,
                    source="device-camera",
                    sequenceDurationMs=415,
                    extractorVersion=MEDIAPIPE_EXTRACTOR_VERSION,
                )
            )

            self.assertEqual(result.model, "temporal-landmark-softmax-v1")
            self.assertEqual(result.modelVersion, "unit-test-model")
            self.assertEqual(result.trainingRunId, "unit-test-run")
            self.assertEqual(result.label, "wait")
            self.assertEqual(result.status, "recognized")
            self.assertEqual(result.extractorVersion, MEDIAPIPE_EXTRACTOR_VERSION)

    def test_invalid_active_model_pointer_falls_back_to_explicit_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            samples_dir = base_dir / "samples"
            model_dir = base_dir / "models"
            pointer_path = model_dir / "active-model.json"

            for sample_index in range(4):
                write_sample(samples_dir, "wait", sample_index)
                write_sample(samples_dir, "stop", sample_index)

            examples = load_training_examples(samples_dir)
            split = stratified_split(examples, validation_ratio=0.25, seed=42)
            artifact, _ = build_artifact(
                train_examples=split.train,
                validation_examples=split.validation,
                epochs=120,
                learning_rate=0.15,
                weight_decay=0.0005,
                model_version="unit-test-fallback-model",
                training_run_id="unit-test-fallback-run",
            )
            paths = write_artifacts(artifact, model_dir)
            pointer_path.write_text(
                json.dumps(
                    {
                        "artifactPath": "data/sign-models/does-not-exist.json",
                        "promotedAt": "2025-04-10T18:00:00Z",
                        "trainingRunId": "broken-run",
                        "notes": "",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            classifier = LandmarkSequenceClassifier(
                samples_dir,
                model_path=Path(paths["versionedModelPath"]),
                active_model_pointer_path=pointer_path,
            )
            stop_example = next(example for example in examples if example.label == "stop")
            result = classifier.predict(
                ClassifySequenceRequest(
                    frames=stop_example.frames,
                    fps=12.0,
                    source="device-camera",
                    sequenceDurationMs=415,
                    extractorVersion=MEDIAPIPE_EXTRACTOR_VERSION,
                )
            )

            self.assertEqual(result.modelVersion, "unit-test-fallback-model")
            self.assertEqual(result.trainingRunId, "unit-test-fallback-run")
            self.assertEqual(result.status, "recognized")


class ExtractorLockTests(unittest.TestCase):
    def test_ts_and_python_extractor_constants_agree(self) -> None:
        """The TypeScript source of truth and the Python mirror must match."""

        ts_file = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "sign"
            / "extractor-version.ts"
        )
        content = ts_file.read_text(encoding="utf-8")
        match = re.search(
            r'MEDIAPIPE_EXTRACTOR_VERSION\s*=\s*"([^"]+)"', content
        )
        self.assertIsNotNone(
            match,
            "Could not find MEDIAPIPE_EXTRACTOR_VERSION in extractor-version.ts",
        )
        assert match is not None  # for type-checkers
        self.assertEqual(match.group(1), MEDIAPIPE_EXTRACTOR_VERSION)

    def test_training_rejects_mixed_extractor_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            samples_dir = base_dir / "samples"

            # Half the samples use the current extractor, half use a
            # hypothetical bumped version — training must abort.
            for sample_index in range(4):
                extractor = (
                    MEDIAPIPE_EXTRACTOR_VERSION
                    if sample_index < 2
                    else "mediapipe-holistic-js@9.9.9999999999+schema-v1"
                )
                write_sample(samples_dir, "wait", sample_index, extractor_version=extractor)
                write_sample(samples_dir, "stop", sample_index, extractor_version=extractor)

            examples = load_training_examples(samples_dir)
            split = stratified_split(examples, validation_ratio=0.25, seed=42)

            with self.assertRaises(ValueError) as err_ctx:
                build_artifact(
                    train_examples=split.train,
                    validation_examples=split.validation,
                    epochs=10,
                    learning_rate=0.15,
                    weight_decay=0.0005,
                    model_version="unit-test-mixed",
                    training_run_id="unit-test-mixed-run",
                )
            self.assertIn("multiple extractor versions", str(err_ctx.exception))

    def test_training_rejects_missing_extractor_version(self) -> None:
        with self.assertRaises(ValueError):
            resolve_single_extractor_version([])

    def test_classifier_refuses_request_with_mismatched_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            samples_dir = base_dir / "samples"
            model_dir = base_dir / "models"

            for sample_index in range(4):
                write_sample(samples_dir, "wait", sample_index)
                write_sample(samples_dir, "stop", sample_index)

            examples = load_training_examples(samples_dir)
            split = stratified_split(examples, validation_ratio=0.25, seed=42)
            artifact, _ = build_artifact(
                train_examples=split.train,
                validation_examples=split.validation,
                epochs=120,
                learning_rate=0.15,
                weight_decay=0.0005,
                model_version="unit-test-extractor-lock",
                training_run_id="unit-test-extractor-lock-run",
            )
            self.assertEqual(artifact["extractor"], MEDIAPIPE_EXTRACTOR_VERSION)

            paths = write_artifacts(artifact, model_dir)
            classifier = LandmarkSequenceClassifier(
                samples_dir,
                active_model_pointer_path=Path(paths["activeModelPath"]),
            )
            wait_example = next(example for example in examples if example.label == "wait")

            # Using an unexpected extractor version — serving must refuse.
            stale_request = ClassifySequenceRequest(
                frames=wait_example.frames,
                fps=12.0,
                source="device-camera",
                sequenceDurationMs=415,
                extractorVersion="mediapipe-holistic-js@0.0.0000000001+schema-v1",
            )
            result = classifier.predict(stale_request)
            self.assertEqual(result.status, "no_sign")
            self.assertIsNotNone(result.reason)
            assert result.reason is not None  # for type-checkers
            self.assertIn("extractor_mismatch", result.reason)
            self.assertEqual(result.extractorVersion, MEDIAPIPE_EXTRACTOR_VERSION)

            # Matching extractor still works.
            fresh_request = ClassifySequenceRequest(
                frames=wait_example.frames,
                fps=12.0,
                source="device-camera",
                sequenceDurationMs=415,
                extractorVersion=MEDIAPIPE_EXTRACTOR_VERSION,
            )
            fresh_result = classifier.predict(fresh_request)
            self.assertEqual(fresh_result.status, "recognized")
            self.assertEqual(fresh_result.label, "wait")

    def test_classifier_refuses_artifact_missing_extractor(self) -> None:
        """An artifact without the `extractor` field must be rejected on load."""

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            samples_dir = base_dir / "samples"
            model_dir = base_dir / "models"

            for sample_index in range(4):
                write_sample(samples_dir, "wait", sample_index)
                write_sample(samples_dir, "stop", sample_index)

            examples = load_training_examples(samples_dir)
            split = stratified_split(examples, validation_ratio=0.25, seed=42)
            artifact, _ = build_artifact(
                train_examples=split.train,
                validation_examples=split.validation,
                epochs=120,
                learning_rate=0.15,
                weight_decay=0.0005,
                model_version="unit-test-unlabeled",
                training_run_id="unit-test-unlabeled-run",
            )
            # Strip the extractor field to simulate a legacy artifact.
            del artifact["extractor"]
            model_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = model_dir / "legacy.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            pointer_path = model_dir / "active-model.json"
            pointer_path.write_text(
                json.dumps(
                    {
                        "artifactPath": str(artifact_path),
                        "promotedAt": "2026-04-10T18:00:00Z",
                        "trainingRunId": "unit-test-unlabeled-run",
                        "notes": "",
                    }
                ),
                encoding="utf-8",
            )

            classifier = LandmarkSequenceClassifier(
                samples_dir,
                active_model_pointer_path=pointer_path,
            )
            self.assertIsNone(classifier.trained_model)


if __name__ == "__main__":
    unittest.main()
