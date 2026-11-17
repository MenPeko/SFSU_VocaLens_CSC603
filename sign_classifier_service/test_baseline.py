from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .classifier import LandmarkSequenceClassifier
from .eval import audit, baseline
from .schemas import ClassifierAlternative, ClassifySequenceResponse
from .test_eval import V1_EXTRACTOR, make_example, make_frames


def _prediction(truth: str, label: str, confidence: float, second: float, status: str = "recognized", group: str = "g1") -> baseline.Prediction:
    response = ClassifySequenceResponse(
        status=status,
        label=label,
        confidence=confidence,
        alternatives=[ClassifierAlternative(label=label, confidence=confidence), ClassifierAlternative(label="no" if label != "no" else "yes", confidence=second)],
    )
    return baseline.Prediction(truth, group, response, latency_ms=1.0)


class ClassificationMetricTests(unittest.TestCase):
    def test_macro_f1_and_missing_prediction(self) -> None:
        pairs = [("yes", "yes"), ("yes", "no"), ("no", "no"), ("no", None)]
        metrics = baseline.classification_metrics(pairs, ["yes", "no"])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["perClass"]["yes"], {"support": 2, "precision": 1.0, "recall": 0.5, "f1": 0.6667})
        self.assertEqual(metrics["perClass"]["no"], {"support": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5})
        self.assertEqual(metrics["macroF1"], 0.5833)
        self.assertEqual(metrics["confusionMatrix"]["no"], {"no": 1, "none": 1})

    def test_selective_metrics_count_false_accepts_against_all_clips(self) -> None:
        predictions = [
            _prediction("yes", "yes", 0.9, 0.1),
            _prediction("no", "yes", 0.9, 0.1),
            _prediction("no", "no", 0.5, 0.4, status="uncertain"),
        ]
        metrics = baseline.selective_metrics(predictions, ["yes", "no"], lambda p: p.status == "recognized")
        self.assertEqual(metrics["acceptanceRate"], 0.6667)
        self.assertEqual(metrics["acceptedAccuracy"], 0.5)
        self.assertEqual(metrics["falseAcceptRate"], 0.3333)
        self.assertEqual(metrics["perClassFalsePositiveRate"], {"yes": 0.5, "no": 0.0})

    def test_sweep_applies_margin(self) -> None:
        predictions = [_prediction("yes", "yes", 0.9, 0.85), _prediction("no", "no", 0.9, 0.1)]
        rows = {(row["confidence"], row["margin"]): row for row in baseline.threshold_sweep(predictions, ["yes", "no"])}
        self.assertEqual(rows[(0.8, 0.0)]["acceptanceRate"], 1.0)
        self.assertEqual(rows[(0.8, 0.15)]["acceptanceRate"], 0.5)

    def test_bootstrap_is_seeded_and_resamples_groups(self) -> None:
        predictions = [_prediction("yes", "yes" if i % 3 else "no", 0.9, 0.1, group=f"g{i % 4}") for i in range(20)]
        first = baseline.cluster_bootstrap(predictions, ["yes", "no"], resamples=200, seed=7)
        second = baseline.cluster_bootstrap(predictions, ["yes", "no"], resamples=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual((first["unit"], first["units"]), ("group", 4))
        single = baseline.cluster_bootstrap([_prediction("yes", "yes", 0.9, 0.1)] * 5, ["yes"], resamples=10, seed=7)
        self.assertEqual((single["unit"], single["units"]), ("clip", 5))


class IsolatedClassifierTests(unittest.TestCase):
    def test_cosine_path_uses_given_references_and_never_reads_disk(self) -> None:
        rng = random.Random(3)
        references = [make_example(label, rng) for label in ("yes", "no") for _ in range(3)]
        with mock.patch.object(LandmarkSequenceClassifier, "reload_runtime_assets", side_effect=AssertionError("disk reload")):
            classifier = baseline.cosine_classifier(references)
            predictions = baseline.score(classifier, [make_example("yes", rng)])
        self.assertEqual(len(classifier.samples), 6)
        self.assertEqual(predictions[0].label, "yes")

    def test_softmax_path_runs_through_production_predict(self) -> None:
        rng = random.Random(4)
        train = [make_example(label, rng, session="a") for label in ("yes", "no", "stop") for _ in range(6)]
        validation = [make_example(label, rng, session="b") for label in ("yes", "no", "stop") for _ in range(2)]
        artifact, _ = baseline.build_artifact(train, validation, epochs=40, learning_rate=0.12, weight_decay=0.0005, model_version="t", training_run_id="t")
        predictions = baseline.score(baseline.softmax_classifier(artifact), [make_example("stop", rng)])
        self.assertEqual(predictions[0].label, "stop")
        self.assertNotEqual(predictions[0].status, "no_sign")


class AuditTests(unittest.TestCase):
    def test_counts_extractors_layout_and_blocks(self) -> None:
        rng = random.Random(5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (label, extractor) in enumerate([("yes", V1_EXTRACTOR), ("yes", V1_EXTRACTOR), ("no", "x+schema-v2")]):
                frames = make_frames(label, rng)
                path = root / "flat" / label / f"{index}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"label": label, "landmarkSequence": frames, "extractorVersion": extractor, "capturedAt": 1_000_000 + index * 10_000}),
                    encoding="utf-8",
                )
            report = audit.audit(flat_dir=root / "flat", sessions_dir=root / "none", archive_dir=root / "none", landmarks_dir=root / "none", asl_dir=root / "none")
        flat = report["sources"][0]
        self.assertEqual(flat["clips"], 3)
        self.assertEqual(flat["extractorVersions"], {V1_EXTRACTOR: 2, "x+schema-v2": 1})
        self.assertEqual(flat["landmarkLayout"]["handPointCounts"], {"21": 36})
        self.assertEqual(flat["missingLandmarks"]["leftHandPresent"], 0.0)
        self.assertEqual([(block["label"], block["clips"]) for block in report["ownFlatRecordingBlocks"]], [("yes", 2), ("no", 1)])
        self.assertEqual(report["rawVideo"]["ownRecordingsVideoFiles"], 0)


if __name__ == "__main__":
    unittest.main()
