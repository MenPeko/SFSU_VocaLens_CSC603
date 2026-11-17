"""Score the existing classifiers on held-out clips without changing them.

    python -m sign_classifier_service.eval.baseline

Classifiers are the two paths in ``LandmarkSequenceClassifier.predict``: the
sample-similarity path (cosine kNN plus heuristics, used when no artifact
loads) and the softmax artifact path. Each runs through the production
``predict`` with the production thresholds; only the per-request disk reload
is disabled so the reference set is the protocol's training split.

Protocols:

- ``asl-citizen-signer-disjoint``: ASL Citizen train/val signers versus the
  fixed held-out test signers, assembled and leak-checked by ``eval.run``.
- ``own-v1-time-holdout``: legacy schema-v1 own clips. One signer, one
  sitting, one sign per block, so no session-disjoint split exists. Each
  sign's block is split by capture time (first 60% train, next 15%
  validation, last 25% test). This is an optimistic upper bound.

Writes reports/baseline-evaluation.json and .md. Aggregate numbers only.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..classifier import (
    CORE_SIGN_LABELS,
    LandmarkSequenceClassifier,
    SampleVector,
    TrainedSequenceModel,
    TrainingExample,
    load_training_examples,
)
from ..schemas import ClassifySequenceRequest, ClassifySequenceResponse
from ..train import build_artifact, group_split
from . import read_json, sha256_file, load_config, write_json
from .audit import FLAT_SAMPLES_DIR, PHASE0_REPORTS_DIR
from .leakage import find_duplicates, landmark_hash
from .run import ASL_TEST, assemble, git_state, load_experiment

CONFIDENCE_GRID = [round(0.40 + 0.05 * step, 2) for step in range(12)]
MARGIN_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
HISTOGRAM_BINS = 10
SCHEMA_V1 = "mediapipe-holistic-js@0.5.1675471629+schema-v1"


class Prediction:
    __slots__ = ("truth", "group", "status", "label", "confidence", "margin", "latency_ms")

    def __init__(self, truth: str, group: str, response: ClassifySequenceResponse, latency_ms: float) -> None:
        self.truth = truth
        self.group = group
        self.status = response.status
        alternatives = response.alternatives
        self.label: Optional[str] = response.label or (alternatives[0].label if alternatives else None)
        self.confidence = float(alternatives[0].confidence if alternatives else response.confidence)
        second = alternatives[1].confidence if len(alternatives) > 1 else 0.0
        self.margin = self.confidence - float(second) if alternatives else 0.0
        self.latency_ms = latency_ms


def _isolated_classifier() -> LandmarkSequenceClassifier:
    classifier = LandmarkSequenceClassifier.__new__(LandmarkSequenceClassifier)
    classifier.samples = []
    classifier.trained_model = None
    classifier.loaded_model_path = None
    classifier.active_model_pointer = None
    classifier.reload_runtime_assets = lambda force=False: None  # type: ignore[method-assign]
    return classifier


def cosine_classifier(references: list[TrainingExample]) -> LandmarkSequenceClassifier:
    classifier = _isolated_classifier()
    classifier.samples = [
        SampleVector(label=example.label, vector=example.vector, source=example.source, sample_path=example.sample_path)
        for example in references
    ]
    return classifier


def softmax_classifier(artifact: dict[str, Any]) -> LandmarkSequenceClassifier:
    classifier = _isolated_classifier()
    model = TrainedSequenceModel.from_payload(artifact, Path(f"{artifact['modelVersion']}.json"))
    if model is None:
        raise RuntimeError("trained artifact failed TrainedSequenceModel validation")
    classifier.trained_model = model
    return classifier


def request_for(example: TrainingExample) -> ClassifySequenceRequest:
    return ClassifySequenceRequest(
        frames=example.frames,
        source="device-camera",
        extractorVersion=example.extractor_version,
    )


def score(classifier: LandmarkSequenceClassifier, examples: list[TrainingExample]) -> list[Prediction]:
    predictions: list[Prediction] = []
    for example in examples:
        request = request_for(example)
        started = time.perf_counter()
        response = classifier.predict(request)
        elapsed = (time.perf_counter() - started) * 1000.0
        predictions.append(Prediction(example.label, example.group_key or "ungrouped", response, elapsed))
    return predictions


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def classification_metrics(pairs: list[tuple[str, Optional[str]]], labels: list[str]) -> dict[str, Any]:
    """Top-1 metrics. A missing prediction counts as wrong for its true class."""

    confusion: dict[str, Counter[str]] = {label: Counter() for label in labels}
    for truth, predicted in pairs:
        confusion.setdefault(truth, Counter())[predicted or "none"] += 1
    per_class: dict[str, dict[str, Any]] = {}
    f1_scores: list[float] = []
    for label in labels:
        support = sum(confusion[label].values())
        true_positive = confusion[label][label]
        predicted_as = sum(row[label] for row in confusion.values())
        precision = true_positive / predicted_as if predicted_as else 0.0
        recall = true_positive / support if support else 0.0
        f1 = _f1(precision, recall)
        if support:
            f1_scores.append(f1)
        per_class[label] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    total = len(pairs)
    return {
        "clips": total,
        "accuracy": round(sum(truth == predicted for truth, predicted in pairs) / total, 4) if total else None,
        "macroF1": round(statistics.fmean(f1_scores), 4) if f1_scores else None,
        "perClass": per_class,
        "confusionMatrix": {label: dict(confusion[label]) for label in labels},
    }


def _accepted(prediction: Prediction, confidence: float, margin: float) -> bool:
    return prediction.label is not None and prediction.confidence >= confidence and prediction.margin >= margin


def selective_metrics(predictions: list[Prediction], labels: list[str], accept: Callable[[Prediction], bool]) -> dict[str, Any]:
    """What the app would emit locally: accepted predictions, the rest go to Gemini."""

    total = len(predictions)
    accepted = [prediction for prediction in predictions if accept(prediction)]
    correct = sum(prediction.label == prediction.truth for prediction in accepted)
    support = Counter(prediction.truth for prediction in predictions)
    false_positive_rate: dict[str, Optional[float]] = {}
    for label in labels:
        negatives = total - support[label]
        wrong = sum(1 for prediction in accepted if prediction.label == label and prediction.truth != label)
        false_positive_rate[label] = round(wrong / negatives, 4) if negatives else None
    return {
        "acceptanceRate": round(len(accepted) / total, 4) if total else None,
        "rejectionRate": round(1 - len(accepted) / total, 4) if total else None,
        "acceptedAccuracy": round(correct / len(accepted), 4) if accepted else None,
        "falseAcceptRate": round((len(accepted) - correct) / total, 4) if total else None,
        "correctAcceptRate": round(correct / total, 4) if total else None,
        "perClassFalsePositiveRate": false_positive_rate,
    }


def threshold_sweep(predictions: list[Prediction], labels: list[str]) -> list[dict[str, Any]]:
    rows = []
    for margin in MARGIN_GRID:
        for confidence in CONFIDENCE_GRID:
            metrics = selective_metrics(predictions, labels, lambda p, c=confidence, m=margin: _accepted(p, c, m))
            metrics.pop("perClassFalsePositiveRate")
            rows.append({"confidence": confidence, "margin": margin, **metrics})
    return rows


def histogram(values: list[float]) -> list[int]:
    counts = [0] * HISTOGRAM_BINS
    for value in values:
        counts[min(HISTOGRAM_BINS - 1, max(0, int(value * HISTOGRAM_BINS)))] += 1
    return counts


def _quantiles(values: list[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]

    return {"p10": round(at(0.1), 4), "p50": round(at(0.5), 4), "p90": round(at(0.9), 4), "mean": round(statistics.fmean(ordered), 4)}


def confidence_distribution(predictions: list[Prediction]) -> dict[str, Any]:
    correct = [prediction for prediction in predictions if prediction.label == prediction.truth]
    wrong = [prediction for prediction in predictions if prediction.label != prediction.truth]
    return {
        "bins": [round(index / HISTOGRAM_BINS, 1) for index in range(HISTOGRAM_BINS)],
        "confidenceCorrect": histogram([p.confidence for p in correct]),
        "confidenceWrong": histogram([p.confidence for p in wrong]),
        "confidenceQuantilesCorrect": _quantiles([p.confidence for p in correct]),
        "confidenceQuantilesWrong": _quantiles([p.confidence for p in wrong]),
        "marginQuantilesCorrect": _quantiles([p.margin for p in correct]),
        "marginQuantilesWrong": _quantiles([p.margin for p in wrong]),
    }


def latency(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": round(ordered[len(ordered) // 2], 3),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "mean": round(statistics.fmean(ordered), 3),
        "max": round(ordered[-1], 3),
    }


def cluster_bootstrap(predictions: list[Prediction], labels: list[str], resamples: int, seed: int) -> dict[str, Any]:
    """95% interval on accuracy and macro-F1, resampling whole groups (signers or sessions).

    With one group it falls back to resampling clips, which ignores within-session correlation.
    """

    by_group: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_group[prediction.group].append(prediction)
    groups = sorted(by_group)
    unit = "group" if len(groups) > 1 else "clip"
    units: list[list[Prediction]] = [by_group[key] for key in groups] if unit == "group" else [[p] for p in predictions]
    rng = random.Random(seed)
    accuracies: list[float] = []
    f1s: list[float] = []
    for _ in range(resamples):
        sample = [prediction for _ in units for prediction in units[rng.randrange(len(units))]]
        metrics = classification_metrics([(p.truth, p.label) for p in sample], labels)
        accuracies.append(metrics["accuracy"])
        f1s.append(metrics["macroF1"])

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [round(ordered[int(0.025 * (len(ordered) - 1))], 4), round(ordered[int(0.975 * (len(ordered) - 1))], 4)]

    return {"unit": unit, "units": len(units), "resamples": resamples, "accuracy95": interval(accuracies), "macroF1_95": interval(f1s)}


def evaluate_classifier(name: str, classifier: LandmarkSequenceClassifier, test: list[TrainingExample], labels: list[str], bootstrap: dict[str, Any]) -> dict[str, Any]:
    predictions = score(classifier, test)
    statuses = Counter(prediction.status for prediction in predictions)
    production = selective_metrics(predictions, labels, lambda p: p.status == "recognized")
    return {
        "classifier": name,
        "topOne": classification_metrics([(p.truth, p.label) for p in predictions], labels),
        "bootstrap": cluster_bootstrap(predictions, labels, int(bootstrap["resamples"]), int(bootstrap["seed"])),
        "statusCounts": {status: statuses.get(status, 0) for status in ("recognized", "uncertain", "no_sign")},
        "productionThresholds": production,
        "thresholdSweep": threshold_sweep(predictions, labels),
        "confidence": confidence_distribution(predictions),
        "latencyMs": latency([p.latency_ms for p in predictions]),
    }


def _fingerprint(examples: list[TrainingExample]) -> str:
    return hashlib.sha256("\n".join(sorted(landmark_hash(example) for example in examples)).encode("utf-8")).hexdigest()


def _counts(examples: list[TrainingExample]) -> dict[str, int]:
    counts = Counter(example.label for example in examples)
    return {label: counts[label] for label in CORE_SIGN_LABELS if counts[label]}


def _duplicate_summary(train: list[TrainingExample], test: list[TrainingExample], validation: list[TrainingExample], config: dict[str, Any]) -> dict[str, Any]:
    result = find_duplicates(train, {"test": test}, config["duplicates"], config["seed"], validation=validation)
    return {
        "droppedWithinTrain": len(result.dropped_within_train),
        "crossSplitFailures": len(result.failures),
        "calibration": result.calibration,
    }


def run_protocol(
    name: str,
    description: str,
    references: list[TrainingExample],
    train: list[TrainingExample],
    validation: list[TrainingExample],
    test: list[TrainingExample],
    config: dict[str, Any],
    caveats: list[str],
) -> dict[str, Any]:
    labels = [label for label in CORE_SIGN_LABELS if any(example.label == label for example in test)]
    duplicates = _duplicate_summary(references, test, [], config)
    if duplicates["crossSplitFailures"]:
        raise RuntimeError(f"{name}: {duplicates['crossSplitFailures']} test clips duplicate a reference clip")

    training = config["training"]
    started = time.perf_counter()
    artifact, _ = build_artifact(
        train_examples=train,
        validation_examples=validation,
        epochs=training["epochs"],
        learning_rate=training["learningRate"],
        weight_decay=training["weightDecay"],
        model_version=f"baseline-{name}",
        training_run_id=f"baseline-{name}",
        early_stopping=training["earlyStopping"],
    )
    training_seconds = time.perf_counter() - started

    return {
        "protocol": name,
        "description": description,
        "caveats": caveats,
        "extractorVersion": test[0].extractor_version if test else None,
        "labels": labels,
        "clips": {
            "cosineReferences": _counts(references),
            "softmaxTrain": _counts(train),
            "softmaxValidation": _counts(validation),
            "test": _counts(test),
        },
        "groups": {
            "references": len({example.group_key for example in references if example.group_key}),
            "test": len({example.group_key for example in test if example.group_key}),
        },
        "testFingerprint": _fingerprint(test),
        "duplicates": duplicates,
        "softmaxTraining": {
            "seconds": round(training_seconds, 1),
            "earlyStopping": artifact["metrics"].get("earlyStopping"),
            "validationAccuracy": artifact["metrics"]["validation"]["accuracy"],
        },
        "classifiers": [
            evaluate_classifier("cosine-knn-heuristic (local-knn-sequence-v1)", cosine_classifier(references), test, labels, config["bootstrap"]),
            evaluate_classifier("softmax (temporal-landmark-softmax-v1)", softmax_classifier(artifact), test, labels, config["bootstrap"]),
        ],
    }


def asl_citizen_protocol(config: dict[str, Any]) -> dict[str, Any]:
    data, report = assemble(load_experiment("plus-asl-citizen"), config)
    if not report.ok:
        raise RuntimeError("ASL Citizen leak check failed:\n- " + "\n- ".join(report.failures))
    pool = sorted(data.train, key=lambda e: (e.group_key, e.source_video_id))
    test = sorted(data.tests[ASL_TEST], key=lambda e: (e.group_key, e.source_video_id))
    split = group_split(pool, 0.2, config["seed"])
    return run_protocol(
        "asl-citizen-signer-disjoint",
        "ASL Citizen clips from the official train/val signers as references and training data; the fixed held-out test signers as the test set. "
        "Softmax early-stops on 20% of the training signers (whole signers).",
        references=pool,
        train=split.train,
        validation=split.validation,
        test=test,
        config=config,
        caveats=[
            "Not the app's camera: studio-style webcam clips from Deaf signers, extracted offline with the app's browser extractor.",
            "Signer-disjoint, so it does measure cross-signer generalization.",
        ],
    )


def own_v1_protocol(config: dict[str, Any], samples_dir: Path = FLAT_SAMPLES_DIR) -> dict[str, Any]:
    examples = [example for example in load_training_examples(samples_dir) if example.extractor_version == SCHEMA_V1]
    captured = {example.sample_path: read_json(example.sample_path).get("capturedAt") or 0 for example in examples}
    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    test: list[TrainingExample] = []
    for label in CORE_SIGN_LABELS:
        block = sorted((example for example in examples if example.label == label), key=lambda e: (captured[e.sample_path], str(e.sample_path)))
        if not block:
            continue
        train_end = int(round(len(block) * 0.60))
        validation_end = int(round(len(block) * 0.75))
        train += block[:train_end]
        validation += block[train_end:validation_end]
        test += block[validation_end:]
    return run_protocol(
        "own-v1-time-holdout",
        "Legacy schema-v1 own clips. Each sign was recorded in one continuous block; the first 60% of each block trains, "
        "the next 15% is validation for softmax early stopping, the last 25% is the test set. Cosine references are train plus validation.",
        references=train + validation,
        train=train,
        validation=validation,
        test=test,
        config=config,
        caveats=[
            "One signer, one sitting, so neither signer- nor session-disjoint. Read as an optimistic upper bound.",
            "Each sign was recorded in its own block, so sign is confounded with capture time, posture drift, and lighting.",
            "No 'wait' class: the schema-v1 wait clips were archived as wrong, and the 6 schema-v2 wait clips cannot be mixed in.",
            "schema-v1 does not match the live extractor, so these numbers do not describe the running app.",
        ],
    )


def production_runtime_check(test: list[TrainingExample]) -> dict[str, Any]:
    """What the running service does today: the real classifier, with its real disk reloads, on current-extractor requests."""

    classifier = LandmarkSequenceClassifier(FLAT_SAMPLES_DIR)
    summary = classifier.runtime_summary()
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    timings: list[float] = []
    for example in test:
        request = request_for(example)
        started = time.perf_counter()
        classifier.reload_runtime_assets()
        response = classifier.predict(request)
        timings.append((time.perf_counter() - started) * 1000.0)
        statuses[response.status] += 1
        reasons[(response.reason or "").split(":")[0]] += 1
    loaded = classifier.trained_model
    return {
        "activeModel": summary["activeModel"],
        "modelVersion": summary["modelVersion"],
        "modelExtractorVersion": loaded.extractor_version if loaded else None,
        "requestExtractorVersion": test[0].extractor_version if test else None,
        "referenceSamplesLoaded": summary["sampleCount"],
        "requests": len(test),
        "statusCounts": dict(statuses),
        "reasons": dict(reasons),
        "latencyMsIncludingBothDiskReloads": latency(timings),
        "artifactRecordedMetrics": {
            "validationSplit": "stratified by clip (train.stratified_split), so validation clips come from the same sitting as training clips",
            "validationAccuracy": loaded.metrics.get("validation", {}).get("accuracy") if loaded and loaded.metrics else None,
            "trainAccuracy": loaded.metrics.get("train", {}).get("accuracy") if loaded and loaded.metrics else None,
        },
    }


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Baseline evaluation",
        "",
        f"Generated {report['createdAt'][:10]} by `python -m sign_classifier_service.eval.baseline` (seed {report['seed']}, "
        f"git {str(report['git']['commit'])[:7]}{' dirty' if report['git']['dirty'] else ''}, Python {report['python']}, {report['machine']}). "
        "Aggregate numbers only. Production code paths and thresholds are used unchanged.",
        "",
        "## What the running app does today",
        "",
    ]
    runtime = report["productionRuntime"]
    lines += [
        f"- Active model: `{runtime['activeModel']}` `{runtime['modelVersion']}`, trained on `{runtime['modelExtractorVersion']}`.",
        f"- Live requests are stamped `{runtime['requestExtractorVersion']}`.",
        f"- Result on {runtime['requests']} current-extractor requests: {runtime['statusCounts']} (reasons: {runtime['reasons']}).",
        f"- Latency per request including the two per-request disk reloads in `app.py`: p50 {runtime['latencyMsIncludingBothDiskReloads']['p50']:.0f} ms, "
        f"p95 {runtime['latencyMsIncludingBothDiskReloads']['p95']:.0f} ms ({runtime['referenceSamplesLoaded']} reference samples re-read each time).",
        f"- The artifact's own recorded validation accuracy ({_pct(runtime['artifactRecordedMetrics']['validationAccuracy'])}) used a clip-level split, so it is not a held-out number.",
    ]

    for protocol in report["protocols"]:
        lines += ["", f"## {protocol['protocol']}", "", protocol["description"], ""]
        lines += [f"- {caveat}" for caveat in protocol["caveats"]]
        clips = protocol["clips"]
        lines += [
            "",
            f"Clips: references {sum(clips['cosineReferences'].values())}, softmax train {sum(clips['softmaxTrain'].values())}, "
            f"softmax validation {sum(clips['softmaxValidation'].values())}, test {sum(clips['test'].values())} "
            f"({protocol['groups']['test'] or 'no'} test groups). Duplicate check: {protocol['duplicates']['crossSplitFailures']} cross-split duplicates, "
            f"calibration {protocol['duplicates']['calibration'].get('status')}.",
            "",
            "| Classifier | Top-1 accuracy (95%) | Macro-F1 (95%) | Recognized | Uncertain | No sign | Accepted accuracy | False accepts | Latency p50 / p95 ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for result in protocol["classifiers"]:
            top, boot, status, prod, lat = result["topOne"], result["bootstrap"], result["statusCounts"], result["productionThresholds"], result["latencyMs"]
            lines.append(
                f"| {result['classifier']} | {_pct(top['accuracy'])} ({_pct(boot['accuracy95'][0])}-{_pct(boot['accuracy95'][1])}) "
                f"| {top['macroF1']:.3f} ({boot['macroF1_95'][0]:.3f}-{boot['macroF1_95'][1]:.3f}) "
                f"| {status['recognized']} | {status['uncertain']} | {status['no_sign']} | {_pct(prod['acceptedAccuracy'])} | {_pct(prod['falseAcceptRate'])} "
                f"| {lat['p50']:.1f} / {lat['p95']:.1f} |"
            )
        lines += ["", f"Bootstrap resamples whole {protocol['classifiers'][0]['bootstrap']['unit']}s.", ""]

        labels = protocol["labels"]
        lines += ["Per-class recall / precision (top-1):", "", "| Classifier | " + " | ".join(labels) + " |", "| --- |" + " ---: |" * len(labels)]
        for result in protocol["classifiers"]:
            cells = [f"{result['topOne']['perClass'][label]['recall']:.2f} / {result['topOne']['perClass'][label]['precision']:.2f}" for label in labels]
            lines.append(f"| {result['classifier']} | " + " | ".join(cells) + " |")

        lines += ["", "Per-class false-positive rate at production thresholds (accepted outputs only):", ""]
        lines += ["| Classifier | " + " | ".join(labels) + " |", "| --- |" + " ---: |" * len(labels)]
        for result in protocol["classifiers"]:
            rates = result["productionThresholds"]["perClassFalsePositiveRate"]
            lines.append(f"| {result['classifier']} | " + " | ".join(_pct(rates[label]) for label in labels) + " |")

        for result in protocol["classifiers"]:
            confidence = result["confidence"]
            lines += [
                "",
                f"### {result['classifier']}",
                "",
                f"Top confidence, correct: {confidence['confidenceQuantilesCorrect']}; wrong: {confidence['confidenceQuantilesWrong']}.",
                f"Top-1 minus top-2 margin, correct: {confidence['marginQuantilesCorrect']}; wrong: {confidence['marginQuantilesWrong']}.",
                "",
                "Confusion (rows true, columns predicted):",
                "",
                "| true \\ pred | " + " | ".join(labels) + " |",
                "| --- |" + " ---: |" * len(labels),
            ]
            for label in labels:
                row = result["topOne"]["confusionMatrix"][label]
                lines.append(f"| {label} | " + " | ".join(str(row.get(other, 0)) for other in labels) + " |")
            lines += [
                "",
                "Threshold sweep (margin 0.15 as in production, and margin 0): coverage = share accepted, then accuracy among accepted.",
                "",
                "| Confidence | Coverage m=0.15 | Accepted acc m=0.15 | Coverage m=0 | Accepted acc m=0 |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
            sweep = {(row["confidence"], row["margin"]): row for row in result["thresholdSweep"]}
            for confidence_value in CONFIDENCE_GRID:
                with_margin, without = sweep[(confidence_value, 0.15)], sweep[(confidence_value, 0.0)]
                lines.append(
                    f"| {confidence_value:.2f} | {_pct(with_margin['acceptanceRate'])} | {_pct(with_margin['acceptedAccuracy'])} "
                    f"| {_pct(without['acceptanceRate'])} | {_pct(without['acceptedAccuracy'])} |"
                )

    lines += [
        "",
        "## Not measurable yet",
        "",
        "- Own-camera, session-disjoint accuracy on the current extractor: the three registered sessions hold 0 clips on this machine.",
        "- Rejection of non-signs: there are no negative (non-sign) clips, so every test clip is a real sign and false accepts here are confusions between signs.",
        "- Calibration: confidences are reported raw. Neither classifier's confidence is calibrated.",
    ]
    return "\n".join(lines) + "\n"


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    asl = asl_citizen_protocol(config)
    own = own_v1_protocol(config)
    data, _ = assemble(load_experiment("plus-asl-citizen"), config)
    runtime = production_runtime_check(sorted(data.tests[ASL_TEST], key=lambda e: (e.group_key, e.source_video_id)))
    return {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "seed": config["seed"],
        "git": git_state(),
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "evalConfigSha256": sha256_file(Path(__file__).resolve().parent / "config.json"),
        "productionThresholds": {"recognizedConfidence": 0.80, "recognizedMargin": 0.15, "uncertainFloor": 0.40},
        "productionRuntime": runtime,
        "protocols": [asl, own],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 baseline for the existing sign classifiers (read-only).")
    parser.add_argument("--output-dir", type=Path, default=PHASE0_REPORTS_DIR)
    args = parser.parse_args()
    report = build_report(load_config())
    write_json(args.output_dir / "baseline-evaluation.json", report)
    (args.output_dir / "baseline-evaluation.md").write_text(render(report), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'baseline-evaluation.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
