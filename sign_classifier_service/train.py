from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .classifier import (
    ACTIVE_MODEL_POINTER_PATH,
    APP_ROOT,
    CORE_SIGN_LABELS,
    TRAINED_MODEL_FAMILY,
    TrainingExample,
    load_training_examples,
    vectorize_sequence,
)
from .schemas import LandmarkFrameModel


@dataclass
class DatasetSplit:
    train: list[TrainingExample]
    validation: list[TrainingExample]


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _std(values: list[float], mean: float) -> float:
    if not values:
        return 1.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) or 1.0


def load_manifest_examples(manifest_paths: list[Path]) -> list[TrainingExample]:
    examples: list[TrainingExample] = []

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue

        if manifest_path.suffix == ".jsonl":
            raw_records = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw_records = payload.get("samples", [])
            elif isinstance(payload, list):
                raw_records = payload
            else:
                raw_records = []

        for index, record in enumerate(raw_records):
            label = record.get("label")
            if label not in CORE_SIGN_LABELS:
                continue
            frames_payload = record.get("landmarkSequence") or record.get("frames") or []
            try:
                frames = [LandmarkFrameModel.model_validate(frame) for frame in frames_payload]
            except Exception:
                continue
            if not frames:
                continue

            extractor_value = record.get("extractorVersion")
            if not isinstance(extractor_value, str) or not extractor_value.strip():
                print(
                    f"WARNING: skipping record {index} in {manifest_path.name} — missing extractorVersion"
                )
                continue

            examples.append(
                TrainingExample(
                    label=label,
                    frames=frames,
                    source=str(record.get("source", "external")),
                    sample_path=Path(f"{manifest_path.name}#{index}"),
                    vector=vectorize_sequence(frames),
                    extractor_version=extractor_value.strip(),
                )
            )

    return examples


def stratified_split(samples: list[TrainingExample], validation_ratio: float, seed: int) -> DatasetSplit:
    rng = random.Random(seed)
    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    grouped: dict[str, list[TrainingExample]] = defaultdict(list)

    for sample in samples:
        grouped[sample.label].append(sample)

    for label in CORE_SIGN_LABELS:
        group = grouped.get(label, [])
        rng.shuffle(group)
        if not group:
            continue

        validation_count = max(1, int(round(len(group) * validation_ratio))) if len(group) > 2 else 1
        validation_count = min(validation_count, max(1, len(group) - 1))
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])

    return DatasetSplit(train=train, validation=validation)


def compute_standardization(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    if not vectors:
        return [], []

    feature_count = len(vectors[0])
    means: list[float] = []
    stds: list[float] = []
    for feature_index in range(feature_count):
        values = [vector[feature_index] for vector in vectors]
        mean = _mean(values)
        means.append(mean)
        stds.append(_std(values, mean))
    return means, stds


def standardize_vector(vector: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [
        (value - mean) / std if abs(std) > 1e-6 else value - mean
        for value, mean, std in zip(vector, means, stds)
    ]


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_values = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def maybe_print_imbalance_warning(label_counts: Counter[str]) -> None:
    positive_counts = [count for count in label_counts.values() if count > 0]
    if not positive_counts:
        return

    max_count = max(positive_counts)
    min_count = min(positive_counts)
    if min_count <= 0 or max_count / min_count <= 2.0:
        return

    print("WARNING: label imbalance detected")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} samples")
    print("Consider collecting more samples for underrepresented labels before training.")


def train_softmax_classifier(
    train_examples: list[TrainingExample],
    labels: list[str],
    feature_mean: list[float],
    feature_std: list[float],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[list[float]], list[float], list[dict[str, float]]]:
    feature_count = len(feature_mean)
    weights = [[0.0 for _ in range(feature_count)] for _ in labels]
    bias = [0.0 for _ in labels]
    label_to_index = {label: index for index, label in enumerate(labels)}
    label_counts = Counter(example.label for example in train_examples)
    maybe_print_imbalance_warning(label_counts)
    total_examples = max(1, len(train_examples))
    class_weights = {
        label: total_examples / (max(1, len(labels)) * count)
        for label, count in label_counts.items()
    }

    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        grad_w = [[0.0 for _ in range(feature_count)] for _ in labels]
        grad_b = [0.0 for _ in labels]
        total_loss = 0.0
        total_weight = 0.0

        for example in train_examples:
            x = standardize_vector(example.vector, feature_mean, feature_std)
            logits = [
                bias[class_index] + sum(weight * value for weight, value in zip(weights[class_index], x))
                for class_index in range(len(labels))
            ]
            probabilities = softmax(logits)
            target_index = label_to_index[example.label]
            sample_weight = class_weights.get(example.label, 1.0)
            total_loss -= math.log(max(probabilities[target_index], 1e-9)) * sample_weight
            total_weight += sample_weight

            for class_index in range(len(labels)):
                error = (probabilities[class_index] - (1.0 if class_index == target_index else 0.0)) * sample_weight
                grad_b[class_index] += error
                for feature_index, value in enumerate(x):
                    grad_w[class_index][feature_index] += error * value

        if total_weight <= 0:
            break

        scale = 1.0 / total_weight
        current_lr = learning_rate * (0.96 ** epoch)
        for class_index in range(len(labels)):
            grad_b[class_index] *= scale
            bias[class_index] -= current_lr * grad_b[class_index]
            for feature_index in range(feature_count):
                grad = grad_w[class_index][feature_index] * scale + weight_decay * weights[class_index][feature_index]
                weights[class_index][feature_index] -= current_lr * grad

        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total_loss / total_weight,
                "learningRate": current_lr,
            }
        )

    return weights, bias, history


def predict_from_artifact(artifact: dict[str, Any], example: TrainingExample) -> tuple[str, float]:
    means = artifact["featureMean"]
    stds = artifact["featureStd"]
    weights = artifact["weights"]
    bias = artifact["bias"]
    labels = artifact["labels"]
    x = standardize_vector(example.vector, means, stds)
    logits = [
        bias[class_index] + sum(weight * value for weight, value in zip(weights[class_index], x))
        for class_index in range(len(labels))
    ]
    probabilities = softmax(logits)
    best_index = max(range(len(labels)), key=lambda index: probabilities[index])
    return labels[best_index], probabilities[best_index]


def evaluate_artifact(artifact: dict[str, Any], examples: list[TrainingExample]) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0

    for example in examples:
        predicted_label, confidence = predict_from_artifact(artifact, example)
        confusion[example.label][predicted_label] += 1
        if predicted_label == example.label:
            correct += 1
        _ = confidence

    total = len(examples)
    per_label: dict[str, dict[str, float]] = {}
    for label in CORE_SIGN_LABELS:
        true_positive = confusion[label][label]
        false_negative = sum(confusion[label].values()) - true_positive
        false_positive = sum(confusion[other][label] for other in CORE_SIGN_LABELS if other != label)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "support": float(sum(confusion[label].values())),
        }

    return {
        "sampleCount": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "confusionMatrix": {label: dict(confusion[label]) for label in CORE_SIGN_LABELS},
        "perLabel": per_label,
    }


def resolve_single_extractor_version(examples: list[TrainingExample]) -> str:
    """Return the single extractor version shared by every example.

    Raises ``ValueError`` if any example is missing the field or if more than
    one distinct version is present. This is the M1 "silent killer" guard:
    training on a mixed extractor set produces a model whose feature space
    quietly disagrees with the runtime, and the symptom is bad predictions
    with no obvious cause.
    """

    versions = {example.extractor_version for example in examples}
    versions.discard("")
    if not versions:
        raise ValueError(
            "No training examples carry an extractorVersion. Capture fresh "
            "samples with the current pipeline and retry."
        )
    if len(versions) > 1:
        summary = ", ".join(sorted(versions))
        raise ValueError(
            "Training set contains multiple extractor versions — refusing to "
            f"train a mixed-lineage model. Found: {summary}. Re-extract to a "
            "single version before retraining."
        )
    return next(iter(versions))


def build_artifact(
    train_examples: list[TrainingExample],
    validation_examples: list[TrainingExample],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    model_version: str,
    training_run_id: str,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    labels = [label for label in CORE_SIGN_LABELS if any(example.label == label for example in train_examples)]
    if len(labels) < 2:
        raise ValueError("Need at least two labels with training samples to train a model")

    extractor_version = resolve_single_extractor_version(train_examples + validation_examples)

    train_vectors = [example.vector for example in train_examples]
    feature_mean, feature_std = compute_standardization(train_vectors)
    weights, bias, history = train_softmax_classifier(
        train_examples=train_examples,
        labels=labels,
        feature_mean=feature_mean,
        feature_std=feature_std,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    artifact = {
        "formatVersion": 1,
        "modelFamily": TRAINED_MODEL_FAMILY,
        "modelVersion": model_version,
        "trainingRunId": training_run_id,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "extractor": extractor_version,
        "labels": labels,
        "featureMean": feature_mean,
        "featureStd": feature_std,
        "weights": weights,
        "bias": bias,
        "thresholds": {
            "recognizedConfidence": 0.80,
            "recognizedMargin": 0.15,
            "uncertainFloor": 0.40,
        },
        "dataset": {
            "trainSamples": len(train_examples),
            "validationSamples": len(validation_examples),
            "labelCounts": dict(Counter(example.label for example in train_examples)),
        },
    }
    artifact["metrics"] = {
        "train": evaluate_artifact(artifact, train_examples),
        "validation": evaluate_artifact(artifact, validation_examples),
        "history": history,
    }
    return artifact, history


def to_app_relative_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(APP_ROOT).as_posix()
    except ValueError:
        return os.path.relpath(resolved_path, APP_ROOT).replace(os.sep, "/")


def write_artifacts(artifact: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_version = artifact["modelVersion"]
    training_run_id = artifact["trainingRunId"]
    versioned_path = output_dir / f"{model_version}.json"
    active_model_path = output_dir / ACTIVE_MODEL_POINTER_PATH.name
    metrics_path = output_dir / f"{training_run_id}-metrics.json"

    versioned_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(artifact.get("metrics", {}), indent=2), encoding="utf-8")
    active_model_payload = {
        "artifactPath": to_app_relative_path(versioned_path),
        "promotedAt": datetime.now(timezone.utc).isoformat(),
        "trainingRunId": training_run_id,
        "notes": "",
    }
    active_model_path.write_text(json.dumps(active_model_payload, indent=2), encoding="utf-8")

    return {
        "versionedModelPath": str(versioned_path),
        "activeModelPath": str(active_model_path),
        "metricsPath": str(metrics_path),
    }


def resolve_default_output_dir() -> Path:
    if os.environ.get("SM_MODEL_DIR"):
        return Path(os.environ["SM_MODEL_DIR"])
    return Path(__file__).resolve().parent.parent / "data" / "sign-models"


def resolve_default_training_inputs() -> tuple[Path, list[Path]]:
    local_samples_dir = Path(__file__).resolve().parent.parent / "data" / "sign-samples"
    manifest_paths: list[Path] = []
    if os.environ.get("SM_CHANNEL_TRAIN"):
        channel_dir = Path(os.environ["SM_CHANNEL_TRAIN"])
        for pattern in ("*.jsonl", "*.json"):
            manifest_paths.extend(sorted(channel_dir.glob(pattern)))
    return local_samples_dir, manifest_paths


def main() -> None:
    default_samples_dir, default_manifests = resolve_default_training_inputs()
    parser = argparse.ArgumentParser(description="Train a core-sign temporal classifier artifact for local or SageMaker use.")
    parser.add_argument("--samples-dir", type=Path, default=default_samples_dir, help="Local captured sample directory.")
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="*",
        default=default_manifests,
        help="Optional external JSON/JSONL manifests to merge with local samples.",
    )
    parser.add_argument("--output-dir", type=Path, default=resolve_default_output_dir(), help="Directory for model artifacts.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Per-label validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting.")
    parser.add_argument("--epochs", type=int, default=180, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.12, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="L2 regularization strength.")
    parser.add_argument("--model-version", type=str, default="", help="Optional explicit model version name.")
    parser.add_argument("--training-run-id", type=str, default="", help="Optional explicit training run id.")
    args = parser.parse_args()

    local_examples = load_training_examples(args.samples_dir)
    external_examples = load_manifest_examples(list(args.manifest))
    all_examples = local_examples + external_examples

    if not all_examples:
        raise SystemExit("No training examples found. Capture local samples or provide --manifest data first.")

    split = stratified_split(all_examples, validation_ratio=args.validation_ratio, seed=args.seed)
    if not split.train or not split.validation:
        raise SystemExit("Could not build both train and validation splits. Add more labeled samples per sign.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    training_run_id = args.training_run_id or f"core-sign-{timestamp}"
    model_version = args.model_version or f"core-sign-{timestamp}"

    artifact, _ = build_artifact(
        train_examples=split.train,
        validation_examples=split.validation,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_version=model_version,
        training_run_id=training_run_id,
    )
    paths = write_artifacts(artifact, args.output_dir)

    summary = {
        "trainingRunId": training_run_id,
        "modelVersion": model_version,
        "extractor": artifact["extractor"],
        "outputDir": str(args.output_dir),
        "paths": paths,
        "dataset": artifact["dataset"],
        "validation": artifact["metrics"]["validation"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
