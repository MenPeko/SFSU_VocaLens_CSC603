from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .classifier import LandmarkSequenceClassifier, get_default_model_path, load_training_examples
from .schemas import ClassifySequenceRequest, LandmarkFrameModel


def freeze_runtime_assets(classifier: LandmarkSequenceClassifier) -> None:
    classifier.reload_runtime_assets(force=True)
    classifier.reload_runtime_assets = lambda force=False: None  # type: ignore[method-assign]


def make_request(example, sequence_duration_ms: int | None = None) -> ClassifySequenceRequest:
    return ClassifySequenceRequest(
        frames=example.frames,
        fps=12.0,
        source=example.source if example.source in ("device-camera", "glasses-stream") else "device-camera",
        sequenceDurationMs=sequence_duration_ms if sequence_duration_ms is not None else 0,
        extractorVersion=example.extractor_version,
    )


def evaluate_examples(
    classifier: LandmarkSequenceClassifier,
    samples_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    examples = load_training_examples(samples_dir)
    confusion: dict[str, dict[str, int]] = {}
    misclassifications_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    correct = 0

    for example in examples:
        result = classifier.predict(make_request(example))
        predicted = result.label or "no_sign"
        confusion.setdefault(example.label, {})
        confusion[example.label][predicted] = confusion[example.label].get(predicted, 0) + 1

        if predicted == example.label:
            correct += 1
            continue

        pair_key = f"{example.label}->{predicted}"
        misclassifications_by_pair[pair_key].append(
            {
                "file": str(example.sample_path.relative_to(samples_dir)),
                "expected": example.label,
                "predicted": predicted,
                "status": result.status,
                "confidence": result.confidence,
                "alternatives": [
                    {"label": alternative.label, "confidence": alternative.confidence}
                    for alternative in result.alternatives
                ],
                "reason": result.reason,
            }
        )

    summary = {
        "runtime": classifier.runtime_summary(),
        "sampleCount": len(examples),
        "accuracy": round(correct / len(examples), 4) if examples else 0.0,
        "confusionMatrix": confusion,
    }
    return summary, dict(sorted(misclassifications_by_pair.items()))


def load_metrics(metrics_path: Path | None) -> dict[str, Any] | None:
    if not metrics_path or not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_active_model_pointer(pointer_path: Path) -> dict[str, Any] | None:
    if not pointer_path.exists():
        return None
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def infer_metrics_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}-metrics.json")


def normalize_model_version(summary: dict[str, Any], model_path: Path) -> str:
    runtime = summary.get("runtime", {})
    if isinstance(runtime, dict):
        version = runtime.get("modelVersion")
        if isinstance(version, str) and version.strip():
            return version.strip()
    return model_path.stem


def build_spot_checks(
    classifier: LandmarkSequenceClassifier,
    sample_paths: list[Path],
    samples_dir: Path,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    for path in sample_paths:
        resolved = path if path.is_absolute() else (Path.cwd() / path).resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        frames = [LandmarkFrameModel.model_validate(frame) for frame in payload.get("landmarkSequence", [])]
        result = classifier.predict(
            ClassifySequenceRequest(
                frames=frames,
                fps=12.0,
                source=payload.get("source", "device-camera"),
                sequenceDurationMs=int(payload.get("sequenceDurationMs") or 0),
                extractorVersion=payload["extractorVersion"],
            )
        )
        checks.append(
            {
                "file": str(resolved.relative_to(samples_dir)),
                "expected": payload.get("label"),
                "status": result.status,
                "predicted": result.label or "no_sign",
                "confidence": result.confidence,
                "alternatives": [
                    {"label": alternative.label, "confidence": alternative.confidence}
                    for alternative in result.alternatives
                ],
                "reason": result.reason,
            }
        )

    return checks


def top_validation_rows(metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not metrics:
        return []
    validation = metrics.get("validation")
    if not isinstance(validation, dict):
        return []
    per_label = validation.get("perLabel")
    if not isinstance(per_label, dict):
        return []

    rows: list[dict[str, Any]] = []
    for label, payload in per_label.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "label": label,
                "precision": payload.get("precision"),
                "recall": payload.get("recall"),
                "support": payload.get("support"),
            }
        )
    return rows


def top_confusion_pairs(confusion_matrix: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for expected, predictions in confusion_matrix.items():
        if not isinstance(predictions, dict):
            continue
        for predicted, count in predictions.items():
            if expected == predicted:
                continue
            pairs.append({"pair": f"{expected}->{predicted}", "count": count})
    return sorted(pairs, key=lambda item: (-item["count"], item["pair"]))


def build_next_batch_recommendations(metrics: dict[str, Any] | None, full_eval: dict[str, Any]) -> list[str]:
    validation_confusions = top_confusion_pairs(
        metrics.get("validation", {}).get("confusionMatrix", {}) if metrics else {}
    )
    full_eval_confusions = top_confusion_pairs(full_eval.get("confusionMatrix", {}))
    ranked_pairs = validation_confusions or full_eval_confusions

    recommendations: list[str] = []
    def normalize_pair_key(pair_key: str) -> str:
        left, _, right = pair_key.partition("->")
        if right == "no_sign":
            return pair_key
        normalized = sorted([left, right])
        return "<->".join(normalized)

    pair_map = {
        "no<->yes": "Capture more contrastive `no` vs `yes` clips with the same framing and tempo.",
        "again<->stop": "Capture more contrastive `stop` vs `again` clips, especially around the motion endpoint.",
        "please<->thank_you": "Capture more contrastive `thank_you` vs `please` clips with consistent torso framing.",
        "stop<->wait": "Review `wait` outliers that collapse into `stop`; add a small corrective batch in clean clothing.",
    }

    for pair in ranked_pairs:
        text = pair_map.get(normalize_pair_key(pair["pair"]))
        if text and text not in recommendations:
            recommendations.append(text)
        if len(recommendations) == 3:
            break

    baseline_note = "Keep most future captures in plain shirts and cleaner backgrounds, then add a smaller hard-mode batch afterward."
    if baseline_note not in recommendations:
        recommendations.append(baseline_note)

    return recommendations


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Sign Model Report: {report['modelVersion']}")
    lines.append("")
    lines.append(f"- Generated: {report['generatedAt']}")
    lines.append(f"- Model path: `{report['modelPath']}`")
    lines.append(f"- Samples evaluated: `{report['fullEvaluation']['sampleCount']}`")
    lines.append(f"- Full-eval accuracy: `{report['fullEvaluation']['accuracy']}`")

    validation = report.get("trainingMetrics", {}).get("validation", {})
    if validation:
        lines.append(f"- Validation accuracy: `{validation.get('accuracy')}`")
    promoted_at = report.get("activeModelPointer", {}).get("promotedAt")
    if promoted_at:
        lines.append(f"- Promoted at: `{promoted_at}`")

    lines.append("")
    lines.append("## Validation Per-Label")
    lines.append("")
    lines.append("| Label | Precision | Recall | Support |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in report["validationTable"]:
        lines.append(
            f"| `{row['label']}` | `{row['precision']:.4f}` | `{row['recall']:.4f}` | `{row['support']:.0f}` |"
        )

    lines.append("")
    lines.append("## Spot Checks")
    lines.append("")
    for check in report["spotChecks"]:
        lines.append(
            f"- `{check['file']}`: expected `{check['expected']}`, predicted `{check['predicted']}`, status `{check['status']}`, confidence `{check['confidence']}`"
        )

    lines.append("")
    lines.append("## Top Misclassification Pairs")
    lines.append("")
    for pair in report["topMisclassificationPairs"][:8]:
        lines.append(f"- `{pair['pair']}`: `{pair['count']}` samples")

    lines.append("")
    lines.append("## Sample Review Buckets")
    lines.append("")
    for pair, items in report["misclassificationsByPair"].items():
        lines.append(f"### `{pair}`")
        lines.append("")
        for item in items[:5]:
            lines.append(
                f"- `{item['file']}`: status `{item['status']}`, confidence `{item['confidence']}`, reason `{item['reason']}`"
            )
        lines.append("")

    lines.append("## Next Batch")
    lines.append("")
    for note in report["nextBatchRecommendations"]:
        lines.append(f"- {note}")

    return "\n".join(lines).rstrip() + "\n"


def write_history_index(history_path: Path, report_summary: dict[str, Any]) -> list[dict[str, Any]]:
    if history_path.exists():
        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
            entries = payload if isinstance(payload, list) else []
        except Exception:
            entries = []
    else:
        entries = []

    filtered = [
        entry
        for entry in entries
        if not (
            isinstance(entry, dict)
            and entry.get("modelVersion") == report_summary["modelVersion"]
            and entry.get("trainingRunId") == report_summary["trainingRunId"]
        )
    ]
    filtered.append(report_summary)
    filtered.sort(key=lambda item: (item.get("generatedAt", ""), item.get("modelVersion", "")))
    history_path.write_text(json.dumps(filtered, indent=2) + "\n", encoding="utf-8")
    return filtered


def render_history_markdown(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Sign Model Accuracy History",
        "",
        "| Model | Training Run | Validation Accuracy | Full Eval Accuracy | Samples | Report Folder |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]

    for entry in entries:
        lines.append(
            "| "
            f"`{entry.get('modelVersion', '')}` | "
            f"`{entry.get('trainingRunId', '')}` | "
            f"`{entry.get('validationAccuracy', '')}` | "
            f"`{entry.get('fullEvalAccuracy', '')}` | "
            f"`{entry.get('sampleCount', '')}` | "
            f"`{entry.get('reportDir', '')}` |"
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot sign-classifier metrics, evaluation results, and review buckets.")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sign-samples",
        help="Directory containing local labeled samples.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional trained model artifact path. Defaults to the active runtime model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sign-reports",
        help="Directory where timestamped model reports should be written.",
    )
    parser.add_argument(
        "--spot-check",
        type=Path,
        action="append",
        default=[],
        help="Optional sample JSON to run through the trained runtime model and include in the report.",
    )
    args = parser.parse_args()

    samples_dir = args.samples_dir.resolve()
    model_path = (args.model_path.resolve() if args.model_path else get_default_model_path().resolve())
    metrics_path = infer_metrics_path(model_path)
    output_dir = args.output_dir.resolve()
    active_pointer_path = model_path.parent / "active-model.json"

    classifier = LandmarkSequenceClassifier(samples_dir, model_path, prefer_active_model_pointer=False)
    freeze_runtime_assets(classifier)

    full_eval, misclassifications_by_pair = evaluate_examples(classifier, samples_dir)
    metrics = load_metrics(metrics_path)
    validation_rows = top_validation_rows(metrics)
    spot_checks = build_spot_checks(classifier, args.spot_check, samples_dir) if args.spot_check else []

    model_version = normalize_model_version(full_eval, model_path)
    training_run_id = full_eval.get("runtime", {}).get("trainingRunId")
    generated_at = datetime.now(timezone.utc).isoformat()

    report = {
        "generatedAt": generated_at,
        "modelPath": str(model_path),
        "modelVersion": model_version,
        "trainingRunId": training_run_id,
        "activeModelPointer": load_active_model_pointer(active_pointer_path) or {},
        "trainingMetrics": metrics or {},
        "validationTable": validation_rows,
        "fullEvaluation": full_eval,
        "topMisclassificationPairs": top_confusion_pairs(full_eval.get("confusionMatrix", {})),
        "misclassificationsByPair": misclassifications_by_pair,
        "spotChecks": spot_checks,
        "nextBatchRecommendations": build_next_batch_recommendations(metrics, full_eval),
    }

    report_dir = output_dir / model_version
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (report_dir / "report.md").write_text(render_markdown_report(report), encoding="utf-8")
    (report_dir / "full-eval.json").write_text(json.dumps(full_eval, indent=2) + "\n", encoding="utf-8")
    (report_dir / "misclassifications-by-pair.json").write_text(
        json.dumps(misclassifications_by_pair, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "spot-checks.json").write_text(json.dumps(spot_checks, indent=2) + "\n", encoding="utf-8")
    if metrics:
        (report_dir / "training-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    history_entries = write_history_index(
        output_dir / "index.json",
        {
            "generatedAt": generated_at,
            "modelVersion": model_version,
            "trainingRunId": training_run_id,
            "validationAccuracy": metrics.get("validation", {}).get("accuracy") if metrics else None,
            "fullEvalAccuracy": full_eval.get("accuracy"),
            "sampleCount": full_eval.get("sampleCount"),
            "reportDir": str(report_dir.relative_to(output_dir)),
        },
    )
    (output_dir / "index.md").write_text(render_history_markdown(history_entries), encoding="utf-8")

    print(
        json.dumps(
            {
                "reportDir": str(report_dir),
                "historyIndex": str(output_dir / "index.json"),
                "modelVersion": model_version,
                "validationAccuracy": metrics.get("validation", {}).get("accuracy") if metrics else None,
                "fullEvalAccuracy": full_eval.get("accuracy"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
