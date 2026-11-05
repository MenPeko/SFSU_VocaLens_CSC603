from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classifier import LandmarkSequenceClassifier, get_default_model_path, load_training_examples
from .schemas import ClassifySequenceRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current sign classifier against local samples.")
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
        help="Optional trained model artifact path.",
    )
    args = parser.parse_args()

    classifier = LandmarkSequenceClassifier(
        args.samples_dir,
        args.model_path or get_default_model_path(),
        prefer_active_model_pointer=args.model_path is None,
    )
    # The evaluator should benchmark one fixed runtime snapshot. Avoid
    # reloading the entire sample + model set on every single prediction,
    # which turns a small local eval into hundreds of redundant filesystem
    # scans.
    classifier.reload_runtime_assets(force=True)
    classifier.reload_runtime_assets = lambda force=False: None  # type: ignore[method-assign]

    examples = load_training_examples(args.samples_dir)
    confusion: dict[str, dict[str, int]] = {}
    correct = 0

    for example in examples:
        request = ClassifySequenceRequest(
            frames=example.frames,
            fps=12.0,
            source=example.source if example.source in ("device-camera", "glasses-stream") else "device-camera",
            sequenceDurationMs=0,
            extractorVersion=example.extractor_version,
        )
        result = classifier.predict(request)
        predicted = result.label or "no_sign"
        confusion.setdefault(example.label, {})
        confusion[example.label][predicted] = confusion[example.label].get(predicted, 0) + 1
        if predicted == example.label:
            correct += 1

    total = len(examples)
    summary = {
        "runtime": classifier.runtime_summary(),
        "sampleCount": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "confusionMatrix": confusion,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
