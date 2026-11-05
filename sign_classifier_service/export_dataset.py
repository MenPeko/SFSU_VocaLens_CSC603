from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .classifier import CORE_SIGN_LABELS, load_training_examples


def export_samples(samples_dir: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = load_training_examples(samples_dir)
    counts = Counter(example.label for example in examples)

    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = json.loads(example.sample_path.read_text(encoding="utf-8"))
            record = {
                "label": example.label,
                "source": example.source,
                "samplePath": str(example.sample_path),
                "sequenceDurationMs": payload.get("sequenceDurationMs", 0),
                "sequenceFrameCount": payload.get("sequenceFrameCount", len(example.frames)),
                "frameTimestamps": payload.get("frameTimestamps", []),
                "landmarkSequence": payload.get("landmarkSequence", []),
            }
            handle.write(json.dumps(record))
            handle.write("\n")

    summary = {
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "samplesDir": str(samples_dir),
        "outputPath": str(output_path),
        "sampleCount": len(examples),
        "labels": {label: counts.get(label, 0) for label in CORE_SIGN_LABELS},
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export local ASL training samples to JSONL for S3/SageMaker upload.")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sign-samples",
        help="Directory containing label-organized local training samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sign-exports" / "core-sign-samples.jsonl",
        help="Destination JSONL path.",
    )
    args = parser.parse_args()

    summary = export_samples(args.samples_dir, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
