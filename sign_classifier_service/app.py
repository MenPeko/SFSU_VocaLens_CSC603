from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from fastapi import FastAPI

from .classifier import LandmarkSequenceClassifier
from .schemas import ClassifySequenceRequest, ClassifySequenceResponse

DATA_DIR = Path(os.environ.get("SIGN_CLASSIFIER_DATA_DIR", Path(__file__).resolve().parent.parent / "data" / "sign-samples"))
classifier = LandmarkSequenceClassifier(DATA_DIR)
app = FastAPI(title="VocaLens Sign Classifier", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Union[str, int]]:
    summary = classifier.runtime_summary()
    return {
        "status": "ok",
        "samples": len(classifier.samples),
        "dataDir": str(DATA_DIR),
        "activeModel": str(summary["activeModel"]),
        "modelLoaded": int(bool(summary["modelLoaded"])),
        "modelPath": str(summary["modelPath"]),
        "modelVersion": str(summary["modelVersion"] or ""),
        "trainingRunId": str(summary["trainingRunId"] or ""),
    }


@app.get("/metadata")
def metadata() -> dict[str, Union[str, int, bool, None]]:
    summary = classifier.runtime_summary()
    return {
        "status": "ok",
        "samples": len(classifier.samples),
        "dataDir": str(DATA_DIR),
        "activeModel": str(summary["activeModel"]),
        "modelLoaded": bool(summary["modelLoaded"]),
        "modelPath": str(summary["modelPath"]),
        "modelVersion": summary["modelVersion"],
        "trainingRunId": summary["trainingRunId"],
    }


@app.post("/classify-sign-sequence", response_model=ClassifySequenceResponse)
def classify_sign_sequence(request: ClassifySequenceRequest) -> ClassifySequenceResponse:
    classifier.reload_runtime_assets()
    return classifier.predict(request)
