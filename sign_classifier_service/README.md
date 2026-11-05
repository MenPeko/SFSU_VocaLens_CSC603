# Sign Classifier Service

This service powers the local-first ASL core-sign classifier for `whiteboard-reader-tts`.

## Runtime

- The FastAPI app serves `POST /classify-sign-sequence`.
- If `data/sign-models/active-model.json` points to a promoted artifact, the service uses that model first.
- If `SIGN_CLASSIFIER_MODEL_PATH` points to a trained artifact, the service uses that as the next fallback.
- If no trained artifact is present, it falls back to the legacy sample-similarity + heuristic classifier.

Default promotion pointer:

```text
data/sign-models/active-model.json
```

## Export Local Samples

Export the captured landmark sequences into a single JSONL manifest that can be uploaded to S3 for SageMaker jobs:

```bash
.venv/bin/python -m sign_classifier_service.export_dataset \
  --samples-dir data/sign-samples \
  --output data/sign-exports/core-sign-samples.jsonl
```

This writes:

- `data/sign-exports/core-sign-samples.jsonl`
- `data/sign-exports/core-sign-samples.summary.json`

## Train A Model Locally

Train a temporal landmark classifier from local samples plus any optional external JSON/JSONL manifests:

```bash
.venv/bin/python -m sign_classifier_service.train \
  --samples-dir data/sign-samples \
  --output-dir data/sign-models
```

The trainer writes:

- a versioned artifact like `data/sign-models/core-sign-20260409T120000Z.json`
- `data/sign-models/active-model.json`
- a metrics file keyed by training run id

## SageMaker Notes

The trainer is compatible with SageMaker-style paths:

- `SM_CHANNEL_TRAIN`
- `SM_MODEL_DIR`

Recommended flow:

1. Export local samples to JSONL.
2. Upload the manifest to S3.
3. Run the trainer in SageMaker against 1-2 shortlisted datasets/models first.
   Do not pass the same local samples both as `--samples-dir` and as a manifest in the same run unless you intentionally want to oversample them.
4. Download the best artifact and update `data/sign-models/active-model.json` to point to it, or copy the generated pointer file alongside the artifact.
5. Restart the classifier service or send another request; it hot-reloads the artifact.

## Evaluate Current Runtime Model

```bash
.venv/bin/python -m sign_classifier_service.evaluate \
  --samples-dir data/sign-samples \
  --model-path data/sign-models/core-sign-20260409T120000Z.json
```

The evaluator prints the active runtime metadata, overall accuracy, and a confusion matrix.

## Snapshot A Report

Write a self-contained report folder that tracks validation accuracy, full-dataset evaluation, spot checks, and grouped misclassified samples:

```bash
.venv/bin/python -m sign_classifier_service.report_results \
  --samples-dir data/sign-samples \
  --output-dir data/sign-reports \
  --spot-check data/sign-samples/yes/<sample>.json
```

There is also a wrapper script in `scripts/` so the app can snapshot reports with the same defaults every time:

```bash
bash scripts/report-model.sh --spot-check data/sign-samples/yes/<sample>.json
```

This writes:

- `data/sign-reports/<model-version>/report.md`
- `data/sign-reports/<model-version>/report.json`
- `data/sign-reports/<model-version>/full-eval.json`
- `data/sign-reports/<model-version>/misclassifications-by-pair.json`
- `data/sign-reports/<model-version>/spot-checks.json`
- `data/sign-reports/index.json`
- `data/sign-reports/index.md`
