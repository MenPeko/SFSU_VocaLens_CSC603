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

## Held-Out Evaluation (baseline vs. public data)

The plan is in `docs/SIGN_EVAL_PLAN.md`. Code lives in `sign_classifier_service/eval/`. Only schema-v2 samples recorded into a session are used.

1. Create a session before recording, then start the app with that session active:

   ```bash
   .venv/bin/python -m sign_classifier_service.eval.sessions new --role train --signer owner --lighting desk-daylight --camera iphone13
   SIGN_TRAINING_SESSION=<sessionId> bun run dev
   ```

   Test sessions use `--role test`, and their IDs must also be added to `sign_classifier_service/eval/test_sessions.json`. `sessions list` prints every session with its role and clip counts.

   Every run also needs at least one validation session (`--role validation --signer owner`). Its clips are never trained on or tested on. They drive early stopping (validation loss, patience set in `eval/config.json`), and the runner keeps the best epoch's weights. A validation session must not share both day and lighting with any test session. With no validation session, the runner stops; it never falls back to the test sets. Test sets are scored once, after training finishes.

2. ASL Citizen (optional, after unpacking under `data/asl-citizen/`):

   ```bash
   .venv/bin/python -m sign_classifier_service.eval.asl_citizen glosses        # review gloss variants, then edit eval/gloss_map.json
   .venv/bin/python -m sign_classifier_service.eval.asl_citizen filter         # writes data/sign-eval/asl-citizen-filtered.csv
   .venv/bin/python -m sign_classifier_service.eval.asl_citizen select-test-signers
   bun run sign:eval-extract                                                  # headless Chromium, same extractor as the app
   ```

3. Run experiments and refresh the summary:

   ```bash
   bun run sign:eval-run -- --experiment baseline
   bun run sign:eval-run -- --experiment plus-asl-citizen
   bun run sign:eval-summarize
   ```

Each run writes `data/sign-reports/eval/<runId>/` (gitignored) and a model under `models/sign/` (gitignored). It never changes `data/sign-models/active-model.json`. Any leak check failure aborts the run before training.

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
