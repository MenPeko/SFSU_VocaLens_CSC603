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

   Held-out signers come only from the official `test.csv`. Training clips come only from `train.csv` and `val.csv`. `filter` records each clip's source CSV, and a run fails if the filtered CSV is missing when ASL Citizen is a training source.

   The extractor prints one line per clip: decoded or not, frames kept, and left-, right-, and both-hand frame counts. At the end it prints the same table again. It also saves one skeleton image of the first clip to `data/sign-reports/asl-citizen-smoke-<label>-skeleton.png` (gitignored) as a visual check.

3. Run experiments and refresh the summary:

   ```bash
   bun run sign:eval-run -- --experiment baseline
   bun run sign:eval-run -- --experiment plus-asl-citizen
   bun run sign:eval-summarize
   ```

Each run writes `data/sign-reports/eval/<runId>/` (gitignored) and a model under `models/sign/` (gitignored). It never changes `data/sign-models/active-model.json`. Any leak check failure aborts the run before training.

## Dataset Audit And Baseline

Two read-only commands describe the data and score the classifiers that already exist. Neither one trains a model for the app or changes the active model.

```bash
.venv/bin/python -m sign_classifier_service.eval.audit
.venv/bin/python -m sign_classifier_service.eval.baseline
```

- `eval.audit` scans every landmark clip on disk: the flat `data/sign-samples/` layout, sessions, the archive, and extracted ASL Citizen landmarks. It reports clips per sign, extractor versions, sequence length, missing landmarks, handedness, and exact duplicates. It writes `reports/dataset-audit.json` and `reports/dataset-audit.md`.
- `eval.baseline` runs both runtime classifier paths through the production `predict` and its thresholds: the cosine kNN plus heuristics, and the softmax artifact. It scores them on two protocols:
  - `asl-citizen-signer-disjoint`: ASL Citizen train/val signers against the held-out test signers, assembled and leak-checked by `eval.run`.
  - `own-v1-time-holdout`: the legacy schema-v1 own clips, split by capture time within each sign. This is one signer in one sitting, so treat it as an optimistic upper bound.

  It reports accuracy and macro-F1 with group bootstrap intervals, per-sign recall and precision, confusion matrices, a confidence and margin threshold sweep, and latency. It also scores the active model on current live requests. It writes `reports/baseline-evaluation.json` and `reports/baseline-evaluation.md`.

`reports/` is committed and holds aggregate numbers only, with no clip names, signer IDs, or landmarks. Rerun both commands after the data changes.

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
