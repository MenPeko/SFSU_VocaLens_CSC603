# Development Roadmap — Live ASL → Text → TTS & Translation

This roadmap covers the path from today's working hybrid pipeline to a **live, continuous, hands-free ASL translation experience** with on-glasses captions, text-to-speech, and multilingual output.

It is split into three sections:

1. **Phase 0 — What's already shipped.** Foundation work that is in `main` today and is the launchpad for everything that follows.
2. **Milestones M1–M7 — The main roadmap.** Sequenced delivery plan for the live experience.
3. **Cross-cutting tracks A & D — Data and fine-tuning.** The dataset acquisition and fine-tuning work that feeds M1/M3 and is deep enough to deserve its own swim lanes.

---

## Phase 0 — Foundation (Completed)

Everything in this section is already implemented and running in VocaLens. This is the baseline the rest of the roadmap builds on.

### 0.1 — Dual-pipeline app shell
- [x] **Whiteboard OCR pipeline:** camera photo → `gemini-ocr.ts` → TTS on glasses
- [x] **ASL sign translation pipeline:** webview camera → landmark capture → classifier + Gemini fallback → TTS on glasses
- [x] Both pipelines converge on the same output surfaces (glasses TTS, SSE feed, webview history)
- [x] Dual-server topology: Express/Mentra AppServer on `:3333` + Bun server on `:3334` (see `ARCHITECTURE.md`)
- [x] Cloudflare dev tunnel via `dev-start.sh` / `dev-stop.sh` / `dev-restart.sh`
- [x] Active session registry so API routes can speak on the currently connected glasses

### 0.2 — Webview capture + landmarking
- [x] React webview with MediaPipe Hands integration (`useSignLanguage.ts`)
- [x] Live hand skeleton overlay
- [x] Fingerpose gesture hints layered on top of raw landmarks
- [x] `4 x 4` chronological sign grid generation for Gemini interpretation
- [x] `SignLanguageCapture.tsx` component wiring capture → API → UI
- [x] Static MediaPipe WASM assets served from `public/`

### 0.3 — Sign recognition backend
- [x] Core-sign label set defined in `src/sign/core-signs.ts` (8 labels: `wait`, `help`, `yes`, `no`, `stop`, `again`, `thank_you`, `please`)
- [x] Python FastAPI classifier service (`sign_classifier_service/`) exposing `POST /classify-sign-sequence`
- [x] Landmark schema + sequence vectorizer (`classifier.py:vectorize_sequence`, 344-dim handcrafted feature vector over a 24-frame resampled window)
- [x] Heuristic + sample-similarity fallback classifier when no trained artifact is available
- [x] Trained linear-softmax classifier path with standardization, class weighting, and history logging (`train.py`)
- [x] Hot-reload via `data/sign-models/active-model.json` pointer — classifier picks up new artifacts without restart
- [x] TypeScript client in `sign-classifier-client.ts` with timeout + disable flag (`SIGN_CLASSIFIER_DISABLED`)

### 0.4 — Hybrid classifier-first routing
- [x] `sign-translation-resolver.ts` tries local classifier first, falls back to Gemini on low confidence or no sign
- [x] Records which recognizer handled each request (`recognizerType: "classifier" | "gemini"`)
- [x] Preserves classifier model version + training run ID in every response for lineage
- [x] Confidence thresholds (`recognizedConfidence: 0.80`, `recognizedMargin: 0.15`, `uncertainFloor: 0.40`) baked into every artifact

### 0.5 — Open-vocabulary fallback
- [x] `gemini-sign-interpreter.ts` — Gemini prompt tuned for ASL grid-of-frames interpretation
- [x] Accepts optional skeleton reference grid + gesture hints as noisy priors
- [x] Returns `NO_SIGN_FOUND` sentinel for ambiguous input
- [x] Shared Gemini plumbing with OCR (retries, model discovery, error translation) via `gemini-ocr.ts`

### 0.6 — Translation + TTS
- [x] `gemini-translate.ts` for target-language translation of interpreter output
- [x] Language selection API + settings (`/api/settings/language`)
- [x] TTS delivery to glasses via Mentra session
- [x] Captions pushed into the same SSE/UI history as whiteboard OCR output

### 0.7 — Data capture & training pipeline
- [x] Local sample capture and storage under `data/sign-samples/` (`sign-training-store.ts`)
- [x] JSONL dataset export (`sign_classifier_service/export_dataset.py`) with summary metadata
- [x] Local trainer with stratified split, class weighting, and L2 regularization (`train.py`)
- [x] Evaluation CLI with per-label precision/recall + confusion matrix (`evaluate.py`)
- [x] SageMaker-compatible paths: `SM_CHANNEL_TRAIN`, `SM_MODEL_DIR` already honored by `train.py:resolve_default_*`
- [x] Versioned artifact format: `core-sign-{timestamp}.json` + metrics sidecar file

### 0.8 — Tests & tooling
- [x] Unit tests across `src/services/__tests__/`, `src/webview/hooks/__tests__/`, `src/sign/__tests__/`
- [x] Python runtime smoke test (`sign_classifier_service/test_runtime.py`)
- [x] Background dev scripts with PID tracking (`.ngrok.pid`, `.sign-classifier.pid`)
- [x] Architecture + how-it-works + structure docs checked in (`ARCHITECTURE.md`, `HOW-IT-WORKS.md`, `STRUCTURE.md`, `TECHSTACK.md`)

### 0.9 — Current known limitations (the "why there's a roadmap" list)
- Capture is button-triggered, not continuous — user must initiate each sign
- Vocabulary is 8 signs in the classifier; everything else routes to Gemini
- Each sign is translated in isolation; no sentence assembly
- Training data is small (target 50/label) and single-signer-heavy
- No external datasets wired in; no SageMaker jobs running yet
- No segmentation layer — can't detect sign boundaries in a continuous stream
- Grammar and tense are lost (signed "ME GO STORE" → "I go store", not "I'm going to the store")

---

## Current State Snapshot

| Dimension | Today | Target |
|---|---|---|
| Trigger | Button press per sign | Continuous hands-free |
| Vocabulary | 8 core signs + Gemini fallback | Broad ASL lexicon with classifier-first routing |
| Latency | Per-sign round-trip to Gemini | <400ms classifier, <1.5s hybrid path |
| Output | Single-sign English text | Assembled sentences, TTS + translation |
| Training data | ~50 samples/label, single source | ≥500 samples/label, multi-signer, multi-source |
| Training infra | Local `train.py` runs | SageMaker base + fine-tune cadence |

---

## Main Roadmap

### Milestone 1 — Reliable Single-Sign Recognition
**Goal:** 8 core signs recognized live with >90% top-1 accuracy from the glasses/webview camera.

- [ ] Collect ≥200 samples/label across multiple signers and lighting conditions
- [ ] Lock the MediaPipe extractor version and record it in every artifact's metadata
- [ ] Move training off-laptop — finalize SageMaker loop, promote via `active-model.json` (depends on Track A + Track D)
- [ ] Add held-out per-signer evaluation; publish confusion matrix per release
- [ ] Surface classifier confidence in `SignLanguageCapture.tsx` so users get feedback on low-confidence reads

**Exit criteria:** promoted model beats heuristic fallback by ≥20 points on held-out signers.

### Milestone 2 — Continuous Sign Segmentation
**Goal:** Stop requiring a button press per sign. Detect sign boundaries from a live landmark stream.

- [ ] Streaming buffer on the glasses camera feed (`glasses-camera-stream.ts`) → rolling landmark window
- [ ] Motion/energy-based gating: detect "signing" vs rest, emit candidate segments
- [ ] Add a `silence` / `transition` class to the classifier so it can explicitly reject non-signs
- [ ] End-of-utterance detection (pause > N ms or hands-down) to flush text
- [ ] Latency budget: <400ms from end-of-sign to classifier result

**Exit criteria:** user can sign 3+ core signs in a row hands-free and see each transcribed.

### Milestone 3 — Open-Vocabulary Hybrid
**Goal:** Cover signs outside the classifier's label set by routing to Gemini, intelligently.

- [ ] Router in `sign-translation-resolver.ts`: if classifier confidence < threshold → Gemini grid (builds on existing routing)
- [ ] Cache recent segments to avoid re-sending; dedupe repeats
- [ ] Budget and cost guardrails (max Gemini calls/minute/session)
- [ ] Track which path produced each token for eval

**Exit criteria:** unknown signs still produce reasonable English within ~1.5s end-to-end.

### Milestone 4 — Sentence Assembly & Grammar
**Goal:** Turn a token stream ("ME GO STORE") into natural English ("I'm going to the store").

- [ ] Sliding-window LLM pass via `gemini-translate.ts` over the last N sign tokens + partial text
- [ ] Distinguish lexical signs from fingerspelling (fingerspell buffer → word)
- [ ] Handle ASL non-manual markers where camera allows (question inflection → "?")
- [ ] Commit/revise protocol: partial captions update, finalize on pause

**Exit criteria:** short conversational utterances read as natural English, not word salad.

### Milestone 5 — TTS & Multilingual Output
**Goal:** Speak the result and translate it, without stepping on itself.

- [ ] Streaming TTS on finalized sentences only (avoid speaking partials)
- [ ] Voice selection + rate control in webview settings
- [ ] Translation target picker (reuse `gemini-translate.ts`)
- [ ] Dual display: source English caption + translated caption on glasses
- [ ] Barge-in: new sign flushes TTS queue

**Exit criteria:** signer → English speech + target-language caption, end-to-end under ~2s.

### Milestone 6 — On-Device / Edge Latency
**Goal:** Drop dependence on round-tripping every segment to the cloud.

- [ ] Colocate classifier with capture in the app server (mostly there already)
- [ ] Evaluate quantized model for phone-side execution
- [ ] Fall back to cloud only for Gemini interpreter + translation
- [ ] Telemetry: p50/p95 latency per stage (capture → classify → sentence → TTS)

### Milestone 7 — Productization
- [ ] Signer calibration flow (quick per-user sample capture at onboarding)
- [ ] Privacy: on-device landmark extraction, no raw video leaves the glasses
- [ ] Error recovery UX: "didn't catch that, please repeat"
- [ ] Evaluation harness + regression dataset checked in (landmarks only, never raw video)
- [ ] Docs: signer guide, supported vocabulary, known limits

---

## Track A — Kaggle Data Acquisition & Ingest

**Why this is its own track:** Phase 0 ships a working pipeline but only ~50 samples per label. M1's exit criteria are not reachable without a real dataset pipeline. This track runs in parallel with M1–M2 and gates M3 scalability.

### A.1 — Dataset Survey & Shortlist

| Dataset | Format | Overlap w/ our 8 signs | Re-extract required? | License notes |
|---|---|---|---|---|
| **Google ASL Signs (Kaggle 2023, "asl-signs")** | Parquet of MediaPipe Holistic landmarks, 250 signs, ~100k clips | High: `yes`, `no`, `please`, `thankyou`, `help`, `wait`, `stop` likely present (verify strings) | Schema adapter only — landmarks already extracted, different layout | Kaggle competition rules; verify commercial use clause before shipping derived models |
| **WLASL-2000** | Video URLs + glosses, 2000 signs | High: all 8 core signs | Yes — run pinned MediaPipe over video | Academic; YouTube source clips expire; don't redistribute videos |
| **MS-ASL** | Video URLs + glosses, 1000 signs | High | Yes — video → MediaPipe | Microsoft research license; link rot |
| **How2Sign** | Continuous ASL video + transcripts | Not isolated signs | N/A for M1; relevant to M2 | CC-BY-NC |
| **Sign Language MNIST / ASL Alphabet** | Static letter images | None (alphabet only) | N/A | Useful for M4 fingerspelling only |

**Decision rule:** start with Google ASL Signs (no re-extraction, largest). Add WLASL second. Defer MS-ASL unless data is thin.

**Tasks**
- [ ] Write `docs/sign-datasets.md` with the shortlist + per-dataset label-overlap table (actual string matches, not guesses)
- [ ] Verify Google ASL Signs includes each of the 8 labels (their vocab strings may differ: `"thank you"` vs `thank_you`)
- [ ] Pull one clip per target label and hand-inspect — confirm the sign matches intended meaning, not a homophone

### A.2 — Label Taxonomy & Mapping
`load_manifest_examples()` silently drops anything outside `CORE_SIGN_LABELS` (`train.py:68`). External datasets use different label strings.

- [ ] Create `sign_classifier_service/label_map.py` with per-dataset dicts (e.g., `GOOGLE_ASL_SIGNS_TO_CORE`, `WLASL_TO_CORE`)
- [ ] Version-control the mapping alongside the dataset version string
- [ ] Unit test that every mapped label round-trips into `CORE_SIGN_LABELS`

### A.3 — Landmark Schema Adapter
This is the **silent killer.** `vectorize_sequence()` expects our `LandmarkFrameModel` layout (`leftHand`, `rightHand`, `pose.leftWrist/rightWrist`, `face.lips`). External data uses different layouts or no landmarks at all.

- [ ] Add `sign_classifier_service/ingest/` with one file per source:
  - `google_asl_signs.py` — parquet → `LandmarkFrameModel` via indexing into MediaPipe Holistic point IDs (pin exact IDs for wrists, lip contour, hand joints)
  - `wlasl.py` — video → pinned MediaPipe Holistic → `LandmarkFrameModel`
- [ ] Pin MediaPipe version in `sign_classifier_service/requirements.txt` and record it in every emitted manifest under a top-level `extractor` field
- [ ] **Reject-on-mismatch:** drop frames missing required points with a counter, do not zero-fill
- [ ] Emit converted data as JSONL matching the exact shape `load_manifest_examples()` reads (records with `label`, `landmarkSequence`, `source`)

### A.4 — Data Quality Gates
Bad external data is worse than no external data. Every manifest passes these gates before upload to S3.

- [ ] **Sanity classifier:** run the current `active-model.json` against the converted manifest; if it scores >30% overall but <5% on a specific label, that label's mapping or extraction is likely broken
- [ ] **Duration filter:** drop clips <300ms or >5s
- [ ] **Dedupe** by cosine similarity on `vectorize_sequence()` output
- [ ] **Signer ID tracking:** carry `signerId` through to the manifest for signer-disjoint eval splits
- [ ] **Audit:** hand-label 20 random converted clips per dataset per label, confirm the sign matches

### A.5 — S3 Dataset Registry
- [ ] Bucket layout: `s3://mentra-sign-datasets/{source}/{dataset_version}/{extractor_version}/manifest.jsonl`
- [ ] Per-dataset `manifest.summary.json` (label counts, signer counts, duration histogram, extractor version)
- [ ] **Versioning:** never overwrite. New extraction = new `extractor_version` directory. This is what protects you when MediaPipe bumps.
- [ ] Registry index `s3://mentra-sign-datasets/index.json` listing all available `(source, version, extractor)` tuples
- [ ] IAM: SageMaker training role gets read-only; upload is done from a separate CI role

### A.6 — Exit Criteria
- Google ASL Signs + local captures available as S3 manifests with matching extractor versions
- Sanity classifier check passes on converted Kaggle data
- Every core label has ≥500 total training samples across all sources
- Signer-disjoint validation split is reproducible from the registry
- Lineage traceable: every sample in S3 → `(source, dataset_version, extractor_version, signer_id)`

---

## Track D — Fine-Tuning & SageMaker Training Loop

**Reality check first.** The current trainer (`train.py:build_artifact`) re-initializes `weights = 0`, recomputes `feature_mean/feature_std` from the train set, and uses a handcrafted feature vector that doesn't change between runs. **There is no learned encoder to transfer.** "Fine-tuning" this architecture means three specific things:

1. **Warm-start the linear head** from a prior artifact's weights and bias
2. **Freeze the standardization basis** so prior weights remain coherent
3. **Pull toward the prior** during continued training so small in-house datasets don't wipe broad Kaggle signal

This gets real lift from the pretrain+fine-tune loop. For true feature transfer, an architectural change is required — flagged as an explicit branch point in D.5.

### D.1 — Warm-Start Support in `train.py`
All changes contained to `sign_classifier_service/train.py` and `classifier.py`.

- [ ] New `--warm-start-artifact PATH` CLI flag on `train.py:main`
- [ ] When set, `build_artifact()`:
  - Seeds `weights` and `bias` from the prior (expand/slice if label set changed — new labels get zero-initialized rows)
  - **Reuses** `featureMean` / `featureStd` from the prior instead of calling `compute_standardization()` on the fine-tune set
  - Reads prior `labels` and rejects fine-tuning if feature dimensionality doesn't match (guards against extractor version mismatch)
- [ ] Record `baseArtifact` metadata on the new artifact: `{ "baseVersion": "...", "baseTrainingRunId": "...", "fineTuneEpochs": N }`
- [ ] Default epochs drop from 180 → ~30 when warm-starting; default LR drops 0.12 → ~0.02
- [ ] **Abort** the run if prior artifact's `extractor` field differs from current — this is the mismatch killer

### D.2 — Regularization Toward Prior (EWC-lite)
Warm-start + small LR isn't enough; on small fine-tune sets the head still drifts. Add an L2 pull toward the prior weights rather than toward zero (which is what current `weight_decay` does at `train.py:215`).

- [ ] New `--prior-pull-strength FLOAT` CLI flag (default 0 cold, ~0.01 warm)
- [ ] In the inner loop at `train.py:215`, modify the weight-decay term:
  ```
  grad = grad_w[...] * scale
       + weight_decay * weights[...]
       + prior_pull * (weights[...] - prior_weights[...])
  ```
- [ ] Ablate on held-out set: `prior_pull ∈ {0, 0.005, 0.02, 0.1}` and pick per-label best
- [ ] Record chosen `priorPull` in artifact metadata

### D.3 — Three-Tier Evaluation
Every fine-tuned artifact gets evaluated on three frozen slices so domain shift is visible, not just aggregate accuracy.

- [ ] Extend `evaluate_artifact()` at `train.py:245` to accept multiple named eval sets and emit them under `metrics`
- [ ] Three slices, all reproducible from the A.5 registry:
  1. **Kaggle held-out** (signer-disjoint from any Kaggle data used in training)
  2. **In-house held-out** (signer-disjoint from `data/sign-samples/`)
  3. **Per-signer micro-eval** — for each in-house signer, accuracy when that signer is held out
- [ ] **Regression gate:** promotion blocked if either (Kaggle held-out, in-house held-out) drops >2pt vs the current `active-model.json`. Per-signer numbers are advisory.
- [ ] Emit `promotion_report.md` with an old-vs-new diff table per run

### D.4 — SageMaker Training Cadence
The fine-tuning playbook this roadmap commits to.

1. **Base run** (weekly, or on dataset change):
   ```
   train.py \
     --samples-dir s3://.../local \
     --manifest s3://.../google_asl_signs \
     --epochs 180 --learning-rate 0.12
   ```
   Produces `core-sign-base-{timestamp}`. Kaggle-heavy cold-start model.

2. **Fine-tune run** (on every in-house capture batch):
   ```
   train.py \
     --samples-dir s3://.../local \
     --warm-start-artifact core-sign-base-{...}.json \
     --epochs 30 --learning-rate 0.02 \
     --prior-pull-strength 0.01
   ```
   Produces `core-sign-ft-{timestamp}`. This is what gets promoted.

3. **Per-user calibration** (M7, separate path — not retraining):
   Compute per-label **prototype vectors** (mean of `vectorize_sequence()` across that user's samples), store alongside the artifact. At inference time, blend prototype cosine similarity into the softmax logits. Cheap, runs in the app server, adapts to one signer in under a minute of capture.

**Infrastructure tasks**
- [ ] Provision SageMaker training role + bucket access
- [ ] Minimal training container wrapping `sign_classifier_service.train`
- [ ] Job launcher script: takes `--samples-dir` (S3) + manifest list, runs `train.py` with `SM_CHANNEL_TRAIN` / `SM_MODEL_DIR`
- [ ] Metrics shipped to CloudWatch or a runs table (keep existing metrics JSON format as source of truth)
- [ ] Artifact promotion: download winner → update `data/sign-models/active-model.json` → classifier hot-reloads
- [ ] `scripts/train-base.sh` and `scripts/train-finetune.sh` encode default hyperparameters
- [ ] Automation: when `data/sign-samples/` changes by >N new samples, enqueue a fine-tune job (not a base job). Base runs stay manual until Kaggle data changes.
- [ ] Document the cadence in `sign_classifier_service/README.md`

### D.5 — Architecture Branch Point (honest caveat)
Everything above is "best fine-tuning you can do with a linear softmax on handcrafted features." The ceiling is real. If A+D delivers 10k+ samples and the linear model plateaus, the next move is architectural, not hyperparameter tuning.

- [ ] **Decision criterion:** if validation accuracy with full Kaggle + local data plateaus below ~92% after D.1–D.3, escalate
- [ ] **Option A:** small 1D temporal conv over the 24-frame resampled landmark sequence (drops `vectorize_sequence` in favor of raw standardized landmarks). Fine-tuning then becomes "freeze conv layers, retrain head on in-house data" — real transfer learning.
- [ ] **Option B:** keep `vectorize_sequence` but add a 2-layer MLP head. Cheaper, modest gain.
- [ ] **Option C:** small Transformer over landmark frames. Higher ceiling, higher cost, probably overkill until the labeled set is much larger.
- [ ] **Do not take this branch preemptively.** Linear softmax on good data is surprisingly competitive for ~10-class isolated-sign recognition and trains in seconds — keep it until the data says you can't.

### D.6 — Exit Criteria
- `train.py` supports warm-start with frozen standardization and prior-pull regularization
- A base artifact trained on Kaggle + local exists, promoted, and hot-loaded by the classifier service
- A fine-tune artifact outperforms the base artifact on **in-house held-out** by ≥3pt without losing >2pt on Kaggle held-out
- Three-slice eval is part of every artifact's `metrics` JSON
- Per-user prototype calibration spec written (implementation deferred to M7)
- Architectural escalation criterion (D.5) documented so future work knows when to spend that complexity

---

## Critical Dependencies & Risks

1. **Dataset volume and diversity** is the single biggest blocker. Everything past M1 is gated on Track A delivering real data.
2. **Landmark extractor stability** — any MediaPipe version bump invalidates trained models. Pin the version and store it in every artifact and every manifest.
3. **Glasses camera framerate / FOV** — continuous capture (M2) needs verified sustained fps from the Mentra Live camera.
4. **Gemini cost and latency** — M3 hybrid routing only pays off if the classifier is strong enough that Gemini is a fallback, not a first-line path.
5. **Segmentation is harder than classification** — budget real time for M2; do not underestimate it.
6. **Kaggle dataset licensing** — WLASL and MS-ASL in particular have restrictions. Verify before shipping any model derived from them.
7. **Linear-head ceiling** — D.5 is a known limit; have the data before deciding whether to cross it.

---

## Suggested Order of Attack

```
Track A (Data)      ──┐
                      ├──> M1 ──┐
Track D (Training)  ──┘         │
                                ├──> M2 + M3 (parallel) ──> M4 ──> M5
                                │
                                └──> M6 / M7 (hardening, after pipeline works end-to-end)
```

- **Phase 0** is already shipped — demonstrates the app shell, hybrid recognizer, training pipeline, and data capture loop all work end-to-end today.
- **Tracks A and D** are prerequisites for M1's exit criteria. They run in parallel with M1 itself because some of M1's work (confidence UI, per-signer eval surfacing) doesn't depend on data volume.
- **M2 and M3** can move in parallel once M1 is green: continuous segmentation is orthogonal to open-vocab routing.
- **M4 → M5** are sequential: you can't do streaming TTS cleanly until sentence assembly has a commit/revise protocol.
- **M6 and M7** are hardening passes. Don't start them until the end-to-end pipeline works on at least one happy path.

---

## Glossary of Key Files (for new contributors)

| File | Role |
|---|---|
| `src/sign/core-signs.ts` | Closed-set label vocabulary (8 signs) |
| `src/services/sign-translation-resolver.ts` | Hybrid classifier-first router with Gemini fallback |
| `src/services/sign-classifier-client.ts` | TS client for the Python classifier service |
| `src/services/gemini-sign-interpreter.ts` | Open-vocabulary grid-of-frames interpreter |
| `src/services/gemini-translate.ts` | Target-language translation |
| `src/services/sign-training-store.ts` | Local sample capture persistence |
| `src/webview/hooks/useSignLanguage.ts` | MediaPipe Hands + landmark streaming |
| `src/webview/components/SignLanguageCapture.tsx` | Capture UI |
| `sign_classifier_service/classifier.py` | Feature extractor, heuristic + trained runtime |
| `sign_classifier_service/train.py` | Trainer, SageMaker entry point |
| `sign_classifier_service/evaluate.py` | Offline eval CLI |
| `sign_classifier_service/export_dataset.py` | Local samples → JSONL manifest |
| `data/sign-models/active-model.json` | Promotion pointer; classifier hot-reloads this |
| `data/sign-samples/` | Captured landmark sequences, one JSON per sample |
| `data/sign-exports/` | Exported JSONL manifests ready for S3 upload |
