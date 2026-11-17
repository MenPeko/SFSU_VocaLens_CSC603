# Baseline evaluation

Generated 2026-11-17 by `python -m sign_classifier_service.eval.baseline` (seed 42, git 7c263fc dirty, Python 3.9.6, Darwin arm64). Aggregate numbers only. Production code paths and thresholds are used unchanged.

## What the running app does today

- Active model: `temporal-landmark-softmax-v1` `core-sign-20260414T210536Z`, trained on `mediapipe-holistic-js@0.5.1675471629+schema-v1`.
- Live requests are stamped `mediapipe-holistic-js@0.5.1675471629+schema-v2`.
- Result on 82 current-extractor requests: {'no_sign': 82} (reasons: {'extractor_mismatch': 82}).
- Latency per request including the two per-request disk reloads in `app.py`: p50 4516 ms, p95 5016 ms (372 reference samples re-read each time).
- The artifact's own recorded validation accuracy (85.0%) used a clip-level split, so it is not a held-out number.

## asl-citizen-signer-disjoint

ASL Citizen clips from the official train/val signers as references and training data; the fixed held-out test signers as the test set. Softmax early-stops on 20% of the training signers (whole signers).

- Not the app's camera: studio-style webcam clips from Deaf signers, extracted offline with the app's browser extractor.
- Signer-disjoint, so it does measure cross-signer generalization.

Clips: references 141, softmax train 107, softmax validation 34, test 82 (10 test groups). Duplicate check: 0 cross-split duplicates, calibration ok.

| Classifier | Top-1 accuracy (95%) | Macro-F1 (95%) | Recognized | Uncertain | No sign | Accepted accuracy | False accepts | Latency p50 / p95 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cosine-knn-heuristic (local-knn-sequence-v1) | 69.5% (61.6%-76.6%) | 0.671 (0.583-0.746) | 0 | 82 | 0 | n/a | 0.0% | 6.3 / 7.2 |
| softmax (temporal-landmark-softmax-v1) | 76.8% (68.3%-85.2%) | 0.757 (0.657-0.844) | 33 | 40 | 9 | 90.9% | 3.7% | 0.6 / 0.8 |

Bootstrap resamples whole groups.

Per-class recall / precision (top-1):

| Classifier | wait | help | yes | no | stop | again | thank_you | please |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine-knn-heuristic (local-knn-sequence-v1) | 1.00 / 0.43 | 0.73 / 1.00 | 1.00 / 0.91 | 0.89 / 1.00 | 0.09 / 0.50 | 0.44 / 0.67 | 0.50 / 0.62 | 0.92 / 0.69 |
| softmax (temporal-landmark-softmax-v1) | 0.90 / 0.90 | 0.82 / 0.90 | 1.00 / 0.77 | 0.67 / 1.00 | 0.73 / 0.62 | 0.78 / 0.78 | 0.30 / 0.75 | 0.92 / 0.65 |

Per-class false-positive rate at production thresholds (accepted outputs only):

| Classifier | wait | help | yes | no | stop | again | thank_you | please |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine-knn-heuristic (local-knn-sequence-v1) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| softmax (temporal-landmark-softmax-v1) | 0.0% | 0.0% | 1.4% | 0.0% | 2.8% | 0.0% | 0.0% | 0.0% |

### cosine-knn-heuristic (local-knn-sequence-v1)

Top confidence, correct: {'p10': 0.973, 'p50': 0.9927, 'p90': 0.9972, 'mean': 0.9887}; wrong: {'p10': 0.9613, 'p50': 0.9741, 'p90': 0.9944, 'mean': 0.9766}.
Top-1 minus top-2 margin, correct: {'p10': 0.0006, 'p50': 0.0024, 'p90': 0.007, 'mean': 0.0035}; wrong: {'p10': 0.0001, 'p50': 0.0024, 'p90': 0.0072, 'mean': 0.0032}.

Confusion (rows true, columns predicted):

| true \ pred | wait | help | yes | no | stop | again | thank_you | please |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wait | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| help | 1 | 8 | 0 | 0 | 0 | 1 | 1 | 0 |
| yes | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 |
| no | 0 | 0 | 1 | 8 | 0 | 0 | 0 | 0 |
| stop | 8 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| again | 4 | 0 | 0 | 0 | 1 | 4 | 0 | 0 |
| thank_you | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 |
| please | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 11 |

Threshold sweep (margin 0.15 as in production, and margin 0): coverage = share accepted, then accuracy among accepted.

| Confidence | Coverage m=0.15 | Accepted acc m=0.15 | Coverage m=0 | Accepted acc m=0 |
| ---: | ---: | ---: | ---: | ---: |
| 0.40 | 0.0% | n/a | 100.0% | 69.5% |
| 0.45 | 0.0% | n/a | 100.0% | 69.5% |
| 0.50 | 0.0% | n/a | 100.0% | 69.5% |
| 0.55 | 0.0% | n/a | 100.0% | 69.5% |
| 0.60 | 0.0% | n/a | 100.0% | 69.5% |
| 0.65 | 0.0% | n/a | 100.0% | 69.5% |
| 0.70 | 0.0% | n/a | 100.0% | 69.5% |
| 0.75 | 0.0% | n/a | 100.0% | 69.5% |
| 0.80 | 0.0% | n/a | 100.0% | 69.5% |
| 0.85 | 0.0% | n/a | 100.0% | 69.5% |
| 0.90 | 0.0% | n/a | 100.0% | 69.5% |
| 0.95 | 0.0% | n/a | 100.0% | 69.5% |

### softmax (temporal-landmark-softmax-v1)

Top confidence, correct: {'p10': 0.4719, 'p50': 0.778, 'p90': 0.9689, 'mean': 0.7477}; wrong: {'p10': 0.3611, 'p50': 0.5097, 'p90': 0.8216, 'mean': 0.5474}.
Top-1 minus top-2 margin, correct: {'p10': 0.1458, 'p50': 0.6184, 'p90': 0.9493, 'mean': 0.5938}; wrong: {'p10': 0.078, 'p50': 0.2966, 'p90': 0.6933, 'mean': 0.3037}.

Confusion (rows true, columns predicted):

| true \ pred | wait | help | yes | no | stop | again | thank_you | please |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wait | 9 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| help | 0 | 9 | 0 | 0 | 2 | 0 | 0 | 0 |
| yes | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 |
| no | 0 | 0 | 3 | 6 | 0 | 0 | 0 | 0 |
| stop | 1 | 1 | 0 | 0 | 8 | 1 | 0 | 0 |
| again | 0 | 0 | 0 | 0 | 2 | 7 | 0 | 0 |
| thank_you | 0 | 0 | 0 | 0 | 0 | 1 | 3 | 6 |
| please | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 11 |

Threshold sweep (margin 0.15 as in production, and margin 0): coverage = share accepted, then accuracy among accepted.

| Confidence | Coverage m=0.15 | Accepted acc m=0.15 | Coverage m=0 | Accepted acc m=0 |
| ---: | ---: | ---: | ---: | ---: |
| 0.40 | 82.9% | 82.3% | 89.0% | 79.5% |
| 0.45 | 81.7% | 82.1% | 87.8% | 79.2% |
| 0.50 | 79.3% | 83.1% | 82.9% | 82.3% |
| 0.55 | 73.2% | 86.7% | 73.2% | 86.7% |
| 0.60 | 68.3% | 91.1% | 68.3% | 91.1% |
| 0.65 | 62.2% | 94.1% | 62.2% | 94.1% |
| 0.70 | 54.9% | 93.3% | 54.9% | 93.3% |
| 0.75 | 45.1% | 91.9% | 45.1% | 91.9% |
| 0.80 | 40.2% | 90.9% | 40.2% | 90.9% |
| 0.85 | 29.3% | 95.8% | 29.3% | 95.8% |
| 0.90 | 18.3% | 93.3% | 18.3% | 93.3% |
| 0.95 | 12.2% | 100.0% | 12.2% | 100.0% |

## own-v1-time-holdout

Legacy schema-v1 own clips. Each sign was recorded in one continuous block; the first 60% of each block trains, the next 15% is validation for softmax early stopping, the last 25% is the test set. Cosine references are train plus validation.

- One signer, one sitting, so neither signer- nor session-disjoint. Read as an optimistic upper bound.
- Each sign was recorded in its own block, so sign is confounded with capture time, posture drift, and lighting.
- No 'wait' class: the schema-v1 wait clips were archived as wrong, and the 6 schema-v2 wait clips cannot be mixed in.
- schema-v1 does not match the live extractor, so these numbers do not describe the running app.

Clips: references 274, softmax train 221, softmax validation 53, test 92 (no test groups). Duplicate check: 0 cross-split duplicates, calibration no_pairs.

| Classifier | Top-1 accuracy (95%) | Macro-F1 (95%) | Recognized | Uncertain | No sign | Accepted accuracy | False accepts | Latency p50 / p95 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cosine-knn-heuristic (local-knn-sequence-v1) | 88.0% (81.5%-94.6%) | 0.882 (0.805-0.947) | 0 | 92 | 0 | n/a | 0.0% | 11.6 / 12.5 |
| softmax (temporal-landmark-softmax-v1) | 73.9% (65.2%-82.6%) | 0.706 (0.602-0.793) | 11 | 80 | 1 | 100.0% | 0.0% | 0.6 / 0.8 |

Bootstrap resamples whole clips.

Per-class recall / precision (top-1):

| Classifier | help | yes | no | stop | again | thank_you | please |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine-knn-heuristic (local-knn-sequence-v1) | 1.00 / 1.00 | 0.92 / 0.92 | 0.92 / 0.92 | 0.77 / 0.83 | 0.83 / 0.91 | 0.77 / 0.91 | 0.94 / 0.75 |
| softmax (temporal-landmark-softmax-v1) | 0.69 / 1.00 | 1.00 / 0.57 | 0.31 / 1.00 | 0.92 / 0.75 | 1.00 / 0.92 | 0.23 / 1.00 | 1.00 / 0.62 |

Per-class false-positive rate at production thresholds (accepted outputs only):

| Classifier | help | yes | no | stop | again | thank_you | please |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosine-knn-heuristic (local-knn-sequence-v1) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| softmax (temporal-landmark-softmax-v1) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### cosine-knn-heuristic (local-knn-sequence-v1)

Top confidence, correct: {'p10': 0.9763, 'p50': 0.9901, 'p90': 0.9969, 'mean': 0.9874}; wrong: {'p10': 0.943, 'p50': 0.9793, 'p90': 0.9938, 'mean': 0.9733}.
Top-1 minus top-2 margin, correct: {'p10': 0.0007, 'p50': 0.0034, 'p90': 0.0102, 'mean': 0.0045}; wrong: {'p10': 0.0004, 'p50': 0.001, 'p90': 0.0047, 'mean': 0.0017}.

Confusion (rows true, columns predicted):

| true \ pred | help | yes | no | stop | again | thank_you | please |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| help | 13 | 0 | 0 | 0 | 0 | 0 | 0 |
| yes | 0 | 11 | 1 | 0 | 0 | 0 | 0 |
| no | 0 | 1 | 12 | 0 | 0 | 0 | 0 |
| stop | 0 | 0 | 0 | 10 | 1 | 0 | 2 |
| again | 0 | 0 | 0 | 2 | 10 | 0 | 0 |
| thank_you | 0 | 0 | 0 | 0 | 0 | 10 | 3 |
| please | 0 | 0 | 0 | 0 | 0 | 1 | 15 |

Threshold sweep (margin 0.15 as in production, and margin 0): coverage = share accepted, then accuracy among accepted.

| Confidence | Coverage m=0.15 | Accepted acc m=0.15 | Coverage m=0 | Accepted acc m=0 |
| ---: | ---: | ---: | ---: | ---: |
| 0.40 | 0.0% | n/a | 100.0% | 88.0% |
| 0.45 | 0.0% | n/a | 100.0% | 88.0% |
| 0.50 | 0.0% | n/a | 100.0% | 88.0% |
| 0.55 | 0.0% | n/a | 100.0% | 88.0% |
| 0.60 | 0.0% | n/a | 100.0% | 88.0% |
| 0.65 | 0.0% | n/a | 100.0% | 88.0% |
| 0.70 | 0.0% | n/a | 100.0% | 88.0% |
| 0.75 | 0.0% | n/a | 100.0% | 88.0% |
| 0.80 | 0.0% | n/a | 100.0% | 88.0% |
| 0.85 | 0.0% | n/a | 100.0% | 88.0% |
| 0.90 | 0.0% | n/a | 100.0% | 88.0% |
| 0.95 | 0.0% | n/a | 96.7% | 89.9% |

### softmax (temporal-landmark-softmax-v1)

Top confidence, correct: {'p10': 0.5046, 'p50': 0.6818, 'p90': 0.8711, 'mean': 0.6765}; wrong: {'p10': 0.497, 'p50': 0.5424, 'p90': 0.707, 'mean': 0.565}.
Top-1 minus top-2 margin, correct: {'p10': 0.0872, 'p50': 0.4079, 'p90': 0.7667, 'mean': 0.4163}; wrong: {'p10': 0.0157, 'p50': 0.1777, 'p90': 0.4415, 'mean': 0.191}.

Confusion (rows true, columns predicted):

| true \ pred | help | yes | no | stop | again | thank_you | please |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| help | 9 | 0 | 0 | 4 | 0 | 0 | 0 |
| yes | 0 | 12 | 0 | 0 | 0 | 0 | 0 |
| no | 0 | 9 | 4 | 0 | 0 | 0 | 0 |
| stop | 0 | 0 | 0 | 12 | 1 | 0 | 0 |
| again | 0 | 0 | 0 | 0 | 12 | 0 | 0 |
| thank_you | 0 | 0 | 0 | 0 | 0 | 3 | 10 |
| please | 0 | 0 | 0 | 0 | 0 | 0 | 16 |

Threshold sweep (margin 0.15 as in production, and margin 0): coverage = share accepted, then accuracy among accepted.

| Confidence | Coverage m=0.15 | Accepted acc m=0.15 | Coverage m=0 | Accepted acc m=0 |
| ---: | ---: | ---: | ---: | ---: |
| 0.40 | 75.0% | 82.6% | 98.9% | 73.6% |
| 0.45 | 73.9% | 83.8% | 97.8% | 74.4% |
| 0.50 | 71.7% | 83.3% | 87.0% | 76.2% |
| 0.55 | 69.6% | 85.9% | 69.6% | 85.9% |
| 0.60 | 59.8% | 89.1% | 59.8% | 89.1% |
| 0.65 | 48.9% | 91.1% | 48.9% | 91.1% |
| 0.70 | 33.7% | 90.3% | 33.7% | 90.3% |
| 0.75 | 20.6% | 89.5% | 20.6% | 89.5% |
| 0.80 | 12.0% | 100.0% | 12.0% | 100.0% |
| 0.85 | 9.8% | 100.0% | 9.8% | 100.0% |
| 0.90 | 3.3% | 100.0% | 3.3% | 100.0% |
| 0.95 | 1.1% | 100.0% | 1.1% | 100.0% |

## Not measurable yet

- Own-camera, session-disjoint accuracy on the current extractor: the three registered sessions hold 0 clips on this machine.
- Rejection of non-signs: there are no negative (non-sign) clips, so every test clip is a real sign and false accepts here are confusions between signs.
- Calibration: confidences are reported raw. Neither classifier's confidence is calibrated.
