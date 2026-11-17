# Dataset audit

Generated 2026-11-17 by `python -m sign_classifier_service.eval.audit`. Aggregate numbers only.
Current extractor: `mediapipe-holistic-js@0.5.1675471629+schema-v2`.

## Sources

| Source | Clips | Signers | Sessions | Days | Extractor versions |
| --- | ---: | ---: | ---: | ---: | --- |
| own-flat | 372 | n/a | n/a | 4 | `schema-v1` 366, `schema-v2` 6 |
| own-sessions | 0 | n/a | n/a | n/a | none |
| own-archive | 50 | n/a | n/a | 1 | `schema-v1` 50 |
| asl-citizen (mediapipe-holistic-js@0.5.1675471629+schema-v2) | 230 | 43 | n/a | n/a | `schema-v2` 230 |

## Clips per sign

| Source | wait | help | yes | no | stop | again | thank_you | please | Largest/smallest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| own-flat | 6 | 51 | 50 | 51 | 51 | 50 | 51 | 62 | 10.333 |
| own-archive | 50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 |
| asl-citizen (mediapipe-holistic-js@0.5.1675471629+schema-v2) | 28 | 31 | 28 | 27 | 31 | 25 | 29 | 31 | 1.240 |

## Sequence length and landmarks

| Source | Frames p50 (min-max) | Duration ms p50 | Effective fps p50 | No hand | Both hands | Arm joint missing | Lips missing | Clips < 6 frames | Exact duplicates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| own-flat | 18 (4-48) | 1313.500 | 12.825 | 0.000 | 0.306 | 0.000 | 0.018 | 1 | 0 |
| own-archive | 27 (9-48) | 1686.000 | 15.640 | 0.000 | 0.836 | 0.000 | 0.147 | 0 | 0 |
| asl-citizen (mediapipe-holistic-js@0.5.1675471629+schema-v2) | 25 (7-48) | 1041.000 | 23.992 | 0.000 | 0.345 | 0.000 | 0.000 | 0 | 0 |

Landmark layout per frame (from the stored samples):

- own-flat: pose keys 7, hand point counts ['21'], face subsets: leftEyebrow 0/5, rightEyebrow 0/5, leftEye 0/6, rightEye 0/6, lips 0/12.
- own-archive: pose keys 7, hand point counts ['21'], face subsets: leftEyebrow 0/5, rightEyebrow 0/5, leftEye 0/6, rightEye 0/6, lips 0/12.
- asl-citizen (mediapipe-holistic-js@0.5.1675471629+schema-v2): pose keys 7, hand point counts ['21'], face subsets: leftEyebrow 0/5, rightEyebrow 0/5, leftEye 0/6, rightEye 0/6, lips 0/12.

## Handedness (share of clips where the right hand is seen in more frames)

| Source | wait | help | yes | no | stop | again | thank_you | please |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| own-flat | 1.000 | 0.818 | 1.000 | 0.961 | 0.489 | 0.938 | 1.000 | 0.984 |
| own-archive | 0.372 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| asl-citizen (mediapipe-holistic-js@0.5.1675471629+schema-v2) | 0.593 | 0.710 | 0.714 | 0.741 | 0.679 | 0.591 | 0.724 | 0.742 |

## Own recordings: capture blocks (flat layout)

| Sign | Clips | Day | Minutes |
| --- | ---: | --- | ---: |
| help | 51 | 2026-04-14 | 6.0 |
| yes | 50 | 2026-04-14 | 5.8 |
| no | 51 | 2026-04-14 | 5.9 |
| stop | 51 | 2026-04-14 | 4.9 |
| again | 50 | 2026-04-14 | 4.6 |
| thank_you | 51 | 2026-04-14 | 5.9 |
| please | 50 | 2026-04-14 | 4.9 |
| please | 10 | 2026-04-15 | 1.1 |
| please | 2 | 2026-04-15 | 0.1 |
| wait | 1 | 2026-04-21 | 0.0 |
| wait | 5 | 2026-04-22 | 2.3 |

## Sessions (held-out harness layout)

| Role | Clips |
| --- | ---: |
| train | 0 |
| validation | 0 |
| test | 0 |

## Raw video

- Own recordings: 0 video files. Stored media is 4x4 RGB grid JPEG and skeleton grid per clip (16 thumbnails), not video.
- ASL Citizen: 83399 video files on disk; 230 extracted clips have their source video.
- ASL Citizen clips by official split: {"test.csv": 89, "train.csv": 110, "val.csv": 31}.

## Trainable on the current extractor

- ownFlat: 6
- ownSessions: 0
- aslCitizen: 230
