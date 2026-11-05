"""Landmark extractor version lock (Python side).

This module intentionally mirrors ``src/sign/extractor-version.ts``. The
canonical value lives in both files because TypeScript and Python cannot share
imports directly — the two files must stay in sync, and
``test_runtime.py`` verifies that they do.

See ``src/sign/extractor-version.ts`` for the full rationale. Summary:

    mediapipe-holistic-js@<package-version>+schema-v<schema-revision>

    * ``@<package-version>`` must equal the pinned ``@mediapipe/holistic``
      version in ``package.json``.
    * ``+schema-v<n>`` tracks *our* emitted landmark subset (face indices,
      hand/pose subsets, configured holistic model complexity) so that we can
      bump it independently of MediaPipe.

Python-side validation never produces landmarks — the classifier service only
consumes samples that the JS extractor has already stamped — so this constant
acts as the "expected" value for:

* unit-test cross-check against the TypeScript constant
* informational display when the service starts (so the pinned extractor shows
  up in logs)

The actual equality check at training time and serving time compares the
stamp carried on each sample / request / artifact against the stamp on the
trained model, so multi-extractor deployments are still possible in principle
— they just have to be explicit.
"""

from __future__ import annotations

MEDIAPIPE_EXTRACTOR_VERSION = "mediapipe-holistic-js@0.5.1675471629+schema-v2"
"""Current pinned extractor identifier — must match the TS constant."""
