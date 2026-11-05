"""Coarse hand gesture classifier (replaces fingerpose).

The original project used ``fingerpose`` via the JS bundle. We don't want a
JS dependency, so this file re-implements the same style of curl-table rule
system directly in Python. Rules live in ``config.GESTURE_RULES``.

The matching policy mirrors fingerpose:
  * For each rule, compute a score in [0,1] equal to the fraction of
    fingers that satisfy the required curl.
  * Return the best-scoring rule whose score is above ``min_score``.
  * Ties break toward the earlier rule (order matters in GESTURE_RULES).
"""

from __future__ import annotations

from typing import Sequence

from .. import config
from ..schema import Point
from .hands import finger_curls


DEFAULT_MIN_SCORE = 0.75


def classify_hand(hand: Sequence[Point], min_score: float = DEFAULT_MIN_SCORE) -> tuple[str, float]:
    """Return (gesture_name, score). ("none", 0.0) when no rule matches."""
    if len(hand) < 21:
        return "none", 0.0

    curls = finger_curls(hand)
    best_name = "none"
    best_score = 0.0

    for rule in config.GESTURE_RULES:
        wanted = rule.curls
        matched = 0
        required = 0
        for want, actual in zip(wanted, curls):
            if want is None:
                continue
            required += 1
            if want == actual:
                matched += 1
        if required == 0:
            continue
        score = matched / required
        if score > best_score:
            best_score = score
            best_name = rule.name

    if best_score < min_score:
        return "none", 0.0
    return best_name, best_score
