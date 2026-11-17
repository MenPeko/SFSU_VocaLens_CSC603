"""Leak checks that run before training. Any failure aborts the run.

See docs/SIGN_EVAL_PLAN.md, section 2.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..classifier import TrainingExample, _resample_frames
from ..schemas import LandmarkFrameModel, PointModel
from .sessions import RecordingSession

ARM_JOINTS = ("leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist")
HAND_POINTS = 21
DUPLICATE_FRAMES = 24


@dataclass
class LeakReport:
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def extend(self, messages: Iterable[str]) -> None:
        self.failures.extend(messages)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "failures": self.failures, "notes": self.notes}


def _describe(example: TrainingExample) -> str:
    return f"{example.dataset}:{example.sample_path}"


def check_extractor(examples: Iterable[TrainingExample], required: str) -> list[str]:
    wrong = [example for example in examples if example.extractor_version != required]
    if not wrong:
        return []
    found = sorted({example.extractor_version for example in wrong})
    return [f"{len(wrong)} clips use extractor {found}, not {required} (first: {_describe(wrong[0])})"]


def check_group_keys(examples: Iterable[TrainingExample]) -> list[str]:
    missing = [example for example in examples if not example.group_key]
    if not missing:
        return []
    return [f"{len(missing)} clips have no session or signer, so they cannot be split by group (first: {_describe(missing[0])})"]


def check_session_registry(sessions: list[RecordingSession], registry: set[str]) -> list[str]:
    failures: list[str] = []
    by_id = {session.session_id: session for session in sessions}
    for session in sessions:
        if session.role == "test" and session.session_id not in registry:
            failures.append(f"session {session.session_id} says role=test but is not in test_sessions.json")
        if session.role != "test" and session.session_id in registry:
            failures.append(
                f"session {session.session_id} is in test_sessions.json but its session.json says role={session.role}"
            )
    for session_id in sorted(registry - set(by_id)):
        failures.append(f"test_sessions.json lists {session_id}, which has no session folder")
    return failures


def check_session_separation(sessions: list[RecordingSession]) -> list[str]:
    """A test session must differ from each training session of the same signer by day or lighting."""

    failures: list[str] = []
    train_sessions = [session for session in sessions if session.role == "train"]
    for test_session in (session for session in sessions if session.role == "test"):
        for train_session in train_sessions:
            if (
                train_session.signer == test_session.signer
                and train_session.recorded_on == test_session.recorded_on
                and train_session.lighting == test_session.lighting
            ):
                failures.append(
                    f"test session {test_session.session_id} has the same day and lighting as training session {train_session.session_id}"
                )
    return failures


def check_validation_sessions(sessions: list[RecordingSession], own_signer: str) -> list[str]:
    """Validation sessions are the owner's, and differ from every test session by day or lighting."""

    failures: list[str] = []
    test_sessions = [session for session in sessions if session.role == "test"]
    for validation in (session for session in sessions if session.role == "validation"):
        if validation.signer != own_signer:
            failures.append(f"validation session {validation.session_id} is from signer {validation.signer}; only {own_signer} validates")
        for test_session in test_sessions:
            if test_session.recorded_on == validation.recorded_on and test_session.lighting == validation.lighting:
                failures.append(
                    f"validation session {validation.session_id} has the same day and lighting as test session {test_session.session_id}"
                )
    return failures


def check_validation_disjoint(
    train: list[TrainingExample],
    validation: list[TrainingExample],
    tests: dict[str, list[TrainingExample]],
    validation_session_ids: set[str],
) -> list[str]:
    """Validation clips appear in neither training nor any test set, by session or by group."""

    failures: list[str] = []
    train_hits = [example for example in train if example.session_id in validation_session_ids]
    if train_hits:
        failures.append(f"{len(train_hits)} training clips come from validation sessions {sorted({e.session_id for e in train_hits})}")
    for name, examples in tests.items():
        hits = [example for example in examples if example.session_id in validation_session_ids]
        if hits:
            failures.append(f"test set {name} has {len(hits)} clips from validation sessions {sorted({e.session_id for e in hits})}")

    validation_groups = {example.group_key for example in validation}
    shared_train = sorted(validation_groups & {example.group_key for example in train})
    if shared_train:
        failures.append(f"validation shares groups with training: {shared_train[:5]}")
    for name, examples in tests.items():
        shared = sorted(validation_groups & {example.group_key for example in examples})
        if shared:
            failures.append(f"validation shares groups with test set {name}: {shared[:5]}")
    return failures


def check_asl_official_splits(
    train: Iterable[TrainingExample],
    test_split_participants: set[str],
    held_out_signers: set[str],
    train_val_participants: set[str],
) -> list[str]:
    """Fail if training includes an official test-split signer, or a held-out signer is in train/val.

    ``check_disjoint`` still rejects held-out signers that show up in the training examples.
    """

    failures: list[str] = []
    hits = [
        example
        for example in train
        if example.dataset == "asl-citizen" and example.signer_id in test_split_participants
    ]
    if hits:
        names = sorted({example.signer_id for example in hits})
        failures.append(f"{len(hits)} training clips come from ASL Citizen test-split signers {names}")
    leaked = sorted(held_out_signers & train_val_participants)
    if leaked:
        failures.append(f"held-out ASL Citizen signers appear in train.csv or val.csv: {leaked}")
    return failures


def check_disjoint(
    train: list[TrainingExample],
    test_session_ids: set[str],
    test_signers: dict[str, set[str]],
) -> list[str]:
    """Fail if any training clip comes from a test session or a test signer.

    ``test_signers`` maps a dataset name to its held-out signer IDs; for own
    recordings these are teammates whose sessions are test-only.
    """

    failures: list[str] = []
    session_hits = [example for example in train if example.session_id and example.session_id in test_session_ids]
    if session_hits:
        sessions = sorted({example.session_id for example in session_hits})
        failures.append(f"{len(session_hits)} training clips come from test sessions {sessions}")
    for dataset, signers in test_signers.items():
        hits = [example for example in train if example.dataset == dataset and example.signer_id in signers]
        if hits:
            names = sorted({example.signer_id for example in hits})
            failures.append(f"{len(hits)} training clips come from held-out {dataset} signers {names}")
    return failures


def check_train_test_groups(train: list[TrainingExample], tests: dict[str, list[TrainingExample]]) -> list[str]:
    train_groups = {example.group_key for example in train}
    failures: list[str] = []
    for name, examples in tests.items():
        shared = sorted(train_groups & {example.group_key for example in examples})
        if shared:
            failures.append(f"test set {name} shares groups with training: {shared[:5]}")
    return failures


def landmark_hash(example: TrainingExample) -> str:
    payload = [frame.model_dump(exclude={"timestampMs"}) for frame in example.frames]

    def rounded(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, list):
            return [rounded(item) for item in value]
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items() if item is not None}
        return value

    encoded = json.dumps(rounded(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


Joint = Optional[tuple[float, float, float]]


def _point(point: Optional[PointModel]) -> Joint:
    return (point.x, point.y, point.z) if point is not None else None


def duplicate_representation(example: TrainingExample) -> list[list[Joint]]:
    """24 resampled frames of 6 arm joints plus both hands (48 joints per frame)."""

    frames: list[LandmarkFrameModel] = _resample_frames(example.frames, DUPLICATE_FRAMES)
    representation: list[list[Joint]] = []
    for frame in frames:
        joints: list[Joint] = [_point(getattr(frame.pose, name)) for name in ARM_JOINTS]
        for hand in (frame.leftHand, frame.rightHand):
            if len(hand) == HAND_POINTS:
                joints.extend(_point(point) for point in hand)
            else:
                joints.extend([None] * HAND_POINTS)
        representation.append(joints)
    return representation


def landmark_distance(left: list[list[Joint]], right: list[list[Joint]], missing_penalty: float) -> float:
    """Mean Euclidean distance per joint per frame, in shoulder-width units.

    A joint present in one clip and missing in the other at the same frame
    counts as ``missing_penalty``. Joints missing in both are skipped.
    """

    total = 0.0
    counted = 0
    for left_frame, right_frame in zip(left, right):
        for a, b in zip(left_frame, right_frame):
            if a is None and b is None:
                continue
            counted += 1
            if a is None or b is None:
                total += missing_penalty
            else:
                total += math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    return total / counted if counted else missing_penalty


@dataclass
class DuplicateResult:
    train: list[TrainingExample]
    failures: list[str]
    dropped_within_train: list[dict[str, str]]
    cross_split_pairs: list[dict[str, Any]]
    calibration: dict[str, Any]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.floor(fraction * (len(ordered) - 1)))))
    return ordered[index]


def find_duplicates(
    train: list[TrainingExample],
    tests: dict[str, list[TrainingExample]],
    settings: dict[str, Any],
    seed: int,
    validation: Optional[list[TrainingExample]] = None,
) -> DuplicateResult:
    """Drop repeats within training, and fail on any clip shared across splits.

    Shared means a test clip that matches training or validation, or a
    validation clip that matches training.
    """

    validation = validation or []

    failures: list[str] = []
    dropped: list[dict[str, str]] = []
    pairs: list[dict[str, Any]] = []

    kept: list[TrainingExample] = []
    seen_video: dict[tuple[str, str], TrainingExample] = {}
    seen_hash: dict[str, TrainingExample] = {}
    hashes: dict[int, str] = {}
    for example in train:
        digest = landmark_hash(example)
        hashes[id(example)] = digest
        video_key = (example.dataset, example.source_video_id)
        if example.source_video_id and video_key in seen_video:
            dropped.append({"clip": _describe(example), "duplicateOf": _describe(seen_video[video_key]), "by": "sourceVideoId"})
            continue
        if digest in seen_hash:
            dropped.append({"clip": _describe(example), "duplicateOf": _describe(seen_hash[digest]), "by": "landmarkHash"})
            continue
        if example.source_video_id:
            seen_video[video_key] = example
        seen_hash[digest] = example
        kept.append(example)

    threshold = float(settings["threshold"])
    penalty = float(settings["handMissingPenalty"])
    representations: dict[int, list[list[Joint]]] = {}

    def representation(example: TrainingExample) -> list[list[Joint]]:
        key = id(example)
        if key not in representations:
            representations[key] = duplicate_representation(example)
        return representations[key]

    def compare(query_name: str, queries: list[TrainingExample], reference_name: str, references: list[TrainingExample]) -> None:
        by_video = {(e.dataset, e.source_video_id): e for e in references if e.source_video_id}
        by_hash = {landmark_hash(e): e for e in references}
        by_label: dict[str, list[TrainingExample]] = defaultdict(list)
        for example in references:
            by_label[example.label].append(example)

        for query in queries:
            pair = {"testSet": query_name, "against": reference_name, "test": _describe(query)}
            video_match = by_video.get((query.dataset, query.source_video_id)) if query.source_video_id else None
            if video_match is not None:
                failures.append(f"{query_name}: video {query.source_video_id} is also in {reference_name}")
                pairs.append({**pair, "train": _describe(video_match), "by": "sourceVideoId"})
                continue
            hash_match = by_hash.get(landmark_hash(query))
            if hash_match is not None:
                failures.append(f"{query_name}: {_describe(query)} has the same landmarks as a {reference_name} clip")
                pairs.append({**pair, "train": _describe(hash_match), "by": "landmarkHash"})
                continue
            for reference in by_label.get(query.label, []):
                distance = landmark_distance(representation(query), representation(reference), penalty)
                if distance < threshold:
                    failures.append(
                        f"{query_name}: {_describe(query)} is a near-duplicate of {reference_name} clip {_describe(reference)} (distance {distance:.4f})"
                    )
                    pairs.append({**pair, "train": _describe(reference), "by": "landmarkDistance", "distance": round(distance, 5)})

    for name, examples in tests.items():
        compare(f"test set {name}", examples, "training", kept)
        compare(f"test set {name}", examples, "validation", validation)
    compare("validation", validation, "training", kept)

    calibration = calibrate(
        kept + validation + [example for examples in tests.values() for example in examples], settings, seed, representation
    )
    if calibration["status"] == "too_close":
        failures.append(
            f"calibration: the {settings['calibrationPercentile']:.1%} distance between distinct clips is "
            f"{calibration['lowPercentileDistance']:.4f}, below the floor {settings['calibrationFloor']}. Review the duplicate threshold."
        )

    return DuplicateResult(train=kept, failures=failures, dropped_within_train=dropped, cross_split_pairs=pairs, calibration=calibration)


def calibrate(examples: list[TrainingExample], settings: dict[str, Any], seed: int, representation) -> dict[str, Any]:
    """Distances between clips known to be distinct: same label, different session or signer."""

    rng = random.Random(seed)
    penalty = float(settings["handMissingPenalty"])
    cap = int(settings["maxCalibrationPairsPerLabel"])
    by_label: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    distances: list[float] = []
    for label in sorted(by_label):
        group = by_label[label]
        candidates = [
            (i, j)
            for i in range(len(group))
            for j in range(i + 1, len(group))
            if group[i].group_key != group[j].group_key
        ]
        if len(candidates) > cap:
            candidates = rng.sample(candidates, cap)
        for i, j in candidates:
            distances.append(landmark_distance(representation(group[i]), representation(group[j]), penalty))

    if not distances:
        return {"status": "no_pairs", "pairs": 0}
    low = _percentile(distances, float(settings["calibrationPercentile"]))
    return {
        "status": "too_close" if low < float(settings["calibrationFloor"]) else "ok",
        "pairs": len(distances),
        "lowPercentileDistance": round(low, 5),
        "minDistance": round(min(distances), 5),
        "medianDistance": round(_percentile(distances, 0.5), 5),
        "threshold": settings["threshold"],
        "floor": settings["calibrationFloor"],
    }
