"""Run one held-out experiment: assemble data, check for leaks, train, score.

    python -m sign_classifier_service.eval.run --experiment baseline

See docs/SIGN_EVAL_PLAN.md, sections 4 to 7.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..classifier import APP_ROOT, CORE_SIGN_LABELS, TrainingExample
from ..train import build_artifact, evaluate_artifact, group_split, maybe_print_imbalance_warning, predict_from_artifact
from . import (
    EXPERIMENTS_DIR,
    GLOSS_MAP_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SESSIONS_DIR,
    TEST_SESSIONS_PATH,
    load_config,
    read_json,
    sha256_file,
    write_json,
)
from . import asl_citizen
from .leakage import (
    LeakReport,
    check_asl_official_splits,
    check_disjoint,
    check_extractor,
    check_group_keys,
    check_session_registry,
    check_session_separation,
    check_train_test_groups,
    check_validation_disjoint,
    check_validation_sessions,
    find_duplicates,
    landmark_hash,
)
from .sessions import load_session_examples, load_sessions, load_test_session_registry

SOURCES = ("own", "asl-citizen")
OWN_TEST = "own-heldout"
ASL_TEST = "asl-citizen-heldout"
TEAM_TEST = "team"
TEST_SETS = (OWN_TEST, ASL_TEST, TEAM_TEST)


class ExperimentError(RuntimeError):
    pass


def load_experiment(name_or_path: str, experiments_dir: Path = EXPERIMENTS_DIR) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.suffix:
        path = experiments_dir / f"{name_or_path}.json"
    if not path.exists():
        raise ExperimentError(f"No experiment config at {path}")
    experiment = read_json(path)
    unknown = sorted(set(experiment["trainSources"]) - set(SOURCES))
    if unknown:
        raise ExperimentError(f"{experiment['name']}: unknown training sources {unknown}")
    if "own" not in experiment["trainSources"]:
        raise ExperimentError(f"{experiment['name']}: every experiment trains on your own recordings")

    parent_name = experiment.get("extends")
    if parent_name:
        parent = load_experiment(parent_name, experiments_dir)
        added = set(experiment["trainSources"]) - set(parent["trainSources"])
        removed = set(parent["trainSources"]) - set(experiment["trainSources"])
        if removed or len(added) != 1:
            raise ExperimentError(
                f"{experiment['name']} must add exactly one dataset to {parent_name}; added {sorted(added)}, removed {sorted(removed)}"
            )
        experiment["addedSource"] = next(iter(added))
    experiment["configPath"] = str(path)
    return experiment


@dataclass
class AssembledData:
    train: list[TrainingExample]
    validation: list[TrainingExample]
    tests: dict[str, list[TrainingExample]]
    own_examples: list[TrainingExample]
    asl_examples: list[TrainingExample]
    provenance: dict[str, Any] = field(default_factory=dict)


def assemble(
    experiment: dict[str, Any],
    config: dict[str, Any],
    sessions_dir: Path = SESSIONS_DIR,
    registry_path: Path = TEST_SESSIONS_PATH,
    asl_landmarks: Optional[Path] = None,
    asl_test_signers_path: Path = asl_citizen.TEST_SIGNERS_PATH,
    asl_filter_summary_path: Path = asl_citizen.FILTER_SUMMARY_PATH,
    asl_filtered_csv_path: Path = asl_citizen.FILTERED_CSV_PATH,
) -> tuple[AssembledData, LeakReport]:
    report = LeakReport()
    sources = set(experiment["trainSources"])
    own_signer = config["ownSigner"]
    gloss_map = asl_citizen.load_gloss_map()

    sessions, problems = load_sessions(sessions_dir)
    report.extend(problems)
    registry = load_test_session_registry(registry_path)
    report.extend(check_session_registry(sessions, registry))
    report.extend(check_session_separation(sessions))
    report.extend(check_validation_sessions(sessions, own_signer))

    test_session_ids = {session.session_id for session in sessions if session.role == "test" or session.session_id in registry}
    validation_session_ids = {
        session.session_id for session in sessions if session.role == "validation" and session.session_id not in test_session_ids
    }
    team_signers = {session.signer for session in sessions if session.session_id in test_session_ids and session.signer != own_signer}
    for session in sessions:
        if session.role == "train" and session.session_id not in test_session_ids and session.signer != own_signer:
            report.fail(f"training session {session.session_id} is from signer {session.signer}; only {own_signer} trains")

    own_map = gloss_map.get("own", {})
    own_train: list[TrainingExample] = []
    own_validation: list[TrainingExample] = []
    own_test: list[TrainingExample] = []
    team_test: list[TrainingExample] = []
    own_dropped = Counter()
    sessions_used: list[dict[str, Any]] = []
    for session in sessions:
        examples, sample_problems = load_session_examples(session)
        report.extend(sample_problems)
        mapped: list[TrainingExample] = []
        for example in examples:
            label = own_map.get(example.label)
            if label is None:
                own_dropped[example.label] += 1
                continue
            example.label = label
            mapped.append(example)
        if session.session_id in test_session_ids:
            role = "test"
            (own_test if session.signer == own_signer else team_test).extend(mapped)
        elif session.session_id in validation_session_ids:
            role = "validation"
            own_validation.extend(mapped)
        else:
            role = "train"
            own_train.extend(mapped)
        sessions_used.append({"sessionId": session.session_id, "role": role, "signer": session.signer, "clips": len(mapped)})

    asl_path = asl_landmarks or asl_citizen.landmarks_path(config["requiredExtractorVersion"])
    asl_examples = asl_citizen.load_examples(config["requiredExtractorVersion"], asl_path)
    asl_train: list[TrainingExample] = []
    asl_test: list[TrainingExample] = []
    held_out: set[str] = set()
    test_split_participants: set[str] = set()
    train_val_participants: set[str] = set()
    provenance: dict[str, Any] = {
        "sessions": sessions_used,
        "glossMapSha256": sha256_file(GLOSS_MAP_PATH),
        "droppedByGlossMap": {"own": sum(own_dropped.values())},
    }

    if "asl-citizen" in sources and not asl_examples:
        report.fail(f"experiment trains on ASL Citizen but {asl_path} has no clips")
    if asl_examples:
        provenance["aslCitizenLandmarksSha256"] = sha256_file(asl_path)
        try:
            signer_file = asl_citizen.load_test_signers(asl_test_signers_path)
            held_out = set(signer_file["testSigners"])
            test_split_participants = set(signer_file.get("testSplitParticipants", []))
            train_val_participants = set(signer_file.get("trainValParticipants", []))
            provenance["aslCitizenTestSignersSha256"] = sha256_file(asl_test_signers_path)
        except asl_citizen.AslCitizenError as error:
            if "asl-citizen" in sources:
                report.fail(str(error))
            else:
                report.notes.append(f"ASL Citizen test set skipped: {error}")
        if asl_filter_summary_path.exists():
            summary = read_json(asl_filter_summary_path)
            provenance["droppedByGlossMap"]["asl-citizen"] = summary.get("droppedRows")
            provenance["aslCitizenGaps"] = summary.get("gaps", [])
            if summary.get("glossMapSha256") != provenance["glossMapSha256"]:
                report.fail("gloss_map.json changed after the ASL Citizen filter ran; rerun filter and extraction")
        elif "asl-citizen" in sources:
            report.fail(f"{asl_filter_summary_path} is missing; run `asl_citizen filter`")
        if held_out:
            asl_test = [example for example in asl_examples if example.signer_id in held_out]
            if "asl-citizen" in sources:
                if not asl_filtered_csv_path.exists():
                    report.fail(f"{asl_filtered_csv_path} is missing; run `asl_citizen filter` before training")
                else:
                    by_video = asl_citizen.source_csv_by_video(asl_filtered_csv_path)
                    asl_train = [
                        example
                        for example in asl_examples
                        if by_video.get(Path(example.source_video_id).name) in asl_citizen.TRAIN_SOURCE_CSVS
                    ]

    train = own_train + asl_train
    tests = {OWN_TEST: own_test, ASL_TEST: asl_test, TEAM_TEST: team_test}
    everything = train + own_validation + own_test + team_test + asl_test

    report.extend(check_extractor(everything, config["requiredExtractorVersion"]))
    report.extend(check_group_keys(everything))
    report.extend(check_asl_official_splits(train, test_split_participants, held_out, train_val_participants))
    report.extend(check_disjoint(train, test_session_ids, {"asl-citizen": held_out, "own": team_signers}))
    report.extend(check_train_test_groups(train, tests))
    report.extend(check_validation_disjoint(train, own_validation, tests, validation_session_ids))

    data = AssembledData(
        train=train,
        validation=own_validation,
        tests=tests,
        own_examples=own_train + own_validation + own_test + team_test,
        asl_examples=asl_examples,
        provenance=provenance,
    )
    return data, report


def _sort_key(example: TrainingExample) -> tuple[str, str, str, str]:
    return (example.dataset, example.group_key, example.source_video_id, str(example.sample_path))


def evaluate_test_set(artifact: dict[str, Any], examples: list[TrainingExample], bootstrap: dict[str, Any]) -> dict[str, Any]:
    confusion: dict[str, Counter] = {label: Counter() for label in CORE_SIGN_LABELS}
    correct_flags: list[int] = []
    for example in examples:
        predicted, _ = predict_from_artifact(artifact, example)
        confusion[example.label][predicted] += 1
        correct_flags.append(int(predicted == example.label))

    per_sign: dict[str, dict[str, Any]] = {}
    for label in CORE_SIGN_LABELS:
        support = sum(confusion[label].values())
        per_sign[label] = {
            "support": support,
            "accuracy": round(confusion[label][label] / support, 4) if support else None,
        }

    total = len(correct_flags)
    accuracy = sum(correct_flags) / total if total else None
    interval: Optional[list[float]] = None
    if total:
        rng = random.Random(bootstrap["seed"])
        resampled = sorted(
            sum(correct_flags[rng.randrange(total)] for _ in range(total)) / total for _ in range(int(bootstrap["resamples"]))
        )
        low = resampled[int(0.025 * (len(resampled) - 1))]
        high = resampled[int(0.975 * (len(resampled) - 1))]
        interval = [round(low, 4), round(high, 4)]

    return {
        "clips": total,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "accuracy95": interval,
        "perSign": per_sign,
        "confusionMatrix": {label: {pred: confusion[label][pred] for pred in CORE_SIGN_LABELS} for label in CORE_SIGN_LABELS},
    }


def train_counts(train: list[TrainingExample]) -> dict[str, Any]:
    by_source: dict[str, Counter] = defaultdict(Counter)
    for example in train:
        by_source[example.dataset][example.label] += 1
    total = Counter(example.label for example in train)
    present = [count for count in total.values() if count > 0]
    ratio = round(max(present) / min(present), 3) if present else None
    return {
        "total": {label: total.get(label, 0) for label in CORE_SIGN_LABELS},
        "bySource": {source: {label: counts.get(label, 0) for label in CORE_SIGN_LABELS} for source, counts in sorted(by_source.items())},
        "missingSigns": [label for label in CORE_SIGN_LABELS if total.get(label, 0) == 0],
        "largestToSmallestRatio": ratio,
        "imbalanced": bool(ratio and ratio > 2.0),
    }


def handedness_check(own: list[TrainingExample], asl: list[TrainingExample], min_clips: int = 5) -> dict[str, Any]:
    """Share of clips where the left hand is seen in more frames than the right, per sign."""

    def left_share(examples: list[TrainingExample]) -> dict[str, tuple[float, int]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for example in examples:
            left = sum(1 for frame in example.frames if frame.leftHand)
            right = sum(1 for frame in example.frames if frame.rightHand)
            grouped[example.label].append(int(left > right))
        return {label: (sum(flags) / len(flags), len(flags)) for label, flags in grouped.items()}

    own_share = left_share(own)
    asl_share = left_share(asl)
    rows: dict[str, Any] = {}
    flagged: list[str] = []
    for label in CORE_SIGN_LABELS:
        if label not in own_share or label not in asl_share:
            continue
        (own_value, own_n), (asl_value, asl_n) = own_share[label], asl_share[label]
        rows[label] = {"ownLeftShare": round(own_value, 3), "aslCitizenLeftShare": round(asl_value, 3), "ownClips": own_n, "aslCitizenClips": asl_n}
        if own_n >= min_clips and asl_n >= min_clips and abs(own_value - asl_value) > 0.5:
            flagged.append(label)
    return {"perSign": rows, "flagged": flagged}


def own_heldout_fingerprint(examples: list[TrainingExample]) -> Optional[str]:
    if not examples:
        return None
    digest = hashlib.sha256("\n".join(sorted(landmark_hash(example) for example in examples)).encode("utf-8"))
    return digest.hexdigest()


def validation_identity(examples: list[TrainingExample]) -> dict[str, Any]:
    return {
        "sessions": sorted({example.session_id for example in examples}),
        "fingerprint": own_heldout_fingerprint(examples),
    }


def find_parent(
    experiment: dict[str, Any],
    fingerprint: Optional[str],
    validation: dict[str, Any],
    reports_dir: Path,
) -> Optional[dict[str, Any]]:
    """Latest parent run scored on the same own held-out clips, or None.

    Raises ExperimentError when that run selected its model on different
    validation sessions, because the two accuracies would not be comparable.
    """

    parent_name = experiment.get("extends")
    if not parent_name or fingerprint is None:
        return None
    parents: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*/metrics.json")):
        payload = read_json(path)
        if payload.get("experiment") == parent_name and payload.get("ownHeldoutFingerprint") == fingerprint:
            parents.append(payload)
    if not parents:
        return None
    parent = max(parents, key=lambda payload: payload["createdAt"])
    parent_validation = parent.get("validationSessions")
    if parent_validation is None:
        raise ExperimentError(
            f"Cannot compare with {parent['runId']}: it does not record which validation sessions it used. "
            f"Rerun `{parent_name}` with the current validation sessions {validation['sessions']}, then rerun {experiment['name']}."
        )
    if parent_validation != validation["sessions"]:
        raise ExperimentError(
            f"Cannot compare with {parent['runId']}: it used validation sessions {parent_validation}, "
            f"this run uses {validation['sessions']}. Both runs must select their model on the same validation sessions. "
            f"Rerun `{parent_name}` with the current sessions, then rerun {experiment['name']}."
        )
    if parent.get("validationFingerprint") != validation["fingerprint"]:
        raise ExperimentError(
            f"Cannot compare with {parent['runId']}: validation sessions {validation['sessions']} have gained, lost, or changed "
            f"clips since that run. Rerun `{parent_name}`, then rerun {experiment['name']}."
        )
    return parent


def decide(
    experiment: dict[str, Any],
    metrics: dict[str, Any],
    fingerprint: Optional[str],
    validation: dict[str, Any],
    reports_dir: Path,
) -> dict[str, Any]:
    parent_name = experiment.get("extends")
    if not parent_name:
        return {"decision": "baseline", "reason": "reference run"}
    own = metrics.get(OWN_TEST)
    if not own or own["accuracy"] is None:
        return {"decision": "undecided", "reason": "no own held-out clips"}

    parent = find_parent(experiment, fingerprint, validation, reports_dir)
    if parent is None:
        return {"decision": "undecided", "reason": f"no {parent_name} run on the same own held-out clips"}
    parent_accuracy = parent["testSets"][OWN_TEST]["accuracy"]
    kept = own["accuracy"] > parent_accuracy
    return {
        "decision": "kept" if kept else "dropped",
        "reason": f"own held-out accuracy {own['accuracy']:.4f} vs {parent_accuracy:.4f} in {parent['runId']}",
        "comparedTo": parent["runId"],
        "parentOwnHeldoutAccuracy": parent_accuracy,
    }


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=APP_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def render_report(run: dict[str, Any]) -> str:
    lines = [
        f"# {run['runId']}",
        "",
        f"Experiment: `{run['experiment']}`. Training sources: {', '.join(run['trainSources'])}. Seed {run['seed']}.",
        f"Decision: **{run['decision']['decision']}** ({run['decision']['reason']}).",
    ]
    stopping = run.get("earlyStopping")
    if stopping:
        lines.append(
            f"Validation: {run['validationClips']} clips from validation sessions, accuracy {run['validation']['accuracy']:.3f}. "
            f"Early stopping kept epoch {stopping['bestEpoch']} of {stopping['stoppedEpoch']} run (max {stopping['maxEpochs']}, patience {stopping['patience']})."
        )
    lines += [
        "",
        "## Test sets",
        "",
        "| Test set | Clips | Accuracy | 95% interval |",
        "| --- | ---: | ---: | --- |",
    ]
    for name in TEST_SETS:
        result = run["testSets"].get(name)
        if not result:
            lines.append(f"| {name} | 0 | n/a | n/a |")
            continue
        interval = "-".join(f"{value:.3f}" for value in result["accuracy95"]) if result["accuracy95"] else "n/a"
        lines.append(f"| {name} | {result['clips']} | {result['accuracy']:.3f} | {interval} |")

    for name in TEST_SETS:
        result = run["testSets"].get(name)
        if not result:
            continue
        lines += ["", f"### {name}: per sign", "", "| Sign | Clips | Accuracy |", "| --- | ---: | ---: |"]
        for label, row in result["perSign"].items():
            accuracy = f"{row['accuracy']:.3f}" if row["accuracy"] is not None else "n/a"
            lines.append(f"| {label} | {row['support']} | {accuracy} |")
        lines += ["", f"### {name}: confusion (rows true, columns predicted)", ""]
        lines.append("| true \\ pred | " + " | ".join(CORE_SIGN_LABELS) + " |")
        lines.append("| --- |" + " ---: |" * len(CORE_SIGN_LABELS))
        for label in CORE_SIGN_LABELS:
            row = result["confusionMatrix"][label]
            lines.append(f"| {label} | " + " | ".join(str(row[pred]) for pred in CORE_SIGN_LABELS) + " |")

    counts = run["trainCounts"]
    lines += ["", "## Training clips per sign", "", "| Sign | " + " | ".join(counts["bySource"]) + " | Total |"]
    lines.append("| --- |" + " ---: |" * (len(counts["bySource"]) + 1))
    for label in CORE_SIGN_LABELS:
        cells = [str(counts["bySource"][source][label]) for source in counts["bySource"]]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {counts['total'][label]} |")
    if counts["imbalanced"]:
        lines += ["", f"Imbalance: the largest sign has {counts['largestToSmallestRatio']}x the clips of the smallest."]
    if counts["missingSigns"]:
        lines += ["", f"No training clips for: {', '.join(counts['missingSigns'])}."]

    lines += [
        "",
        "## Dropped and checked",
        "",
        f"- Dropped by the gloss map: {json.dumps(run['droppedByGlossMap'])}",
        f"- Duplicates dropped within training: {run['duplicates']['droppedWithinTrain']}",
        f"- Duplicate calibration: {json.dumps(run['duplicates']['calibration'])}",
    ]
    if run.get("handedness", {}).get("flagged"):
        lines.append(f"- Handedness differs between own and ASL Citizen clips for: {', '.join(run['handedness']['flagged'])}")
    for note in run.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def run_experiment(
    experiment_name: str,
    sessions_dir: Path = SESSIONS_DIR,
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
    registry_path: Path = TEST_SESSIONS_PATH,
    asl_landmarks: Optional[Path] = None,
    asl_test_signers_path: Path = asl_citizen.TEST_SIGNERS_PATH,
    asl_filter_summary_path: Path = asl_citizen.FILTER_SUMMARY_PATH,
    asl_filtered_csv_path: Path = asl_citizen.FILTERED_CSV_PATH,
    experiments_dir: Path = EXPERIMENTS_DIR,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    config = load_config()
    experiment = load_experiment(experiment_name, experiments_dir)
    data, report = assemble(
        experiment,
        config,
        sessions_dir=sessions_dir,
        registry_path=registry_path,
        asl_landmarks=asl_landmarks,
        asl_test_signers_path=asl_test_signers_path,
        asl_filter_summary_path=asl_filter_summary_path,
        asl_filtered_csv_path=asl_filtered_csv_path,
    )

    own_train = [example for example in data.train if example.dataset == "own"]
    if not own_train and report.ok:
        raise ExperimentError(
            f"No training sessions found under {sessions_dir}. Create one with "
            "`python -m sign_classifier_service.eval.sessions new --role train ...` and record into it."
        )
    if not data.validation and report.ok:
        raise ExperimentError(
            f"No validation session found under {sessions_dir}. Early stopping uses validation sessions only and never "
            "falls back to the test sets. Create one with `python -m sign_classifier_service.eval.sessions new "
            "--role validation --signer owner ...` on a different day or lighting from every test session, and record into it."
        )

    created_at = datetime.now(timezone.utc)
    run_id = run_id or f"{experiment['name']}-{created_at.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = reports_dir / run_id

    duplicates = None
    if report.ok:
        duplicates = find_duplicates(data.train, data.tests, config["duplicates"], config["seed"], validation=data.validation)
        report.extend(duplicates.failures)
        write_json(
            run_dir / "duplicates.json",
            {
                "droppedWithinTrain": duplicates.dropped_within_train,
                "crossSplitPairs": duplicates.cross_split_pairs,
                "calibration": duplicates.calibration,
            },
        )
    write_json(run_dir / "leak-check.json", report.to_dict())
    if not report.ok:
        raise ExperimentError(f"Leak check failed ({len(report.failures)} problems); see {run_dir / 'leak-check.json'}:\n- " + "\n- ".join(report.failures[:20]))

    assert duplicates is not None
    train = sorted(duplicates.train, key=_sort_key)
    validation = sorted(data.validation, key=_sort_key)
    fingerprint = own_heldout_fingerprint(data.tests[OWN_TEST])
    validation_id = validation_identity(validation)
    find_parent(experiment, fingerprint, validation_id, reports_dir)
    # validationRatio > 0 holds whole training groups out as a diagnostic only;
    # early stopping and model selection see validation sessions alone.
    split = group_split(train, config["validationRatio"], config["seed"])
    maybe_print_imbalance_warning(Counter(example.label for example in split.train))
    training = config["training"]
    artifact, _ = build_artifact(
        train_examples=split.train,
        validation_examples=validation,
        epochs=training["epochs"],
        learning_rate=training["learningRate"],
        weight_decay=training["weightDecay"],
        model_version=run_id,
        training_run_id=run_id,
        early_stopping=training["earlyStopping"],
    )
    if split.validation:
        artifact["metrics"]["trainingGroupHoldout"] = evaluate_artifact(artifact, split.validation)
    model_path = models_dir / f"{run_id}.json"
    write_json(model_path, artifact)

    # Test sets are scored once, here, after training and model selection are final.
    test_metrics = {
        name: evaluate_test_set(artifact, examples, config["bootstrap"]) for name, examples in data.tests.items() if examples
    }
    decision = decide(experiment, test_metrics, fingerprint, validation_id, reports_dir)
    counts = train_counts(split.train)

    run_config = {
        "runId": run_id,
        "createdAt": created_at.isoformat(),
        "experiment": experiment,
        "evalConfig": config,
        "seed": config["seed"],
        "git": git_state(),
        "extractorVersion": artifact["extractor"],
        "provenance": data.provenance,
        "trainClips": len(split.train),
        "validationClips": len(validation),
        "trainingGroupHoldoutClips": len(split.validation),
        "testClips": {name: len(examples) for name, examples in data.tests.items()},
        "modelPath": str(model_path.relative_to(APP_ROOT)) if model_path.is_relative_to(APP_ROOT) else str(model_path),
    }
    run = {
        "runId": run_id,
        "createdAt": created_at.isoformat(),
        "experiment": experiment["name"],
        "extends": experiment.get("extends"),
        "trainSources": experiment["trainSources"],
        "seed": config["seed"],
        "trainClips": len(split.train),
        "trainCounts": counts,
        "testSets": test_metrics,
        "ownHeldoutFingerprint": fingerprint,
        "validationSessions": validation_id["sessions"],
        "validationFingerprint": validation_id["fingerprint"],
        "decision": decision,
        "droppedByGlossMap": data.provenance["droppedByGlossMap"],
        "aslCitizenGaps": data.provenance.get("aslCitizenGaps", []),
        "duplicates": {"droppedWithinTrain": len(duplicates.dropped_within_train), "calibration": duplicates.calibration},
        "handedness": handedness_check(data.own_examples, data.asl_examples) if data.asl_examples else {},
        "validationClips": len(validation),
        "validation": artifact["metrics"]["validation"],
        "earlyStopping": artifact["metrics"].get("earlyStopping"),
        "notes": report.notes,
    }
    write_json(run_dir / "config.json", run_config)
    write_json(run_dir / "train-counts.json", counts)
    write_json(run_dir / "metrics.json", run)
    (run_dir / "report.md").write_text(render_report(run), encoding="utf-8")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a held-out sign-classifier experiment.")
    parser.add_argument("--experiment", required=True, help="Name in eval/experiments/ or a path to a config.")
    parser.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    try:
        run = run_experiment(args.experiment, sessions_dir=args.sessions_dir, reports_dir=args.reports_dir, models_dir=args.models_dir)
    except ExperimentError as error:
        raise SystemExit(str(error))

    print(f"Run {run['runId']}: decision {run['decision']['decision']} ({run['decision']['reason']})")
    for name in TEST_SETS:
        result = run["testSets"].get(name)
        if result:
            print(f"  {name}: accuracy {result['accuracy']:.3f} on {result['clips']} clips, 95% {result['accuracy95']}")
        else:
            print(f"  {name}: no clips")
    print(f"Report: {args.reports_dir / run['runId'] / 'report.md'}")
    print("Refresh docs/SIGN_RESULTS.md with: python -m sign_classifier_service.eval.summarize")


if __name__ == "__main__":
    main()
