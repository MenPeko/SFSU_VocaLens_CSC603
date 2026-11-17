"""ASL Citizen: gloss filtering, held-out signer selection, and landmark loading.

Raw files stay under ``data/asl-citizen/`` and derived files under
``data/sign-eval/`` and ``data/landmarks/``; all are gitignored. See
docs/DATA_LICENSES.md (Decisions). Cite Desai et al. 2023 (``desai2023asl``).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..classifier import APP_ROOT, TrainingExample
from ..train import load_manifest_examples
from . import (
    ASL_CITIZEN_DIR,
    GLOSS_MAP_PATH,
    LANDMARKS_DIR,
    SIGN_EVAL_DATA_DIR,
    load_config,
    read_json,
    sha256_file,
    write_json,
)

DATASET = "asl-citizen"
FILTERED_CSV_PATH = SIGN_EVAL_DATA_DIR / "asl-citizen-filtered.csv"
FILTER_SUMMARY_PATH = SIGN_EVAL_DATA_DIR / "asl-citizen-filter-summary.json"
TEST_SIGNERS_PATH = SIGN_EVAL_DATA_DIR / "asl-citizen-test-signers.json"
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
FILTERED_COLUMNS = ["participantId", "videoFile", "sourceGloss", "label", "sourceCsv", "videoPath"]
TRAIN_SOURCE_CSVS = frozenset({"train.csv", "val.csv"})
TEST_SOURCE_CSV = "test.csv"
SPLIT_SOURCE_CSVS = ("train.csv", "val.csv", "test.csv")


class AslCitizenError(RuntimeError):
    pass


def landmarks_path(extractor_version: str) -> Path:
    return LANDMARKS_DIR / DATASET / extractor_version / "clips.jsonl"


def load_gloss_map(path: Path = GLOSS_MAP_PATH) -> dict[str, dict[str, str]]:
    payload = read_json(path)
    classes = set(payload["classes"])
    for dataset, mapping in payload["datasets"].items():
        unknown = sorted(set(mapping.values()) - classes)
        if unknown:
            raise AslCitizenError(f"gloss_map.json maps {dataset} labels to unknown classes: {unknown}")
    return payload["datasets"]


def _columns() -> tuple[str, str, str]:
    settings = load_config()["aslCitizen"]
    return settings["participantColumn"], settings["videoColumn"], settings["glossColumn"]


def find_split_csvs(root: Path = ASL_CITIZEN_DIR) -> list[Path]:
    """Every CSV under the dataset root that has the participant, video, and gloss columns."""

    required = set(_columns())
    found: list[Path] = []
    for path in sorted(root.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
        if required.issubset({column.strip() for column in header}):
            found.append(path)
    return found


def read_rows(csv_paths: list[Path]) -> list[dict[str, str]]:
    participant_column, video_column, gloss_column = _columns()
    rows: list[dict[str, str]] = []
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                record = {key.strip(): (value or "").strip() for key, value in raw.items() if key}
                rows.append(
                    {
                        "participantId": record[participant_column],
                        "videoFile": record[video_column],
                        "sourceGloss": record[gloss_column],
                        "sourceCsv": path.name,
                    }
                )
    return rows


def _normalize_gloss(gloss: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", gloss.upper())


def gloss_candidates(rows: list[dict[str, str]], classes: list[str]) -> dict[str, dict[str, list[tuple[str, int]]]]:
    """Glosses that look like each class, for a person to review before editing gloss_map.json.

    ``likely``: the gloss without punctuation and trailing digits equals the
    class word (``THANK YOU``, ``STOP1``). ``contains``: the class word appears
    anywhere in the gloss; for short words this list is noisy by design.
    """

    counts = Counter(row["sourceGloss"] for row in rows)
    result: dict[str, dict[str, list[tuple[str, int]]]] = {}
    for label in classes:
        token = label.replace("_", "").upper()
        likely: list[tuple[str, int]] = []
        contains: list[tuple[str, int]] = []
        for gloss, count in sorted(counts.items()):
            normalized = _normalize_gloss(gloss)
            if normalized.rstrip("0123456789") == token:
                likely.append((gloss, count))
            elif token in normalized:
                contains.append((gloss, count))
        result[label] = {"likely": likely, "contains": contains}
    return result


def video_index_roots(csv_root: Path) -> list[Path]:
    """Directories whose video files can satisfy the split CSVs.

    ``--root`` is often the splits folder. The clips live in the sibling
    ``videos`` directory of the unpacked dataset.
    """

    roots = [csv_root]
    sibling = csv_root.parent / "videos"
    if sibling.is_dir() and sibling.resolve() != csv_root.resolve():
        roots.append(sibling)
    return roots


def index_videos(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file():
                index.setdefault(path.name, path)
    return index


def filter_rows(
    rows: list[dict[str, str]],
    mapping: dict[str, str],
    classes: list[str],
    video_index: Optional[dict[str, Path]] = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    kept: list[dict[str, str]] = []
    dropped = Counter()
    missing_videos = 0
    for row in rows:
        label = mapping.get(row["sourceGloss"])
        if label is None:
            dropped[row["sourceGloss"]] += 1
            continue
        video_path = ""
        if video_index is not None:
            located = video_index.get(Path(row["videoFile"]).name)
            if located is None:
                missing_videos += 1
            else:
                video_path = _app_relative(located)
        kept.append({**row, "label": label, "videoPath": video_path})

    tokens = {label: label.replace("_", "").upper() for label in classes}
    english_drops = Counter(
        {
            gloss: count
            for gloss, count in dropped.items()
            if any(token in _normalize_gloss(gloss) for token in tokens.values())
        }
    )
    per_label = Counter(row["label"] for row in kept)
    per_split: dict[str, Counter[str]] = {label: Counter() for label in classes}
    for row in kept:
        per_split[row["label"]][row["sourceCsv"]] += 1
    summary = {
        "totalRows": len(rows),
        "mappedRows": len(kept),
        "droppedRows": sum(dropped.values()),
        "droppedGlossCount": len(dropped),
        "topDroppedEnglishMatches": english_drops.most_common(20),
        "perLabel": {label: per_label.get(label, 0) for label in classes},
        "perLabelBySplit": {
            label: {split: per_split[label].get(split, 0) for split in SPLIT_SOURCE_CSVS} for label in classes
        },
        "gaps": [label for label in classes if per_label.get(label, 0) == 0],
        "missingVideos": missing_videos,
    }
    return kept, summary


def select_test_signers(
    rows: list[dict[str, str]],
    seed: int,
    count: int,
    required_labels: Optional[set[str]] = None,
) -> tuple[list[str], list[str]]:
    """Pick ``count`` signers from ``test.csv`` who have every mapped class there.

    A signer who appears only in ``train.csv`` or ``val.csv`` is not eligible.
    Labels on those rows do not count. Returns ``(selected, eligible)``.
    Deterministic for a given row set and seed.
    """

    if required_labels is None:
        required = {row["label"] for row in rows if row.get("label")}
    else:
        required = set(required_labels)
    labels_by_signer: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("sourceCsv") != TEST_SOURCE_CSV:
            continue
        labels_by_signer[row["participantId"]].add(row["label"])
    eligible = sorted(signer for signer, labels in labels_by_signer.items() if required <= labels)
    if len(eligible) < count:
        raise AslCitizenError(
            f"Only {len(eligible)} test-split signers have every mapped class; {count} test signers were requested."
        )
    selected = sorted(random.Random(seed).sample(eligible, count))
    return selected, eligible


def official_split_pools(
    rows: list[dict[str, str]],
    seed: int,
    count: int,
    required_labels: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Held-out signers from ``test.csv``; training signers from ``train.csv`` and ``val.csv``.

    A ``test.csv`` signer left out of the held-out list, including one who lacks
    a mapped class, is in neither pool and is counted as excluded.
    """

    selected, eligible = select_test_signers(rows, seed, count, required_labels)
    test_split = sorted({row["participantId"] for row in rows if row.get("sourceCsv") == TEST_SOURCE_CSV})
    train_val = sorted({row["participantId"] for row in rows if row.get("sourceCsv") in TRAIN_SOURCE_CSVS})
    selected_set = set(selected)
    excluded = [signer for signer in test_split if signer not in selected_set]
    return {
        "selected": selected,
        "eligible": eligible,
        "testSplitParticipants": test_split,
        "trainValParticipants": train_val,
        "excluded": excluded,
        "excludedCount": len(excluded),
    }


def _app_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(APP_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_filtered_csv(rows: list[dict[str, str]], path: Path = FILTERED_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FILTERED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in FILTERED_COLUMNS})


def read_filtered_csv(path: Path = FILTERED_CSV_PATH) -> list[dict[str, str]]:
    if not path.exists():
        raise AslCitizenError(f"{path} does not exist. Run `asl_citizen filter` first.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_csv_hashes(csv_paths: list[Path]) -> dict[str, str]:
    return {_app_relative(path): sha256_file(path) for path in csv_paths}


def participants_by_split(csv_hashes: dict[str, str]) -> dict[str, list[str]]:
    """Every participant ID in each hashed split CSV, including unmapped glosses."""

    participant_column, _, _ = _columns()
    grouped: dict[str, set[str]] = defaultdict(set)
    for relative in csv_hashes:
        path = Path(relative) if Path(relative).is_absolute() else APP_ROOT / relative
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                record = {key.strip(): (value or "").strip() for key, value in raw.items() if key}
                grouped[path.name].add(record[participant_column])
    return {name: sorted(signers) for name, signers in grouped.items()}


def source_csv_by_video(path: Path = FILTERED_CSV_PATH) -> dict[str, str]:
    """Map a filtered video file name to its official split CSV."""

    mapping: dict[str, str] = {}
    for row in read_filtered_csv(path):
        mapping[Path(row["videoFile"]).name] = row["sourceCsv"]
    return mapping


def write_test_signers(
    selected: list[str],
    eligible: list[str],
    seed: int,
    count: int,
    csv_hashes: dict[str, str],
    path: Path = TEST_SIGNERS_PATH,
    test_split_participants: Optional[list[str]] = None,
    train_val_participants: Optional[list[str]] = None,
    excluded_count: int = 0,
) -> None:
    if path.exists():
        raise AslCitizenError(f"{path} already exists. The held-out signer list is fixed and is never regenerated.")
    write_json(
        path,
        {
            "dataset": DATASET,
            "seed": seed,
            "count": count,
            "eligibilityRule": "test.csv signer has at least one clip of every class mapped in gloss_map.json",
            "eligibleCount": len(eligible),
            "excludedCount": excluded_count,
            "testSigners": selected,
            "testSplitParticipants": list(test_split_participants or []),
            "trainValParticipants": list(train_val_participants or []),
            "sourceCsvSha256": csv_hashes,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        },
    )


def load_test_signers(path: Path = TEST_SIGNERS_PATH, verify_hashes: bool = True) -> dict[str, Any]:
    """Load the fixed held-out list. Fails if the source CSVs changed since selection."""

    if not path.exists():
        raise AslCitizenError(f"{path} does not exist. Run `asl_citizen select-test-signers` first.")
    payload = read_json(path)
    if verify_hashes:
        for relative, expected in payload["sourceCsvSha256"].items():
            csv_path = Path(relative) if Path(relative).is_absolute() else APP_ROOT / relative
            if not csv_path.exists():
                raise AslCitizenError(f"{relative} listed in {path.name} is missing.")
            if sha256_file(csv_path) != expected:
                raise AslCitizenError(f"{relative} changed since the held-out signers were chosen.")
    return payload


def load_examples(extractor_version: str, path: Optional[Path] = None) -> list[TrainingExample]:
    manifest = path or landmarks_path(extractor_version)
    if not manifest.exists():
        return []
    examples = load_manifest_examples([manifest])
    for example in examples:
        example.dataset = DATASET
    return examples


def _print_candidates(rows: list[dict[str, str]], classes: list[str]) -> None:
    for label, groups in gloss_candidates(rows, classes).items():
        print(f"\n{label}")
        if not groups["likely"]:
            print("  likely: none")
        for gloss, count in groups["likely"]:
            print(f"  likely:   {gloss!r} ({count} clips)")
        for gloss, count in groups["contains"][:30]:
            print(f"  contains: {gloss!r} ({count} clips)")
        if len(groups["contains"]) > 30:
            print(f"  contains: ... {len(groups['contains']) - 30} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASL Citizen for the held-out sign evaluation.")
    parser.add_argument("--root", type=Path, default=ASL_CITIZEN_DIR, help="Unpacked ASL Citizen folder.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("glosses", help="Print gloss strings that look like each of the eight classes.")
    subparsers.add_parser("filter", help="Write the filtered clip list using gloss_map.json.")
    subparsers.add_parser("select-test-signers", help="Choose and save the fixed held-out signer list.")
    args = parser.parse_args()

    classes = read_json(GLOSS_MAP_PATH)["classes"]
    try:
        if args.command in ("glosses", "filter"):
            csv_paths = find_split_csvs(args.root)
            if not csv_paths:
                raise AslCitizenError(f"No ASL Citizen split CSVs found under {args.root}.")
            rows = read_rows(csv_paths)
            if args.command == "glosses":
                print(f"{len(rows)} rows from {len(csv_paths)} CSVs.")
                _print_candidates(rows, classes)
                return
            mapping = load_gloss_map().get(DATASET, {})
            if not mapping:
                raise AslCitizenError("gloss_map.json has no asl-citizen entries yet. Run `glosses` and fill it in.")
            kept, summary = filter_rows(rows, mapping, classes, index_videos(video_index_roots(args.root)))
            summary["sourceCsvSha256"] = source_csv_hashes(csv_paths)
            summary["glossMapSha256"] = sha256_file(GLOSS_MAP_PATH)
            write_filtered_csv(kept)
            write_json(FILTER_SUMMARY_PATH, summary)
            print(f"Kept {summary['mappedRows']} of {summary['totalRows']} clips; dropped {summary['droppedRows']}.")
            print("Per class: " + ", ".join(f"{label}={n}" for label, n in summary["perLabel"].items()))
            for label, counts in summary["perLabelBySplit"].items():
                print(
                    f"  {label}: train={counts['train.csv']} val={counts['val.csv']} test={counts['test.csv']}"
                )
            if summary["gaps"]:
                print(f"Gaps (no ASL Citizen clips): {', '.join(summary['gaps'])}.")
            else:
                print("Gaps: none.")
            if summary["missingVideos"]:
                print(f"WARNING: {summary['missingVideos']} mapped rows have no video file on disk.")
            return

        rows = read_filtered_csv()
        settings = load_config()["aslCitizen"]
        summary = read_json(FILTER_SUMMARY_PATH)
        required = set(load_gloss_map().get(DATASET, {}).values())
        pools = official_split_pools(rows, settings["testSignerSeed"], settings["testSignerCount"], required)
        by_split = participants_by_split(summary["sourceCsvSha256"])
        test_split = by_split.get(TEST_SOURCE_CSV, [])
        train_val = sorted(set(by_split.get("train.csv", [])) | set(by_split.get("val.csv", [])))
        selected_set = set(pools["selected"])
        excluded = sum(1 for signer in test_split if signer not in selected_set)
        write_test_signers(
            pools["selected"],
            pools["eligible"],
            settings["testSignerSeed"],
            settings["testSignerCount"],
            summary["sourceCsvSha256"],
            test_split_participants=test_split,
            train_val_participants=train_val,
            excluded_count=excluded,
        )
        excluded_word = "signer" if excluded == 1 else "signers"
        print(
            f"Saved {len(pools['selected'])} held-out signers "
            f"(of {len(pools['eligible'])} eligible). "
            f"Excluded {excluded} test-split {excluded_word}."
        )
    except AslCitizenError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
