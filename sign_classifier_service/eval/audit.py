"""Read-only audit of every landmark clip on disk.

    python -m sign_classifier_service.eval.audit

Writes reports/dataset-audit.json and reports/dataset-audit.md. Aggregate
numbers only: no clip names, signer IDs, or landmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import ValidationError

from ..classifier import APP_ROOT, CORE_SIGN_LABELS
from ..extractor_version import MEDIAPIPE_EXTRACTOR_VERSION
from ..schemas import LandmarkFrameModel
from . import ASL_CITIZEN_DIR, LANDMARKS_DIR, SESSIONS_DIR, read_json, write_json
from . import asl_citizen

FLAT_SAMPLES_DIR = APP_ROOT / "data" / "sign-samples"
ARCHIVE_DIR = APP_ROOT / "data" / "sign-samples-archive"
PHASE0_REPORTS_DIR = APP_ROOT / "reports"
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
ARM_JOINTS = ("leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist")


def _stats(values: list[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 3),
        "p50": round(statistics.median(ordered), 3),
        "mean": round(statistics.fmean(ordered), 3),
        "max": round(ordered[-1], 3),
    }


def _share(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def _landmark_digest(frames: list[dict[str, Any]]) -> str:
    def rounded(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, list):
            return [rounded(item) for item in value]
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items() if key != "timestampMs" and item is not None}
        return value

    encoded = json.dumps(rounded(frames), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SourceAudit:
    """Accumulates per-clip facts for one data source."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.clips = 0
        self.labels: Counter[str] = Counter()
        self.extractors: Counter[str] = Counter()
        self.extractors_by_label: dict[str, Counter[str]] = defaultdict(Counter)
        self.signers: set[str] = set()
        self.sessions: set[str] = set()
        self.recording_days: set[str] = set()
        self.frame_counts: list[float] = []
        self.durations_ms: list[float] = []
        self.fps: list[float] = []
        self.pose_keys: Counter[str] = Counter()
        self.hand_sizes: Counter[int] = Counter()
        self.face_sizes: dict[str, Counter[int]] = defaultdict(Counter)
        self.frames = 0
        self.frames_left = 0
        self.frames_right = 0
        self.frames_both = 0
        self.frames_none = 0
        self.frames_partial_hand = 0
        self.frames_missing_arm_joint = 0
        self.frames_missing_lips = 0
        self.clips_without_hands = 0
        self.clips_under_min_frames = 0
        self.right_dominant: dict[str, list[int]] = defaultdict(list)
        self.schema_failures = 0
        self.ineligible: Counter[str] = Counter()
        self.digests: Counter[str] = Counter()
        self.has_rgb_grid = 0
        self.has_skeleton_grid = 0

    def add(self, record: dict[str, Any], label: str, signer: str = "", session: str = "", day: str = "") -> None:
        self.clips += 1
        self.labels[label] += 1
        extractor = str(record.get("extractorVersion") or "missing")
        self.extractors[extractor] += 1
        self.extractors_by_label[label][extractor] += 1
        if signer:
            self.signers.add(signer)
        if session:
            self.sessions.add(session)
        if day:
            self.recording_days.add(day)
        if record.get("base64Grid"):
            self.has_rgb_grid += 1
        if record.get("base64SkeletonGrid"):
            self.has_skeleton_grid += 1

        if isinstance(record.get("labelSource"), str) and record["labelSource"] not in ("", "manual_capture"):
            self.ineligible["labelSource not manual_capture"] += 1
        if record.get("recognizerType") == "gemini" or record.get("fallbackUsed") is True or "rawGeminiOutput" in record:
            self.ineligible["Gemini-derived metadata"] += 1
        if extractor == "missing":
            self.ineligible["missing extractorVersion"] += 1

        raw_frames = record.get("landmarkSequence") or []
        self.digests[_landmark_digest(raw_frames)] += 1
        try:
            frames = [LandmarkFrameModel.model_validate(frame) for frame in raw_frames]
        except ValidationError:
            self.schema_failures += 1
            return

        self.frame_counts.append(len(frames))
        if len(frames) < 6:
            self.clips_under_min_frames += 1
        timestamps = record.get("frameTimestamps") or [frame.timestampMs for frame in frames]
        duration = record.get("sequenceDurationMs")
        if not duration and len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]
        if duration:
            self.durations_ms.append(float(duration))
            if len(frames) > 1:
                self.fps.append((len(frames) - 1) * 1000.0 / float(duration))

        left_frames = right_frames = 0
        for frame in frames:
            self.frames += 1
            has_left, has_right = bool(frame.leftHand), bool(frame.rightHand)
            left_frames += has_left
            right_frames += has_right
            self.frames_left += has_left
            self.frames_right += has_right
            self.frames_both += has_left and has_right
            self.frames_none += not has_left and not has_right
            for hand in (frame.leftHand, frame.rightHand):
                if hand:
                    self.hand_sizes[len(hand)] += 1
                    if len(hand) != 21:
                        self.frames_partial_hand += 1
            for key, value in frame.pose.model_dump().items():
                if value is not None:
                    self.pose_keys[key] += 1
            if any(getattr(frame.pose, joint) is None for joint in ARM_JOINTS):
                self.frames_missing_arm_joint += 1
            for part, points in frame.face.model_dump().items():
                self.face_sizes[part][len(points)] += 1
            if not frame.face.lips:
                self.frames_missing_lips += 1
        if left_frames == 0 and right_frames == 0:
            self.clips_without_hands += 1
        elif left_frames != right_frames:
            self.right_dominant[label].append(int(right_frames > left_frames))

    def to_dict(self) -> dict[str, Any]:
        duplicate_copies = sum(count - 1 for count in self.digests.values() if count > 1)
        present = [count for count in self.labels.values() if count > 0]
        return {
            "name": self.name,
            "description": self.description,
            "clips": self.clips,
            "perLabel": {label: self.labels.get(label, 0) for label in CORE_SIGN_LABELS},
            "largestToSmallestRatio": round(max(present) / min(present), 3) if present else None,
            "extractorVersions": dict(self.extractors),
            "extractorVersionsByLabel": {label: dict(counts) for label, counts in self.extractors_by_label.items()},
            "uniqueSigners": len(self.signers) if self.signers else None,
            "sessions": len(self.sessions) if self.sessions else None,
            "recordingDays": len(self.recording_days) if self.recording_days else None,
            "framesPerClip": _stats(self.frame_counts),
            "durationMs": _stats(self.durations_ms),
            "effectiveFps": _stats(self.fps),
            "clipsUnderMinFrames": self.clips_under_min_frames,
            "landmarkLayout": {
                "poseKeysPresentFrames": dict(self.pose_keys),
                "handPointCounts": {str(size): count for size, count in sorted(self.hand_sizes.items())},
                "faceSubsetSizes": {part: {str(size): count for size, count in sorted(sizes.items())} for part, sizes in self.face_sizes.items()},
            },
            "missingLandmarks": {
                "frames": self.frames,
                "leftHandPresent": _share(self.frames_left, self.frames),
                "rightHandPresent": _share(self.frames_right, self.frames),
                "bothHandsPresent": _share(self.frames_both, self.frames),
                "noHandPresent": _share(self.frames_none, self.frames),
                "partialHand": _share(self.frames_partial_hand, self.frames),
                "anyArmJointMissing": _share(self.frames_missing_arm_joint, self.frames),
                "lipsMissing": _share(self.frames_missing_lips, self.frames),
                "clipsWithoutAnyHand": self.clips_without_hands,
            },
            "rightHandDominantShare": {
                label: {"share": round(sum(flags) / len(flags), 3), "clips": len(flags)}
                for label, flags in sorted(self.right_dominant.items())
            },
            "schemaValidationFailures": self.schema_failures,
            "trainingIneligible": dict(self.ineligible),
            "exactDuplicateCopies": duplicate_copies,
            "rgbGridThumbnails": self.has_rgb_grid,
            "skeletonGridThumbnails": self.has_skeleton_grid,
        }


def _day(captured_at: Any) -> str:
    if isinstance(captured_at, (int, float)) and captured_at > 0:
        return datetime.fromtimestamp(captured_at / 1000, tz=timezone.utc).date().isoformat()
    return ""


def _sample_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*/*.json")):
        if path.name != "session.json" and path.parent.name in CORE_SIGN_LABELS:
            yield path


def recording_blocks(paths: list[Path], gap_ms: int = 120_000) -> list[dict[str, Any]]:
    """Contiguous capture runs, split wherever two clips are more than ``gap_ms`` apart or the label changes."""

    rows = []
    for path in paths:
        record = read_json(path)
        rows.append((record.get("capturedAt") or 0, record.get("label") or path.parent.name))
    rows.sort()
    blocks: list[dict[str, Any]] = []
    previous = None
    for captured_at, label in rows:
        if not blocks or blocks[-1]["label"] != label or (previous is not None and captured_at - previous > gap_ms):
            blocks.append({"label": label, "clips": 0, "day": _day(captured_at), "start": captured_at, "end": captured_at})
        blocks[-1]["clips"] += 1
        blocks[-1]["end"] = captured_at
        previous = captured_at
    return [
        {"label": block["label"], "clips": block["clips"], "day": block["day"], "minutes": round((block["end"] - block["start"]) / 60000, 1)}
        for block in blocks
    ]


def _count_videos(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)


def audit(
    flat_dir: Path = FLAT_SAMPLES_DIR,
    sessions_dir: Path = SESSIONS_DIR,
    archive_dir: Path = ARCHIVE_DIR,
    landmarks_dir: Path = LANDMARKS_DIR,
    asl_dir: Path = ASL_CITIZEN_DIR,
) -> dict[str, Any]:
    sources: list[SourceAudit] = []

    flat = SourceAudit("own-flat", "data/sign-samples/<label>/: legacy flat layout, no session or signer fields")
    flat_paths = list(_sample_files(flat_dir))
    for path in flat_paths:
        record = read_json(path)
        flat.add(record, str(record.get("label") or path.parent.name), day=_day(record.get("capturedAt")))
    sources.append(flat)

    session_rows: list[dict[str, Any]] = []
    sessions = SourceAudit("own-sessions", "data/sign-samples/sessions/<id>/: schema-v2 session layout used by the held-out harness")
    for session_path in sorted(sessions_dir.glob("*/session.json")):
        meta = read_json(session_path)
        clip_paths = list(_sample_files(session_path.parent))
        session_rows.append({"role": meta.get("role"), "extractorVersion": meta.get("extractorVersion"), "clips": len(clip_paths)})
        for path in clip_paths:
            record = read_json(path)
            sessions.add(
                record,
                str(record.get("label") or path.parent.name),
                signer=str(meta.get("signer") or ""),
                session=str(meta.get("sessionId") or ""),
                day=str(meta.get("recordedOn") or ""),
            )
    sources.append(sessions)

    archive = SourceAudit("own-archive", "data/sign-samples-archive/: clips set aside by hand; never loaded by the trainer")
    for path in sorted(archive_dir.rglob("*.json")):
        record = read_json(path)
        archive.add(record, str(record.get("label") or "unknown"), day=_day(record.get("capturedAt")))
    sources.append(archive)

    asl_split_counts: Counter[str] = Counter()
    asl_videos_present = 0
    by_video: dict[str, str] = {}
    if asl_citizen.FILTERED_CSV_PATH.exists():
        by_video = asl_citizen.source_csv_by_video(asl_citizen.FILTERED_CSV_PATH)
    video_names = {path.name for path in asl_dir.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES} if asl_dir.exists() else set()
    for manifest in sorted(landmarks_dir.glob("asl-citizen/*/clips.jsonl")):
        source = SourceAudit(f"asl-citizen ({manifest.parent.name})", "data/landmarks/asl-citizen/: offline extraction with the app's browser extractor")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            source.add(record, str(record.get("label")), signer=str(record.get("signerId") or ""))
            video = Path(str(record.get("sourceVideoId") or "")).name
            asl_split_counts[by_video.get(video, "unmapped")] += 1
            asl_videos_present += video in video_names
        sources.append(source)

    flat_v2 = flat.extractors.get(MEDIAPIPE_EXTRACTOR_VERSION, 0)
    return {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "currentExtractorVersion": MEDIAPIPE_EXTRACTOR_VERSION,
        "classes": list(CORE_SIGN_LABELS),
        "sources": [source.to_dict() for source in sources],
        "ownFlatRecordingBlocks": recording_blocks(flat_paths),
        "ownSessions": session_rows,
        "aslCitizenClipsByOfficialSplit": dict(asl_split_counts),
        "rawVideo": {
            "ownRecordingsVideoFiles": _count_videos(flat_dir) + _count_videos(archive_dir),
            "ownRecordingsStoredMedia": "4x4 RGB grid JPEG and skeleton grid per clip (16 thumbnails), not video",
            "aslCitizenVideoFilesOnDisk": len(video_names),
            "aslCitizenExtractedClipsWithVideoOnDisk": asl_videos_present,
        },
        "trainableOnCurrentExtractor": {
            "ownFlat": flat_v2,
            "ownSessions": sessions.extractors.get(MEDIAPIPE_EXTRACTOR_VERSION, 0),
            "aslCitizen": sum(source.extractors.get(MEDIAPIPE_EXTRACTOR_VERSION, 0) for source in sources if source.name.startswith("asl-citizen")),
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset audit",
        "",
        f"Generated {report['createdAt'][:10]} by `python -m sign_classifier_service.eval.audit`. Aggregate numbers only.",
        f"Current extractor: `{report['currentExtractorVersion']}`.",
        "",
        "## Sources",
        "",
        "| Source | Clips | Signers | Sessions | Days | Extractor versions |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for source in report["sources"]:
        versions = ", ".join(f"`{name.split('+')[-1]}` {count}" for name, count in source["extractorVersions"].items()) or "none"
        lines.append(
            f"| {source['name']} | {source['clips']} | {_fmt(source['uniqueSigners'])} | {_fmt(source['sessions'])} | {_fmt(source['recordingDays'])} | {versions} |"
        )

    lines += ["", "## Clips per sign", "", "| Source | " + " | ".join(report["classes"]) + " | Largest/smallest |"]
    lines.append("| --- |" + " ---: |" * (len(report["classes"]) + 1))
    for source in report["sources"]:
        if source["clips"]:
            cells = [str(source["perLabel"][label]) for label in report["classes"]]
            lines.append(f"| {source['name']} | " + " | ".join(cells) + f" | {_fmt(source['largestToSmallestRatio'])} |")

    lines += ["", "## Sequence length and landmarks", ""]
    lines.append("| Source | Frames p50 (min-max) | Duration ms p50 | Effective fps p50 | No hand | Both hands | Arm joint missing | Lips missing | Clips < 6 frames | Exact duplicates |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for source in report["sources"]:
        if not source["clips"]:
            continue
        frames, missing = source["framesPerClip"], source["missingLandmarks"]
        lines.append(
            f"| {source['name']} | {frames['p50']:.0f} ({frames['min']:.0f}-{frames['max']:.0f}) | {_fmt(source['durationMs'] and source['durationMs']['p50'])} "
            f"| {_fmt(source['effectiveFps'] and source['effectiveFps']['p50'])} | {_fmt(missing['noHandPresent'])} | {_fmt(missing['bothHandsPresent'])} "
            f"| {_fmt(missing['anyArmJointMissing'])} | {_fmt(missing['lipsMissing'])} | {source['clipsUnderMinFrames']} | {source['exactDuplicateCopies']} |"
        )

    lines += ["", "Landmark layout per frame (from the stored samples):", ""]
    for source in report["sources"]:
        if source["clips"]:
            layout = source["landmarkLayout"]
            faces = ", ".join(f"{part} {'/'.join(sizes)}" for part, sizes in layout["faceSubsetSizes"].items())
            lines.append(
                f"- {source['name']}: pose keys {len(layout['poseKeysPresentFrames'])}, hand point counts {list(layout['handPointCounts'])}, face subsets: {faces}."
            )

    lines += ["", "## Handedness (share of clips where the right hand is seen in more frames)", ""]
    lines.append("| Source | " + " | ".join(report["classes"]) + " |")
    lines.append("| --- |" + " ---: |" * len(report["classes"]))
    for source in report["sources"]:
        if source["clips"]:
            shares = source["rightHandDominantShare"]
            lines.append(f"| {source['name']} | " + " | ".join(_fmt(shares.get(label, {}).get("share")) for label in report["classes"]) + " |")

    lines += ["", "## Own recordings: capture blocks (flat layout)", "", "| Sign | Clips | Day | Minutes |", "| --- | ---: | --- | ---: |"]
    for block in report["ownFlatRecordingBlocks"]:
        lines.append(f"| {block['label']} | {block['clips']} | {block['day']} | {block['minutes']} |")

    lines += ["", "## Sessions (held-out harness layout)", ""]
    if report["ownSessions"]:
        lines += ["| Role | Clips |", "| --- | ---: |"]
        lines += [f"| {row['role']} | {row['clips']} |" for row in report["ownSessions"]]
    else:
        lines.append("No session folders.")

    raw = report["rawVideo"]
    lines += [
        "",
        "## Raw video",
        "",
        f"- Own recordings: {raw['ownRecordingsVideoFiles']} video files. Stored media is {raw['ownRecordingsStoredMedia']}.",
        f"- ASL Citizen: {raw['aslCitizenVideoFilesOnDisk']} video files on disk; {raw['aslCitizenExtractedClipsWithVideoOnDisk']} extracted clips have their source video.",
        f"- ASL Citizen clips by official split: {json.dumps(report['aslCitizenClipsByOfficialSplit'])}.",
        "",
        "## Trainable on the current extractor",
        "",
    ]
    lines += [f"- {name}: {count}" for name, count in report["trainableOnCurrentExtractor"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit landmark clips on disk (read-only).")
    parser.add_argument("--output-dir", type=Path, default=PHASE0_REPORTS_DIR)
    args = parser.parse_args()
    report = audit()
    write_json(args.output_dir / "dataset-audit.json", report)
    (args.output_dir / "dataset-audit.md").write_text(render(report), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'dataset-audit.md'}")


if __name__ == "__main__":
    main()
