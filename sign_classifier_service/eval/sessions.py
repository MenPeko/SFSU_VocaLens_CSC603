"""Recording sessions: one signer, one day, one setup.

A session folder is ``data/sign-samples/sessions/<sessionId>/`` with a
``session.json`` and one subfolder per label. The capture server writes into
the folder named by ``SIGN_TRAINING_SESSION`` (see
``src/services/sign-training-store.ts``).
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from ..classifier import CORE_SIGN_LABELS, TrainingExample, load_training_examples
from . import SESSIONS_DIR, TEST_SESSIONS_PATH, load_config, read_json, write_json

ROLES = ("train", "validation", "test")
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class RecordingSession:
    session_id: str
    role: str
    signer: str
    recorded_on: str
    lighting: str
    camera: str
    extractor_version: str
    path: Path


def slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"{value!r} has no letters or digits")
    return slug


def make_session_id(recorded_on: str, signer: str, lighting: str) -> str:
    return f"{recorded_on}_{slugify(signer)}_{slugify(lighting)}"


def create_session(
    sessions_dir: Path,
    role: str,
    signer: str,
    lighting: str,
    camera: str,
    recorded_on: Optional[str] = None,
) -> RecordingSession:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    own_signer = load_config()["ownSigner"]
    if role == "validation" and slugify(signer) != own_signer:
        raise ValueError(f"validation sessions are recorded by {own_signer} only, not {slugify(signer)}")
    recorded_on = recorded_on or date.today().isoformat()
    date.fromisoformat(recorded_on)
    session_id = make_session_id(recorded_on, signer, lighting)
    session_dir = sessions_dir / session_id
    session_path = session_dir / "session.json"
    if session_path.exists():
        raise FileExistsError(f"{session_path} already exists; sessions are never overwritten")

    payload = {
        "sessionId": session_id,
        "role": role,
        "signer": slugify(signer),
        "recordedOn": recorded_on,
        "lighting": slugify(lighting),
        "camera": slugify(camera),
        "extractorVersion": load_config()["requiredExtractorVersion"],
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    write_json(session_path, payload)
    for label in CORE_SIGN_LABELS:
        (session_dir / label).mkdir(exist_ok=True)
    return _session_from_payload(payload, session_dir)


def _session_from_payload(payload: dict, session_dir: Path) -> RecordingSession:
    return RecordingSession(
        session_id=str(payload["sessionId"]),
        role=str(payload["role"]),
        signer=str(payload["signer"]),
        recorded_on=str(payload["recordedOn"]),
        lighting=str(payload["lighting"]),
        camera=str(payload.get("camera", "")),
        extractor_version=str(payload.get("extractorVersion", "")),
        path=session_dir,
    )


def load_sessions(sessions_dir: Path = SESSIONS_DIR) -> tuple[list[RecordingSession], list[str]]:
    """Return every session plus a list of problems (folders without a valid session.json)."""

    sessions: list[RecordingSession] = []
    problems: list[str] = []
    if not sessions_dir.exists():
        return sessions, problems

    for session_dir in sorted(path for path in sessions_dir.iterdir() if path.is_dir()):
        session_path = session_dir / "session.json"
        if not session_path.exists():
            problems.append(f"{session_dir.name}: no session.json")
            continue
        try:
            session = _session_from_payload(read_json(session_path), session_dir)
        except (KeyError, ValueError) as error:
            problems.append(f"{session_dir.name}: unreadable session.json ({error})")
            continue
        if session.session_id != session_dir.name:
            problems.append(f"{session_dir.name}: session.json says sessionId={session.session_id}")
            continue
        if session.role not in ROLES:
            problems.append(f"{session_dir.name}: role={session.role!r} is not one of {', '.join(ROLES)}")
            continue
        sessions.append(session)
    return sessions, problems


def load_test_session_registry(path: Path = TEST_SESSIONS_PATH) -> set[str]:
    return set(read_json(path).get("testSessions", []))


def load_session_examples(session: RecordingSession) -> tuple[list[TrainingExample], list[str]]:
    """Load a session's samples, stamped with its id and signer.

    A sample whose own ``sessionId`` names a different session was moved by
    hand between folders; that is reported as a problem and the sample is not
    returned.
    """

    examples: list[TrainingExample] = []
    problems: list[str] = []
    for example in load_training_examples(session.path):
        if example.session_id and example.session_id != session.session_id:
            problems.append(
                f"{example.sample_path}: stamped sessionId={example.session_id} but stored in {session.session_id}"
            )
            continue
        example.dataset = "own"
        example.session_id = session.session_id
        example.signer_id = session.signer
        examples.append(example)
    return examples, problems


def _print_sessions(sessions_dir: Path) -> None:
    sessions, problems = load_sessions(sessions_dir)
    registry = load_test_session_registry()
    if not sessions:
        print(f"No sessions under {sessions_dir}.")
    for session in sessions:
        examples, _ = load_session_examples(session)
        counts = Counter(example.label for example in examples)
        registered = "registered" if session.session_id in registry else "not in test_sessions.json"
        role_note = f"{session.role} ({registered})" if session.role == "test" else session.role
        print(f"{session.session_id}  role={role_note}  signer={session.signer}  clips={len(examples)}")
        print("   " + ", ".join(f"{label}={counts.get(label, 0)}" for label in CORE_SIGN_LABELS))
    for problem in problems:
        print(f"PROBLEM: {problem}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and list sign recording sessions.")
    parser.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="Create a session folder and its session.json.")
    new_parser.add_argument("--role", choices=ROLES, required=True)
    new_parser.add_argument(
        "--signer", required=True, help="Use 'owner' for your own recordings. Validation sessions must be 'owner'."
    )
    new_parser.add_argument("--lighting", required=True, help="Short tag, e.g. desk-daylight.")
    new_parser.add_argument("--camera", required=True, help="Short tag, e.g. iphone13.")
    new_parser.add_argument("--date", default=None, help="Recording day, YYYY-MM-DD. Defaults to today.")

    subparsers.add_parser("list", help="List sessions, roles, and clip counts.")
    args = parser.parse_args()

    if args.command == "new":
        try:
            session = create_session(
                args.sessions_dir, args.role, args.signer, args.lighting, args.camera, recorded_on=args.date
            )
        except (FileExistsError, ValueError) as error:
            raise SystemExit(str(error))
        print(f"Created {session.path}")
        print(f"Record with: SIGN_TRAINING_SESSION={session.session_id} bun run dev")
        if session.role == "test":
            print(f'Add "{session.session_id}" to {TEST_SESSIONS_PATH} and commit it.')
        elif session.role == "validation":
            print("Validation clips are used only for early stopping. Record on a different day or lighting from every test session.")
    else:
        _print_sessions(args.sessions_dir)


if __name__ == "__main__":
    main()
