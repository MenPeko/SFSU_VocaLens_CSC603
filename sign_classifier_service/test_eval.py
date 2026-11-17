from __future__ import annotations

import csv
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from typing import Optional

from .classifier import CORE_SIGN_LABELS, TrainingExample, vectorize_sequence
from .eval import GLOSS_MAP_PATH, load_config, sha256_file
from .eval import asl_citizen, leakage, run as eval_run
from .eval.sessions import create_session, load_sessions
from .extractor_version import MEDIAPIPE_EXTRACTOR_VERSION
from .schemas import LandmarkFrameModel
from .train import build_artifact, group_split

V1_EXTRACTOR = "mediapipe-holistic-js@0.5.1675471629+schema-v1"


def _point(x: float, y: float, z: float = 0.0) -> dict:
    return {"x": x, "y": y, "z": z}


def make_frames(label: str, rng: random.Random, frame_count: int = 12, jitter: float = 0.025) -> list[dict]:
    """Synthetic landmark frames: each sign has its own wrist height and hand spread.

    Every frame moves the whole hand rigidly by a random offset. Separate clips
    then sit well above the duplicate threshold, while the hand-shape features
    (which depend on distances inside the hand) stay the same for a sign.
    """

    index = CORE_SIGN_LABELS.index(label)
    frames = []
    for frame_index in range(frame_count):
        dx, dy, dz = (rng.gauss(0.0, jitter) for _ in range(3))
        wrist_y = 0.6 - index * 0.12 + frame_index * 0.01 + dy
        hand = [
            _point(0.35 + dx + (k % 5) * 0.02 * (1 + index * 0.2), wrist_y - (k // 5) * 0.04, dz)
            for k in range(21)
        ]
        frames.append(
            {
                "timestampMs": 1000 + frame_index * 42,
                "pose": {
                    "leftShoulder": _point(-0.5, 0.0),
                    "rightShoulder": _point(0.5, 0.0),
                    "leftElbow": _point(-0.45, 0.3),
                    "rightElbow": _point(0.42, 0.3),
                    "leftWrist": _point(-0.4, 0.6),
                    "rightWrist": _point(0.35 + dx, wrist_y, dz),
                },
                "leftHand": [],
                "rightHand": hand,
                "face": {"leftEyebrow": [], "rightEyebrow": [], "leftEye": [], "rightEye": [], "lips": [_point(0.0, -0.3)]},
            }
        )
    return frames


def write_session_clip(session_dir: Path, session_id: str, label: str, name: str, frames: list[dict], extractor: str = MEDIAPIPE_EXTRACTOR_VERSION) -> Path:
    path = session_dir / label / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "labelSource": "manual_capture",
        "source": "device-camera",
        "landmarkSequence": frames,
        "frameTimestamps": [frame["timestampMs"] for frame in frames],
        "extractorVersion": extractor,
        "sessionId": session_id,
        "signer": "owner",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_example(label: str, rng: random.Random, dataset: str = "own", signer: str = "owner", session: str = "s1", video: str = "") -> TrainingExample:
    frames = [LandmarkFrameModel.model_validate(frame) for frame in make_frames(label, rng)]
    return TrainingExample(
        label=label,
        frames=frames,
        source="test",
        sample_path=Path(f"{dataset}-{signer}-{session}-{video}-{rng.random()}.json"),
        vector=vectorize_sequence(frames),
        extractor_version=MEDIAPIPE_EXTRACTOR_VERSION,
        dataset=dataset,
        signer_id=signer,
        session_id=session if dataset == "own" else "",
        source_video_id=video,
    )


class Workspace:
    """Temporary sessions, registry, reports, and models folders for one test."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.sessions = root / "sessions"
        self.registry = root / "test_sessions.json"
        self.reports = root / "reports"
        self.models = root / "models"
        self.asl_landmarks = root / "asl" / "clips.jsonl"
        self.asl_signers = root / "asl-citizen-test-signers.json"
        self.asl_summary = root / "asl-citizen-filter-summary.json"
        self.asl_filtered = root / "asl-citizen-filtered.csv"
        self.write_registry([])

    def write_registry(self, session_ids: list[str]) -> None:
        self.registry.write_text(json.dumps({"testSessions": session_ids}), encoding="utf-8")

    def add_session(self, role: str, recorded_on: str, lighting: str, clips_per_sign: int, seed: int, signer: str = "owner") -> str:
        session = create_session(self.sessions, role, signer, lighting, "iphone13", recorded_on=recorded_on)
        rng = random.Random(seed)
        for label in CORE_SIGN_LABELS:
            for clip in range(clips_per_sign):
                write_session_clip(session.path, session.session_id, label, f"{label}-{clip}", make_frames(label, rng))
        return session.session_id

    def run(self, experiment: str = "baseline", run_id: Optional[str] = None) -> dict:
        return eval_run.run_experiment(
            experiment,
            sessions_dir=self.sessions,
            reports_dir=self.reports,
            models_dir=self.models,
            registry_path=self.registry,
            asl_landmarks=self.asl_landmarks,
            asl_test_signers_path=self.asl_signers,
            asl_filter_summary_path=self.asl_summary,
            asl_filtered_csv_path=self.asl_filtered,
            run_id=run_id,
        )


class GroupSplitTests(unittest.TestCase):
    def test_no_group_lands_on_both_sides(self) -> None:
        rng = random.Random(1)
        examples = [make_example(label, rng, session=f"s{session}") for session in range(5) for label in CORE_SIGN_LABELS]
        split = group_split(examples, validation_ratio=0.3, seed=7)
        self.assertTrue(split.validation)
        self.assertFalse({e.group_key for e in split.train} & {e.group_key for e in split.validation})
        self.assertEqual(group_split(examples, 0.3, 7).validation, split.validation)

    def test_single_group_or_zero_ratio_keeps_everything_in_training(self) -> None:
        rng = random.Random(2)
        examples = [make_example(label, rng) for label in CORE_SIGN_LABELS]
        self.assertEqual(len(group_split(examples, 0.5, 1).train), len(examples))
        self.assertEqual(group_split(examples, 0.0, 1).validation, [])

    def test_harness_never_splits_by_clip(self) -> None:
        source = Path(eval_run.__file__).read_text(encoding="utf-8")
        self.assertNotIn("stratified_split", source)


class SessionTests(unittest.TestCase):
    def test_session_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            session = create_session(sessions_dir, "test", "Owner", "Kitchen Evening", "iphone13", recorded_on="2026-11-14")
            self.assertEqual(session.session_id, "2026-11-14_owner_kitchen-evening")
            with self.assertRaises(FileExistsError):
                create_session(sessions_dir, "train", "owner", "kitchen-evening", "iphone13", recorded_on="2026-11-14")
            loaded, problems = load_sessions(sessions_dir)
            self.assertEqual([s.role for s in loaded], ["test"])
            self.assertEqual(problems, [])

    def test_validation_role_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            session = create_session(sessions_dir, "validation", "owner", "office-overcast", "iphone13", recorded_on="2026-11-12")
            self.assertEqual(session.role, "validation")
            with self.assertRaises(ValueError):
                create_session(sessions_dir, "validation", "juan", "lab", "iphone13", recorded_on="2026-11-12")
            loaded, problems = load_sessions(sessions_dir)
            self.assertEqual([(s.session_id, s.role) for s in loaded], [("2026-11-12_owner_office-overcast", "validation")])
            self.assertEqual(problems, [])

    def test_unknown_role_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            session = create_session(sessions_dir, "train", "owner", "desk", "iphone13", recorded_on="2026-11-12")
            payload = json.loads((session.path / "session.json").read_text())
            payload["role"] = "holdout"
            (session.path / "session.json").write_text(json.dumps(payload))
            loaded, problems = load_sessions(sessions_dir)
            self.assertEqual(loaded, [])
            self.assertIn("is not one of train, validation, test", problems[0])


class EarlyStoppingTests(unittest.TestCase):
    def _examples(self, seed: int, per_sign: int, session: str) -> list[TrainingExample]:
        rng = random.Random(seed)
        return [make_example(label, rng, session=session) for label in CORE_SIGN_LABELS for _ in range(per_sign)]

    def test_returns_best_epoch_weights_and_stops_after_patience(self) -> None:
        train = self._examples(1, 3, "train")
        # Permuted labels make validation loss rise as training fits, which forces a stop.
        validation = self._examples(2, 2, "validation")
        for example in validation:
            example.label = CORE_SIGN_LABELS[(CORE_SIGN_LABELS.index(example.label) + 1) % len(CORE_SIGN_LABELS)]
        common = dict(train_examples=train, validation_examples=validation, learning_rate=0.12, weight_decay=0.0005, model_version="m", training_run_id="r")
        stopped, history = build_artifact(epochs=60, early_stopping={"patience": 3, "minDelta": 0.0}, **common)
        selection = stopped["metrics"]["earlyStopping"]
        self.assertTrue(selection["stoppedEarly"])
        self.assertEqual(len(history), selection["stoppedEpoch"])
        self.assertEqual(selection["stoppedEpoch"] - selection["bestEpoch"], 3)
        losses = [entry["validationLoss"] for entry in history]
        self.assertEqual(selection["bestEpoch"], losses.index(min(losses)) + 1)

        replay, _ = build_artifact(epochs=selection["bestEpoch"], **common)
        self.assertEqual(stopped["weights"], replay["weights"])
        self.assertEqual(stopped["bias"], replay["bias"])
        self.assertNotIn("earlyStopping", replay["metrics"])

    def test_without_early_stopping_runs_every_epoch(self) -> None:
        _, history = build_artifact(
            train_examples=self._examples(3, 2, "train"),
            validation_examples=self._examples(4, 1, "validation"),
            epochs=7,
            learning_rate=0.12,
            weight_decay=0.0005,
            model_version="m",
            training_run_id="r",
        )
        self.assertEqual(len(history), 7)
        self.assertNotIn("validationLoss", history[0])


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.ws = Workspace(Path(self._temp.name))

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _standard_sessions(self) -> tuple[str, str]:
        train_id = self.ws.add_session("train", "2026-11-10", "desk-daylight", clips_per_sign=6, seed=10)
        self.validation_id = self.ws.add_session("validation", "2026-11-12", "office-overcast", clips_per_sign=3, seed=30)
        test_id = self.ws.add_session("test", "2026-11-14", "kitchen-evening", clips_per_sign=3, seed=20)
        self.ws.write_registry([test_id])
        return train_id, test_id

    def _hand_made_session(self, like: str, session_id: str, **fields: str) -> Path:
        """A session folder copied by hand, the way a manual edit would; the CLI cannot create these."""

        session_dir = self.ws.sessions / session_id
        session_dir.mkdir()
        payload = json.loads((self.ws.sessions / like / "session.json").read_text())
        payload.update({"sessionId": session_id, **fields})
        (session_dir / "session.json").write_text(json.dumps(payload))
        rng = random.Random(len(session_id))
        for label in CORE_SIGN_LABELS:
            write_session_clip(session_dir, session_id, label, f"{label}-0", make_frames(label, rng))
        return session_dir

    def test_no_training_sessions_stops_before_writing(self) -> None:
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("No training sessions found", str(context.exception))
        self.assertFalse(self.ws.reports.exists())

    def test_no_validation_session_stops_instead_of_using_test_sets(self) -> None:
        self.ws.add_session("train", "2026-11-10", "desk-daylight", clips_per_sign=6, seed=10)
        test_id = self.ws.add_session("test", "2026-11-14", "kitchen-evening", clips_per_sign=3, seed=20)
        self.ws.write_registry([test_id])
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("No validation session found", str(context.exception))
        self.assertIn("never falls back to the test sets", str(context.exception))
        self.assertFalse(self.ws.reports.exists())
        self.assertFalse(self.ws.models.exists())

    def test_validation_session_drives_early_stopping_and_tests_are_scored_once_after(self) -> None:
        train_id, test_id = self._standard_sessions()
        calls: list[tuple[str, object]] = []
        real_build, real_evaluate = eval_run.build_artifact, eval_run.evaluate_test_set

        def build(**kwargs):
            calls.append(("build", sorted({e.session_id for e in kwargs["validation_examples"]})))
            return real_build(**kwargs)

        def evaluate(artifact, examples, bootstrap):
            calls.append(("score", sorted({e.session_id for e in examples})))
            return real_evaluate(artifact, examples, bootstrap)

        with mock.patch.object(eval_run, "build_artifact", side_effect=build), mock.patch.object(
            eval_run, "evaluate_test_set", side_effect=evaluate
        ):
            run = self.ws.run(run_id="with-validation")

        self.assertEqual(calls[0], ("build", [self.validation_id]))
        self.assertEqual(calls[1:], [("score", [test_id])])
        self.assertEqual(run["validationClips"], 3 * len(CORE_SIGN_LABELS))
        self.assertEqual(run["trainClips"], 6 * len(CORE_SIGN_LABELS))
        self.assertEqual(run["earlyStopping"]["monitor"], "validationLoss")
        self.assertLessEqual(run["earlyStopping"]["bestEpoch"], run["earlyStopping"]["stoppedEpoch"])
        config = json.loads((self.ws.reports / "with-validation" / "config.json").read_text())
        roles = {entry["sessionId"]: entry["role"] for entry in config["provenance"]["sessions"]}
        self.assertEqual(roles, {train_id: "train", self.validation_id: "validation", test_id: "test"})
        self.assertIn("Early stopping kept epoch", (self.ws.reports / "with-validation" / "report.md").read_text())

    def test_validation_session_with_same_day_and_lighting_as_test_fails(self) -> None:
        _, test_id = self._standard_sessions()
        self._hand_made_session(test_id, f"{test_id}-val", role="validation")
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn(f"validation session {test_id}-val has the same day and lighting as test session {test_id}", str(context.exception))

    def test_validation_session_listed_as_test_fails(self) -> None:
        _, test_id = self._standard_sessions()
        self.ws.write_registry([test_id, self.validation_id])
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("says role=validation", str(context.exception))

    def test_teammate_validation_session_fails(self) -> None:
        self._standard_sessions()
        self._hand_made_session(self.validation_id, "2026-11-13_juan_lab", role="validation", signer="juan", recordedOn="2026-11-13", lighting="lab")
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("only owner validates", str(context.exception))

    def test_clip_copied_from_test_into_validation_fails(self) -> None:
        _, test_id = self._standard_sessions()
        payload = json.loads((self.ws.sessions / test_id / "help" / "help-0.json").read_text())
        payload["sessionId"] = self.validation_id
        (self.ws.sessions / self.validation_id / "help" / "copied.json").write_text(json.dumps(payload))
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("test set own-heldout", str(context.exception))
        self.assertIn("same landmarks as a validation clip", str(context.exception))

    def test_clip_copied_from_training_into_validation_fails(self) -> None:
        train_id, _ = self._standard_sessions()
        payload = json.loads((self.ws.sessions / train_id / "please" / "please-0.json").read_text())
        payload["sessionId"] = self.validation_id
        (self.ws.sessions / self.validation_id / "please" / "copied.json").write_text(json.dumps(payload))
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("validation:", str(context.exception))
        self.assertIn("same landmarks as a training clip", str(context.exception))

    def test_baseline_writes_reports_and_scores_each_test_set(self) -> None:
        self._standard_sessions()
        run = self.ws.run(run_id="baseline-test")
        run_dir = self.ws.reports / "baseline-test"
        for name in ("config.json", "metrics.json", "leak-check.json", "duplicates.json", "train-counts.json", "report.md"):
            self.assertTrue((run_dir / name).exists(), name)
        self.assertTrue((self.ws.models / "baseline-test.json").exists())
        own = run["testSets"][eval_run.OWN_TEST]
        self.assertEqual(own["clips"], 3 * len(CORE_SIGN_LABELS))
        self.assertGreater(own["accuracy"], 0.5)
        self.assertEqual(set(own["perSign"]), set(CORE_SIGN_LABELS))
        self.assertEqual(run["trainCounts"]["total"]["yes"], 6)
        self.assertEqual(run["decision"]["decision"], "baseline")
        config = json.loads((run_dir / "config.json").read_text())
        self.assertEqual(config["seed"], load_config()["seed"])
        self.assertIn("glossMapSha256", config["provenance"])

    def test_test_session_missing_from_registry_fails(self) -> None:
        self._standard_sessions()
        self.ws.write_registry([])
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("not in test_sessions.json", str(context.exception))
        leak = json.loads(next(self.ws.reports.glob("*/leak-check.json")).read_text())
        self.assertFalse(leak["ok"])

    def test_registry_entry_for_training_session_fails(self) -> None:
        train_id, test_id = self._standard_sessions()
        self.ws.write_registry([test_id, train_id])
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("says role=train", str(context.exception))

    def test_same_day_and_lighting_fails(self) -> None:
        train_id = self.ws.add_session("train", "2026-11-10", "desk-daylight", clips_per_sign=4, seed=1)
        test_id = f"{train_id}-retake"
        self._hand_made_session(train_id, test_id, role="test")
        self.ws.write_registry([test_id])
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("same day and lighting", str(context.exception))

    def test_clip_copied_into_test_session_fails(self) -> None:
        train_id, test_id = self._standard_sessions()
        source = self.ws.sessions / train_id / "yes" / "yes-0.json"
        payload = json.loads(source.read_text())
        payload["sessionId"] = test_id
        (self.ws.sessions / test_id / "yes" / "copied.json").write_text(json.dumps(payload))
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("same landmarks as a training clip", str(context.exception))

    def test_moved_clip_with_foreign_session_stamp_fails(self) -> None:
        train_id, test_id = self._standard_sessions()
        source = self.ws.sessions / train_id / "no" / "no-0.json"
        (self.ws.sessions / test_id / "no" / "moved.json").write_text(source.read_text())
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("stamped sessionId", str(context.exception))

    def test_near_duplicate_fails(self) -> None:
        train_id, test_id = self._standard_sessions()
        payload = json.loads((self.ws.sessions / train_id / "stop" / "stop-0.json").read_text())
        rng = random.Random(5)
        for frame in payload["landmarkSequence"]:
            for point in frame["rightHand"]:
                point["x"] += rng.uniform(-0.001, 0.001)
        payload["sessionId"] = test_id
        (self.ws.sessions / test_id / "stop" / "reexported.json").write_text(json.dumps(payload))
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("near-duplicate", str(context.exception))

    def test_schema_v1_clip_fails(self) -> None:
        train_id, _ = self._standard_sessions()
        write_session_clip(self.ws.sessions / train_id, train_id, "help", "old", make_frames("help", random.Random(9)), extractor=V1_EXTRACTOR)
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("schema-v1", str(context.exception))

    def test_teammate_training_session_fails(self) -> None:
        self._standard_sessions()
        self.ws.add_session("train", "2026-11-12", "lab", clips_per_sign=2, seed=3, signer="juan")
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run()
        self.assertIn("only owner trains", str(context.exception))

    def _write_asl(self, signers: list[str], clips_per_sign: int = 2, held_out: Optional[list[str]] = None) -> None:
        rng = random.Random(42)
        held = set(held_out or [])
        filtered: list[dict[str, str]] = []
        self.ws.asl_landmarks.parent.mkdir(parents=True, exist_ok=True)
        with self.ws.asl_landmarks.open("w", encoding="utf-8") as handle:
            for signer in signers:
                source_csv = "test.csv" if signer in held else "train.csv"
                for label in CORE_SIGN_LABELS:
                    for clip in range(clips_per_sign):
                        video = f"{signer}-{label}-{clip}.mp4"
                        record = {
                            "dataset": "asl-citizen",
                            "label": label,
                            "signerId": signer,
                            "sourceVideoId": video,
                            "extractorVersion": MEDIAPIPE_EXTRACTOR_VERSION,
                            "landmarkSequence": make_frames(label, rng),
                        }
                        handle.write(json.dumps(record) + "\n")
                        filtered.append(
                            {
                                "participantId": signer,
                                "videoFile": video,
                                "sourceGloss": label,
                                "label": label,
                                "sourceCsv": source_csv,
                                "videoPath": "",
                            }
                        )
        asl_citizen.write_filtered_csv(filtered, self.ws.asl_filtered)
        csv_path = self.ws.root / "split.csv"
        csv_path.write_text("Participant ID,Video file,Gloss\n", encoding="utf-8")
        self.ws.asl_summary.write_text(
            json.dumps({"droppedRows": 17, "gaps": [], "glossMapSha256": sha256_file(GLOSS_MAP_PATH)}), encoding="utf-8"
        )
        if held_out is not None:
            self.ws.asl_signers.write_text(
                json.dumps(
                    {
                        "testSigners": held_out,
                        "testSplitParticipants": sorted(held),
                        "trainValParticipants": sorted(signer for signer in signers if signer not in held),
                        "sourceCsvSha256": {str(csv_path): sha256_file(csv_path)},
                    }
                ),
                encoding="utf-8",
            )

    def test_plus_asl_citizen_keeps_held_out_signers_out_of_training(self) -> None:
        self._standard_sessions()
        self._write_asl(["P1", "P2", "P3", "P4"], held_out=["P2", "P4"])
        baseline = self.ws.run("baseline", run_id="baseline-1")
        self.assertEqual(baseline["testSets"][eval_run.ASL_TEST]["clips"], 2 * 2 * len(CORE_SIGN_LABELS))
        run = self.ws.run("plus-asl-citizen", run_id="plus-1")
        self.assertEqual(run["trainCounts"]["bySource"]["asl-citizen"]["yes"], 4)
        self.assertEqual(run["droppedByGlossMap"]["asl-citizen"], 17)
        self.assertIn(run["decision"]["decision"], ("kept", "dropped"))
        self.assertEqual(run["decision"]["comparedTo"], "baseline-1")

    def test_comparison_with_parent_on_other_validation_sessions_fails_before_training(self) -> None:
        self._standard_sessions()
        self._write_asl(["P1", "P2", "P3", "P4"], held_out=["P2", "P4"])
        self.ws.run("baseline", run_id="baseline-1")
        old_validation = self.validation_id
        shutil.rmtree(self.ws.sessions / old_validation)
        new_validation = self.ws.add_session("validation", "2026-11-13", "desk-evening", clips_per_sign=3, seed=31)
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run("plus-asl-citizen", run_id="plus-1")
        message = str(context.exception)
        self.assertIn("Cannot compare with baseline-1", message)
        self.assertIn(f"it used validation sessions ['{old_validation}']", message)
        self.assertIn(f"this run uses ['{new_validation}']", message)
        self.assertFalse((self.ws.models / "plus-1.json").exists())
        self.assertFalse((self.ws.reports / "plus-1" / "metrics.json").exists())

    def test_comparison_fails_when_validation_session_clips_changed(self) -> None:
        self._standard_sessions()
        self._write_asl(["P1", "P2", "P3", "P4"], held_out=["P2", "P4"])
        self.ws.run("baseline", run_id="baseline-1")
        write_session_clip(
            self.ws.sessions / self.validation_id, self.validation_id, "help", "extra", make_frames("help", random.Random(77))
        )
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run("plus-asl-citizen", run_id="plus-1")
        self.assertIn("gained, lost, or changed clips", str(context.exception))
        self.assertFalse((self.ws.models / "plus-1.json").exists())

    def test_asl_experiment_without_held_out_list_fails(self) -> None:
        self._standard_sessions()
        self._write_asl(["P1", "P2"], held_out=None)
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run("plus-asl-citizen")
        self.assertIn("select-test-signers", str(context.exception))

    def test_changed_source_csv_fails(self) -> None:
        self._standard_sessions()
        self._write_asl(["P1", "P2"], held_out=["P2"])
        (self.ws.root / "split.csv").write_text("Participant ID,Video file,Gloss\nP9,x.mp4,YES\n", encoding="utf-8")
        with self.assertRaises(eval_run.ExperimentError) as context:
            self.ws.run("plus-asl-citizen")
        self.assertIn("changed since the held-out signers were chosen", str(context.exception))


class LeakCheckTests(unittest.TestCase):
    def test_held_out_signer_in_training_is_reported(self) -> None:
        rng = random.Random(3)
        train = [make_example("yes", rng, dataset="asl-citizen", signer="P7", video="a.mp4")]
        failures = leakage.check_disjoint(train, set(), {"asl-citizen": {"P7"}})
        self.assertEqual(len(failures), 1)
        self.assertIn("P7", failures[0])

    def test_training_clip_from_test_split_signer_fails(self) -> None:
        rng = random.Random(3)
        train = [make_example("yes", rng, dataset="asl-citizen", signer="P7", video="a.mp4")]
        failures = leakage.check_asl_official_splits(train, {"P7"}, set(), set())
        self.assertEqual(len(failures), 1)
        self.assertIn("test-split", failures[0])
        self.assertIn("P7", failures[0])
        leaked = leakage.check_asl_official_splits([], set(), {"P8"}, {"P8"})
        self.assertEqual(len(leaked), 1)
        self.assertIn("train.csv or val.csv", leaked[0])

    def test_shared_video_id_fails_and_repeats_in_training_are_dropped(self) -> None:
        rng = random.Random(4)
        first = make_example("no", rng, dataset="asl-citizen", signer="P1", video="v1.mp4")
        repeat = make_example("no", rng, dataset="asl-citizen", signer="P1", video="v1.mp4")
        other = make_example("no", rng, dataset="asl-citizen", signer="P2", video="v2.mp4")
        test = make_example("no", rng, dataset="asl-citizen", signer="P3", video="v1.mp4")
        result = leakage.find_duplicates([first, repeat, other], {"asl": [test]}, load_config()["duplicates"], seed=1)
        self.assertEqual(len(result.train), 2)
        self.assertEqual(result.dropped_within_train[0]["by"], "sourceVideoId")
        self.assertTrue(any("v1.mp4" in failure for failure in result.failures))

    def test_distinct_clips_are_well_above_threshold(self) -> None:
        rng = random.Random(6)
        a, b = make_example("help", rng), make_example("help", rng)
        settings = load_config()["duplicates"]
        distance = leakage.landmark_distance(
            leakage.duplicate_representation(a), leakage.duplicate_representation(b), settings["handMissingPenalty"]
        )
        self.assertGreater(distance, settings["calibrationFloor"])
        self.assertEqual(
            leakage.landmark_distance(leakage.duplicate_representation(a), leakage.duplicate_representation(a), 1.0), 0.0
        )


class AslCitizenTests(unittest.TestCase):
    def test_unmapped_glosses_are_dropped_and_counted(self) -> None:
        rows = [
            {"participantId": "P1", "videoFile": "1.mp4", "sourceGloss": "YES", "sourceCsv": "train.csv"},
            {"participantId": "P1", "videoFile": "2.mp4", "sourceGloss": "STOP1", "sourceCsv": "train.csv"},
            {"participantId": "P1", "videoFile": "3.mp4", "sourceGloss": "APPLE", "sourceCsv": "train.csv"},
        ]
        kept, summary = asl_citizen.filter_rows(rows, {"YES": "yes"}, list(CORE_SIGN_LABELS))
        self.assertEqual([row["label"] for row in kept], ["yes"])
        self.assertEqual(summary["droppedRows"], 2)
        self.assertEqual(summary["topDroppedEnglishMatches"], [("STOP1", 1)])
        self.assertIn("wait", summary["gaps"])

    def test_gloss_candidates_find_variants(self) -> None:
        rows = [{"sourceGloss": gloss} for gloss in ("THANK YOU", "THANKYOU2", "STOP", "NOTHING", "KNOW")]
        candidates = asl_citizen.gloss_candidates(rows, list(CORE_SIGN_LABELS))
        self.assertEqual([g for g, _ in candidates["thank_you"]["likely"]], ["THANK YOU", "THANKYOU2"])
        self.assertEqual([g for g, _ in candidates["no"]["likely"]], [])
        self.assertIn("NOTHING", [g for g, _ in candidates["no"]["contains"]])

    def test_held_out_selection_is_seeded_and_never_regenerated(self) -> None:
        rows = []
        for signer in range(20):
            labels = ["yes", "no"] if signer % 4 else ["yes"]
            rows += [{"participantId": f"P{signer:02d}", "label": label, "sourceCsv": "test.csv"} for label in labels]
        selected, eligible = asl_citizen.select_test_signers(rows, seed=603, count=5)
        self.assertEqual(len(eligible), 15)
        self.assertEqual(selected, asl_citizen.select_test_signers(list(reversed(rows)), seed=603, count=5)[0])
        self.assertTrue(set(selected) <= set(eligible))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "signers.json"
            asl_citizen.write_test_signers(selected, eligible, 603, 5, {}, path=path)
            with self.assertRaises(asl_citizen.AslCitizenError):
                asl_citizen.write_test_signers(selected, eligible, 603, 5, {}, path=path)

    def test_split_csvs_are_found_by_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "splits").mkdir()
            with (root / "splits" / "train.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Participant ID", "Video file", "Gloss", "ASL-LEX Code"])
                writer.writerow(["P1", "a.mp4", "YES", "x"])
            (root / "demographics.csv").write_text("Participant ID,Age\nP1,30\n")
            found = asl_citizen.find_split_csvs(root)
            self.assertEqual([path.name for path in found], ["train.csv"])
            self.assertEqual(asl_citizen.read_rows(found)[0]["sourceGloss"], "YES")

    def test_split_participant_sets_include_unmapped_signers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            train = Path(temp_dir) / "train.csv"
            with train.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Participant ID", "Video file", "Gloss"])
                writer.writerow(["P1", "a.mp4", "APPLE"])
            found = asl_citizen.participants_by_split({str(train): "unused"})
            self.assertEqual(found["train.csv"], ["P1"])

    def test_video_index_includes_sibling_videos_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            splits = root / "splits"
            videos = root / "videos"
            splits.mkdir()
            videos.mkdir()
            (videos / "155-YES.mp4").write_bytes(b"clip")
            found = asl_citizen.index_videos(asl_citizen.video_index_roots(splits))
            self.assertEqual(found["155-YES.mp4"], videos / "155-YES.mp4")

    def test_train_only_signer_is_ineligible_and_incomplete_test_signer_is_in_neither_pool(self) -> None:
        rows = []
        for label in ("yes", "no"):
            rows.append({"participantId": "train-only", "label": label, "sourceCsv": "train.csv"})
            rows.append({"participantId": "complete", "label": label, "sourceCsv": "test.csv"})
        rows.append({"participantId": "missing-no", "label": "yes", "sourceCsv": "test.csv"})

        selected, eligible = asl_citizen.select_test_signers(rows, seed=603, count=1)
        self.assertEqual(eligible, ["complete"])
        self.assertEqual(selected, ["complete"])
        self.assertNotIn("train-only", eligible)

        pools = asl_citizen.official_split_pools(rows, seed=603, count=1)
        self.assertIn("train-only", pools["trainValParticipants"])
        self.assertNotIn("missing-no", pools["selected"])
        self.assertNotIn("missing-no", pools["trainValParticipants"])
        self.assertEqual(pools["excluded"], ["missing-no"])
        self.assertEqual(pools["excludedCount"], 1)


class DecisionTests(unittest.TestCase):
    def test_kept_only_when_own_heldout_accuracy_rises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports = Path(temp_dir)
            parent = {
                "runId": "baseline-x",
                "experiment": "baseline",
                "createdAt": "2026-11-20T00:00:00+00:00",
                "ownHeldoutFingerprint": "abc",
                "validationSessions": ["val-1"],
                "validationFingerprint": "v1",
                "testSets": {eval_run.OWN_TEST: {"accuracy": 0.8}},
            }
            (reports / "baseline-x").mkdir()
            (reports / "baseline-x" / "metrics.json").write_text(json.dumps(parent))
            experiment = {"name": "plus-asl-citizen", "extends": "baseline"}
            validation = {"sessions": ["val-1"], "fingerprint": "v1"}
            same = eval_run.decide(experiment, {eval_run.OWN_TEST: {"accuracy": 0.8}}, "abc", validation, reports)
            better = eval_run.decide(experiment, {eval_run.OWN_TEST: {"accuracy": 0.85}}, "abc", validation, reports)
            other_set = eval_run.decide(experiment, {eval_run.OWN_TEST: {"accuracy": 0.9}}, "different", validation, reports)
            self.assertEqual(same["decision"], "dropped")
            self.assertEqual(better["decision"], "kept")
            self.assertEqual(other_set["decision"], "undecided")

            with self.assertRaises(eval_run.ExperimentError) as context:
                eval_run.decide(experiment, {eval_run.OWN_TEST: {"accuracy": 0.9}}, "abc", {"sessions": ["val-2"], "fingerprint": "v2"}, reports)
            self.assertIn("it used validation sessions ['val-1'], this run uses ['val-2']", str(context.exception))

            del parent["validationSessions"]
            (reports / "baseline-x" / "metrics.json").write_text(json.dumps(parent))
            with self.assertRaises(eval_run.ExperimentError) as context:
                eval_run.decide(experiment, {eval_run.OWN_TEST: {"accuracy": 0.9}}, "abc", validation, reports)
            self.assertIn("does not record which validation sessions", str(context.exception))

    def test_experiment_must_add_exactly_one_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments = Path(temp_dir)
            (experiments / "baseline.json").write_text(json.dumps({"name": "baseline", "extends": None, "trainSources": ["own"]}))
            (experiments / "same.json").write_text(json.dumps({"name": "same", "extends": "baseline", "trainSources": ["own"]}))
            with self.assertRaises(eval_run.ExperimentError):
                eval_run.load_experiment("same", experiments)
            self.assertEqual(eval_run.load_experiment("plus-asl-citizen")["addedSource"], "asl-citizen")


if __name__ == "__main__":
    unittest.main()
