"""One-shot faster-whisper transcription. Prints the transcript to stdout."""

import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


SILENCE_RMS = 0.005
MIN_DURATION_SECONDS = 0.2


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def load_audio(path: str):
    import numpy as np
    from faster_whisper.audio import decode_audio

    audio = decode_audio(path)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        return audio, 0.0, 0.0
    duration = audio.size / 16000
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return audio, duration, rms


def transcribe(audio) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        without_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=None,
        log_prob_threshold=None,
        compression_ratio_threshold=None,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: transcribe_local.py AUDIO", file=sys.stderr)
        return 2

    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError as error:
        return fail(
            f"faster-whisper is not installed for {sys.executable}. {error}. Install it with: {sys.executable} -m pip install faster-whisper"
        )

    try:
        audio, duration, rms = load_audio(sys.argv[1])
    except Exception as error:
        return fail(f"The recording could not be decoded. {error}")

    if duration < MIN_DURATION_SECONDS:
        return fail(
            f"Recording is too short ({duration:.2f}s). Hold the button while you speak the question."
        )
    if rms < SILENCE_RMS:
        return fail(
            "The microphone recording was silent. Hold the button and speak the question."
        )

    try:
        text = transcribe(audio)
    except Exception as error:
        return fail(f"Local transcription failed. {error}")

    if not text:
        return fail(
            f"No words were recognized in the {duration:.1f}s recording. Hold the button and speak the question again."
        )

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
