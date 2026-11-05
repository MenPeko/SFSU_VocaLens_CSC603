"""Entry point: capture -> MediaPipe -> features -> transport loop.

Run me with::

    python -m touchdesigner.main                       # sends OSC to 127.0.0.1:7400
    python -m touchdesigner.main --camera 0 --fps 24
    python -m touchdesigner.main --print --no-osc      # debug, prints signals to stdout
    python -m touchdesigner.main --preview             # cv2.imshow debug window
    python -m touchdesigner.main --websocket 7401      # also serve JSON on ws://localhost:7401
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import asdict

import cv2

from . import config
from .capture.webcam import WebcamCapture, now_ms
from .classifier_adapter import SignClassifierAdapter
from .extractor_version import MEDIAPIPE_EXTRACTOR_VERSION
from .pipeline import PipelineState, process_frame
from .schema import DerivedSignals
from .tracking.holistic import HolisticTracker
from .transport.osc import OscBroadcaster


_should_exit = False


def _install_signal_handlers() -> None:
    def _handler(_signum, _frame):
        global _should_exit
        _should_exit = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _parse_args(argv: list[str] | None = None) -> config.RuntimeConfig:
    parser = argparse.ArgumentParser(description="TouchDesigner vision pipeline")
    parser.add_argument("--camera", type=int, default=0, help="cv2 camera index")
    parser.add_argument("--fps", type=float, default=config.DEFAULT_CAPTURE_FPS)
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=config.HOLISTIC_MODEL_COMPLEXITY,
        choices=[0, 1, 2],
        help="Holistic model complexity: 0 lite, 1 full, 2 heavy",
    )
    parser.add_argument("--osc-host", default=config.DEFAULT_OSC_HOST)
    parser.add_argument("--osc-port", type=int, default=config.DEFAULT_OSC_PORT)
    parser.add_argument("--no-osc", action="store_true", help="Disable OSC output")
    parser.add_argument("--websocket", type=int, default=None, help="Broadcast JSON on this port")
    parser.add_argument(
        "--viewer",
        type=int,
        default=None,
        metavar="PORT",
        help="Serve a live HTML viewer on http://localhost:PORT (and auto-start websocket on PORT+1)",
    )
    parser.add_argument("--print", dest="print_signals", action="store_true", help="Print signals to stdout")
    parser.add_argument("--preview", action="store_true", help="Open a cv2 preview window with the raw camera feed")
    parser.add_argument("--classifier-model", default=None, help="Path to a trained classifier artifact (optional)")
    parser.add_argument("--no-flip", action="store_true", help="Do not mirror the webcam (default is on)")
    parser.add_argument("--log-timings", action="store_true", help="Print per-frame timing to stderr")
    parser.add_argument("--disable-face", action="store_true", help="Skip face mesh for lower latency")

    ns = parser.parse_args(argv)
    # --viewer implies --websocket on viewer-port + 1 unless one is already set.
    websocket_port = ns.websocket
    viewer_port = ns.viewer
    if viewer_port is not None and websocket_port is None:
        websocket_port = viewer_port + 1

    return config.RuntimeConfig(
        camera_index=ns.camera,
        fps_target=ns.fps,
        model_complexity=ns.model_complexity,
        osc_host=ns.osc_host,
        osc_port=ns.osc_port,
        osc_enabled=not ns.no_osc,
        websocket_port=websocket_port,
        viewer_port=viewer_port,
        print_signals=ns.print_signals,
        show_preview=ns.preview,
        classifier_model_path=ns.classifier_model,
        flip_horizontal=not ns.no_flip,
        log_timings=ns.log_timings,
        disable_face=ns.disable_face,
    )


def _print_signals_line(signals: DerivedSignals) -> None:
    # Compact single-line log for quick manual sanity-checking.
    print(
        f"f={signals.frame_index:05d} fps={signals.fps:5.1f} "
        f"L[{'on' if signals.left.present else '--'} "
        f"o={signals.left.openness:.2f} p={signals.left.pinch_distance:.2f} "
        f"P={int(signals.left.pinch_state)} g={signals.left.gesture}] "
        f"R[{'on' if signals.right.present else '--'} "
        f"o={signals.right.openness:.2f} p={signals.right.pinch_distance:.2f} "
        f"P={int(signals.right.pinch_state)} g={signals.right.gesture}] "
        f"sess={'on' if signals.session.active else '--'}({signals.session.duration_ms}ms)",
        flush=True,
    )


def _draw_preview_overlay(frame_bgr, signals: DerivedSignals) -> None:
    text = (
        f"fps={signals.fps:.1f}  "
        f"L={'on' if signals.left.present else '--'}({signals.left.gesture})  "
        f"R={'on' if signals.right.present else '--'}({signals.right.gesture})  "
        f"sess={'ACTIVE' if signals.session.active else 'idle'}"
    )
    cv2.rectangle(frame_bgr, (8, 8), (8 + 12 * len(text), 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame_bgr,
        text,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run(cfg: config.RuntimeConfig) -> int:
    _install_signal_handlers()

    print(f"[touchdesigner] extractor_version={MEDIAPIPE_EXTRACTOR_VERSION}", file=sys.stderr)
    print(
        f"[touchdesigner] osc={'on' if cfg.osc_enabled else 'off'} "
        f"{cfg.osc_host}:{cfg.osc_port}  model_complexity={cfg.model_complexity}",
        file=sys.stderr,
    )

    webcam = WebcamCapture(
        index=cfg.camera_index,
        width=config.REQUESTED_CAPTURE_WIDTH,
        height=config.REQUESTED_CAPTURE_HEIGHT,
        fps=cfg.fps_target,
        flip_horizontal=cfg.flip_horizontal,
    )
    info = webcam.open()
    print(
        f"[touchdesigner] camera {cfg.camera_index}: {info.width}x{info.height} "
        f"(requested fps={info.fps_requested}, reported fps={info.fps_actual})",
        file=sys.stderr,
    )

    osc = OscBroadcaster(host=cfg.osc_host, port=cfg.osc_port, enabled=cfg.osc_enabled)

    ws = None
    if cfg.websocket_port is not None:
        try:
            from .transport.websocket import WebsocketBroadcaster

            ws = WebsocketBroadcaster(port=cfg.websocket_port)
            ws.start()
            print(f"[touchdesigner] websocket ws://127.0.0.1:{cfg.websocket_port}", file=sys.stderr)
        except Exception as e:
            print(f"[touchdesigner] websocket disabled: {e}", file=sys.stderr)
            ws = None

    viewer = None
    if cfg.viewer_port is not None and cfg.websocket_port is not None:
        try:
            from .transport.http_viewer import HttpViewerServer

            viewer = HttpViewerServer(port=cfg.viewer_port, websocket_port=cfg.websocket_port)
            viewer.start()
            print(f"[touchdesigner] viewer  {viewer.url}", file=sys.stderr)
        except Exception as e:
            print(f"[touchdesigner] viewer disabled: {e}", file=sys.stderr)
            viewer = None

    classifier = SignClassifierAdapter(model_path=cfg.classifier_model_path)
    if classifier.enabled:
        print("[touchdesigner] sign classifier available (predict on session end)", file=sys.stderr)
    else:
        print(f"[touchdesigner] sign classifier disabled: {classifier.unavailable_reason}", file=sys.stderr)

    pipeline_state = PipelineState()

    try:
        with HolisticTracker(
            model_complexity=cfg.model_complexity,
            disable_face=cfg.disable_face,
        ) as tracker:
            target_dt = 1.0 / max(cfg.fps_target, 1.0)
            while not _should_exit:
                loop_start = time.monotonic()

                ok, frame_bgr = webcam.read()
                if not ok or frame_bgr is None:
                    pipeline_state.dropped_frames += 1
                    time.sleep(0.005)
                    continue

                t_track0 = time.monotonic()
                tracked = tracker.process(frame_bgr, timestamp_ms=now_ms())
                t_track1 = time.monotonic()

                signals = process_frame(pipeline_state, tracked, timestamp_ms=now_ms())

                osc.broadcast(signals)
                if ws is not None:
                    ws.broadcast(asdict(signals))

                if signals.session.ended and classifier.enabled:
                    result = classifier.predict(list(pipeline_state.landmark_history))
                    if result is not None:
                        print(f"[touchdesigner] classifier: {result}", file=sys.stderr)
                        osc._send("/classifier/label", str(result.get("label") or "none"))
                        osc._send("/classifier/status", str(result.get("status") or "unknown"))
                        osc._send("/classifier/confidence", float(result.get("confidence") or 0.0))

                if cfg.print_signals:
                    _print_signals_line(signals)

                if cfg.show_preview:
                    _draw_preview_overlay(frame_bgr, signals)
                    cv2.imshow("touchdesigner preview", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

                if cfg.log_timings:
                    loop_dt = (time.monotonic() - loop_start) * 1000
                    track_dt = (t_track1 - t_track0) * 1000
                    print(
                        f"[timings] loop={loop_dt:5.1f}ms holistic={track_dt:5.1f}ms",
                        file=sys.stderr,
                    )

                # Light pacing so we don't burn CPU if the camera returns
                # faster than our target fps (rare, but cheap insurance).
                elapsed = time.monotonic() - loop_start
                remaining = target_dt - elapsed
                if remaining > 0.001:
                    time.sleep(remaining)

    finally:
        webcam.release()
        osc.close()
        if ws is not None:
            ws.stop()
        if viewer is not None:
            viewer.stop()
        if cfg.show_preview:
            cv2.destroyAllWindows()

    print("[touchdesigner] exited cleanly", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv)
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
