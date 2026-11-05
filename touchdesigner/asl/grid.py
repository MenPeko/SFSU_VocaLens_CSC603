"""Compose the 4x4 chronological sign-frame grid.

Port of ``createFrameCanvas`` + ``redrawFrameGridCanvas`` from
``src/webview/hooks/useSignLanguage.ts``. Produces the base64 JPEG that
Gemini's ASL interpreter expects — same layout the original app used, so
Gemini prompts/responses behave the same.

Each cell is a 160x160 center-cropped frame; the full grid is 640x640.
"""

from __future__ import annotations

import base64
from typing import Sequence

import cv2
import numpy as np


GRID_COLUMNS: int = 4
GRID_ROWS: int = 4
GRID_CELL_SIZE: int = 160
MAX_GRID_FRAMES: int = GRID_COLUMNS * GRID_ROWS


def _center_crop_cover(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Crop to a square aspect and resize to (size, size) — the Python
    equivalent of the ``drawFrameCover`` canvas helper."""
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = frame_bgr[y0:y0 + side, x0:x0 + side]
    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return cropped


def _resample_to_grid(frames: Sequence[np.ndarray], target: int) -> list[np.ndarray]:
    """Evenly subsample / resample ``frames`` down to exactly ``target``
    frames. Mirrors the browser's "last N frames with uniform spacing"
    pattern — reflects the whole session, not just the tail."""
    n = len(frames)
    if n == 0:
        return []
    if n <= target:
        return list(frames)
    out: list[np.ndarray] = []
    for i in range(target):
        idx = min(n - 1, round(i * (n - 1) / max(target - 1, 1)))
        out.append(frames[idx])
    return out


def compose_sign_grid_jpeg(
    frames: Sequence[np.ndarray],
    cell_size: int = GRID_CELL_SIZE,
    jpeg_quality: int = 88,
) -> tuple[str, str, int]:
    """Compose a 4x4 chronological frame grid as a JPEG.

    Returns ``(base64_data, data_url, frame_count_used)``.
    """
    sampled = _resample_to_grid(frames, MAX_GRID_FRAMES)
    grid = np.zeros((GRID_ROWS * cell_size, GRID_COLUMNS * cell_size, 3), dtype=np.uint8)
    # Dark background so empty cells don't flash white in Gemini's input.
    grid[:] = (11, 7, 5)  # BGR equivalent of #05070b

    for i, frame in enumerate(sampled):
        row = i // GRID_COLUMNS
        col = i % GRID_COLUMNS
        cell = _center_crop_cover(frame, cell_size)
        y = row * cell_size
        x = col * cell_size
        grid[y:y + cell_size, x:x + cell_size] = cell

    ok, buf = cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed to produce a JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    return b64, data_url, len(sampled)
