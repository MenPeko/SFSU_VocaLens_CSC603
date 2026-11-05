"""Minimal .env loader.

Replaces the Bun/Node-side `process.env` behaviour for our Python runtime
without pulling in a full dotenv dependency. Only values that are NOT
already set in the real environment are loaded, so explicit shell overrides
always win.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Load key=value pairs from a dotenv-style file into ``os.environ``.
    Returns the dict of keys that were actually inserted. Missing files are
    silently ignored.
    """
    p = Path(path)
    if not p.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Strip surrounding quotes if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded
