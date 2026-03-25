"""Runtime guards for repository entry points."""

from __future__ import annotations

import sys

MIN_PYTHON = (3, 12)


def require_python_312() -> None:
    """Exit early with a clear message if the interpreter is too old."""
    if sys.version_info >= MIN_PYTHON:
        return

    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    message = (
        f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required for astro-reason; "
        f"found {version} at {sys.executable}.\n"
        "Activate the project environment and rerun with `.venv/bin/python` or `python` from the 3.12 venv."
    )
    raise SystemExit(message)
