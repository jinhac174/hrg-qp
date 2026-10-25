"""Repository root, and paths in configs (relative to it)."""

from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path: str) -> str:
    """``path`` if absolute, else relative to the repository root."""
    path = str(path)
    return path if os.path.isabs(path) else os.path.join(REPO, path)
