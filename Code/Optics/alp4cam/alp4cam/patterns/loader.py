from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..core.exceptions import PatternError


def discover_pattern_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        raise PatternError(f"Pattern directory not found: {directory}")
    paths = sorted(p for p in root.rglob("*.png") if p.is_file())
    if not paths:
        raise PatternError(f"No PNG pattern files found in: {directory}")
    return paths


def load_pattern_stack(directory: str | Path) -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    for path in discover_pattern_files(directory):
        image = Image.open(path)
        array = np.asarray(image.convert("L"))
        patterns.append((array > 127).astype(np.uint8))
    return patterns
