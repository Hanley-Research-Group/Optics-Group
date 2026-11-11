from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .core.exceptions import WriteError


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_pattern_image(pattern: np.ndarray, path: Path) -> Path:
    array = np.asarray(pattern, dtype=np.uint8)
    if array.ndim != 2:
        raise WriteError(f"Pattern image must be 2D, got shape {array.shape}.")
    _ensure_parent(path)
    Image.fromarray(array, mode="L").save(path)
    return path


def save_pattern_batch(patterns: Sequence[np.ndarray], output_dir: str | Path, prefix: str = "pattern") -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, pattern in enumerate(patterns):
        target = root / f"{prefix}_{index:04d}.png"
        saved.append(write_pattern_image(pattern, target))
    return saved
