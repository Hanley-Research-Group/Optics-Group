from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ..core.exceptions import PatternError


class DMDCalibrationGenerator:
    """Generate binary DMD calibration patterns using a Hadamard basis and Lee-hologram carrier."""

    def __init__(
        self,
        dmd_res: tuple[int, int] = (768, 1024),
        grid_size: tuple[int, int] = (16, 16),
        active_radius_px: int = 450,
        carrier_period_px: float = 6.0,
        carrier_angle_deg: float = 45.0,
    ):
        self.H, self.W = dmd_res
        self.Ny, self.Nx = grid_size
        self.n_modes = self.Ny * self.Nx
        self.radius = max(1, min(active_radius_px, max(1, min(self.H, self.W) // 2)))

        y = np.arange(self.H) - (self.H // 2)
        x = np.arange(self.W) - (self.W // 2)
        self.Y, self.X = np.meshgrid(y, x, indexing="ij")
        self.Aperture = (self.X**2 + self.Y**2) <= (self.radius**2)

        theta = np.deg2rad(carrier_angle_deg)
        fx = np.cos(theta) / carrier_period_px
        fy = np.sin(theta) / carrier_period_px
        self.Carrier = 2.0 * np.pi * (fx * self.X + fy * self.Y)

        box_half = max(1, int(self.radius / np.sqrt(2)))
        self.y_start = max(0, (self.H // 2) - box_half)
        self.y_end = min(self.H, (self.H // 2) + box_half)
        self.x_start = max(0, (self.W // 2) - box_half)
        self.x_end = min(self.W, (self.W // 2) + box_half)

    def _upsample_to_dmd(self, phase_grid_2d: np.ndarray) -> np.ndarray:
        box_h = max(1, self.y_end - self.y_start)
        box_w = max(1, self.x_end - self.x_start)
        y_idx = np.linspace(0, self.Ny - 1, num=box_h, dtype=int)
        x_idx = np.linspace(0, self.Nx - 1, num=box_w, dtype=int)
        upsampled_phase = phase_grid_2d[np.ix_(y_idx, x_idx)]

        full_phase = np.zeros((self.H, self.W), dtype=np.float32)
        full_phase[self.y_start : self.y_end, self.x_start : self.x_end] = upsampled_phase

        binary_frame = ((np.cos(self.Carrier + full_phase) > 0.0) & self.Aperture).astype(np.uint8)
        return binary_frame

    def generate_hadamard_basis(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_modes & (self.n_modes - 1):
            raise PatternError("Ny * Nx must be a power of 2 for Hadamard patterns.")

        h_matrix = hadamard_matrix(self.n_modes)
        complex_inputs = np.zeros((self.n_modes, self.Ny, self.Nx), dtype=np.complex64)
        dmd_frames: list[np.ndarray] = []

        for idx in range(self.n_modes):
            mode_2d = h_matrix[idx].reshape((self.Ny, self.Nx))
            complex_inputs[idx] = mode_2d.astype(np.complex64)
            phase_2d = np.where(mode_2d > 0, 0.0, np.pi).astype(np.float32)
            dmd_frames.append(self._upsample_to_dmd(phase_2d))

        return complex_inputs, np.array(dmd_frames, dtype=np.uint8)

    def generate_phaseless_random_masks(self, oversampling_factor: int = 4, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_frames = oversampling_factor * self.n_modes

        complex_inputs = np.zeros((n_frames, self.Ny, self.Nx), dtype=np.complex64)
        dmd_frames: list[np.ndarray] = []

        for idx in range(n_frames):
            phase_2d = rng.uniform(-np.pi, np.pi, size=(self.Ny, self.Nx)).astype(np.float32)
            complex_inputs[idx] = np.exp(1j * phase_2d)
            dmd_frames.append(self._upsample_to_dmd(phase_2d))

        return complex_inputs, np.array(dmd_frames, dtype=np.uint8)


def _reshape_square(values: np.ndarray, side: int) -> np.ndarray:
    if values.size != side * side:
        raise PatternError(f"Expected {side * side} pixels, got {values.size}.")
    return values.reshape(side, side)


def hadamard_matrix(size: int) -> np.ndarray:
    if size < 1:
        raise PatternError("Hadamard matrix size must be positive.")
    h = np.array([[1]], dtype=np.int8)
    while h.shape[0] < size:
        h = np.block([[h, h], [h, -h]])
    return h[:size, :size]


def generate_test_patterns(side: int = 32) -> list[np.ndarray]:
    black = np.zeros((side, side), dtype=np.uint8)
    white = np.full((side, side), 255, dtype=np.uint8)
    half = np.full((side, side), 128, dtype=np.uint8)
    checker = np.indices((side, side)).sum(axis=0) % 2
    checker = (checker * 255).astype(np.uint8)
    return [black, white, half, checker]


def generate_hadamard_set(mode_side: int = 32, macro: int = 2) -> list[np.ndarray]:
    if mode_side <= 0:
        raise PatternError("mode_side must be > 0")

    n_modes = mode_side * mode_side
    total_patterns = max(1, int(macro) * n_modes)

    generator = DMDCalibrationGenerator(
        dmd_res=(mode_side, mode_side),
        grid_size=(mode_side, mode_side),
        active_radius_px=max(8, mode_side // 2),
        carrier_period_px=6.0,
        carrier_angle_deg=45.0,
    )

    _, hadamard_frames = generator.generate_hadamard_basis()
    patterns: list[np.ndarray] = [
        (frame.astype(np.uint8) * 255) for frame in hadamard_frames[: min(len(hadamard_frames), total_patterns)]
    ]

    if len(patterns) < total_patterns:
        _, random_frames = generator.generate_phaseless_random_masks(oversampling_factor=max(1, int(macro)), seed=42)
        for frame in random_frames[: total_patterns - len(patterns)]:
            patterns.append((frame.astype(np.uint8) * 255))

    return patterns


def save_pattern_array(array: np.ndarray, path: Path) -> None:
    bitmap = np.asarray(array, dtype=np.uint8)
    if bitmap.max() <= 1:
        bitmap = bitmap * 255
    image = Image.fromarray(bitmap)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_pattern_set(patterns: Sequence[np.ndarray], out_dir: str | Path, prefix: str = "pattern") -> list[Path]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for idx, pattern in enumerate(patterns):
        target = root / f"{prefix}_{idx:04d}.png"
        save_pattern_array(pattern, target)
        saved.append(target)
    return saved
