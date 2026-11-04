import os
import numpy as np
from PIL import Image
from scipy.linalg import hadamard


class DMDCalibrationGenerator:
    """Generates 1-bit Binary DMD Probing Patterns (Walsh-Hadamard + Random Phase Masks)

    encoded via Off-Axis Lee Holography for MMF Transmission Matrix & ML
    Training.
    """

    def __init__(
        self,
        dmd_res=(768, 1024),  # (H, W) physical mirrors (e.g., TI DLP6500 is 1080x1920)
        grid_size=(16, 16),  # (Ny, Nx) logical mode superpixels (must be power of 2 for Hadamard)
        active_radius_px=450,  # Radius of the laser beam / fiber aperture on the DMD
        carrier_period_px=6.0,  # Grating period in DMD mirrors (typically 4 to 8 pixels)
        carrier_angle_deg=45.0,  # Off-axis diffraction angle to separate 1st order in 4f plane
    ):
        self.H, self.W = dmd_res
        self.Ny, self.Nx = grid_size
        self.n_modes = self.Ny * self.Nx
        self.radius = active_radius_px

        # 1. Build physical DMD coordinate grid centered on the chip
        y = np.arange(self.H) - (self.H // 2)
        x = np.arange(self.W) - (self.W // 2)
        self.Y, self.X = np.meshgrid(y, x, indexing="ij")

        # 2. Circular pupil mask (turns OFF unused mirrors outside the laser beam)
        self.Aperture = (self.X**2 + self.Y**2) <= (self.radius**2)

        # 3. Precompute the 2D Off-Axis Spatial Carrier Grating: 2*pi*(fx*X + fy*Y)
        theta = np.deg2rad(carrier_angle_deg)
        fx = np.cos(theta) / carrier_period_px
        fy = np.sin(theta) / carrier_period_px
        self.Carrier = 2.0 * np.pi * (fx * self.X + fy * self.Y)

        # 4. Map central square inside the circular aperture to the (Ny, Nx) superpixel grid
        box_half = int(self.radius / np.sqrt(2))
        self.y_start, self.y_end = (self.H // 2) - box_half, (
            self.H // 2
        ) + box_half
        self.x_start, self.x_end = (self.W // 2) - box_half, (
            self.W // 2
        ) + box_half
        self.box_size = 2 * box_half

    def _upsample_to_dmd(self, phase_grid_2d: np.ndarray) -> np.ndarray:
        """Upsamples an (Ny, Nx) phase array to the DMD active region using nearest-neighbor

        superpixels and encodes the 1-bit binary Lee Hologram.
        """
        # Nearest-neighbor Kronecker-like expansion to match box_size
        y_idx = (np.arange(self.box_size) * self.Ny) // self.box_size
        x_idx = (np.arange(self.box_size) * self.Nx) // self.box_size
        upsampled_phase = phase_grid_2d[np.ix_(y_idx, x_idx)]

        # Embed into full DMD canvas
        full_phase = np.zeros((self.H, self.W), dtype=np.float32)
        full_phase[self.y_start : self.y_end, self.x_start : self.x_end] = (
            upsampled_phase
        )

        # Binary Lee Hologram Threshold: cos(2*pi*(fx*x + fy*y) + phi(x,y)) > 0
        binary_frame = (
            (np.cos(self.Carrier + full_phase) > 0.0) & self.Aperture
        ).astype(np.uint8)
        return binary_frame

    def generate_hadamard_basis(self):
        """Generates all N = Ny * Nx Walsh-Hadamard 2D basis patterns.

        Returns both the complex mode vectors (for PyTorch X_train) and 1-bit
        DMD frames.
        """
        assert (
            self.n_modes & (self.n_modes - 1)
        ) == 0, "Ny * Nx must be a power of 2 for Hadamard!"
        H_mat = hadamard(self.n_modes)  # Entries in {-1, +1}

        dmd_frames = []
        complex_inputs = np.zeros(
            (self.n_modes, self.Ny, self.Nx), dtype=np.complex64
        )

        for i in range(self.n_modes):
            # Reshape 1D Hadamard row into 2D (Ny, Nx) grid
            mode_2d = H_mat[i].reshape((self.Ny, self.Nx))
            complex_inputs[i] = mode_2d.astype(np.complex64)

            # Map +1 -> 0 rad phase, -1 -> pi rad phase
            phase_2d = np.where(mode_2d > 0, 0.0, np.pi).astype(np.float32)
            dmd_frames.append(self._upsample_to_dmd(phase_2d))

        return complex_inputs, np.array(dmd_frames, dtype=np.uint8)

    def generate_phaseless_random_masks(
        self, oversampling_factor=4, seed=42
    ):
        """Generates M = oversampling_factor * N uniform random phase patterns.

        Required for phaseless calibration (complexPyTorch Amplitude Flow) to
        avoid local minima.
        """
        rng = np.random.default_rng(seed)
        n_frames = oversampling_factor * self.n_modes

        dmd_frames = []
        complex_inputs = np.zeros(
            (n_frames, self.Ny, self.Nx), dtype=np.complex64
        )

        for i in range(n_frames):
            # Random continuous phase in [-pi, pi)
            phase_2d = rng.uniform(-np.pi, np.pi, size=(self.Ny, self.Nx)).astype(
                np.float32
            )
            complex_inputs[i] = np.exp(1j * phase_2d)
            dmd_frames.append(self._upsample_to_dmd(phase_2d))

        return complex_inputs, np.array(dmd_frames, dtype=np.uint8)

    @staticmethod
    def export_for_dmd_hardware(
        dmd_frames: np.ndarray,
        complex_inputs: np.ndarray,
        out_dir="dmd_calibration_set",
    ):
        """Saves 1-bit PNG images (for GUI loaders), bit-packed uint8 buffers (for ALP4/pycrafter),

        and the exact complex input tensor (for complexPyTorch training).
        """
        os.makedirs(out_dir, exist_ok=True)

        # 1. Save exact complex field inputs for PyTorch DataLoader
        np.save(
            os.path.join(out_dir, "X_train_complex_inputs.npy"), complex_inputs
        )

        # 2. Save bit-packed hardware buffer (8 mirrors per byte along width axis)
        packed_buffer = np.packbits(dmd_frames, axis=-1)
        np.save(
            os.path.join(out_dir, "dmd_bitpacked_stream.npy"), packed_buffer
        )

        # 3. Save sample 1-bit BMP/PNG images for visual inspection or TI GUI upload
        for i in range(min(10, len(dmd_frames))):
            img = Image.fromarray(dmd_frames[i] * 255, mode="L").convert("1")
            img.save(os.path.join(out_dir, f"probe_pattern_{i:04d}.bmp"))

        print(f"Exported {len(dmd_frames)} calibration frames to './{out_dir}/'")
        print(
            f"  -> PyTorch Complex Input Tensor: {complex_inputs.shape} (complex64)"
        )
        print(
            f"  -> Bit-Packed DMD RAM Buffer:    {packed_buffer.shape} (uint8)"
        )


# --- Example Usage ---
if __name__ == "__main__":
    # Configure for a 16x16 = 256 mode calibration
    gen = DMDCalibrationGenerator(
        grid_size=(16, 16),
        active_radius_px=400,
        carrier_period_px=6.0,
    )

    # 1. Generate 1x Hadamard Basis (256 frames) + 3x Random Phase Basis (768 frames) = 1024 total frames
    X_had, frames_had = gen.generate_hadamard_basis()
    X_rnd, frames_rnd = gen.generate_phaseless_random_masks(
        oversampling_factor=3
    )

    X_combined = np.concatenate([X_had, X_rnd], axis=0)
    frames_combined = np.concatenate([frames_had, frames_rnd], axis=0)

    # 2. Export for PyTorch and DMD Controller
    gen.export_for_dmd_hardware(frames_combined, X_combined)
