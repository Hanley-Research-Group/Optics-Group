from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from ALP4 import ALP4 as ALP4Driver

from ..core.exceptions import DependencyError, DMDHardwareError, PatternError


class DMDController:
    def __init__(self, library_dir: str | None = None, version: str = "4.2"):
        self.library_dir = library_dir
        self.version = version
        self.device = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def open(self):
        """Create the ALP4 hardware driver and initialize it if supported."""
        self.device = ALP4Driver(version=self.version, libDir=self.library_dir)
        return self.device

    def close(self):
        """Stop the device and release any allocated device memory when the controller exits."""
        if self.device is None:
            return

        try:
            if hasattr(self.device, "Halt"):
                self.device.Halt()

            if hasattr(self.device, "FreeSeq"):
                self.device.FreeSeq()

            if hasattr(self.device, "Free"):
                self.device.Free()
        finally:
            self.device = None

    def _validate_patterns(self, patterns: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Reject empty or malformed pattern inputs before they are sent to the DMD."""
        if not patterns:
            raise PatternError("No pattern images were supplied.")

        validated: list[np.ndarray] = []
        for index, pattern in enumerate(patterns):
            array = np.asarray(pattern)
            if array.ndim != 2:
                raise PatternError(f"Pattern #{index} is not 2D.")
            validated.append(array.astype(np.uint8))
        return validated

    def load_patterns(
        self,
        patterns: Sequence[np.ndarray],
        picture_time_us: int = 10000,
        once: bool = False,
        duration_s: int = 0,
    ):
        """Upload a sequence of 2D binary patterns and start the projection loop."""
        if self.device is None:
            self.open()

        validated = self._validate_patterns(patterns)
        flat = np.concatenate([pattern.astype(np.uint8).ravel() for pattern in validated])

        if hasattr(self.device, "SeqAlloc"):
            self.device.SeqAlloc(nbImg=len(validated), bitDepth=1)
        if not hasattr(self.device, "SeqPut"):
            raise DMDHardwareError("ALP4lib does not expose SeqPut().")
        self.device.SeqPut(imgData=flat)

        if not hasattr(self.device, "SetTiming"):
            raise DMDHardwareError("ALP4lib does not expose SetTiming().")
        self.device.SetTiming(pictureTime=picture_time_us)

        if not hasattr(self.device, "Run"):
            raise DMDHardwareError("ALP4lib does not expose Run().")
        self.device.Run(loop=not once)

        if duration_s and hasattr(self.device, "ProjWait"):
            time.sleep(float(duration_s))
        return len(validated)
