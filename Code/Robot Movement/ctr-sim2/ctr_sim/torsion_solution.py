from dataclasses import dataclass

import numpy as np

from ctr_sim.torsion_segment_solution import (
    TorsionSegmentSolution,
)


@dataclass
class TorsionSolution:
    """
    Piecewise segment-aware torsion solution.
    """

    base_theta_dot: np.ndarray

    segments: list[
        TorsionSegmentSolution
    ]

    base_moment: np.ndarray | None = None
    """
    Internal moment at the base in the backbone frame (N·m). Only set
    by the loaded (tip force) solver; None for unloaded solutions.
    """

    shooting_vector: np.ndarray | None = None
    """
    Full shooting solution of the loaded solver. None for unloaded
    solutions. Use `warm_start` rather than reading this directly.
    """

    shooting_jacobian: np.ndarray | None = None
    """
    Approximate Jacobian of the loaded shooting residual at the
    solution. Reused when this TorsionSolution is passed as the
    initial guess of a nearby loaded solve. None for unloaded
    solutions.
    """

    @property
    def warm_start(self) -> np.ndarray:
        """
        Initial guess to pass as `torsion_initial_guess` to the next,
        nearby forward solve. Works for both loaded and unloaded
        solutions: under load it also carries the base bending moment,
        which makes warm-started loaded solves much faster than passing
        `base_theta_dot` alone.
        """

        if self.shooting_vector is not None:
            return self.shooting_vector

        return self.base_theta_dot
