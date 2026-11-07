from dataclasses import dataclass

import numpy as np

from scipy.integrate._ivp.ivp import OdeResult

from .segment import Segment


@dataclass
class SegmentSolution:
    """
    Complete mechanics solution for one backbone segment.
    """

    segment: Segment

    theta: np.ndarray

    curvature: np.ndarray

    ivp_solution: OdeResult