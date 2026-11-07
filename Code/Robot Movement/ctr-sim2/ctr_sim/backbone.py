from dataclasses import dataclass

from ctr_sim.segment_solution import SegmentSolution
from ctr_sim.torsion_solution import TorsionSolution


@dataclass
class Backbone:
    """
    Segment-wise mechanics solution.
    """

    segments: list[SegmentSolution]
    torsion_solution: TorsionSolution | None = None

    