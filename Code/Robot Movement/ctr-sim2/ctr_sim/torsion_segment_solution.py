from dataclasses import dataclass

from ctr_sim.segment import Segment


@dataclass
class TorsionSegmentSolution:
    """
    Torsion solution over one backbone segment.
    """

    segment: Segment
    ivp_solution: object