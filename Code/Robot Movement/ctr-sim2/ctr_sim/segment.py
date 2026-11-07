from dataclasses import dataclass

from ctr_sim.tube import Tube


@dataclass
class Segment:
    """
    A contiguous interval of the backbone over which both the set of
    active tubes and the precurvature of every active tube remain
    constant.

    `curvatures[k]` is the precurvature (1/m) of `active_tubes[k]` over
    this segment. A tube can be active in a segment while being straight
    there (curvature 0): it still contributes bending and torsional
    stiffness, but no precurvature.

    If `curvatures` is omitted it defaults to each tube's `precurvature`,
    which is only valid for tubes with uniform curvature; segments built
    from tubes with a straight/curved/straight profile must supply it
    (`backbone_segments` always does).
    """

    start: float
    end: float
    active_tubes: list[Tube]
    curvatures: list[float] | None = None

    def __post_init__(self):

        if self.curvatures is None:

            for tube in self.active_tubes:

                if not tube.has_uniform_curvature:
                    raise ValueError(
                        f"Tube {tube.name} has a straight/curved "
                        "profile, so the segment curvatures must be "
                        "given explicitly."
                    )

            self.curvatures = [
                tube.precurvature
                for tube in self.active_tubes
            ]

        elif len(self.curvatures) != len(self.active_tubes):

            raise ValueError(
                "curvatures must contain one value per active tube."
            )
