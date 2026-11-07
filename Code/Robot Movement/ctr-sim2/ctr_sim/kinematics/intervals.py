from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.segment import Segment
from ctr_sim.types import Interval


def occupied_intervals(
    robot: ConcentricTubeRobot,
) -> list[Interval]:
    """
    Compute the occupied interval of each tube.

    Each interval is returned as

        (proximal_end, distal_tip)

    where

        proximal_end = βᵢ - Lᵢ

        distal_tip = βᵢ

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Robot whose tube intervals are to be computed.

    Returns
    -------
    list[tuple[float, float]]
        List of (proximal_end, distal_tip) intervals.
    """

    intervals = []

    for tube, beta in zip(robot.tubes, robot.state.insertions):

        proximal = beta - tube.length
        distal = beta

        intervals.append((proximal, distal))

    return intervals


# Backbone boundaries closer together than this (m) are treated as one
# boundary. This prevents numerically zero-length segments when, for
# example, a curve boundary of one tube coincides with the tip or curve
# boundary of another tube (or with the base, s = 0).
_BOUNDARY_TOL = 1e-12


def curved_intervals(
    robot: ConcentricTubeRobot,
) -> list[Interval]:
    """
    Compute the arc-length extent of each tube's curved section.

    Each interval is returned in robot arc length s as

        (curve_start, curve_end)

    with

        curve_start = βᵢ - Lᵢ + length_before_curve
        curve_end   = curve_start + curve_length

    Tubes with no curved section return a zero-length interval.
    """

    intervals = []

    for tube, beta in zip(robot.tubes, robot.state.insertions):

        sigma_start, sigma_end = tube.curve_interval

        proximal = beta - tube.length

        intervals.append(
            (
                proximal + sigma_start,
                proximal + sigma_end,
            )
        )

    return intervals


def backbone_segments(
    robot: ConcentricTubeRobot,
) -> list[Segment]:
    """
    Partition the robot backbone into contiguous segments.

    Within each segment, the set of active tubes and the precurvature of
    every active tube are constant. Segment boundaries are therefore
    placed at

        * the distal tip of every tube,
        * the start and end of every tube's curved section, and
        * the robot base (s = 0).

    Tube material behind the base (s < 0) lies inside a rigid guide and
    is not part of the free backbone.
    """

    intervals = occupied_intervals(robot)
    curves = curved_intervals(robot)

    # Collect all interval endpoints
    boundaries = []

    for start, end in intervals:
        boundaries.append(start)
        boundaries.append(end)

    for start, end in curves:

        # A straight tube (or zero-length curve) has no curvature
        # change to resolve.
        if end - start > _BOUNDARY_TOL:
            boundaries.append(start)
            boundaries.append(end)

    # The robot backbone begins at the robot base, s=0.
    # Tube material behind the base remains inside a rigid guide and
    # is not part of the free backbone.
    boundaries = [
        boundary
        for boundary in boundaries
        if boundary >= 0.0
    ]

    # The free robot backbone begins at the base frame.
    boundaries.append(0.0)

    # The backbone ends at the most distal tube tip. Curved sections
    # never extend past their own tube's tip, so clip any boundary
    # that does only through round-off.
    s_end = max(end for _, end in intervals)

    boundaries = sorted(
        boundary
        for boundary in boundaries
        if boundary <= s_end + _BOUNDARY_TOL
    )

    # Merge boundaries that coincide to within tolerance.
    merged = [boundaries[0]]

    for boundary in boundaries[1:]:

        if boundary - merged[-1] > _BOUNDARY_TOL:
            merged.append(boundary)

    boundaries = merged

    segments = []

    # Build each backbone segment
    for i in range(len(boundaries) - 1):

        start = boundaries[i]
        end = boundaries[i + 1]

        midpoint = (start + end) / 2

        active_tubes = []
        curvatures = []

        for tube, (tube_start, tube_end) in zip(
            robot.tubes,
            intervals,
        ):

            if tube_start <= midpoint <= tube_end:

                active_tubes.append(tube)

                # Tube-local arc length of the segment midpoint,
                # measured from the tube's proximal end.
                sigma = midpoint - tube_start

                curvatures.append(
                    tube.curvature_at(sigma)
                )

        segments.append(
            Segment(
                start=start,
                end=end,
                active_tubes=active_tubes,
                curvatures=curvatures,
            )
        )

    return segments
