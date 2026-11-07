import numpy as np

from .torsion import (
    evaluate_torsion_solution_sampled,
    solve_torsion_bvp
)
from .integration import (
    integrate_segments,
    resultant_curvature
)

from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.solution import CTRSolution

def solve_forward_kinematics(
    robot: ConcentricTubeRobot,
    ds: float = 1e-3,
) -> CTRSolution:
    """
    Solve the unloaded forward kinematics of a concentric tube robot.

    This is the original (V1) sampled pipeline, which assumes every tube
    has a single uniform precurvature. Tubes with a straight/curved/
    straight profile are only supported by the segment-aware V2 pipeline
    (ctr_sim.mechanics.forward_v2.solve_forward_kinematics_v2).
    """

    for tube in robot.tubes:

        if not tube.has_uniform_curvature:
            raise NotImplementedError(
                f"Tube {tube.name} has a straight/curved profile, which "
                "the V1 mechanics pipeline cannot represent. Use "
                "solve_forward_kinematics_v2 instead."
            )

    bvp_solution = solve_torsion_bvp(robot)

    s = np.arange(
        0.0,
        robot.state.insertions[-1] + ds,
        ds,
    )

    theta = evaluate_torsion_solution_sampled(
        bvp_solution,
        s,
    )

    u = resultant_curvature(
        robot,
        theta,
        s,
    )

    backbone = integrate_segments(
        u,
        s,
        robot,
    )

    return CTRSolution(
        robot=robot,
        backbone=backbone,
        torsion=theta,
        curvature=u,
        s=s,
        tip_position=backbone.position[-1],
        tip_rotation=backbone.rotation[-1],
    )