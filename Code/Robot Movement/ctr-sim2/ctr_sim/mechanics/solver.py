from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.solution import CTRSolution
from ctr_sim.backbone import Backbone

def solve_forward_kinematics(
    robot: ConcentricTubeRobot,
) -> CTRSolution:
    """
    Solve the unloaded concentric tube robot boundary value problem.

    Parameters
    ----------
    robot : ConcentricTubeRobot
        Robot configuration defined by tube properties and
        current state (α, β).

    Returns
    -------
    CTRSolution
        Complete solution of the forward mechanics problem.
    """

    raise NotImplementedError