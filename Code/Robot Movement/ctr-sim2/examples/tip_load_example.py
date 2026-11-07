"""
Teleoperation-style loop with a weight attached to the robot tip.

Shows the three calls a teleop script makes, each with the same
tip_force:

    solve_forward_kinematics_v2(robot, warm_start, tip_force=load)
    numerical_position_jacobian_v2(robot, ..., tip_force=load)
    resolved_rate_step_v2(robot, dx, jacobian=J, tip_force=load)

Tube and material values are placeholders.
"""

import numpy as np

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)
from ctr_sim.mechanics.forward_v2 import solve_forward_kinematics_v2
from ctr_sim.mechanics.sampling import tip_position
from ctr_sim.mechanics.loaded import tip_weight
from ctr_sim.control.jacobian import numerical_position_jacobian_v2
from ctr_sim.control.resolved_rate import resolved_rate_step_v2
from ctr_sim.control.constraints import constrain_joint_step


nylon = Material(
    name="Nylon 12",
    youngs_modulus=1.3e9,
    shear_modulus=1.3e9 / 2.8,
)

robot = ConcentricTubeRobot(
    tubes=[
        Tube.from_lengths("OuterTube", 0.060, 0.050, 0.030, 18.0,
                          4.0e-3, 3.2e-3, nylon),
        Tube.from_lengths("MiddleTube", 0.080, 0.045, 0.035, 22.0,
                          2.8e-3, 2.1e-3, nylon),
        Tube.from_lengths("InnerTube", 0.100, 0.040, 0.040, 25.0,
                          1.8e-3, 0.8e-3, nylon),
    ],
    state=CTRState(
        insertions=[0.09, 0.12, 0.15],
        rotations=[0.5, -1.0, 2.0],
    ),
)


# ----------------------------------------
# Tip load
#
# 15 g attached to the tip. Gravity direction is expressed in the robot
# base frame, in which the robot extends from the base along +z:
#   robot pointing up    -> (0, 0, -1)
#   robot pointing down  -> (0, 0, +1)
#   robot horizontal     -> e.g. (0, -1, 0)
# Use load = None for no load.
# ----------------------------------------

load = tip_weight(0.015, gravity_direction=(0, -1, 0))

unloaded_tip = tip_position(solve_forward_kinematics_v2(robot))
backbone = solve_forward_kinematics_v2(robot, tip_force=load)

print("Tip without load (mm):", np.round(unloaded_tip * 1e3, 2))
print("Tip with 15 g    (mm):", np.round(tip_position(backbone) * 1e3, 2))


# ----------------------------------------
# Teleop loop: move the tip 0.3 mm per step along +x.
#
# Expect a few steps to go off-direction, with or without a load. With
# sharp straight/curved transitions, the tip position has a kink (its
# slope changes abruptly) whenever one tube's curve start/end passes
# another tube's tip or curve boundary; a Jacobian computed on one side
# of the kink is wrong on the other.
# ----------------------------------------

n = len(robot.tubes)
jacobian = None
jacobian_update_interval = 3

for step in range(10):

    # Refresh the Jacobian every few steps (it is the expensive part).
    if step % jacobian_update_interval == 0:
        jacobian = numerical_position_jacobian_v2(
            robot,
            torsion_initial_guess=backbone.torsion_solution,
            tip_force=load,
        )

    dx = np.array([3e-4, 0.0, 0.0])

    dq = resolved_rate_step_v2(
        robot,
        dx,
        jacobian=jacobian,
        tip_force=load,
    )
    dq = constrain_joint_step(robot, dq)

    robot.state.insertions = list(np.array(robot.state.insertions) + dq[:n])
    robot.state.rotations = list(np.array(robot.state.rotations) + dq[n:])
    robot.validate_configuration()

    # Warm start from the previous solution: pass the whole
    # torsion_solution (under load it also reuses the solver's Jacobian).
    previous_tip = tip_position(backbone)

    backbone = solve_forward_kinematics_v2(
        robot,
        backbone.torsion_solution,
        tip_force=load,
    )

    moved = (tip_position(backbone) - previous_tip) * 1e3

    print(f"step {step}: tip moved {np.round(moved, 3)} mm")
