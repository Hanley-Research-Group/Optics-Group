import numpy as np

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)

from ctr_sim.mechanics import solve_forward_kinematics
from ctr_sim.control import resolved_rate_step


#
# Materials
#

nitinol = Material(
    name="Nitinol",
    youngs_modulus=60e9,
    shear_modulus=23e9,
)

fiber = Material(
    name="Fiber",
    youngs_modulus=15e9,
    shear_modulus=6.4e9,
)

#
# Tubes
#

outer = Tube(
    name="Outer",
    length=0.16,
    precurvature=15.0,
    outer_diameter=3.0e-3,
    inner_diameter=2.8e-3,
    material=nitinol,
)

middle = Tube(
    name="Middle",
    length=0.18,
    precurvature=10.0,
    outer_diameter=2.0e-3,
    inner_diameter=1.8e-3,
    material=nitinol,
)

inner = Tube(
    name="Inner",
    length=0.20,
    precurvature=0.0,
    outer_diameter=1.6e-3,
    inner_diameter=0.0,
    material=fiber,
)

#
# Robot
#

robot = ConcentricTubeRobot(
    tubes=[outer, middle, inner],
    state=CTRState(
        insertions=[0.10, 0.12, 0.14],
        rotations=[
            0.0,
            np.deg2rad(30.0),
            0.0,
        ],
    ),
)

#
# Initial forward kinematics
#

solution0 = solve_forward_kinematics(robot)

#
# Desired Cartesian displacement
#

dx = np.array([
    1e-3,
    0.0,
    0.0,
])

#
# Compute joint increment
#

dq = resolved_rate_step(
    robot,
    dx,
)

print("\nDesired Cartesian Motion")
print(dx)

print("\nComputed Joint Increment")
print(dq)

#
# Apply joint increment
#

n = len(robot.tubes)

robot.state.insertions = (
    np.array(robot.state.insertions)
    + dq[:n]
).tolist()

robot.state.rotations = (
    np.array(robot.state.rotations)
    + dq[n:]
).tolist()

#
# New forward kinematics
#

solution1 = solve_forward_kinematics(robot)

actual_dx = (
    solution1.tip_position
    - solution0.tip_position
)

print("\nInitial Tip Position")
print(solution0.tip_position)

print("\nNew Tip Position")
print(solution1.tip_position)

print("\nActual Cartesian Motion")
print(actual_dx)

print("\nPosition Error")
print(dx - actual_dx)