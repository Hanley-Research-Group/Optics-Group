import numpy as np

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)

from ctr_sim.control import numerical_position_jacobian


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
# Robot state
#

state = CTRState(
    insertions=[0.10, 0.12, 0.14],
    rotations=[
        0.0,
        np.deg2rad(30.0),
        0.0,
    ],
)

robot = ConcentricTubeRobot(
    tubes=[outer, middle, inner],
    state=state,
)


#
# Numerical Jacobian
#

J = numerical_position_jacobian(robot)

np.set_printoptions(
    precision=6,
    suppress=True,
)

print("\n========== Numerical Position Jacobian ==========\n")

print("Rows:")
print("  x")
print("  y")
print("  z\n")

print("Columns:")
print("  β1 β2 β3 α1 α2 α3\n")

print(J)