from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)
from ctr_sim.mechanics.torsion import solve_torsion_bvp
import numpy as np


nitinol = Material(
    name="Nitinol",
    youngs_modulus=60e9,
    shear_modulus=23e9,
)

fiber = Material(
    name="SilicaFiber",
    youngs_modulus=15e9,
    shear_modulus=6.4e9,
)

outer = Tube(
    name="OuterTube",
    length=0.16,
    precurvature=15.0,
    outer_diameter=3.0e-3,
    inner_diameter=2.8e-3,
    material=nitinol,
)

middle = Tube(
    name="MiddleTube",
    length=0.18,
    precurvature=10.0,
    outer_diameter=2.0e-3,
    inner_diameter=1.8e-3,
    material=nitinol,
)

inner = Tube(
    name="InnerTube",
    length=0.20,
    precurvature=0,
    outer_diameter=1.6e-3,
    inner_diameter=0,
    material=fiber,
)

state = CTRState(
    insertions=[0.10, 0.12, 0.14],
    rotations=[0.0, np.pi/6, 0.0],
)

robot = ConcentricTubeRobot(
    tubes=[outer, middle, inner],
    state=state,
)


bvp_solution = solve_torsion_bvp(robot)

print("Solver status:", bvp_solution.status)
print("Solver message:", bvp_solution.message)
print("Number of mesh points:", len(bvp_solution.x))

theta = bvp_solution.y[0::2, :]
theta_dot = bvp_solution.y[1::2, :]

print("\nTheta:")
print(theta)

print("\nTheta dot:")
print(theta_dot)