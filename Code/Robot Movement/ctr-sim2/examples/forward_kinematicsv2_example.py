import numpy as np

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)

from ctr_sim.mechanics.forward_v2 import (
    solve_forward_kinematics_v2,
)

# ----------------------------------------
# Materials
# ----------------------------------------

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

# ----------------------------------------
# Robot
# ----------------------------------------

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
    precurvature=0.0,
    outer_diameter=1.6e-3,
    inner_diameter=0.0,
    material=fiber,
)

state = CTRState(
    insertions=[0.10, 0.12, 0.14],
    rotations=[0.0, np.pi / 6, 0.0],
)

robot = ConcentricTubeRobot(
    tubes=[outer, middle, inner],
    state=state,
)

# ----------------------------------------
# Solve using Mechanics V2
# ----------------------------------------

backbone = solve_forward_kinematics_v2(robot)

print(f"\nNumber of solved segments: {len(backbone.segments)}")

for i, segment_solution in enumerate(backbone.segments):

    print("\n" + "=" * 60)
    print(f"Segment {i}")
    print("=" * 60)

    print(
        f"Interval: "
        f"{segment_solution.segment.start:.4f} "
        f"-> "
        f"{segment_solution.segment.end:.4f}"
    )

    print(
        "Active tubes:",
        [tube.name for tube in segment_solution.segment.active_tubes],
    )

    print("\nTheta:")
    print(segment_solution.theta)

    print("\nCurvature:")
    print(segment_solution.curvature)

    print("\nIVP solution:")

    print("t interval:")
    print(segment_solution.ivp_solution.t)

    print("\nState matrix shape:")
    print(segment_solution.ivp_solution.y.shape)

    print("\nDense output available:")
    print(segment_solution.ivp_solution.sol is not None)