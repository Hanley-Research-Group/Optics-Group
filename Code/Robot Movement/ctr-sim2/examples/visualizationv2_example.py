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

from ctr_sim.mechanics.sampling import (
    sample_backbone,
)

from ctr_sim.visualization import (
    plot_backbone_samples,
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
    insertions=[0.10, 0.18, 0.19],
    rotations=[np.pi/2, 0.0, 0.0],
)

robot = ConcentricTubeRobot(
    tubes=[outer, middle, inner],
    state=state,
)


# ----------------------------------------
# Mechanics V2
# ----------------------------------------

backbone = solve_forward_kinematics_v2(
    robot,
)

print("\nV2 segment curvatures:")

for i, segment_solution in enumerate(backbone.segments):
    print(
        f"Segment {i}:",
        segment_solution.curvature,
    )

print("\nActive tubes:\n")

for i, segment_solution in enumerate(backbone.segments):

    print(
        i,
        segment_solution.segment.start,
        segment_solution.segment.end,
        [tube.name for tube in segment_solution.segment.active_tubes],
    )

samples = sample_backbone(
    backbone,
    ds=1e-3,
)

print(samples.position[-1])


# ----------------------------------------
# Visualization
# ----------------------------------------

plot_backbone_samples(
    samples,
    robot,
)