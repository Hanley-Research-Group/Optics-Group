from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)
from ctr_sim.kinematics.intervals import occupied_intervals
from ctr_sim.kinematics.intervals import backbone_segments

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
    insertions=[0.0, 0.0, 0.0],
    rotations=[0.0, 0.0, 0.0],
)

robot = ConcentricTubeRobot(
    tubes=[outer, middle, inner],
    state=state,
)

segments = backbone_segments(robot)

print()

for segment in segments:

    names = [tube.name for tube in segment.tubes]

    print(
        f"[{segment.start:.3f}, {segment.end:.3f}] : {names}"
    )



