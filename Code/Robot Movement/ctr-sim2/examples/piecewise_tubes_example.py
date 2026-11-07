"""
Concentric tube robot whose tubes are straight / curved / straight.

Each tube is defined by the lengths of its three sections, measured from
the proximal (actuator) end to the distal tip:

    |<- length_before_curve ->|<- curve_length ->|<- length_after_curve ->|
    |        straight         |   precurvature   |        straight        |

All numbers below are PLACEHOLDERS. Replace the material properties,
diameters, section lengths and precurvatures with your measured values.
"""

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
# Material  (placeholder nylon values)
# ----------------------------------------

nylon = Material(
    name="Nylon",
    youngs_modulus=2.7e9,
    shear_modulus=1.0e9,
)


# ----------------------------------------
# Tubes (outermost first)
#
# Tube.from_lengths(
#     name,
#     length_before_curve,   # straight, from the proximal end
#     curve_length,          # curved section
#     length_after_curve,    # straight, out to the distal tip
#     precurvature,          # curvature of the curved section (1/m)
#     outer_diameter,
#     inner_diameter,
#     material,
# )
# ----------------------------------------

outer = Tube.from_lengths(
    name="OuterTube",
    length_before_curve=0.060,
    curve_length=0.050,
    length_after_curve=0.030,
    precurvature=18.0,
    outer_diameter=4.0e-3,
    inner_diameter=3.2e-3,
    material=nylon,
)

middle = Tube.from_lengths(
    name="MiddleTube",
    length_before_curve=0.080,
    curve_length=0.045,
    length_after_curve=0.035,
    precurvature=22.0,
    outer_diameter=2.8e-3,
    inner_diameter=2.1e-3,
    material=nylon,
)

inner = Tube.from_lengths(
    name="InnerTube",
    length_before_curve=0.100,
    curve_length=0.040,
    length_after_curve=0.040,
    precurvature=25.0,
    outer_diameter=1.8e-3,
    inner_diameter=0.8e-3,
    material=nylon,
)


# ----------------------------------------
# Robot state
#
# insertion beta_i: how far tube i's distal tip is past the robot base.
# 0 <= beta_i <= tube length.
# ----------------------------------------

state = CTRState(
    insertions=[0.10, 0.14, 0.18],
    rotations=[0.5, -1.0, 2.0],
)

robot = ConcentricTubeRobot(
    tubes=[outer, middle, inner],
    state=state,
)


# ----------------------------------------
# Solve
# ----------------------------------------

backbone = solve_forward_kinematics_v2(robot)

print("\nTube sections (m):")

for tube in robot.tubes:
    print(
        f"  {tube.name:11s} before={tube.length_before_curve:.3f}  "
        f"curve={tube.curve_length:.3f}  "
        f"after={tube.length_after_curve:.3f}  "
        f"total={tube.length:.3f}"
    )

print("\nMechanics segments "
      "(tube curvatures within each, 1/m):")

for i, solution in enumerate(backbone.segments):

    segment = solution.segment

    print(
        f"  {i}: s=[{segment.start:.4f}, {segment.end:.4f}]  "
        + ", ".join(
            f"{tube.name}={kappa:g}"
            for tube, kappa in zip(
                segment.active_tubes,
                segment.curvatures,
            )
        )
    )

samples = sample_backbone(
    backbone,
    ds=1e-3,
)

print("\nTip position (m):", samples.position[-1])


# ----------------------------------------
# Visualization
# ----------------------------------------

plot_backbone_samples(
    samples,
    robot,
)
