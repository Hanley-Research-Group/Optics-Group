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

from ctr_sim.control.resolved_rate import (
    resolved_rate_step_v2,
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
# Desired tip position
# ----------------------------------------

target = np.array([
    0.15,
    -0.02,
    0.09,
])


# ----------------------------------------
# Resolved-rate iterations
# ----------------------------------------

for k in range(25):

    backbone = solve_forward_kinematics_v2(
        robot,
    )

    samples = sample_backbone(
        backbone,
        ds=1e-3,
    )

    tip = samples.position[-1]

    error = target - tip

    print(
        f"Iteration {k:2d}",
        "tip =",
        np.round(tip, 4),
        "error =",
        np.round(error, 4),
    )

    dq = resolved_rate_step_v2(
        robot,
        error,
    )

    n = len(robot.tubes)

    robot.state.insertions += dq[:n]
    robot.state.rotations += dq[n:]


# ----------------------------------------
# Final visualization
# ----------------------------------------

backbone = solve_forward_kinematics_v2(
    robot,
)

samples = sample_backbone(
    backbone,
    ds=1e-3,
)

plot_backbone_samples(
    samples,
    robot,
)