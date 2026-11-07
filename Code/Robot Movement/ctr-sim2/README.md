# ctr-sim

A Python and ROS 2 simulation framework for modeling, controlling, and teleoperating concentric tube robots (CTRs).

![CTR-SIM overview](docs/media/CTR-SIM_Overview_Screenshot.png)

`ctr-sim` implements segment-aware mechanics for concentric tube robots together with numerical task-space control and real-time ROS 2 visualization. The simulator models tube geometry, elastic properties, torsional interaction, backbone curvature, and spatial backbone integration, then exposes the resulting robot through a Cartesian teleoperation interface.

An Xbox controller can be used to command the CTR tip in Cartesian space while the robot backbone, mechanics segments, coordinate frames, tip pose, and tip trajectory are visualized live in RViz.

> **Status:** Research/development software. The current implementation focuses on unloaded CTR mechanics, numerical resolved-rate control, and interactive simulation.

## Demo

![CTR Cartesian teleoperation demo](docs/media/bestVideoGIF.gif)

The demo shows live Xbox-controller teleoperation of the CTR tip in RViz. Individual mechanics segments are rendered in distinct colors, the moving tip frame is published through TF, and the white line records the Cartesian tip trajectory.

## Features

### CTR modeling and mechanics

- Material definitions with Young's and shear moduli
- Tube geometry, length, diameter, precurvature, insertion, and rotation
- Straight / curved / straight tubes defined by the length before the curve, the curve length, and the length after the curve
- Multi-tube robot state representation
- Backbone segmentation at tube tips and at the start and end of every tube's curved section
- Segment-aware torsion solution
- Resultant backbone curvature computation
- Spatial backbone integration
- Dense continuous segment solutions
- Backbone sampling and direct tip-state evaluation
- Warm-started mechanics solves
- Optional force at the robot tip (e.g. the weight of an attached object)

### Cartesian control

- Numerical Cartesian position Jacobian
- Insertion and rotation finite differences
- Resolved-rate inverse kinematics
- Joint-step insertion constraints
- Cartesian velocity command watchdog
- Warm-started Jacobian mechanics
- Jacobian caching for faster interactive control

### ROS 2 teleoperation

- Xbox controller input
- Windows-to-WSL joystick bridge over UDP
- ROS 2 `sensor_msgs/Joy` publication
- Cartesian `geometry_msgs/Twist` commands
- CTR simulator ROS node
- Published tip pose
- Static `world -> ctr_base` transform
- Dynamic `ctr_base -> ctr_tip` transform
- One-command ROS 2 launch workflow

### Visualization

- Matplotlib backbone visualization
- RViz backbone visualization
- Distinct colors for mechanics segments
- TF frame visualization
- Live Cartesian tip trajectory
- Saved RViz configuration

### Testing and validation

- Automated tests with `pytest`
- Forward-mechanics validation against an independent MATLAB implementation
- Tests for robot geometry, segmentation, torsion, curvature, forward mechanics, sampling, Jacobians, resolved-rate control, and constraints
- Timing and Jacobian-reuse diagnostic examples

## System Architecture

```text
Xbox Controller
      |
      | XInput
      v
Windows joy_sender.py
      |
      | UDP
      v
ROS 2 joy_receiver
      |
      | /joy
      v
cartesian_teleop
      |
      | /ctr/cartesian_velocity
      v
ctr_simulator
      |
      +---- segment-aware CTR mechanics
      |
      +---- resolved-rate Cartesian control
      |
      +---- /ctr/backbone
      +---- /ctr/tip_pose
      +---- /ctr/tip_trajectory
      +---- /tf
      +---- /tf_static
               |
               v
              RViz
```

The Cartesian controller operates in the `ctr_base` frame. Controller inputs request Cartesian tip velocity along the X, Y, and Z axes of that frame.

## Mechanics

The V2 mechanics pipeline represents the backbone as a sequence of geometric/mechanics segments rather than treating the entire robot as one uniform interval.

Tube boundaries determine the backbone segmentation. Within each segment, the active tube set is fixed, allowing the torsional state and resultant curvature to be evaluated consistently with the local tube geometry.

The forward-mechanics pipeline is approximately:

```text
Robot configuration
      |
      v
Backbone segmentation
      |
      v
Segment-aware torsion solution
      |
      v
Segment curvature
      |
      v
Spatial backbone integration
      |
      v
Backbone / tip pose
```

The torsion problem is solved using a shooting formulation with continuation when required. Previous torsion solutions are reused as initial guesses during interactive operation, substantially reducing the cost of nearby mechanics solves.

## Tubes with Straight and Curved Sections

A tube does not have to be curved along its whole length. Measured from its proximal (actuator) end to its distal tip, a tube can be straight, then curved at a fixed precurvature, then straight again:

```text
|<- length_before_curve ->|<-  curve_length  ->|<- length_after_curve ->|
|        straight         |  curvature kappa   |        straight        |
proximal end                                                       distal tip
```

Define such a tube from its three section lengths:

```python
tube = Tube.from_lengths(
    name="MiddleTube",
    length_before_curve=0.080,   # m
    curve_length=0.045,          # m
    length_after_curve=0.035,    # m
    precurvature=22.0,           # 1/m, curved section only
    outer_diameter=2.8e-3,
    inner_diameter=2.1e-3,
    material=nylon,
)
```

The tube length is the sum of the three sections. The existing constructor still works: `Tube(name, length, precurvature, ...)` without the optional `length_before_curve` / `curve_length` arguments is a tube that is curved over its whole length, exactly as before.

All sections of a tube have the same bending and torsional stiffness, so a straight section still stiffens the robot and transmits torsion; it just contributes no precurvature. The backbone is split into mechanics segments wherever a tube tip or a curve start/end lies in the free backbone, and the torsion and curvature solutions use each tube's curvature within each segment.

See `examples/piecewise_tubes_example.py`. In the ROS 2 simulator the tubes are defined in the `TUBE_CONFIG` table at the top of `ctr_simulator.py`.

Only the segment-aware V2 mechanics support these tubes; the original V1 pipeline raises `NotImplementedError` for them.

## Tip Load

A force at the robot tip, such as the weight of an object attached to it, is passed as `tip_force` (N, in the robot base frame) to the forward solver, the Jacobian and the resolved-rate step:

```python
from ctr_sim.mechanics.loaded import tip_weight

# 15 g on the tip; gravity direction in the base frame
# (the robot extends from the base along +z).
load = tip_weight(0.015, gravity_direction=(0, -1, 0))

backbone = solve_forward_kinematics_v2(robot, previous.torsion_solution, tip_force=load)
J = numerical_position_jacobian_v2(robot, torsion_initial_guess=backbone.torsion_solution, tip_force=load)
dq = resolved_rate_step_v2(robot, dx, jacobian=J, tip_force=load)
```

Use the same load in all three calls. Without `tip_force` (or with `None`) the original unloaded solver runs, with unchanged results.

Under load the backbone shape and the tube torsion are coupled, so the loaded solver integrates position, orientation, internal moment and tube angles together and shoots on the base torsional strains plus the two bending moments at the base (the model of Rucker, Jones & Webster, IEEE T-RO 2010). Pass the previous `torsion_solution` as the warm start: under load it also reuses the solver's internal Jacobian, which makes nearby solves several times faster than a cold start. A loaded solve is still slower than an unloaded one (typically ~40 ms warm versus ~13 ms, and ~0.2 s versus ~0.1 s for a Jacobian on the development machine). A cold solve at an unusual configuration can take seconds.

Unlike the unloaded shape, the loaded shape depends on the absolute Young's modulus, so it should be accurate.

See `examples/tip_load_example.py`.

## Cartesian Resolved-Rate Control

For robot configuration

```text
q = [beta_1 ... beta_n alpha_1 ... alpha_n]
```

the numerical position Jacobian approximates

```text
J(q) = d p_tip / d q
```

where the first set of columns corresponds to tube insertions and the second set corresponds to tube rotations.

Given a desired Cartesian displacement `dx`, the controller computes a joint-space increment using a pseudoinverse-based resolved-rate step:

```text
dq = J(q)^+ dx
```

Insertion constraints are then applied before updating the simulated robot configuration.

### Jacobian caching

Computing the numerical Jacobian is the primary computational bottleneck because each column requires an additional nonlinear forward-mechanics solve.

To improve interactive performance, the ROS simulator caches the current Jacobian and reuses it for several nearby control updates before recomputing it. Diagnostic experiments showed that the Jacobian changes gradually over small Cartesian motions, making short-term reuse useful for teleoperation.

On the development system, cached control iterations were substantially faster than iterations requiring a full Jacobian refresh. Exact performance depends on robot geometry and hardware.

## Controller Mapping

The current Xbox mapping is:

| Controller input | Cartesian command |
| --- | --- |
| Right trigger | Deadman / motion enable |
| Left stick left/right | `ctr_base` X |
| Left stick up/down | `ctr_base` Y |
| Right stick up/down | `ctr_base` Z |

The commanded motion is expressed in the **CTR base frame**, not the moving tip frame.

In RViz, the standard TF axis colors are:

- **X:** red
- **Y:** green
- **Z:** blue

## Installation

### Requirements

The current development environment uses:

- Python 3.10
- NumPy
- SciPy
- ROS 2 Humble
- RViz 2
- WSL/Ubuntu for the ROS 2 environment
- Windows for Xbox/XInput acquisition

The core Python mechanics library can be used independently of ROS 2.

### Clone the repository

```bash
git clone <https://github.com/JackyP1937/ctr-sim>
cd ctr-sim
```

### Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the core simulator in editable mode:

```bash
python3 -m pip install --no-build-isolation -e .
```

### Build the ROS 2 package

From the ROS workspace:

```bash
cd ros2_ws
colcon build --packages-select ctr_teleop
source install/setup.bash
```

## Running the ROS 2 Simulator

### Ubuntu / WSL

From `ros2_ws`:

```bash
source install/setup.bash
ros2 launch ctr_teleop ctr_sim.launch.py
```

The launch file starts:

- `joy_receiver`
- `cartesian_teleop`
- `ctr_simulator`
- RViz with the saved CTR visualization configuration

### Windows

Run the Xbox-controller sender:

```powershell
python windows\joy_sender.py
```

The Windows process reads the Xbox controller and sends joystick state over UDP to the ROS 2 receiver running under WSL.

Hold the **right trigger** to enable robot motion.

## ROS 2 Interfaces

Important topics include:

| Topic | Purpose |
| --- | --- |
| `/joy` | Xbox joystick state |
| `/ctr/cartesian_velocity` | Desired Cartesian tip velocity |
| `/ctr/backbone` | Colored backbone mechanics-segment markers |
| `/ctr/tip_pose` | Current CTR tip pose |
| `/ctr/tip_trajectory` | Historical tip trajectory marker |
| `/tf` | Dynamic transforms including `ctr_base -> ctr_tip` |
| `/tf_static` | Static `world -> ctr_base` transform |

For example, the live tip position can be inspected with:

```bash
ros2 topic echo /ctr/tip_pose --field pose.position
```

The dynamic tip transform can be inspected with:

```bash
ros2 run tf2_ros tf2_echo ctr_base ctr_tip
```

## RViz Visualization

The included RViz configuration displays:

- a reference grid;
- the CTR backbone;
- mechanics segments using distinct colors;
- the `world`, `ctr_base`, and `ctr_tip` TF frames; and
- an optional tip trajectory.

The **Tip Trajectory** display can be toggled directly from the RViz Displays panel.

## Python Examples

The `examples/` directory contains demonstrations and diagnostics for:

- materials and tube definitions;
- robot construction;
- coordinate transforms;
- backbone segmentation;
- torsion mechanics;
- forward kinematics;
- visualization;
- numerical Jacobians;
- resolved-rate control;
- teleoperation;
- MATLAB validation cases;
- mechanics timing; and
- Jacobian-reuse behavior.

For example:

```bash
python examples/timing_v3.py
```

benchmarks warm forward mechanics, backbone sampling, direct tip extraction, and cold/warm numerical Jacobian computation.

## Testing

Run the test suite from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests -v
```

The test suite covers both the original mechanics implementation and the newer segment-aware V2 pipeline.

## Repository Structure

```text
ctr-sim/
|
+-- ctr_sim/
|   +-- control/          # Jacobians, resolved-rate control, constraints
|   +-- kinematics/       # Intervals and transformations
|   +-- mechanics/        # Torsion, curvature, integration, forward solves
|   +-- visualization/    # Python plotting utilities
|   +-- robot.py
|   +-- tube.py
|   +-- state.py
|   +-- segment.py
|   +-- backbone.py
|   +-- ...
|
+-- ros2_ws/
|   +-- src/
|       +-- ctr_teleop/   # ROS 2 teleoperation/simulation package
|
+-- windows/
|   +-- joy_sender.py     # Windows Xbox-controller sender
|
+-- examples/             # Examples, validation cases, and timing diagnostics
+-- tests/                # Automated test suite
+-- docs/                 # Design notes, roadmap, and media
+-- pyproject.toml
+-- README.md
```

## Known Limitations

- The mechanics model supports a force at the tip only: tip moments (an object whose centre of mass is offset from the tip), the tubes' own weight, external contact and distributed environmental loads are not modeled, and the stability of a loaded equilibrium (buckling, snap-through) is not checked.
- With sharp straight/curved transitions, the tip position has a kink (abrupt slope change) wherever one tube's curve start/end passes another tube's tip or curve boundary. A numerical Jacobian evaluated near such a point can be badly wrong, which shows up in teleoperation as occasional off-direction steps.
- Transitions between straight and curved sections are modeled as sharp (a step in precurvature), and tube rotation is imposed at the robot base (s = 0); torsional wind-up of tube material behind the base is not modeled.
- The numerical Jacobian requires multiple nonlinear forward solves and remains computationally expensive.
- Jacobian caching improves interactive performance but means some control updates use a Jacobian evaluated at a nearby configuration.
- The torsion shooting/continuation solver is not guaranteed to converge for arbitrary robot geometries or all finite-difference perturbations.
- Joint constraints are currently enforced by constraining the resolved-rate joint step rather than solving a fully constrained inverse-kinematics optimization.
- The Windows Xbox bridge is currently separate from the ROS 2 launch process.

## Future Work

Potential extensions include:

- constraint-aware inverse kinematics;
- asynchronous Jacobian computation;
- ROS-parameter-based configuration;
- joint-state publication;
- external loading and contact;
- additional CTR mechanics validation;
- higher-performance Jacobian computation; and
- hardware integration.

## Documentation

Additional development notes are available in:

- [`docs/design.md`](docs/design.md)
- [`docs/roadmap.md`](docs/roadmap.md)
