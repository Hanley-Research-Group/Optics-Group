# ctr-sim Roadmap

This roadmap summarizes the major development milestones completed in the current `ctr-sim` research simulator and identifies possible future extensions.

## Phase 1 - Project Foundation

- [x] Create Git repository
- [x] Create Python package architecture
- [x] Create ROS 2 workspace
- [x] Establish automated testing with `pytest`
- [x] Add project design and roadmap documentation
- [x] Create Windows Xbox-controller bridge

## Phase 2 - CTR Robot Model

- [x] Material representation
- [x] Tube geometry and mechanical properties
- [x] CTR state representation
- [x] Multi-tube robot representation
- [x] Tube nesting and configuration validation
- [x] Arc-length interval utilities
- [x] Backbone representation
- [x] Mechanics segment representation
- [x] Straight / curved / straight tube geometry

## Phase 3 - Forward Mechanics

- [x] Unloaded forward kinematics
- [x] Tube torsion boundary-value solver
- [x] Resultant backbone curvature
- [x] Spatial backbone integration
- [x] Backbone reconstruction
- [x] MATLAB validation cases
- [x] Segment-aware backbone mechanics
- [x] Segment-aware torsion solution
- [x] Dense continuous segment solutions
- [x] Warm-started torsion solves
- [x] Direct tip-state evaluation
- [x] Curve-aware backbone segmentation, torsion, and curvature

## Phase 4 - Cartesian Control

- [x] Numerical position Jacobian
- [x] Segment-aware numerical Jacobian
- [x] Insertion and rotation finite differences
- [x] Resolved-rate Cartesian control
- [x] Insertion constraints
- [x] Cartesian command watchdog
- [x] Warm-started Jacobian mechanics
- [x] Jacobian caching and reuse diagnostics
- [x] Interactive Cartesian tip teleoperation

## Phase 5 - ROS 2 Integration

- [x] Windows Xbox/XInput sender
- [x] UDP joystick bridge to WSL
- [x] ROS 2 `Joy` publication
- [x] Cartesian `Twist` command publication
- [x] CTR simulator ROS node
- [x] Tip-pose publication
- [x] Static `world -> ctr_base` TF
- [x] Dynamic `ctr_base -> ctr_tip` TF
- [x] Command watchdog
- [x] Independent state-publication timer
- [x] ROS 2 launch workflow

## Phase 6 - Visualization

- [x] Matplotlib backbone visualization
- [x] RViz backbone visualization
- [x] Segment-colored backbone markers
- [x] Stale-marker cleanup
- [x] TF frame visualization
- [x] Live tip-pose visualization
- [x] Tip-trajectory visualization
- [x] Toggleable RViz trajectory display
- [x] Saved RViz configuration
- [x] GitHub screenshot and animated demonstration

## Phase 7 - Performance and Diagnostics

- [x] Mechanics timing diagnostics
- [x] Warm-start timing evaluation
- [x] Numerical Jacobian profiling
- [x] Direct tip-extraction optimization
- [x] Jacobian-reuse experiment
- [x] Cached Jacobian control updates
- [x] Increase interactive controller rate to 10 Hz

## Phase 8 - Documentation and Reproducibility

- [x] One-command ROS 2 launch workflow
- [x] Document system architecture
- [x] Document controller mapping
- [x] Document ROS 2 topics and TF hierarchy
- [x] Document installation and startup workflow
- [x] Document known numerical limitations
- [x] Add GitHub demo media

## Potential Future Work

The following are possible extensions rather than requirements for the current milestone.

### Mechanics

- [x] Tip force (e.g. weight of an attached object)
- [ ] Tip moment, self-weight and distributed loading
- [ ] Smooth straight/curved transitions
- [ ] Environmental contact
- [ ] Additional experimental/mechanical validation
- [ ] Improved robustness of the torsion shooting/continuation solver
- [ ] Additional tube constitutive models

### Control

- [ ] Constraint-aware inverse kinematics
- [ ] Analytical or semi-analytical Jacobian
- [ ] Asynchronous Jacobian computation
- [ ] Adaptive Jacobian-refresh criteria
- [ ] Velocity and acceleration limiting
- [ ] Tip-frame Cartesian jogging mode

### ROS 2

- [ ] Publish CTR joint state
- [ ] Move tuning values to ROS parameters/YAML
- [ ] ROS bag recording workflows
- [ ] Additional controller diagnostics
- [ ] Integrate or replace the Windows joystick bridge where appropriate

### Planning and Simulation

- [ ] Motion planning
- [ ] Collision checking
- [ ] Multi-goal trajectory generation
- [ ] External simulation-engine integration where useful
- [ ] Obstacle/contact visualization

### Hardware

- [ ] Hardware abstraction/interface
- [ ] Motor-command mapping
- [ ] Encoder/state feedback
- [ ] Hardware-in-the-loop validation
- [ ] Experimental CTR teleoperation