# CTR control using ctr-sim physics framework
#
# Requires the ctr-sim patches 0001-0003 (straight/curved tubes, torsion
# fix, tip force).
#
# Usage:
#   python ctr-controlv2.py                      # test mode, no load
#   python ctr-controlv2.py --tip-mass-g 12.5    # test mode, 12.5 g on the tip
#   python ctr-controlv2.py --live --tip-mass-g 12.5


import numpy as np
import pygame
import serial
import serial.tools.list_ports
import time
import argparse

from scipy.optimize import minimize, LinearConstraint, Bounds

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
    tip_position as model_tip_position,
)

from ctr_sim.mechanics.loaded import (
    tip_weight,
)

nylon = Material(
    name="Nylon 12",
    youngs_modulus=1.3e9,
    shear_modulus=480e6,
)

# These tubes are modelled as curved over their whole length. If your tubes
# are straight / curved / straight, add the section lengths, e.g.
#
#   Tube(..., length=0.185, length_before_curve=..., curve_length=...)
#
# (the length after the curve is length - length_before_curve - curve_length).
outer = Tube(
    name="OuterTube",
    length=0.180,  # Tube moved back 10mm
    precurvature=17,
    outer_diameter=0.003302,
    inner_diameter=0.002794,
    material=nylon,
    length_before_curve=0.110,
    curve_length=0.050,
)

middle = Tube(
    name="MiddleTube",
    length=0.260,
    precurvature=22,
    outer_diameter=0.0026416,
    inner_diameter=0.0021336,
    material=nylon,
    length_before_curve=0.190,
    curve_length=0.050,
)

inner = Tube(
    name="InnerTube",
    length=0.330,
    precurvature=29,
    outer_diameter=0.0019812,
    inner_diameter=0.0014732,
    material=nylon,
    length_before_curve=0.270,
    curve_length=0.050,
)


# Initial state. Ensure robot matches this.
state = CTRState(
    insertions=[0.0248, 0.0526, 0.0701],
    rotations=[0.0175, 0.0, 0.0], # This is in radians (converted to degrees is 1deg, 0, 0)
)
# Ensure last motor position matches initial state
lastMotorPos = np.concatenate([state.insertions, state.rotations])
# New pos: x, y, 330
START_MOTOR_MM = np.array([190, 170, 150]) # The absolute postions necessary to get the insertions to the correct positions
START_INS_M    = np.array(state.insertions, dtype=float)
AXIS_SIGN      = np.array([1.0, 1.0, 1.0])

robot = ConcentricTubeRobot(
    tubes=[outer, middle, inner],
    state=state,
)

# Tip speed while a key is held: cartesian_step metres per second
# (each frame moves cartesian_step * frame time).
cartesian_step = 1e-3  # 1 mm/s; adjust this for slower/faster speed
sampling_ds = 1e-3

hw = None # Robot

GCODE_AXES = ("X", "Y", "Z", "A", "B", "C")
FEEDRATE = 120 # Adjust for speed
MAX_IN_FLIGHT = 4


# ----------------------------------------------------------------------------
# Tip load
# ----------------------------------------------------------------------------
#
# Direction of gravity in the robot base frame (ctr_base). In that frame the
# tubes extend from the base along +z (the lead screw axis, horizontal here),
# and a tube at rotation 0 curves toward -y. Rotation 0 points UP on this
# robot, so up = -y and gravity (down) = +y.
GRAVITY_DIRECTION_BASE = np.array([0.0, 1.0, 0.0])

# Force on the tip (N, base frame); set from --tip-mass-g in main.
# None = no load.
TIP_FORCE = None


def set_tip_mass(mass_g):
    """Set the attached tip mass (grams). 0 removes the load."""
    global TIP_FORCE
    if mass_g < 0:
        raise ValueError("Tip mass cannot be negative.")
    TIP_FORCE = (
        None if mass_g == 0
        else tip_weight(mass_g / 1000.0, GRAVITY_DIRECTION_BASE)
    )


# ------------
# Constraints
# ------------

# All in meters, remember
# Following values are all to be setup according to the initial position
CLEAR_OM = 0.0205 # Free space between outer and middle carriage body
CLEAR_MI = 0.0224 # Free space between middle and inner carriage body
FRONT_ROOM_OUTER = 0.123 # How far outer tube carriage can move before hitting plexiglass
BACK_ROOM_INNER = 0.172 # How far inner tube carriage can move before hitting back
MARGIN = 0.002 # safety buffer on every limit (Should set this to at least the screw length that sticks out or something)
PLEXI_MIN = 0.0005  # Keep outer tube slightly out of the origin hole


# ------------------------
# Constraints Calculations
# ------------------------

# Max allowed (q[i+1] - q[i]): start-pose difference + free clearance - margin
GAP_OM = (state.insertions[1] - state.insertions[0]) + CLEAR_OM - MARGIN
GAP_MI = (state.insertions[2] - state.insertions[1]) + CLEAR_MI - MARGIN

# The model rejects any tube with insertion (beta) < 0, so every tube needs a
# strictly positive floor, otherwise the rail limits below let the middle/inner
# tube slide to a negative insertion and the step gets rejected.
MODEL_MIN = 1e-3  # 1 mm of slack above beta = 0 (covers solver/projection tolerance)

TUBE_LENGTHS = np.array([tube.length for tube in robot.tubes])

INS_MIN = np.zeros(3)
INS_MIN[0] = PLEXI_MIN
INS_MIN[2] = state.insertions[2] - BACK_ROOM_INNER + MARGIN
INS_MIN[1] = INS_MIN[2] - GAP_MI        # middle can't go back further than the inner allows
INS_MIN = np.maximum(INS_MIN, MODEL_MIN)

INS_MAX = np.zeros(3)
INS_MAX[0] = state.insertions[0] + FRONT_ROOM_OUTER - MARGIN
INS_MAX[1] = INS_MAX[0] + GAP_OM
INS_MAX[2] = INS_MAX[1] + GAP_MI
# The model also rejects beta > tube length.
INS_MAX = np.minimum(INS_MAX, TUBE_LENGTHS - MODEL_MIN)

ROT_LIM = 2 * np.pi
Q_MIN = np.concatenate([INS_MIN, np.full(3, -ROT_LIM)])
Q_MAX = np.concatenate([INS_MAX, np.full(3,  ROT_LIM)])

# Carriage-collision constraint for project_insertions()
D = np.array([[-1.0, 1.0, 0.0],
              [ 0.0,-1.0, 1.0]])
LEAD_MIN = np.array([-np.inf, -np.inf])
LEAD_MAX = np.array([GAP_OM, GAP_MI])

assert np.all(state.insertions >= INS_MIN) and np.all(state.insertions <= INS_MAX), "start pose outside limits"
assert np.all(D @ state.insertions <= LEAD_MAX), "start pose violates carriage gaps"

def insertions_to_motor_mm(ins_m):
    return START_MOTOR_MM + AXIS_SIGN * (np.asarray(ins_m) - START_INS_M) * 1000.0

# Extra clearance kept between carriages on top of MARGIN (metres). SLSQP only
# satisfies constraints to ~1e-6, and the carriages are physically big, so we
# keep the target strictly inside the feasible set.
COLLISION_BUFFER = 0.001

def project_insertions(q_des):
    """Closest feasible insertion vector to q_des.

    Returns None if no feasible point could be found (caller must skip the
    move) instead of silently returning a point that violates a limit.
    """
    lo = INS_MIN
    hi = INS_MAX
    lead_max = LEAD_MAX - COLLISION_BUFFER
    q_des = np.asarray(q_des, dtype=float)
    x0 = np.clip(q_des, lo, hi)
    res = minimize(
        lambda q: 0.5 * np.sum((q - q_des) ** 2),
        x0,
        jac=lambda q: q - q_des,
        bounds=Bounds(lo, hi),
        constraints=[LinearConstraint(D, LEAD_MIN, lead_max)],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 200},
    )
    q = np.clip(res.x, lo, hi)
    # Hard verification: never hand the hardware a pose that breaks a limit.
    if np.any(D @ q > lead_max + 1e-9):
        return None
    return q

# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------
class OctopusController:
    """
    Sends ABSOLUTE G1 moves to a Marlin board with 6 axes, with 'ok' flow control so the
    firmware queue never builds up (a deep queue = robot keeps moving after you let go).
    """

    def __init__(self, q0, port=None, baud=115200, live=False):
        self.live = live
        self.q = np.array(q0, dtype=float)    # last pose that was actually sent
        self.in_flight = 0
        self._rx = b""
        self.ser = None

        if not live:
            print("TEST MODE: commands are printed, nothing is sent.")
            return

        port = port or self._find_port()
        if port is None:
            raise ConnectionError("Could not find the Octopus board")
        print(f"Connecting to Octopus on {port}")
        self.ser = serial.Serial(port, baud, timeout=0, write_timeout=1.0)

        time.sleep(2.5)                       # opening the port usually resets the board
        self.ser.reset_input_buffer()         # discard boot banner
        self._command_blocking("G90")                                   # absolute positioning
        self._command_blocking("G92 " + self._rot_words(self.q))   # A/B/C only


        self._command_blocking("M92 X6760 Y6760 Z6760") # The Z steps are also hard coded in the X and Y homing functions
        self._command_blocking("M914 X115 Y120 Z120")

        self._command_blocking(f"M203 X{FEEDRATE/60} Y{FEEDRATE/60} Z{FEEDRATE/60}")



        print("Firmware position synchronised to start pose.")

    @staticmethod
    def _find_port():
        for p in serial.tools.list_ports.comports():
            dev, desc = p.device.lower(), (p.description or "").upper()
            if "usbmodem" in dev or "ttyacm" in dev or "MARLIN" in desc:
                return p.device
        return None

    @staticmethod
    def _axis_words(q):
        vals = np.concatenate([insertions_to_motor_mm(q[:3]), np.degrees(q[3:])])
        return " ".join(f"{a}{v:.3f}" for a, v in zip(GCODE_AXES, vals))

    @staticmethod
    def _rot_words(q):
        return " ".join(f"{a}{np.degrees(v):.3f}" for a, v in zip(GCODE_AXES[3:], q[3:]))

    def _poll(self):
        """Non-blocking read; counts 'ok' acknowledgements, prints errors."""
        data = self.ser.read(self.ser.in_waiting)
        if not data:
            return
        self._rx += data
        while b"\n" in self._rx:
            line, self._rx = self._rx.split(b"\n", 1)
            text = line.decode(errors="replace").strip()
            if text.lower().startswith("ok"):
                self.in_flight = max(0, self.in_flight - 1)
            elif text.startswith(("Error", "!!")):
                print(f"[MARLIN] {text}")

    def _command_blocking(self, cmd, timeout=5.0):
        self.ser.write((cmd + "\n").encode())
        self.in_flight += 1
        deadline = time.monotonic() + timeout
        while self.in_flight > 0:
            self._poll()
            if time.monotonic() > deadline:
                raise TimeoutError(f"No 'ok' from firmware for {cmd!r}")
            time.sleep(0.005)

    def ready(self):
        """True if a new move can be queued right now."""
        if not self.live:
            return True
        self._poll()
        return self.in_flight < MAX_IN_FLIGHT

    def move_to(self, q_target):
        """Send an absolute pose. Returns False (and does nothing) if the firmware queue is busy."""
        cmd = f"G1 {self._axis_words(q_target)} F{FEEDRATE}"
        if self.live:
            if not self.ready():
                return False
            self.ser.write((cmd + "\n").encode())
            self.in_flight += 1
        else:
            print(f"[VIRTUAL] {cmd}")
        self.q = np.array(q_target, dtype=float)
        return True

    def shutdown(self, emergency=False, reset_angles=True, target_rotations=(0.0175, 0.0, 0.0)):
        if not self.live or self.ser is None:
            if reset_angles and not emergency:
                rot_cmd = " ".join(f"{a}{np.degrees(v):.3f}" for a, v in zip(GCODE_AXES[3:], target_rotations))
                print(f"[VIRTUAL] Resetting tube angles: G1 {rot_cmd} F{FEEDRATE}")
            return
        try:
            if emergency:
                self.ser.write(b"M410\n")             # quick-stop: drop queued moves
            else:
                self._command_blocking("M400", timeout=10.0)   # let queued moves finish
                if reset_angles:
                    print("Resetting tube angles (A, B, C)...")
                    rot_cmd = " ".join(f"{a}{np.degrees(v):.3f}" for a, v in zip(GCODE_AXES[3:], target_rotations))
                    self._command_blocking(f"G90")
                    self._command_blocking(f"G1 {rot_cmd} F{FEEDRATE}", timeout=15.0)
                    self._command_blocking("M400", timeout=30.0) # wait for rotation reset to finish
                    print("Tube angles reset complete.")
        finally:
            self.ser.close()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def solve_robot(torsion_initial_guess=None):
    """Solve the model (with the tip load, if any) and sample the backbone."""

    backbone = solve_forward_kinematics_v2(
        robot,
        torsion_initial_guess=torsion_initial_guess,
        tip_force=TIP_FORCE,
    )

    samples = sample_backbone(
        backbone,
        ds=sampling_ds,
    )

    return backbone, samples


def tip_position(guess=None):
    """Model tip position for the current robot.state (no backbone sampling)."""
    backbone = solve_forward_kinematics_v2(
        robot,
        torsion_initial_guess=guess,
        tip_force=TIP_FORCE,
    )
    return np.asarray(model_tip_position(backbone), dtype=float)


# ----------------------------------------
# Capture inputs
# ----------------------------------------
def captureInputs():
    pygame.init()
    pygame.display.set_mode((300, 300))
    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            dt = clock.tick(20) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            k = pygame.key.get_pressed()
            dx = np.array([
                int(k[pygame.K_d]) - int(k[pygame.K_a]),
                int(k[pygame.K_s]) - int(k[pygame.K_w]),
                int(k[pygame.K_e]) - int(k[pygame.K_q]),
            ], dtype=float)
            if dx.any():
                updateRobot(dx * cartesian_step * dt)
    except KeyboardInterrupt:
        print("\nCtrl+C detected, exiting teleop...")
    finally:
        pygame.quit()


# Rotation weighting. The stock resolved-rate step is a minimum-norm solution in
# mixed units (metres for insertions, radians for rotations). A radian only moves
# the tip ~LEVER metres, so the minimum-norm answer almost never uses the
# rotation axes (~0.002 deg per mm of tip motion) and the tubes barely turn.
# Treating a rotation of 1 rad as costing ROT_LEVER metres of insertion makes the
# two kinds of joint comparable. Lower ROT_LEVER => more rotation, higher => less.
ROT_LEVER = 0.03      # m/rad
FD_INS = 1e-4         # finite-difference step for insertions (m)
FD_ROT = 1e-3         # finite-difference step for rotations (rad)
DAMPING = 1e-4        # damped-least-squares term (in weighted units)


def weighted_rate_step(dx, current_backbone):
    """Damped, weighted resolved-rate step: returns dq = [d_ins(3), d_rot(3)].

    The Jacobian is computed with the same tip load as the model, about the
    current configuration (whose solved backbone is current_backbone).
    """
    n = len(robot.tubes)
    ins0 = list(robot.state.insertions)
    rot0 = list(robot.state.rotations)

    # Warm start every perturbed solve from the current solution. Passing the
    # whole torsion_solution (not just base_theta_dot) is several times
    # faster under load.
    guess = current_backbone.torsion_solution
    p0 = np.asarray(model_tip_position(current_backbone), dtype=float)

    J = np.zeros((3, 2 * n))
    try:
        for j in range(2 * n):
            h = FD_INS if j < n else FD_ROT
            # one-sided: step toward the side that stays inside the limits
            sign = 1.0
            q = np.concatenate([ins0, rot0])
            if q[j] + h > Q_MAX[j]:
                sign = -1.0
            robot.state.insertions[:] = ins0
            robot.state.rotations[:] = rot0
            if j < n:
                robot.state.insertions[j] = ins0[j] + sign * h
            else:
                robot.state.rotations[j - n] = rot0[j - n] + sign * h
            J[:, j] = sign * (tip_position(guess) - p0) / h
    finally:
        robot.state.insertions[:] = ins0
        robot.state.rotations[:] = rot0

    # Weighted min-norm: minimise dq^T W dq subject to J dq = dx
    w = np.concatenate([np.ones(n), np.full(n, ROT_LEVER ** 2)])   # cost of 1 rad = ROT_LEVER m
    Jw = J / np.sqrt(w)                       # J W^-1/2
    dqw = Jw.T @ np.linalg.solve(Jw @ Jw.T + DAMPING * np.eye(3), dx)
    return dqw / np.sqrt(w)


def updateRobot(dx):
    global backbone
    n = len(robot.tubes)

    # Only compute a step if the hardware can take it, so the model never
    # gets ahead of the robot.
    if not hw.ready():
        return

    try:
        dq = weighted_rate_step(dx, backbone)
    except Exception as e:
        print("Jacobian failed, step skipped:", e)
        return

    if np.max(np.abs(dq[:n])) > 2e-3 or np.max(np.abs(dq[n:])) > np.radians(10):
        print("Near-singular step rejected:", dq)
        return

    ins_old = list(robot.state.insertions)
    rot_old = list(robot.state.rotations)

    q_new = np.concatenate([ins_old, rot_old]) + dq
    ins_new = project_insertions(q_new[:n])        # rail limits, plexiglass, carriage gaps
    if ins_new is None:
        print("No collision-free insertion found, step skipped")
        return
    q_new[:n] = ins_new
    q_new[n:] = np.clip(q_new[n:], Q_MIN[n:], Q_MAX[n:])

    robot.state.insertions[:] = q_new[:n]
    robot.state.rotations[:] = q_new[n:]

    try:
        robot.validate_configuration()
        new_backbone, samples = solve_robot(torsion_initial_guess=backbone.torsion_solution)
    except Exception as e:
        # roll back so the sim never diverges from the hardware
        robot.state.insertions[:] = ins_old
        robot.state.rotations[:] = rot_old
        print("Rejected, model says invalid:", e)
        return

    if not moveRobot(q_new):
        # Hardware did not take the move: keep the model at the old pose.
        robot.state.insertions[:] = ins_old
        robot.state.rotations[:] = rot_old
        return

    backbone = new_backbone
    print("Tip:", np.round(samples.position[-1], 6))


def moveRobot(q_target):
    """Send q_target to the hardware. Returns True if the robot is (or will be) there."""
    global lastMotorPos
    if np.all(np.abs(q_target - lastMotorPos) <= 1e-9):
        return True
    if hw.move_to(q_target):
        lastMotorPos = q_target.copy()
        return True
    print("Hardware busy, move skipped")
    return False



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="talk to real hardware (default: test mode)")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--tip-mass-g",
        type=float,
        default=0.0,
        help="mass attached to the tube tip, in grams (default: 0, no load). "
             "Its weight acts downward, i.e. opposite to the curve direction at rotation 0.",
    )
    return ap.parse_args()


# Homing Functions:

def home_z_inverted(controller, step_mm=-1.0, feedrate=150, max_travel_mm=250.0):
    """
    Custom Z homing routine for OctopusController when the limit switch logic is inverted.
    Moves Z incrementally and polls M119 until the switch reads 'open'.

    Args:
        controller: Instance of OctopusController
        step_mm: How far to move per step in mm (negative moves toward Z-min)
        feedrate: Speed of the probing moves in mm/min
        max_travel_mm: Safety limit to prevent infinite crashing
    """
    if not controller.live or controller.ser is None:
        print("[VIRTUAL] Custom inverted Z homing executed.")
        # controller.q[2] = 0.0  # Update internal array (Z is index 2)
        return True

    print("Starting custom Z homing (looking for 'open' state)...")

    # 1. Wait for any existing moves in the queue to finish
    controller._command_blocking("M400")

    # 2. Disable software endstops so Marlin allows negative movement
    controller._command_blocking("M211 S0")

    # 3. Switch to Relative Positioning mode
    controller._command_blocking("G91")

    moved = 0.0
    homed = False

    while abs(moved) < max_travel_mm:
        # We write directly to serial instead of using _command_blocking()
        # so we don't increment in_flight, and we handle the read manually.
        controller.ser.write(b"M119\n")

        z_is_open = False
        response_complete = False

        # Custom read loop: controller._poll() ignores non-error text,
        # so we intercept the M119 output directly from the RX buffer.
        while not response_complete:
            data = controller.ser.read(controller.ser.in_waiting or 1)
            if data:
                controller._rx += data

            while b"\n" in controller._rx:
                # Extract one line at a time
                line_bytes, controller._rx = controller._rx.split(b"\n", 1)
                text = line_bytes.decode(errors="replace").strip()

                # Look for the Z endstop state
                if text.startswith("z_min: open") or text.startswith("z_max: open"):
                    z_is_open = True

                # 'ok' marks the end of the M119 response block
                elif text.lower().startswith("ok"):
                    response_complete = True
                    break

        if z_is_open:
            print(f"Limit switch reached! Homing complete after {moved:.1f} mm.")
            homed = True
            break

        # 4. If not reached, step the motor
        controller._command_blocking(f"G1 Z{step_mm} F{feedrate}")
        controller._command_blocking("M400")  # Block until the physical move is done
        moved += step_mm

    if not homed:
        print("WARNING: Reached max travel limit without hitting the switch!")

    # 5. Zero out the coordinates at this physical location
    if homed:
        controller._command_blocking("G92 Z0")

        # Sync the Python class's internal state array.
        # Note: Your class expects q[] in METERS, so 0.0 is correct.
        # controller.q[2] = 0.0

    # 6. Clean up: Restore Absolute Positioning and Software Endstops
    controller._command_blocking("G90")
    controller._command_blocking("M211 S1")

    return homed

def home_x_normal(controller, home_coordinate_mm=0.0, timeout=30.0):
    """
    Homes the X axis using Marlin's standard G28 command.
    Blocks until the hardware finishes the cycle, then returns a boolean.

    Args:
        controller: Instance of OctopusController.
        home_coordinate_mm: The physical position Marlin assigns to X after homing (usually 0).
        timeout: Maximum time in seconds to wait for the physical move to finish.
    """
    if not controller.live or controller.ser is None:
        print("[VIRTUAL] Homed X axis normally.")
        # controller.q[1] = home_coordinate_mm / 1000.0  # Y is index 1
        return True

    print("Homing X axis (Standard G28)...")

    try:
        # 1. Wait for any existing moves to finish to clear the planner
        controller._command_blocking("M400")

        # 2. Command the standard homing cycle.
        # The script halts here until Marlin replies with 'ok' after the bump.
        controller._command_blocking("M92 Z0.00001")
        controller._command_blocking("G28 X", timeout=timeout)

        # 3. Ensure software endstops are active (standard after homing)
        controller._command_blocking("M211 S1")

        # 4. Sync the Python class's internal array so future moves are accurate
        # Note: Your class expects the first 3 indices (X, Y, Z) in METERS.
        # controller.q[0] = home_coordinate_mm / 1000.0

        controller._command_blocking("M92 Z6760")
        controller._command_blocking("G92 Z0")
        print("X axis successfully homed!")
        return True

    except TimeoutError:
        print(f"ERROR: Homing X axis timed out after {timeout} seconds.")
        # If it times out, flush the serial and queue to prevent locking the app
        if controller.ser:
            controller.ser.reset_input_buffer()
            controller.in_flight = 0
        return False

    except Exception as e:
        print(f"ERROR during X homing: {e}")
        return False

def home_y_normal(controller, home_coordinate_mm=0.0, timeout=30.0):
    """
    Homes the Y axis using Marlin's standard G28 command.
    Blocks until the hardware finishes the cycle, then returns a boolean.

    Args:
        controller: Instance of OctopusController.
        home_coordinate_mm: The physical position Marlin assigns to Y after homing (usually 0).
        timeout: Maximum time in seconds to wait for the physical move to finish.
    """
    if not controller.live or controller.ser is None:
        print("[VIRTUAL] Homed Y axis normally.")
        # controller.q[1] = home_coordinate_mm / 1000.0  # Y is index 1
        return True

    print("Homing Y axis (Standard G28)...")

    try:
        # 1. Wait for any existing moves to finish to clear the planner
        controller._command_blocking("M400")

        # 2. Command the standard homing cycle.
        # The script halts here until Marlin replies with 'ok' after the bump.
        controller._command_blocking("M92 Z0.00001")
        controller._command_blocking("G28 Y", timeout=timeout)

        # 3. Ensure software endstops are active (standard after homing)
        controller._command_blocking("M211 S1")

        # 4. Sync the Python class's internal array so future moves are accurate
        # Note: Your class expects the first 3 indices (X, Y, Z) in METERS.
        # controller.q[1] = home_coordinate_mm / 1000.0

        controller._command_blocking("M92 Z6760")
        controller._command_blocking("G92 Z0")
        print("Y axis successfully homed!")
        return True

    except TimeoutError:
        print(f"ERROR: Homing Y axis timed out after {timeout} seconds.")
        # If it times out, flush the serial and queue to prevent locking the app
        if controller.ser:
            controller.ser.reset_input_buffer()
            controller.in_flight = 0
        return False

    except Exception as e:
        print(f"ERROR during Y homing: {e}")
        return False


# Main

if __name__ == '__main__':
    ap = parse_args()

    unloaded_tip = tip_position()   # before the load is set
    set_tip_mass(ap.tip_mass_g)

    backbone, samples = solve_robot() # Initial solve (with the tip load, if any)
    if TIP_FORCE is not None:
        sag = samples.position[-1] - unloaded_tip
        print(f"Tip load: {ap.tip_mass_g:g} g, force {np.round(TIP_FORCE, 5)} N (base frame, +y = down).")
        print(f"Model tip moved by the load (mm): {np.round(sag * 1e3, 3)}")

    hw = OctopusController(
        q0=np.concatenate([state.insertions, state.rotations]),
        port=ap.port, baud=ap.baud, live=ap.live,
    )

    clean_exit = False
    try:
        # Homing
        homed = home_z_inverted(hw, step_mm=-2.0, feedrate=FEEDRATE, max_travel_mm=300) and home_y_normal(hw, timeout=300) and home_x_normal(hw, timeout=300)
        # and home_x_top_stallguard(hw, max_distance_mm=375, feedrate=150, top_coordinate_mm=358.5)

        if not homed:
            raise SystemExit("Homing failed, not starting teleop.")

        if hw.live:
            #hw._command_blocking(f"G1 X{START_MOTOR_MM[0]/2} F{FEEDRATE*1.5}")
            #hw._command_blocking(f"G1 Y{START_MOTOR_MM[1]/2} F{FEEDRATE*1.5}")
            hw._command_blocking(f"G1 X30 F{FEEDRATE*1.5}")
            hw._command_blocking(f"G1 Y15 F{FEEDRATE*1.5}")

        hw.move_to(lastMotorPos)

        if hw.live:
            hw._command_blocking("M400", timeout=300)  # wait for arrival

        captureInputs()
        clean_exit = True

    finally:
        # Normal exit: finish queued moves and reset the tube angles.
        # Anything else (homing failure, error): quick-stop.
        hw.shutdown(emergency=not clean_exit)
