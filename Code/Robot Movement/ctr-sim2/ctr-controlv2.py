# CTR control using ctr-sim physics framework

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
)


nitinol = Material(
    name="Nitinol",
    youngs_modulus=60e9,
    shear_modulus=23e9,
)

nylon = Material(
    name="Nylon 12",
    youngs_modulus=1.3e9,
    shear_modulus=480e6,
)

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
    length=0.340,
    precurvature=29,
    outer_diameter=0.0019812,
    inner_diameter=0.0014732,
    material=nylon,
    length_before_curve=0.270,
    curve_length=0.050,
)

# Initial state. Ensure robot matches this.
state = CTRState(
    insertions=[0.0005, 0.027, 0.0305],
    rotations=[0.0175, 0.0, 0.0],  # Radians (~1 deg, 0, 0)
)

lastMotorPos = np.concatenate([state.insertions, state.rotations])
START_MOTOR_MM = np.array([250.0, 220.0, 75.0])  # Absolute motor positions (mm) for initial insertions
START_INS_M    = np.array(state.insertions, dtype=float)
AXIS_SIGN      = np.array([1.0, 1.0, 1.0])

robot = ConcentricTubeRobot(
    tubes=[outer, middle, inner],
    state=state,
)

# Teleop speeds and IK weighting
cartesian_speed = 4e-3   # 4 mm/s tip velocity when holding WASDQE
direct_rot_speed = np.radians(45.0)  # 45 deg/s for direct tube rotation keys (U/J, I/K, O/L)
ROT_WEIGHT = 35.0        # Scales rotational Jacobian columns (rad vs m) so tubes actively rotate
IK_DAMPING = 1e-2
sampling_ds = 1e-3

hw = None
GCODE_AXES = ("X", "Y", "Z", "A", "B", "C")
FEEDRATE = 120.0         # Max linear feedrate in mm/min (2 mm/s)
ROT_FEEDRATE = 1800.0    # Max rotational feedrate in deg/min (30 deg/s)
# Set to your A/B/C steps-per-degree if Marlin's firmware default is wrong (or None to keep firmware M92)
ROT_STEPS_PER_DEG = None # e.g., (8.8889, 8.8889, 8.8889) for 16x microstepping with 1:1 gearing
MAX_IN_FLIGHT = 4

# ------------
# Constraints
# ------------
# Because Z homes to 0, Y homes against Z (Y=0), and X homes against Y (X=0),
# the true physical clearances at START_MOTOR_MM are derived directly from START_MOTOR_MM:
CLEAR_OM = (START_MOTOR_MM[0] - START_MOTOR_MM[1]) / 1000.0  # 0.020 m at [240, 220, 75]
CLEAR_MI = (START_MOTOR_MM[1] - START_MOTOR_MM[2]) / 1000.0  # 0.145 m at [240, 220, 75]
BACK_ROOM_INNER = START_MOTOR_MM[2] / 1000.0                 # 0.075 m at [240, 220, 75]
FRONT_ROOM_OUTER = 0.066  # Free travel before outer carriage hits front plexiglass

MARGIN = 0.005       # 5 mm safety buffer between carriages and rail ends
PLEXI_MIN = 0.0005   # Minimum insertion (0.5 mm) so beta is strictly > 0
TIP_MIN_GAP = 0.0005 # Keep each inner tube tip at least 0.5 mm ahead of the tube outside it

# ------------------------
# Constraints Calculations
# ------------------------
TUBE_LENGTHS = np.array([t.length for t in robot.tubes], dtype=float)

# Max allowed (q[i+1] - q[i]): start-pose difference + free carriage clearance - safety margin,
# also capped by exposed proximal tube length difference (L[i+1] - L[i]).
GAP_OM = min(
    (state.insertions[1] - state.insertions[0]) + CLEAR_OM - MARGIN,
    TUBE_LENGTHS[1] - TUBE_LENGTHS[0] - TIP_MIN_GAP,
)
GAP_MI = min(
    (state.insertions[2] - state.insertions[1]) + CLEAR_MI - MARGIN,
    TUBE_LENGTHS[2] - TUBE_LENGTHS[1] - TIP_MIN_GAP,
)

INS_MIN = np.zeros(3)
INS_MIN[0] = PLEXI_MIN
INS_MIN[2] = max(PLEXI_MIN + 2 * TIP_MIN_GAP, state.insertions[2] - BACK_ROOM_INNER + MARGIN)
INS_MIN[1] = max(PLEXI_MIN + TIP_MIN_GAP, INS_MIN[2] - GAP_MI)

INS_MAX = np.zeros(3)
INS_MAX[0] = min(TUBE_LENGTHS[0] - MARGIN, state.insertions[0] + FRONT_ROOM_OUTER - MARGIN)
INS_MAX[1] = min(TUBE_LENGTHS[1] - MARGIN, INS_MAX[0] + GAP_OM)
INS_MAX[2] = min(TUBE_LENGTHS[2] - MARGIN, INS_MAX[1] + GAP_MI)

ROT_LIM = 2 * np.pi
Q_MIN = np.concatenate([INS_MIN, np.full(3, -ROT_LIM)])
Q_MAX = np.concatenate([INS_MAX, np.full(3,  ROT_LIM)])

# Linear constraint on insertions: LEAD_MIN <= [q1 - q0, q2 - q1] <= LEAD_MAX
D = np.array([
    [-1.0,  1.0, 0.0],
    [ 0.0, -1.0, 1.0],
])
LEAD_MIN = np.array([TIP_MIN_GAP, TIP_MIN_GAP])
LEAD_MAX = np.array([GAP_OM, GAP_MI])

assert np.all(state.insertions >= INS_MIN) and np.all(state.insertions <= INS_MAX), (
    f"Start pose {state.insertions} outside limits [{INS_MIN}, {INS_MAX}]"
)
assert np.all(D @ state.insertions >= LEAD_MIN) and np.all(D @ state.insertions <= LEAD_MAX), (
    f"Start pose violates carriage/telescoping gaps: {D @ state.insertions} not in [{LEAD_MIN}, {LEAD_MAX}]"
)


def insertions_to_motor_mm(ins_m):
    return START_MOTOR_MM + AXIS_SIGN * (np.asarray(ins_m) - START_INS_M) * 1000.0


def project_insertions(q_des):
    """Closest feasible insertion vector to q_des that satisfies all tube and carriage bounds."""
    res = minimize(
        lambda q: 0.5 * np.sum((q - q_des) ** 2),
        np.clip(q_des, INS_MIN, INS_MAX),
        jac=lambda q: q - q_des,
        bounds=Bounds(INS_MIN, INS_MAX),
        constraints=[LinearConstraint(D, LEAD_MIN, LEAD_MAX)],
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 100},
    )
    q_proj = np.clip(res.x, INS_MIN, INS_MAX)
    # Enforce strict telescoping and carriage gap limits against solver tolerance drift
    q_proj[1] = np.clip(q_proj[1], q_proj[0] + TIP_MIN_GAP, min(INS_MAX[1], q_proj[0] + GAP_OM))
    q_proj[2] = np.clip(q_proj[2], q_proj[1] + TIP_MIN_GAP, min(INS_MAX[2], q_proj[1] + GAP_MI))
    return np.clip(q_proj, INS_MIN, INS_MAX)


# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------
class OctopusController:
    """
    Sends ABSOLUTE G1 moves to a Marlin board with 6 axes, with 'ok' flow control so the
    firmware queue never builds up.
    """

    def __init__(self, q0, port=None, baud=115200, live=False):
        self.live = live
        self.q = np.array(q0, dtype=float)
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

        time.sleep(2.5)
        self.ser.reset_input_buffer()
        self._command_blocking("G90")
        self._command_blocking("G92 " + self._rot_words(self.q))

        m92_cmd = "M92 X6760 Y6760 Z6760"
        if ROT_STEPS_PER_DEG is not None:
            m92_cmd += (
                f" A{ROT_STEPS_PER_DEG[0]:.4f}"
                f" B{ROT_STEPS_PER_DEG[1]:.4f}"
                f" C{ROT_STEPS_PER_DEG[2]:.4f}"
            )
        self._command_blocking(m92_cmd)
        self._command_blocking("M914 X115 Y120 Z120")

        lin_mms = FEEDRATE / 60.0
        rot_degs = ROT_FEEDRATE / 60.0
        self._command_blocking(
            f"M203 X{lin_mms:.2f} Y{lin_mms:.2f} Z{lin_mms:.2f} "
            f"A{rot_degs:.2f} B{rot_degs:.2f} C{rot_degs:.2f}"
        )

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

    def _compute_feedrate(self, q_target):
        """
        Marlin applies F to the 6D Euclidean distance sqrt(dX^2 + dY^2 + dZ^2 + dA^2 + dB^2 + dC^2).
        Compute F so linear motion respects FEEDRATE (mm/min) and rotational motion respects ROT_FEEDRATE (deg/min).
        """
        cur_mm = insertions_to_motor_mm(self.q[:3])
        tgt_mm = insertions_to_motor_mm(q_target[:3])
        d_lin = np.linalg.norm(tgt_mm - cur_mm)

        cur_deg = np.degrees(self.q[3:])
        tgt_deg = np.degrees(q_target[3:])
        d_rot = np.linalg.norm(tgt_deg - cur_deg)

        d_total = np.hypot(d_lin, d_rot)
        if d_total < 1e-6:
            return FEEDRATE

        t_min = max(d_lin / FEEDRATE, d_rot / ROT_FEEDRATE)
        return np.clip(d_total / t_min, FEEDRATE, ROT_FEEDRATE)

    def _poll(self):
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

    def move_to(self, q_target):
        """Send an absolute pose with a hard motor-space collision guard."""
        motor_mm = insertions_to_motor_mm(q_target[:3])
        margin_mm = MARGIN * 1000.0
        if (motor_mm[0] - motor_mm[1] < margin_mm - 0.1) or (motor_mm[1] - motor_mm[2] < margin_mm - 0.1):
            print(f"[SAFETY] Blocked carriage collision in motor space (mm): {motor_mm}")
            return False

        eff_feed = self._compute_feedrate(q_target)
        cmd = f"G1 {self._axis_words(q_target)} F{eff_feed:.1f}"
        if self.live:
            self._poll()
            if self.in_flight >= MAX_IN_FLIGHT:
                return False
            self.ser.write((cmd + "\n").encode())
            self.in_flight += 1
        else:
            print(f"[VIRTUAL] {cmd}")
        self.q = np.array(q_target, dtype=float)
        return True

    def shutdown(self, emergency=False, reset_angles=True, target_rotations=(0.0175, 0.0, 0.0)):
        if not self.live or self.ser is None:
            if reset_angles:
                rot_cmd = " ".join(f"{a}{np.degrees(v):.3f}" for a, v in zip(GCODE_AXES[3:], target_rotations))
                print(f"[VIRTUAL] Resetting tube angles: G1 {rot_cmd} F{ROT_FEEDRATE:.1f}")
            return
        try:
            if emergency:
                self.ser.write(b"M410\n")
            else:
                self._command_blocking("M400", timeout=10.0)
                if reset_angles:
                    print("Resetting tube angles (A, B, C)...")
                    rot_cmd = " ".join(f"{a}{np.degrees(v):.3f}" for a, v in zip(GCODE_AXES[3:], target_rotations))
                    self._command_blocking("G90")
                    self._command_blocking(f"G1 {rot_cmd} F{ROT_FEEDRATE:.1f}", timeout=15.0)
                    self._command_blocking("M400", timeout=30.0)
                    print("Tube angles reset complete.")
        finally:
            self.ser.close()


def solve_robot(torsion_initial_guess=None):
    backbone = solve_forward_kinematics_v2(
        robot,
        torsion_initial_guess=torsion_initial_guess,
    )
    samples = sample_backbone(
        backbone,
        ds=sampling_ds,
    )
    return backbone, samples


def weighted_resolved_rate_step(dx, p0, guess):
    """
    Computes a joint-weighted damped least-squares step dq so rotational joints (rad)
    are not drowned out by insertion joints (m).
    """
    n = len(robot.tubes)
    ins_0 = robot.state.insertions.copy()
    rot_0 = robot.state.rotations.copy()

    J = np.zeros((3, 2 * n), dtype=float)
    eps_ins = 1e-4  # 0.1 mm finite difference
    eps_rot = 1e-3  # ~0.057 deg finite difference

    try:
        for i in range(n):
            robot.state.insertions[:] = ins_0
            robot.state.rotations[:] = rot_0
            robot.state.insertions[i] += eps_ins
            _, s_pert = solve_robot(torsion_initial_guess=guess)
            J[:, i] = (s_pert.position[-1] - p0) / eps_ins

        for i in range(n):
            robot.state.insertions[:] = ins_0
            robot.state.rotations[:] = rot_0
            robot.state.rotations[i] += eps_rot
            _, s_pert = solve_robot(torsion_initial_guess=guess)
            J[:, n + i] = (s_pert.position[-1] - p0) / eps_rot
    finally:
        robot.state.insertions[:] = ins_0
        robot.state.rotations[:] = rot_0

    # Weight matrix W scales rotational columns (m/rad) to match insertion columns (m/m)
    w = np.concatenate([np.ones(n), np.full(n, ROT_WEIGHT)])
    J_w = J * w[np.newaxis, :]
    JJT = J_w @ J_w.T + (IK_DAMPING ** 2) * np.eye(3)
    dq = w * (J_w.T @ np.linalg.solve(JJT, dx))
    return dq


# ----------------------------------------
# Capture inputs
# ----------------------------------------
def captureInputs():
    pygame.init()
    pygame.display.set_mode((300, 300))
    pygame.display.set_caption("CTR Teleop (WASDQE: Tip | UJ/IK/OL: Rotate A/B/C)")
    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            dt = clock.tick(20) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            k = pygame.key.get_pressed()

            # Cartesian tip motion: WASD (X/Y), QE (Z)
            dx = np.array([
                int(k[pygame.K_d]) - int(k[pygame.K_a]),
                int(k[pygame.K_w]) - int(k[pygame.K_s]),
                int(k[pygame.K_e]) - int(k[pygame.K_q]),
            ], dtype=float)

            # Direct tube rotation override/test keys: U/J (Outer A), I/K (Middle B), O/L (Inner C)
            drot = np.array([
                int(k[pygame.K_u]) - int(k[pygame.K_j]),
                int(k[pygame.K_i]) - int(k[pygame.K_k]),
                int(k[pygame.K_o]) - int(k[pygame.K_l]),
            ], dtype=float)

            if dx.any() or drot.any():
                updateRobot(dx * cartesian_speed * dt, drot * direct_rot_speed * dt)
    except KeyboardInterrupt:
        print("\nCtrl+C detected, exiting teleop...")
    finally:
        pygame.quit()


def updateRobot(dx, drot=None):
    global backbone, samples
    n = len(robot.tubes)
    guess = backbone.torsion_solution.base_theta_dot
    p0 = samples.position[-1]

    dq = np.zeros(2 * n, dtype=float)
    if np.any(dx):
        dq += weighted_resolved_rate_step(dx, p0, guess)
    if drot is not None and np.any(drot):
        dq[n:] += drot

    # Clamp per-step increments instead of dropping the whole step
    max_ins_step = 1.5e-3
    max_rot_step = np.radians(8.0)
    ins_scale = min(1.0, max_ins_step / (np.max(np.abs(dq[:n])) + 1e-12))
    rot_scale = min(1.0, max_rot_step / (np.max(np.abs(dq[n:])) + 1e-12))
    dq *= min(ins_scale, rot_scale)

    ins_old = robot.state.insertions.copy()
    rot_old = robot.state.rotations.copy()

    q_new = np.concatenate([ins_old, rot_old]) + dq
    q_new[:n] = project_insertions(q_new[:n])
    q_new[n:] = np.clip(q_new[n:], Q_MIN[n:], Q_MAX[n:])

    robot.state.insertions[:] = q_new[:n]
    robot.state.rotations[:] = q_new[n:]

    try:
        robot.validate_configuration()
        backbone_new, samples_new = solve_robot(torsion_initial_guess=guess)
    except Exception as e:
        robot.state.insertions[:] = ins_old
        robot.state.rotations[:] = rot_old
        print("Rejected, model says invalid:", e)
        return

    # Only commit simulation state if the hardware queue accepted the command
    if moveRobot(q_new):
        backbone, samples = backbone_new, samples_new
        print(
            "Tip (mm):", np.round(samples.position[-1] * 1000.0, 2),
            "| Ins (mm):", np.round(q_new[:n] * 1000.0, 2),
            "| Rot (deg):", np.round(np.degrees(q_new[n:]), 2),
        )
    else:
        robot.state.insertions[:] = ins_old
        robot.state.rotations[:] = rot_old


def moveRobot(q_target):
    global lastMotorPos
    if np.any(np.abs(q_target - lastMotorPos) > 1e-9):
        if hw.move_to(q_target):
            lastMotorPos = q_target.copy()
            return True
        else:
            print("Hardware busy or blocked, move skipped")
            return False
    return True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="talk to real hardware (default: test mode)")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    return ap.parse_args()


# Homing Functions:
def home_z_inverted(controller, step_mm=-1.0, feedrate=150, max_travel_mm=250.0):
    if not controller.live or controller.ser is None:
        print("[VIRTUAL] Custom inverted Z homing executed.")
        return True

    print("Starting custom Z homing (looking for 'open' state)...")
    controller._command_blocking("M400")
    controller._command_blocking("M211 S0")
    controller._command_blocking("G91")

    moved = 0.0
    homed = False

    while abs(moved) < max_travel_mm:
        controller.ser.write(b"M119\n")

        z_is_open = False
        response_complete = False

        while not response_complete:
            data = controller.ser.read(controller.ser.in_waiting or 1)
            if data:
                controller._rx += data

            while b"\n" in controller._rx:
                line_bytes, controller._rx = controller._rx.split(b"\n", 1)
                text = line_bytes.decode(errors="replace").strip()

                if text.startswith("z_min: open") or text.startswith("z_max: open"):
                    z_is_open = True
                elif text.lower().startswith("ok"):
                    response_complete = True
                    break

        if z_is_open:
            print(f"Limit switch reached! Homing complete after {moved:.1f} mm.")
            homed = True
            break

        controller._command_blocking(f"G1 Z{step_mm} F{feedrate}")
        controller._command_blocking("M400")
        moved += step_mm

    if not homed:
        print("WARNING: Reached max travel limit without hitting the switch!")

    if homed:
        controller._command_blocking("G92 Z0")

    controller._command_blocking("G90")
    controller._command_blocking("M211 S1")
    return homed


def home_x_normal(controller, home_coordinate_mm=0.0, timeout=30.0):
    if not controller.live or controller.ser is None:
        print("[VIRTUAL] Homed X axis normally.")
        return True

    print("Homing X axis (Standard G28)...")
    try:
        controller._command_blocking("M400")
        controller._command_blocking("M92 Z0.00001")
        controller._command_blocking("G28 X", timeout=timeout)
        controller._command_blocking("M211 S1")
        controller._command_blocking("M92 Z6760")
        controller._command_blocking("G92 Z0")
        print("X axis successfully homed!")
        return True
    except TimeoutError:
        print(f"ERROR: Homing X axis timed out after {timeout} seconds.")
        if controller.ser:
            controller.ser.reset_input_buffer()
            controller.in_flight = 0
        return False
    except Exception as e:
        print(f"ERROR during X homing: {e}")
        return False


def home_y_normal(controller, home_coordinate_mm=0.0, timeout=30.0):
    if not controller.live or controller.ser is None:
        print("[VIRTUAL] Homed Y axis normally.")
        return True

    print("Homing Y axis (Standard G28)...")
    try:
        controller._command_blocking("M400")
        controller._command_blocking("M92 Z0.00001")
        controller._command_blocking("G28 Y", timeout=timeout)
        controller._command_blocking("M211 S1")
        controller._command_blocking("M92 Z6760")
        controller._command_blocking("G92 Z0")
        print("Y axis successfully homed!")
        return True
    except TimeoutError:
        print(f"ERROR: Homing Y axis timed out after {timeout} seconds.")
        if controller.ser:
            controller.ser.reset_input_buffer()
            controller.in_flight = 0
        return False
    except Exception as e:
        print(f"ERROR during Y homing: {e}")
        return False


if __name__ == '__main__':
    ap = parse_args()

    backbone, samples = solve_robot()
    hw = OctopusController(
        q0=np.concatenate([state.insertions, state.rotations]),
        port=ap.port, baud=ap.baud, live=ap.live,
    )

    homed = (
        home_z_inverted(hw, step_mm=-2.0, feedrate=FEEDRATE, max_travel_mm=300)
        and home_y_normal(hw, timeout=300)
        and home_x_normal(hw, timeout=300)
    )

    if not homed:
        hw.shutdown(emergency=True)
        raise SystemExit("Homing failed, not starting teleop.")
    else:
        if hw.live:
            # Move front-to-back sequentially with M400 barriers so stacked carriages never collide
            hw._command_blocking(f"G1 X{START_MOTOR_MM[0]:.3f} F{FEEDRATE}", timeout=300)
            hw._command_blocking("M400", timeout=300)
            hw._command_blocking(f"G1 Y{START_MOTOR_MM[1]:.3f} F{FEEDRATE}", timeout=300)
            hw._command_blocking("M400", timeout=300)
            hw._command_blocking(f"G1 Z{START_MOTOR_MM[2]:.3f} F{FEEDRATE}", timeout=300)
            hw._command_blocking("M400", timeout=300)

        hw.move_to(lastMotorPos)

    if hw.live:
        hw._command_blocking("M400", timeout=300)

    try:
        captureInputs()
    finally:
        hw.shutdown()