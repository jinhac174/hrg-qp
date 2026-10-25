"""Robotiq 2F-85 gripper: geometry and ranges shared by the sim, the hardware
driver, the visualizer and the grasp planners.

The controllers talk to the gripper in metres and newtons (``gripper_cmd_t``:
move / grasp to a width, with a speed and a force), as they did for the
Franka Hand. This module converts between that and the two other ways the
2F-85 is described:

* the **driver angle** (rad, 0 = open .. ~0.785 = closed): the one joint of
  the URDF / MuJoCo model (``robotiq_driver_joint``); the rest of the linkage
  follows it;
* the **register position** (0 = open .. 255 = closed) the real gripper takes
  and reports over Modbus (see ``robot_platform/hardware/franka_fr3/robotiq.py``).

Width = distance between the two pad faces. The pads move on an arc (four-bar
linkage), so their height along the approach axis changes with the width:
``pad_z``. Both tables come from a sweep of ``fr3_robotiq_robot.xml``.
"""

from __future__ import annotations

import numpy as np

MAX_WIDTH = 0.085  # fully open (m); the model opens to 85.2 mm
FORCE_RANGE = (20.0, 235.0)  # grip force (N), datasheet
SPEED_RANGE = (0.02, 0.15)  # finger closing speed (m/s), datasheet
MASS = 0.925  # kg incl. coupling (datasheet); enter it as the load in Desk

# TCP (URDF / MuJoCo frame ``fr3_gripper_tcp``) along the approach axis z of
# ``fr3_gripper`` (= the flange frame: the DROID mount, not turned about z). The pad centres pass through it at a ~55 mm opening.
TCP_Z = 0.15

# driver angle (rad) -> pad-face width (m) and pad-face height (m, in
# fr3_gripper). Sweep of the MuJoCo model; the last row is extrapolated to
# touching pads.
_ANGLE = np.array([0.0, 0.0526, 0.1025, 0.1525, 0.2024, 0.2524, 0.3023, 0.3523, 0.4022,
                   0.4522, 0.5021, 0.5521, 0.602, 0.652, 0.7019, 0.7519, 0.785])
_WIDTH = np.array([0.0853, 0.0806, 0.0758, 0.0709, 0.0658, 0.0606, 0.0553, 0.0499, 0.0443,
                   0.0387, 0.033, 0.0272, 0.0214, 0.0155, 0.0097, 0.0038, 0.0])
_PAD_Z = np.array([0.1416, 0.1433, 0.145, 0.1465, 0.1479, 0.1492, 0.1504, 0.1514, 0.1523,
                   0.1531, 0.1537, 0.1542, 0.1545, 0.1547, 0.1547, 0.1546, 0.1546])
CLOSED_ANGLE = float(_ANGLE[-1])


def width_from_angle(angle):
    """Driver angle (rad) -> pad-face width (m)."""
    return np.interp(angle, _ANGLE, _WIDTH)


def angle_from_width(width):
    """Pad-face width (m) -> driver angle (rad)."""
    return np.interp(width, _WIDTH[::-1], _ANGLE[::-1])


def pad_z(width):
    """Height of the pad-face centres along the approach axis (m, in
    ``fr3_gripper``) at a given width; ``pad_z(w) - TCP_Z`` is how far they
    sit beyond the TCP."""
    return np.interp(width, _WIDTH[::-1], _PAD_Z[::-1])


# Sim only: squeeze force at the pads (N) = slope * actuator force + offset,
# measured on boxes 20 / 40 / 60 / 80 mm wide. The offset is the linkage's
# springs, which squeeze a little even with the motor off.
_SQ_WIDTH = np.array([0.02, 0.04, 0.06, 0.08])
_SQ_SLOPE = np.array([8.61, 8.86, 9.43, 10.61])
_SQ_OFFSET = np.array([1.54, 1.78, 2.13, 2.71])


def sim_min_squeeze(width) -> float:
    """Sim only: the squeeze (N) the model's springs give with the motor off."""
    return float(np.interp(width, _SQ_WIDTH, _SQ_OFFSET))


def drive_for_squeeze(squeeze, width):
    """Sim only: MuJoCo gripper actuator force for a squeeze force (N) at
    the pads at this width. Squeezes below the spring offset (~2 N) give 0."""
    k = np.interp(width, _SQ_WIDTH, _SQ_SLOPE)
    b = np.interp(width, _SQ_WIDTH, _SQ_OFFSET)
    return max(float(squeeze) - float(b), 0.0) / float(k)


# --- MuJoCo model (fr3_robotiq_robot.xml) helpers, shared by SimFrankaFR3 and
# run_grasp_quality.py --------------------------------------------------------
DRIVER_JOINT = "robotiq_right_driver_joint"
PAD_SITES = ("robotiq_left_pad_face", "robotiq_right_pad_face")
# Linkage joint angle = multiplier * driver angle with nothing in the hand
# (same as the <mimic> multipliers in urdf/fr3_robotiq.urdf; couplers ~0).
LINKAGE = {
    f"robotiq_{s}_{j}_joint": k
    for s in ("right", "left")
    for j, k in (("driver", 1.0), ("coupler", 0.0), ("spring_link", 1.0), ("follower", -0.963))
}


def mj_width(model, data) -> float:
    """Opening = distance between the two pad faces (m)."""
    a, b = (data.site(n).xpos for n in PAD_SITES)
    return float(np.linalg.norm(a - b))


def mj_set_width(model, data, width: float) -> float:
    """Put the linkage at an opening (on reset); returns the driver angle."""
    angle = float(angle_from_width(width))
    for name, k in LINKAGE.items():
        j = model.joint(name)
        data.qpos[j.qposadr[0]] = k * angle
        data.qvel[j.dofadr[0]] = 0.0
    return angle


def mj_actuate(model, data, actuator_id: int, width_ref: float, squeeze_cap: float, width: float):
    """Width set-point (m) and squeeze-force cap (N) -> the gripper actuator
    (a position servo on the driver angle, force-limited)."""
    data.ctrl[actuator_id] = angle_from_width(width_ref)
    drive = drive_for_squeeze(squeeze_cap, width)
    model.actuator_forcerange[actuator_id] = (-drive, drive)
