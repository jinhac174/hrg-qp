"""Robotiq 2F-85 on the FR3: fingertips, collision spheres and boxes.

Frames
------
TCP frame (URDF ``fr3_gripper_tcp``): ``+z`` approach (from the gripper toward
the object), ``+y`` closing axis, origin midway between the pad faces.

Fingertips (the paper's ``x_i(q)``)
-----------------------------------
The two pad-face centres. They depend on the 7 arm joints *and* on the
opening width ``w`` (pad face to pad face), the gripper's one joint::

    x_1 = (0, -w/2, pad_z(w) - TCP_Z)     left pad  (robotiq_left_pad_face)
    x_2 = (0, +w/2, pad_z(w) - TCP_Z)     right pad (robotiq_right_pad_face)

in the TCP frame (the pads ride on an arc, ``pad_z`` is the 2F-85 linkage,
``fr3.robot_platform.gripper.robotiq_2f85``). So ``x_2 - x_1 = w y_tcp``. With
the QP's decision variable ``z = (qdot (7), wdot)`` every point attached to
the hand has a 3x8 Jacobian: the rigid-body part from the TCP Jacobian, plus
``R_tcp d(point)/dw`` in the last column.

Collision geometry
------------------
* ``hand_spheres``: the paper's point set ``X(q)`` approximating the gripper
  for the QP's ``d(X(q), O) >= eps_O`` (eq. 11). Finger spheres move with the
  width, the linkage spheres too, the palm spheres are fixed.
* ``hand_boxes`` / ``HandCollisionChecker``: the gripper as boxes, checked
  against an object mesh with FCL when the fingertip targets are generated
  (what the simulation collides with).
"""

from __future__ import annotations

import fcl
import numpy as np
import trimesh

from fr3.robot_platform.gripper import robotiq_2f85 as rq
from fr3.utils.lie.so3 import skew

N_ARM = 7
N_Z = N_ARM + 1  # QP decision variables: 7 joint velocities + the width rate
MAX_WIDTH = rq.MAX_WIDTH  # fully open (m)
MAX_WIDTH_RATE = rq.SPEED_RANGE[1]  # m/s, the 2F-85's fastest finger speed
TCP_Z = rq.TCP_Z  # fr3_gripper -> fr3_gripper_tcp along z (m)
FINGER_SIGN = np.array([-1.0, 1.0])  # y side of fingertip 1 (left) / 2 (right)
# Axis of each pad's contact surface (its inward normal), TCP frame: the left
# pad (at -y) presses toward +y, the right pad toward -y (eq. 6).
PAD_CONTACT_AXES = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])

# Box model, measured from the MuJoCo model. One fingertip (follower link +
# pad): inner face at |y| = w/2, from 38 mm below to 19 mm beyond the pad centre.
FINGER_EXTENTS = np.array([0.022, 0.032, 0.057])  # x, y (thickness), z (length)
FINGER_CENTRE_DZ = -0.0095  # box centre relative to the pad centre, along z
# Gripper body + the linkage roots between the fingers, up to 110 mm above the
# flange (frame fr3_gripper; TCP frame = minus TCP_Z along z).
PALM_EXTENTS = np.array([0.075, 0.085, 0.110])
PALM_CENTRE = np.array([0.0, 0.0, 0.055 - TCP_Z])
# Outer linkage on each side (driver, coupler, spring link): from the pad face
# line out to |y| = 75 mm, 56-125 mm above the flange.
LINKAGE_X = 0.039
LINKAGE_Y_OUT = 0.075
LINKAGE_Z = (0.056, 0.125)


# ---------------------------------------------------------------- fingertips
def pad_dz(w) -> float:
    """Pad-face centre beyond the TCP along z (m) at width ``w``."""
    return float(rq.pad_z(w)) - TCP_Z


def pad_dz_slope(w, h: float = 1e-4) -> float:
    """d pad_dz / dw (central difference of the tabulated linkage arc)."""
    lo, hi = max(w - h, 0.0), min(w + h, MAX_WIDTH)
    return (float(rq.pad_z(hi)) - float(rq.pad_z(lo))) / max(hi - lo, 1e-9)


def fingertips_local(w: float):
    """Fingertip positions in the TCP frame and their derivative wrt ``w``:
    ``(P (2, 3), dP_dw (2, 3))``."""
    dz, s = pad_dz(w), pad_dz_slope(w)
    P = np.array([[0.0, 0.5 * sgn * w, dz] for sgn in FINGER_SIGN])
    dP = np.array([[0.0, 0.5 * sgn, s] for sgn in FINGER_SIGN])
    return P, dP


def to_world(T_tcp: np.ndarray, P_local: np.ndarray) -> np.ndarray:
    return P_local @ T_tcp[:3, :3].T + T_tcp[:3, 3]


def point_jacobians(T_tcp: np.ndarray, J_tcp: np.ndarray, P_local: np.ndarray, dP_dw: np.ndarray):
    """3x8 Jacobians of hand points.

    ``J_tcp`` is the 6x7 TCP Jacobian stacked (angular; linear), world
    aligned (``hrg.robot.RobotModel.jacobian``). ``P_local`` / ``dP_dw`` are
    ``(K, 3)`` in the TCP frame. Returns ``(K, 3, 8)``.
    """
    R = T_tcp[:3, :3]
    J_R, J_p = J_tcp[:3, :N_ARM], J_tcp[3:, :N_ARM]
    out = np.zeros((len(P_local), 3, N_Z))
    for k, (r, dr) in enumerate(zip(P_local, dP_dw)):
        out[k, :, :N_ARM] = J_p - skew(R @ r) @ J_R
        out[k, :, N_ARM] = R @ dr
    return out


def fingertip_state(T_tcp: np.ndarray, J_tcp: np.ndarray, w: float):
    """World fingertip positions ``(2, 3)`` and Jacobians ``(2, 3, 8)``."""
    P, dP = fingertips_local(w)
    return to_world(T_tcp, P), point_jacobians(T_tcp, J_tcp, P, dP)


def angular_jacobian(J_tcp: np.ndarray) -> np.ndarray:
    """3x8 angular Jacobian of the hand (both fingertips share it: the 2F-85
    fingers translate along y, they do not rotate the pads)."""
    J = np.zeros((3, N_Z))
    J[:, :N_ARM] = J_tcp[:3, :N_ARM]
    return J


# ---------------------------------------------------------------- collision spheres
# Finger: two columns of r = 11 mm spheres cover the 32 mm thickness, four rows
# the 57 mm length; the inner surface of the first column is exactly the pad
# face, so a pre-grasp target 1.5 cm off the object leaves the spheres outside
# a 1 cm margin.
FINGER_R = 0.011
_FINGER_COLS = (FINGER_R, FINGER_EXTENTS[1] - FINGER_R)  # sphere centre beyond the pad face
_FINGER_ROWS = np.linspace(-0.5 * FINGER_EXTENTS[2] + FINGER_R, 0.5 * FINGER_EXTENTS[2] - FINGER_R, 4)
# Outer linkages above each finger: their inner edge follows the pad face
# (as ``hand_boxes``), so they move with the width.
_LINK_R = 0.02
_LINK_Z = -0.045  # TCP frame
# Palm: fixed (centre in the TCP frame, radius).
_PALM_SPHERES = [
    ((-0.02, -0.025, -0.07), 0.025),
    ((-0.02, 0.025, -0.07), 0.025),
    ((0.02, -0.025, -0.07), 0.025),
    ((0.02, 0.025, -0.07), 0.025),
]


def hand_spheres_local(w: float):
    """Collision spheres of the hand at width ``w``, TCP frame.

    Returns ``(P (K, 3), dP_dw (K, 3), radius (K,), is_finger (K,))``:
    16 finger spheres, 2 linkage spheres, 4 palm spheres.
    """
    tip_z = pad_dz(w) + FINGER_CENTRE_DZ  # finger box centre along z
    s = pad_dz_slope(w)
    P, dP, r, fin = [], [], [], []
    for sgn in FINGER_SIGN:
        for col in _FINGER_COLS:
            for row in _FINGER_ROWS:
                P.append((0.0, sgn * (0.5 * w + col), tip_z + row))
                dP.append((0.0, 0.5 * sgn, s))
                r.append(FINGER_R)
                fin.append(True)
    for sgn in FINGER_SIGN:
        P.append((0.0, sgn * (0.5 * w + _LINK_R), _LINK_Z))
        dP.append((0.0, 0.5 * sgn, 0.0))
        r.append(_LINK_R)
        fin.append(False)
    for c, rad in _PALM_SPHERES:
        P.append(c)
        dP.append((0.0, 0.0, 0.0))
        r.append(rad)
        fin.append(False)
    return np.asarray(P), np.asarray(dP), np.asarray(r), np.asarray(fin)


def hand_spheres(T_tcp: np.ndarray, J_tcp: np.ndarray, w: float):
    """World sphere centres ``(K, 3)``, Jacobians ``(K, 3, 8)``, radii ``(K,)``."""
    P, dP, r, _ = hand_spheres_local(w)
    return to_world(T_tcp, P), point_jacobians(T_tcp, J_tcp, P, dP), r


# ---------------------------------------------------------------- collision boxes
def hand_boxes(width: float):
    """Gripper body, two fingertips and two outer linkages as
    ``[(extents, centre)]`` in the TCP frame, opened to ``width``."""
    half = 0.5 * width + 0.5 * FINGER_EXTENTS[1]
    tip_z = float(rq.pad_z(width)) + FINGER_CENTRE_DZ - TCP_Z
    lo = max(0.5 * width, 0.008)
    link_ext = np.array([LINKAGE_X, LINKAGE_Y_OUT - lo, LINKAGE_Z[1] - LINKAGE_Z[0]])
    link_y = 0.5 * (lo + LINKAGE_Y_OUT)
    link_z = 0.5 * (LINKAGE_Z[0] + LINKAGE_Z[1]) - TCP_Z
    return [
        (PALM_EXTENTS, PALM_CENTRE),
        (FINGER_EXTENTS, np.array([0.0, half, tip_z])),
        (FINGER_EXTENTS, np.array([0.0, -half, tip_z])),
        (link_ext, np.array([0.0, link_y, link_z])),
        (link_ext, np.array([0.0, -link_y, link_z])),
    ]


class HandCollisionChecker:
    """FCL check of the open gripper (``hand_boxes``) against an object mesh."""

    def __init__(self, mesh: trimesh.Trimesh):
        bvh = fcl.BVHModel()
        bvh.beginModel(len(mesh.vertices), len(mesh.faces))
        bvh.addSubModel(mesh.vertices, mesh.faces)
        bvh.endModel()
        self._obj = fcl.CollisionObject(bvh, fcl.Transform())

    def collides(self, T_obj_tcp: np.ndarray, width: float, margin: float = 0.0) -> bool:
        """True if the gripper (opened to ``width``) comes within ``margin``
        of the object, with the TCP at ``T_obj_tcp`` (object frame)."""
        R, p = T_obj_tcp[:3, :3], T_obj_tcp[:3, 3]
        for ext, ctr in hand_boxes(width):
            box = fcl.CollisionObject(fcl.Box(*(ext + 2.0 * margin)), fcl.Transform(R, R @ ctr + p))
            if fcl.collide(box, self._obj, fcl.CollisionRequest(), fcl.CollisionResult()):
                return True
        return False
