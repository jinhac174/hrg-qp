"""FR3 kinematics, dynamics and collision distances for the planner.

* Pinocchio model of the FR3 with the 2F-85 rigidly attached (the gripper's
  width is not a URDF joint here; ``hrg.hand`` adds it): FK, Jacobians
  (stacked angular; linear, world aligned), joint limits, ``g(q)``,
  ``C(q, qdot) qdot``.
* ``Gamma(q)`` of eq. (11): distances between the robot's collision primitives
  and the environment (walls, floor) and between non-adjacent links
  (self-collision), with FCL, linearised for the QP.

The collision primitives are the ones the sim env / ``run_hardware.py``
publish on the ``robot_col_info`` / ``static_col_info`` channels (link-local
and world-fixed boxes / capsules); ``set_collision_geometry`` takes them.

A capsule is a segment (its core) grown by its radius: FCL measures the core,
and the radius is subtracted. GJK on the polytopes (segments, boxes) gives
exact distances and nearest points, so each row ``n . J(p)`` is the exact
gradient of the distance (``p`` the nearest point, ``n`` the unit normal; the
envelope theorem). On the capsule's curved surface FCL's GJK stops at its
tolerance with the nearest point off by millimetres, which tilts the rows by
~1 %; and the core's normal stays defined while a capsule's surface is inside
a wall (recovery), where FCL reports penetration without a direction.
"""

from __future__ import annotations

import fcl
import numpy as np
import pinocchio as pin

from fr3.utils.fcl.getter import get_fcl_geom
from fr3.utils.lie.kinematics import get_point_Jacobian
from fr3.utils.pinocchio.getter import get_fk_link_pose, get_link_Jacobian
from hrg.hand import N_ARM, N_Z
from hrg.paths import resolve
from hrg.qp import DistanceRows

TCP_LINK = "fr3_gripper_tcp"


def _pad(A7: np.ndarray) -> np.ndarray:
    """Arm-joint rows -> rows over ``z = (qdot, wdot)`` (zero width column)."""
    A = np.zeros((A7.shape[0], N_Z))
    A[:, :N_ARM] = A7[:, :N_ARM]
    return A


def fcl_geoms(col_info: dict) -> dict:
    """``{name: {fcl_geom, offset (4x4), radius}}`` from a ``ColInfoData``
    dict; a capsule becomes its core segment with its ``radius`` kept apart."""
    out = {}
    for name, g in col_info.items():
        size = np.asarray(g["size"], dtype=float)
        if g["type"] == "capsule":  # size = (radius, length of the segment)
            geom, radius = fcl.Capsule(0.0, float(size[1])), float(size[0])
        else:
            geom, radius = get_fcl_geom(g["type"], size), 0.0
        out[name] = {"fcl_geom": geom, "offset": np.asarray(g["offset"], dtype=float).reshape(4, 4), "radius": radius}
    return out


class RobotModel:
    """One Pinocchio ``Data`` per user thread: ``data()`` makes a new one."""

    def __init__(self, urdf_path: str, joint_velocity_limit, env_pairs=(), self_pairs=(),
                 env_margin: float = 0.02, self_margin: float = 0.02):
        self.model = pin.buildModelFromUrdf(resolve(urdf_path))
        if self.model.nq != N_ARM:
            raise ValueError(f"{urdf_path}: expected {N_ARM} joints, got {self.model.nq}")
        self.q_lower = self.model.lowerPositionLimit.copy()
        self.q_upper = self.model.upperPositionLimit.copy()
        self.qd_limit = np.asarray(joint_velocity_limit, dtype=float)
        self.env_pairs = [tuple(p) for p in env_pairs]
        self.self_pairs = [tuple(p) for p in self_pairs]
        self.env_margin = float(env_margin)
        self.self_margin = float(self_margin)
        self._robot_geoms: dict = {}
        self._static: dict = {}

    def data(self):
        return self.model.createData()

    # ------------------------------------------------------------ kinematics
    def pose(self, q, data, link: str = TCP_LINK) -> np.ndarray:
        return get_fk_link_pose(q, self.model, data, link)

    def jacobian(self, q, data, link: str = TCP_LINK) -> np.ndarray:
        """6x7, rows (angular; linear), world aligned, at the link origin."""
        return get_link_Jacobian(q, self.model, data, link)

    # ------------------------------------------------------------ dynamics
    def gravity_coriolis(self, q, qd, data) -> np.ndarray:
        """``g(q) + C(q, qdot) qdot``."""
        g = pin.computeGeneralizedGravity(self.model, data, q)
        pin.computeCoriolisMatrix(self.model, data, q, qd)
        return g + data.C @ qd

    # ------------------------------------------------------------ limits
    def velocity_bounds(self, q, horizon: float):
        """Joint-velocity box: velocity limits and the position limits reached
        within ``horizon`` (``q`` clipped into the range first, so a slight
        overshoot of a limit cannot make the box empty)."""
        H = max(float(horizon), 1e-4)
        qc = np.clip(q, self.q_lower, self.q_upper)
        lo = np.maximum(-self.qd_limit, (self.q_lower - qc) / H)
        hi = np.minimum(self.qd_limit, (self.q_upper - qc) / H)
        bad = hi < lo
        mid = 0.5 * (lo + hi)
        return np.where(bad, mid, lo), np.where(bad, mid, hi)

    # ------------------------------------------------------------ collision
    def set_collision_geometry(self, robot_col_info: dict, static_col_info: dict) -> None:
        self._robot_geoms = fcl_geoms(robot_col_info)
        self._static = {
            name: {"obj": fcl.CollisionObject(g["fcl_geom"], fcl.Transform(g["offset"][:3, :3], g["offset"][:3, 3])),
                   "radius": g["radius"]}
            for name, g in fcl_geoms(static_col_info).items()
        }

    @property
    def has_collision_geometry(self) -> bool:
        return bool(self._robot_geoms)

    def _link(self, q, data, name: str) -> dict:
        T = get_fk_link_pose(q, self.model, data, name)
        g = self._robot_geoms[name]
        tf = T @ g["offset"]
        return {"obj": fcl.CollisionObject(g["fcl_geom"], fcl.Transform(tf[:3, :3], tf[:3, 3])),
                "radius": g["radius"], "T": T, "J": get_link_Jacobian(q, self.model, data, name)}

    @staticmethod
    def _point_jac(link: dict, p: np.ndarray) -> np.ndarray:
        """3x7 Jacobian of the world point ``p`` attached to ``link``."""
        R, o, J = link["T"][:3, :3], link["T"][:3, 3], link["J"]
        return get_point_Jacobian(R, J[3:], J[:3], R.T @ (p - o))

    def gamma_rows(self, q, data, rows: DistanceRows) -> None:
        """Add ``Gamma(q) >= eps_Gamma`` (eq. 11): robot vs environment, then
        robot vs itself (both links move: ``n . (J_a(p_a) - J_b(p_b))``)."""
        links: dict = {}

        def link(name):
            if name not in links:
                links[name] = self._link(q, data, name)
            return links[name]

        for a, obs in self.env_pairs:
            d, n, pa, _ = _distance(link(a), self._static[obs])
            rows.add([d], _pad((n @ self._point_jac(link(a), pa)).reshape(1, -1)), self.env_margin, f"{a}-{obs}")
        for a, b in self.self_pairs:
            d, n, pa, pb = _distance(link(a), link(b))
            Jab = self._point_jac(link(a), pa) - self._point_jac(link(b), pb)
            rows.add([d], _pad((n @ Jab).reshape(1, -1)), self.self_margin, f"{a}-{b}")


def _distance(a: dict, b: dict):
    """Distance between two primitives (cores + radii), the unit normal from
    ``b`` to ``a`` and the nearest points of the cores."""
    res = fcl.DistanceResult()
    fcl.distance(a["obj"], b["obj"], fcl.DistanceRequest(enable_nearest_points=True), res)
    pa, pb = (np.asarray(p, dtype=float) for p in res.nearest_points)
    d_core = float(np.linalg.norm(pa - pb))
    n = (pa - pb) / max(d_core, 1e-9)
    return d_core - a["radius"] - b["radius"], n, pa, pb


def scene_collision_info(scene_xml_path: str):
    """The collision geometry the sim env publishes for ``scene_xml_path``
    (robot primitives, scene floor / walls), without LCM: for kinematic
    rollouts and tests. Returns ``(robot_col_info, static_col_info)`` dicts."""
    import mujoco as mj

    from fr3.data_type.basic_types.ColInfoData import ColInfoData
    from fr3.robot_platform.sim.franka_fr3.SimFrankaFR3 import fr3_col_primitives
    from fr3.utils.collision.wall_geoms import extract_static_col_geoms

    robot, static = ColInfoData(), ColInfoData()
    for g in fr3_col_primitives():
        robot.add_col_geom(**g)
    for g in extract_static_col_geoms(mj.MjModel.from_xml_path(resolve(scene_xml_path))):
        static.add_col_geom(**g)
    return robot.get_data()[1], static.get_data()[1]
