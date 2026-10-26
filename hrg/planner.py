"""One planning step of the method (paper Sec. II), as a function of the state.

``ReachPlanner.step(q, w, T_obj, obstacles)`` -- the reach, every plan:

1. Fingertips ``x(q, w)`` and their Jacobians (``hand.fingertip_state``).
2. Target selection, eq. (12), among the candidates valid at the live object
   pose (``targets.FingertipTargets.valid``).
3. Fingertip velocity fields: eqs. (3)-(5) (``field: path``) or the eq. (2)
   baseline (``field: linear``).
4. Auxiliary fields: grasp stability, eqs. (6)-(8) (identically zero on the
   2F-85, ``fields.grasp_stability_field``), gripper posture, eqs. (9)-(10).
5. The QP of eq. (11) over ``z = (qdot, wdot)``::

       min  sum_i |xdot*_i - J_i z|^2 + alpha(g) sum_i |omega*_i - J_w z|^2
            + beta(h) |wdot* - S z|^2 + reg |z|^2
       s.t. Gamma(q) + dGamma/dz z H >= eps_Gamma      robot vs walls, itself
            d(X) + dd/dz z H >= eps_O + r               hand spheres vs object,
                                                        table, obstacles
            joint / width limits over H, velocity limits

The controller (``hrg.controller``) calls it at 100 Hz at the measured state;
``scripts/rollout.py`` calls it in a loop with perfect tracking.

``lift_step`` / ``home_step``: after the grasp / on the home key, the same QP
with a TCP pose field / a joint-space field as the task.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from hrg.fields import (
    FingertipPathField,
    grasp_stability_field,
    gripper_width_field,
    linear_fingertip_field,
    tanh_weight,
)
from hrg.hand import (
    MAX_WIDTH,
    MAX_WIDTH_RATE,
    N_ARM,
    N_Z,
    PAD_CONTACT_AXES,
    angular_jacobian,
    fingertip_state,
    hand_spheres,
)
from hrg.qp import DistanceRows, solve_tracking_qp
from hrg.robot import RobotModel
from hrg.sdf import HalfSpaceSDF, PosedSDF
from hrg.targets import FingertipTargets, select_target


@dataclass
class ReachStep:
    """Output of one reach plan."""

    qd: np.ndarray  # (7,) joint velocities
    wd: float  # width rate (m/s)
    x: np.ndarray  # (2, 3) fingertips at the planning state
    x_tgt: np.ndarray  # (2, 3) selected targets (in the fingertips' order)
    target: int  # index into the object's FingertipTargets
    swapped: bool  # finger order of the selected pair
    contact_width: float  # |p2 - p1| of the selected pair (what to close on)
    paths: np.ndarray | None  # (2, T, 3) eq. (3) solution (path field only)
    qp_ok: bool
    info: dict = field(default_factory=dict)  # alpha, beta, g, h, n_rows, binding (rows at their margin, joint limits)


@dataclass
class HRGParams:
    """The method's parameters (``hrg:`` section of ``configs/controller.yaml``)."""

    field: str = "path"  # path: eqs. (3)-(5); linear: the eq. (2) baseline
    # eq. (5)
    v_const: float = 0.15
    delta_x: float = 0.05
    # eq. (2) baseline
    linear_gain: float = 1.0
    # eq. (3)
    num_knots: int = 100
    num_via_samples: int = 100
    via_dist: float = 0.10
    sqp_iters: int = 3
    sqp_step_bound: float = 0.02
    max_separation: float = MAX_WIDTH
    eps_obj: float = 0.01
    eps_obs: float = 0.05
    eps_table: float = 0.01
    # eq. (12)
    select_w_angle: float = 0.05
    # eqs. (6)-(8)
    eta_g: float = 0.15
    T_g: float = 0.05
    # eqs. (9)-(10)
    w_nominal: float = MAX_WIDTH
    v_gr: float = 0.10
    delta_gr: float = 0.02
    eta_h: float = 0.06
    T_h: float = 0.02
    # eq. (11)
    horizon: float = 0.10  # above the closed-loop latency, docs/method.md eq. (11)
    qp_reg: float = 1e-3
    qp_eps_obj: float = 0.01
    qp_eps_obs: float = 0.02
    qp_eps_table: float = 0.005
    sep_rate_max: float = 0.25

    @classmethod
    def from_config(cls, cfg) -> "HRGParams":
        known = cls.__dataclass_fields__
        unknown = set(cfg) - set(known)
        if unknown:
            raise ValueError(f"unknown hrg parameters: {sorted(unknown)}")
        p = cls(**{k: cfg[k] for k in cfg})
        if p.field not in ("path", "linear"):
            raise ValueError(f"hrg.field must be 'path' or 'linear', got {p.field!r}")
        return p


class ReachPlanner:
    def __init__(self, params: HRGParams, robot: RobotModel, targets: FingertipTargets,
                 obj_sdf: PosedSDF, table_z: float = 0.0):
        self.p = params
        self.robot = robot
        self.targets = targets
        self.obj_sdf = obj_sdf
        self.table = HalfSpaceSDF(table_z)
        self.path_field = FingertipPathField(
            v_const=params.v_const, delta_x=params.delta_x, num_knots=params.num_knots,
            num_via_samples=params.num_via_samples, via_dist=params.via_dist,
            sqp_iters=params.sqp_iters, step_bound=params.sqp_step_bound,
            max_separation=params.max_separation,
        )
        self._data = robot.data()  # Pinocchio data of the planning thread

    def reset(self) -> None:
        """New reach: forget the previous path solution."""
        self.path_field.reset()

    def tcp_pose(self, q: np.ndarray) -> np.ndarray:
        return self.robot.pose(q, self._data)

    # ------------------------------------------------------------ reach
    def step(self, q: np.ndarray, w: float, T_obj: np.ndarray, obstacles=()) -> ReachStep | None:
        """One plan at joints ``q``, width ``w``, object pose ``T_obj``;
        ``obstacles``: posed ``hrg.sdf`` shapes. ``None`` if no target is valid."""
        p = self.p
        T_tcp = self.robot.pose(q, self._data)
        J_tcp = self.robot.jacobian(q, self._data)

        # 1-2. fingertips, target selection (eq. 12)
        x, Jx = fingertip_state(T_tcp, J_tcp, w)
        self.obj_sdf.set_pose(T_obj)
        tg = self.targets.transformed(T_obj)
        valid = self.targets.valid(T_obj, self.table.z0, p.qp_eps_table)
        j, swap = select_target(x, tg.x1, tg.x2, p.select_w_angle, valid)
        if j is None:
            return None
        x_tgt = np.stack([tg.x2[j], tg.x1[j]] if swap else [tg.x1[j], tg.x2[j]])

        # 3. fingertip velocity fields
        constraints = [(self.obj_sdf, p.eps_obj), (self.table, p.eps_table)]
        constraints += [(o, p.eps_obs) for o in obstacles]
        paths = None
        if p.field == "path":
            xd_des, paths = self.path_field(x, x_tgt, constraints, key=(j, swap))
        else:
            xd_des = linear_fingertip_field(x, x_tgt, p.linear_gain)

        # 4. auxiliary fields; both pads are fixed in the TCP frame
        omega_des, g = grasp_stability_field(x, np.stack([T_tcp[:3, :3]] * 2), PAD_CONTACT_AXES)
        alpha = float(tanh_weight(g, p.eta_g, p.T_g))
        h = float(np.linalg.norm(x - x_tgt))
        beta = float(tanh_weight(h, p.eta_h, p.T_h))
        wd_star = gripper_width_field(w, p.w_nominal, p.v_gr, p.delta_gr)

        # 5. QP (eq. 11)
        S = np.zeros((1, N_Z))
        S[0, N_ARM] = 1.0
        J_ang = angular_jacobian(J_tcp)
        tasks = [(Jx[i], xd_des[i], 1.0) for i in range(2)]
        tasks += [(J_ang, omega_des[i], alpha) for i in range(2)]
        tasks.append((S, [wd_star], beta))
        lb, ub = self._bounds(q, w)
        rows = self._rows(q, T_tcp, J_tcp, w, obstacles, with_object=True)
        z = solve_tracking_qp(tasks, lb, ub, rows, reg=p.qp_reg)
        ok = z is not None
        if not ok:
            z = np.zeros(N_Z)
        return ReachStep(
            qd=z[:N_ARM].copy(), wd=float(z[N_ARM]), x=x, x_tgt=x_tgt, target=j, swapped=swap,
            contact_width=float(tg.contact_width[j]), paths=paths, qp_ok=ok,
            info=dict(alpha=alpha, beta=beta, g=g, h=h, n_rows=len(rows), binding=self.binding(q, rows)),
        )

    # ------------------------------------------------------------ after the grasp
    def lift_step(self, q: np.ndarray, w: float, T_goal: np.ndarray, v_max: float, w_max: float,
                  pos_gain: float, rot_gain: float, rot_weight: float):
        """Joint velocities toward the TCP pose ``T_goal`` (6-D pose field,
        saturated at ``v_max`` / ``w_max``), the width held, under ``Gamma``
        and the hand vs table / obstacles (not the grasped object)."""
        T = self.robot.pose(q, self._data)
        J = self.robot.jacobian(q, self._data)
        v = _saturate(pos_gain * (T_goal[:3, 3] - T[:3, 3]), v_max)
        om = _saturate(rot_gain * Rotation.from_matrix(T_goal[:3, :3] @ T[:3, :3].T).as_rotvec(), w_max)
        W = np.r_[np.full(3, rot_weight), np.ones(3)]
        J6 = np.zeros((6, N_Z))
        J6[:, :N_ARM] = W[:, None] * J
        lb, ub = self._bounds(q, w)
        lb[N_ARM] = ub[N_ARM] = 0.0
        z = solve_tracking_qp([(J6, W * np.r_[om, v], 1.0)], lb, ub,
                              self._rows(q, T, J, w, (), with_object=False), reg=self.p.qp_reg)
        return np.zeros(N_ARM) if z is None else z[:N_ARM].copy()

    def home_step(self, q: np.ndarray, w: float, q_home: np.ndarray, gain: float):
        """Joint velocities toward ``q_home`` (``gain (q_home - q)``) under
        the same constraints as ``lift_step``."""
        T = self.robot.pose(q, self._data)
        J = self.robot.jacobian(q, self._data)
        Jq = np.zeros((N_ARM, N_Z))
        Jq[:, :N_ARM] = np.eye(N_ARM)
        lb, ub = self._bounds(q, w)
        lb[N_ARM] = ub[N_ARM] = 0.0
        z = solve_tracking_qp([(Jq, gain * (q_home - q), 1.0)], lb, ub,
                              self._rows(q, T, J, w, (), with_object=False), reg=self.p.qp_reg)
        return np.zeros(N_ARM) if z is None else z[:N_ARM].copy()

    # ------------------------------------------------------------ QP pieces
    def _bounds(self, q, w):
        H = max(self.p.horizon, 1e-4)
        lb, ub = np.zeros(N_Z), np.zeros(N_Z)
        lb[:N_ARM], ub[:N_ARM] = self.robot.velocity_bounds(q, H)
        wc = float(np.clip(w, 0.0, MAX_WIDTH))
        lb[N_ARM] = max(-MAX_WIDTH_RATE, (0.0 - wc) / H)
        ub[N_ARM] = min(MAX_WIDTH_RATE, (MAX_WIDTH - wc) / H)
        return lb, ub

    def _rows(self, q, T_tcp, J_tcp, w, obstacles, with_object: bool) -> DistanceRows:
        p = self.p
        rows = DistanceRows(N_Z, p.horizon, p.sep_rate_max)
        if self.robot.has_collision_geometry:
            self.robot.gamma_rows(q, self._data, rows)  # Gamma(q)
        # d(X(q, w), O): hand spheres vs target object, table, obstacles
        C, JC, r = hand_spheres(T_tcp, J_tcp, w)
        shapes = [("object", self.obj_sdf, p.qp_eps_obj)] if with_object else []
        shapes.append(("table", self.table, p.qp_eps_table))
        shapes += [(f"obstacle{k}", o, p.qp_eps_obs) for k, o in enumerate(obstacles)]
        for name, sdf, eps in shapes:
            d, grad = sdf.dist_grad(C)
            rows.add(d, np.einsum("ki,kij->kj", grad, JC), eps + r, [f"{name}:sphere{i}" for i in range(len(C))])
        return rows

    def binding(self, q, rows: DistanceRows, tol: float = 1e-3, joint_tol: float = 0.02) -> dict:
        """What limits the motion at ``q``: the distance rows at their margin
        (``DistanceRows.active``) and the joints within ``joint_tol`` rad of a
        position limit. The paper's Fig. 12: a reach that stalls does so
        against these."""
        q = np.asarray(q, dtype=float)
        lo, hi = self.robot.q_lower, self.robot.q_upper
        joints = [f"joint{i + 1}{'-' if q[i] - lo[i] < hi[i] - q[i] else '+'}" for i in range(N_ARM)
                  if min(q[i] - lo[i], hi[i] - q[i]) < joint_tol]
        return dict(rows=rows.active(tol), joint_limits=joints)


def binding_text(binding: dict, max_rows: int = 4) -> str:
    """``ReachStep.info['binding']`` as a log fragment ('' if nothing binds)."""
    parts = []
    if binding["rows"]:
        rows = ", ".join(f"{lab} ({s * 1e3:+.1f} mm)" for lab, s in binding["rows"][:max_rows])
        parts.append(f"at margin: {rows}" + (" ..." if len(binding["rows"]) > max_rows else ""))
    if binding["joint_limits"]:
        parts.append("joint limits: " + ", ".join(binding["joint_limits"]))
    return "; " + "; ".join(parts) if parts else ""


def _saturate(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v * (vmax / n) if n > vmax else v

