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
            + sum_n alpha(theta_n) |J_r,b z - w_des,b,n|^2  (oriented reach:
            + kappa |qdot - k_q (q_grasp - q)|^2             alignment, posture)
            + beta(h) |wdot* - S z|^2 + reg |z|^2
       s.t. Gamma(q) + dGamma/dz z H >= eps_Gamma      robot vs walls, itself
            d(X) + dd/dz z H >= eps_O + r               hand spheres vs object,
                                                        table, obstacles
            joint / width limits over H, velocity limits

The oriented reach (``hrg.orient``, not in the paper; design:
``docs/superpowers/specs/2026-10-28-oriented-reach-design.md``): every pair
has hand orientations in the object frame, one per admissible approach
direction and finger order (``targets``); online ``R* = R_obj R_grasp``.
Step 2 becomes a choice in joint space (``_select_oriented``): the grasp
whose configuration (IK) is nearest the current joints, plus a cost for
leaving the object's preferred approach. Step 5 adds the hand's alignment
with the grasp (``_alignment_tasks``: the pads' normal onto the wall's
normal, the approach onto the grasp's, each ``alpha(theta) |J_r,b z -
w_des,b|^2`` in the body frame), and a weak posture task toward the chosen
grasp configuration
(it fixes how the arm's redundancy is resolved: the joint-space choice is
only meaningful if the arm ends there). The fingertip paths end with a
straight ``approach_dist`` along ``z*`` (``fields.FingertipPathField``).

The controller (``hrg.controller``) calls it at 100 Hz at the measured state;
``scripts/rollout.py`` calls it in a loop with perfect tracking.

``lift_step`` / ``home_step``: after the grasp / on the home key, the same QP
with a TCP pose field / a joint-space field as the task. ``lift_step`` keeps
a held object's points above the table (``held``); ``upright_goal`` is the
hand pose that puts the held object back upright.
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
    pad_dz,
    point_jacobians,
    to_world,
)
from hrg.qp import DistanceRows, solve_tracking_qp
from hrg.robot import RobotModel
from hrg.sdf import HalfSpaceSDF, PosedSDF
from hrg.targets import SWAP, FingertipTargets, select_target


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
    R_star: np.ndarray | None = None  # (3, 3) target hand orientation (oriented reach)


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
    # the oriented reach (docs/superpowers/specs/2026-10-28-oriented-reach-design.md)
    orient: bool = True  # false: the paper's reach (fingertip points only)
    rot_weight: float = 0.1  # alpha's full weight
    rot_gain: float = 2.0  # 1/s, k in w_des = sat(k (R n_b) x n_t, rot_max)
    rot_max: float = 1.0  # rad/s
    rot_threshold: float = 30.0  # deg: alpha full above, linear to 0 below (2 sin 30 deg = the cap)
    approach_dist: float = 0.10  # m, straight final approach along z*
    posture_weight: float = 0.05  # the chosen grasp configuration, joint space
    posture_gain: float = 1.0  # 1/s
    select_w_pref: float = 2.0  # rad of joint motion per rad off the object's preferred approach
    select_shortlist: int = 6  # new candidates solved by IK per plan

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
            approach_dist=params.approach_dist if params.orient else 0.0,
        )
        self._data = robot.data()  # Pinocchio data of the planning thread
        w_tab = np.linspace(0.0, MAX_WIDTH, 86)
        self._pad_dz_table = (w_tab, np.array([pad_dz(w) for w in w_tab]))  # pad_dz, vectorised
        # (pair, direction, swapped) -> {q: grasp configuration or None (unreachable),
        # T_obj: the object pose it was solved at, checked: the configuration last found clear}
        self._ik_cache: dict[tuple, dict] = {}
        self._choice = None  # the (pair, direction, swapped) chosen last plan
        self._q_grasp = None  # the chosen grasp's configuration (posture task)

    def reset(self) -> None:
        """New reach: forget the previous path solution and grasp configurations."""
        self.path_field.reset()
        self._ik_cache.clear()
        self._choice = None
        self._q_grasp = None

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
        R_star = approach = None
        k = -1
        if p.orient:
            sel = self._select_oriented(q, T_tcp, J_tcp, T_obj, tg)
            if sel is None:
                return None
            j, k, swap, R_star, self._q_grasp = sel
            approach = R_star[:, 2]
        else:
            valid = self.targets.valid(T_obj, self.table.z0, p.qp_eps_table)
            j, swap = select_target(x, tg.x1, tg.x2, p.select_w_angle, valid)
            if j is None:
                return None
        x_tgt = np.stack([tg.x2[j], tg.x1[j]] if swap else [tg.x1[j], tg.x2[j]])

        # 3. fingertip velocity fields
        constraints = [(self.obj_sdf, p.eps_obj), (self.table, p.eps_table)]
        constraints += [(o, p.eps_obs) for o in obstacles]
        paths = None
        h = float(np.linalg.norm(x - x_tgt))
        if p.field == "path":
            xd_des, paths = self.path_field(x, x_tgt, constraints, key=(j, k, swap), approach=approach)
            if approach is not None:
                h = self.path_field.h_pre  # the opening is the targets' from the pre-grasp on
        else:
            xd_des = linear_fingertip_field(x, x_tgt, p.linear_gain)

        # 4. auxiliary fields; both pads are fixed in the TCP frame
        omega_des, g = grasp_stability_field(x, np.stack([T_tcp[:3, :3]] * 2), PAD_CONTACT_AXES)
        alpha = float(tanh_weight(g, p.eta_g, p.T_g))
        beta = float(tanh_weight(h, p.eta_h, p.T_h))
        wd_star = gripper_width_field(w, p.w_nominal, p.v_gr, p.delta_gr)

        # 5. QP (eq. 11)
        S = np.zeros((1, N_Z))
        S[0, N_ARM] = 1.0
        J_ang = angular_jacobian(J_tcp)
        tasks = [(Jx[i], xd_des[i], 1.0) for i in range(2)]
        tasks += [(J_ang, omega_des[i], alpha) for i in range(2)]
        rot_err = np.nan
        if R_star is not None:
            rot_err = float(np.linalg.norm(Rotation.from_matrix(R_star @ T_tcp[:3, :3].T).as_rotvec()))
            tasks += self._alignment_tasks(T_tcp[:3, :3], R_star, J_ang)
            if self._q_grasp is not None:
                Jq = np.zeros((N_ARM, N_Z))
                Jq[:, :N_ARM] = np.eye(N_ARM)
                tasks.append((Jq, p.posture_gain * (self._q_grasp - q), p.posture_weight))
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
            info=dict(alpha=alpha, beta=beta, g=g, h=h, n_rows=len(rows), binding=self.binding(q, rows),
                      rot_err=rot_err, approach=k),
            R_star=R_star,
        )

    def _alignment_tasks(self, R_sb: np.ndarray, R_star: np.ndarray, J_ang: np.ndarray) -> list:
        """The hand's alignment with the grasp, in the body frame: for each
        vector pair, gripper ``n_b`` -> target ``n_t``,

            alpha(theta) |J_r,b z - w_des,b|^2,   w_des,b = R_sb^T sat(k (R_sb n_b) x n_t, w_max)

        (``J_r,b = R_sb^T J_r``). The pairs: the closing axis (the pads' normal,
        TCP y) -> the pair axis (for an antipodal pair the wall's normal), and
        the approach (TCP z) -> the grasp's approach. The first turns the pads
        onto the wall, the second the roll about the wall's normal, i.e. from
        where the hand comes. ``alpha(theta) = rot_weight min(1, theta /
        rot_threshold)``: full weight while the vectors are far apart (where
        the speed is at its cap), decaying linearly to 0 as they align."""
        p = self.p
        J_b = R_sb.T @ J_ang
        tasks = []
        for col in (1, 2):
            n_s, n_t = R_sb[:, col], R_star[:, col]
            c = np.cross(n_s, n_t)
            theta = float(np.arctan2(np.linalg.norm(c), n_s @ n_t))
            w_des_b = R_sb.T @ _saturate(p.rot_gain * c, p.rot_max)
            tasks.append((J_b, w_des_b, p.rot_weight * min(1.0, theta / np.radians(p.rot_threshold))))
        return tasks

    def _select_oriented(self, q, T_tcp, J_tcp, T_obj, tg):
        """Step 2 of the oriented reach, in joint space: ``(j, k, swapped,
        R*, q_grasp)``, or ``None`` if no grasp is usable and reachable.

        Candidates: every usable (pair ``j``, direction ``k``) in both finger
        orders, each a hand pose at the grasp. Cost: the joint motion to its
        configuration, ``|q_grasp - q|``, plus ``select_w_pref`` per rad
        between its approach and the object's preferred one (the object frame
        decides where to come from; the joint motion, which way round). Every
        plan the cheapest by the first-order estimate ``|J^+ e|`` (``e`` the
        pose error) are solved by IK (``_grasp_config``), cheapest first, until
        a reachable one is found and ``select_shortlist`` are compared; the
        cheapest reachable one is taken, ``None`` if none is.

        The choice is a grasp in the object frame, and it is kept while it
        stays reachable: its targets move with the object (the reach follows
        a moving object every plan), but the decision does not change with
        the pose estimate's noise. Re-choosing every plan (the paper's eq.
        12) over the hundreds of nearly equivalent grasps of a rim jumped
        between them on a noisy pose (2 mm / 1 deg: 50-90 switches per
        reach, targets jumping up to 13 cm). A new choice is made when the
        chosen grasp stops being reachable, or at a new reach (``reset``)."""
        p = self.p
        if self._choice is not None:  # committed: kept while it stays reachable
            j, k, swap = self._choice
            R = self.targets.grasp_rotations(T_obj, np.array([j]), np.array([k]))[0]
            R = R @ SWAP if swap else R
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = (0.5 * (tg.x1[j] + tg.x2[j])
                        - R[:, 2] * np.interp(np.linalg.norm(tg.x2[j] - tg.x1[j]), *self._pad_dz_table))
            q_g = self._grasp_config(self._choice, q, T, T_obj)
            if q_g is not None:
                return j, k, swap, R, q_g
            print("[HRG] The chosen grasp is no longer reachable: choosing again.")
            self._choice = None
        jj, kk = np.nonzero(self.targets.usable(T_obj, self.table.z0, p.qp_eps_table, p.approach_dist))
        if not len(jj):
            return None
        n = len(jj)
        jj, kk = np.r_[jj, jj], np.r_[kk, kk]
        ss = np.r_[np.zeros(n, dtype=bool), np.ones(n, dtype=bool)]
        R = self.targets.grasp_rotations(T_obj, jj, kk)
        R[ss] = R[ss] @ SWAP
        mid = 0.5 * (tg.x1[jj] + tg.x2[jj])
        dz = np.interp(np.linalg.norm(tg.x2[jj] - tg.x1[jj], axis=1), *self._pad_dz_table)
        P = mid - R[:, :, 2] * dz[:, None]  # TCP positions at the grasp
        e = np.hstack([Rotation.from_matrix(R @ T_tcp[:3, :3].T).as_rotvec(), P - T_tcp[:3, 3]])
        J = J_tcp[:, :N_ARM]
        dq_est = np.linalg.norm(e @ np.linalg.solve(J @ J.T + 1e-2 * np.eye(6), J), axis=1)
        pref = p.select_w_pref * np.arccos(np.clip(R[:, :, 2] @ (T_obj[:3, :3] @ self.targets.approach_axis), -1, 1))

        keys = list(zip(jj.tolist(), kk.tolist(), ss.tolist()))
        best, best_cost = None, np.inf

        def evaluate(i):
            nonlocal best, best_cost
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R[i], P[i]
            q_g = self._grasp_config(keys[i], q, T, T_obj)
            if q_g is not None and np.linalg.norm(q_g - q) + pref[i] < best_cost:
                best, best_cost = (i, q_g), np.linalg.norm(q_g - q) + pref[i]

        # the cheapest by the estimate (until one is reachable, as many as it takes)
        for n_eval, i in enumerate(np.argsort(dq_est + pref)):
            if n_eval >= p.select_shortlist and best is not None:
                break
            evaluate(i)
        if best is None:  # every candidate unreachable: no oriented grasp at this object pose
            return None
        i, q_g = best
        self._choice = keys[i]
        return int(jj[i]), int(kk[i]), bool(ss[i]), R[i], q_g

    def _grasp_config(self, key, q, T_goal, T_obj):
        """The arm configuration at a candidate grasp pose, or ``None`` if it
        is unreachable (no IK solution in the joint range, or ``Gamma`` not
        clear there). Cached per candidate: a solution is tracked from its
        last value; an unreachable mark holds while the object stays where it
        was found unreachable (within 1 cm, 5 deg)."""
        c = self._ik_cache.get(key)
        if c is not None and c["q"] is None:
            moved = (np.linalg.norm(T_obj[:3, 3] - c["T_obj"][:3, 3]) > 0.01
                     or np.linalg.norm(Rotation.from_matrix(T_obj[:3, :3] @ c["T_obj"][:3, :3].T).as_rotvec())
                     > np.radians(5.0))
            if not moved:
                return None
            c = None
        q_g = self._ik(c["q"], T_goal, iters=10) if c is not None else None
        if q_g is None:
            q_g = self._ik(q, T_goal)
        checked = None if c is None else c["checked"]  # the configuration last found clear
        if q_g is not None and (checked is None or np.linalg.norm(q_g - checked) > 0.1):
            checked = q_g if self._clear(q_g) else None
            if checked is None:
                q_g = None
        self._ik_cache[key] = dict(q=q_g, T_obj=T_obj.copy(), checked=checked)
        return q_g

    def _clear(self, q) -> bool:
        """No ``Gamma`` pair (robot vs walls / floor / itself) inside its margin at ``q``."""
        if not self.robot.has_collision_geometry:
            return True
        rows = DistanceRows(N_Z, self.p.horizon, self.p.sep_rate_max)
        self.robot.gamma_rows(q, self.robot.data(), rows)
        return not rows.slack or min(rows.slack) >= -1e-3

    def _ik(self, q0: np.ndarray, T_goal: np.ndarray, iters: int = 80, damping: float = 1e-2):
        """Damped least-squares IK of the TCP pose from ``q0`` (joint limits
        clamped); ``None`` if it ends more than 2 cm / 5 deg off."""
        q = np.array(q0, dtype=float)
        data = self.robot.data()
        for _ in range(iters):
            T = self.robot.pose(q, data)
            e = np.r_[Rotation.from_matrix(T_goal[:3, :3] @ T[:3, :3].T).as_rotvec(), T_goal[:3, 3] - T[:3, 3]]
            if np.linalg.norm(e[3:]) < 1e-3 and np.linalg.norm(e[:3]) < 1e-2:
                return q
            J = self.robot.jacobian(q, data)
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            q = np.clip(q + dq, self.robot.q_lower, self.robot.q_upper)
        T = self.robot.pose(q, data)
        ok = (np.linalg.norm(T_goal[:3, 3] - T[:3, 3]) < 0.02
              and np.linalg.norm(Rotation.from_matrix(T_goal[:3, :3] @ T[:3, :3].T).as_rotvec()) < np.radians(5.0))
        return q if ok else None

    # ------------------------------------------------------------ after the grasp
    def lift_step(self, q: np.ndarray, w: float, T_goal: np.ndarray, v_max: float, w_max: float,
                  pos_gain: float, rot_gain: float, rot_weight: float, held=None, held_eps: float = 0.02):
        """Joint velocities toward the TCP pose ``T_goal`` (6-D pose field,
        saturated at ``v_max`` / ``w_max``), the width held, under ``Gamma``
        and the hand vs table / obstacles (not the grasped object).
        ``held``: points of a held object in the TCP frame (M, 3), kept
        ``held_eps`` above the table."""
        T = self.robot.pose(q, self._data)
        J = self.robot.jacobian(q, self._data)
        v = _saturate(pos_gain * (T_goal[:3, 3] - T[:3, 3]), v_max)
        om = _saturate(rot_gain * Rotation.from_matrix(T_goal[:3, :3] @ T[:3, :3].T).as_rotvec(), w_max)
        W = np.r_[np.full(3, rot_weight), np.ones(3)]
        J6 = np.zeros((6, N_Z))
        J6[:, :N_ARM] = W[:, None] * J
        lb, ub = self._bounds(q, w)
        lb[N_ARM] = ub[N_ARM] = 0.0
        rows = self._rows(q, T, J, w, (), with_object=False)
        if held is not None and len(held):
            self.held_rows(T, J, held, held_eps, rows)
        z = solve_tracking_qp([(J6, W * np.r_[om, v], 1.0)], lb, ub, rows, reg=self.p.qp_reg)
        return np.zeros(N_ARM) if z is None else z[:N_ARM].copy()

    def held_rows(self, T_tcp: np.ndarray, J_tcp: np.ndarray, P_tcp: np.ndarray, eps: float,
                  rows: DistanceRows, reach: float = 0.15) -> None:
        """Rows keeping the held object's points (TCP frame) ``eps`` above
        the table; only points within ``reach`` of it (the QP stays small)."""
        C = to_world(T_tcp, P_tcp)
        near = C[:, 2] - self.table.z0 < eps + reach
        if not np.any(near):
            return
        Jp = point_jacobians(T_tcp, J_tcp, P_tcp[near], np.zeros((int(near.sum()), 3)))
        d, grad = self.table.dist_grad(C[near])
        rows.add(d, np.einsum("ki,kij->kj", grad, Jp), eps, "held:table")

    def upright_goal(self, T_tcp: np.ndarray, T_tcp_obj: np.ndarray, P_obj: np.ndarray, clearance: float,
                     position=None):
        """The hand pose that puts a held object back upright: the object
        (``T_tcp_obj``, its pose in the TCP frame) rotated by the smallest
        tilt that turns its ``+z`` up, about the TCP, with the TCP at
        ``position`` (default: where it is) raised until the object's points
        ``P_obj`` (object frame) clear the table by ``clearance``. Also
        returns the tilt (rad)."""
        R_obj = T_tcp[:3, :3] @ T_tcp_obj[:3, :3]
        up = R_obj[:, 2]
        tilt = float(np.arccos(np.clip(up[2], -1.0, 1.0)))
        axis = np.cross(up, [0.0, 0.0, 1.0])
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 1e-9 else T_tcp[:3, 0]  # upside down: about the hand's x
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(axis * tilt).as_matrix() @ T_tcp[:3, :3]
        T[:3, 3] = T_tcp[:3, 3] if position is None else position
        z_low = float(np.min((P_obj @ (T[:3, :3] @ T_tcp_obj[:3, :3]).T)[:, 2] + (T[:3, :3] @ T_tcp_obj[:3, 3])[2]))
        T[2, 3] = max(T[2, 3], self.table.z0 + clearance - z_low)
        return T, tilt

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

