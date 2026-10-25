"""High-level velocity fields (paper Sec. II-A, II-B).

Fingertip linear velocities, eqs. (3)-(5)
-----------------------------------------
Every plan, eq. (3) is solved again from the current fingertips to the
selected targets (``path_opt``); only the paths' initial direction is used
(eq. 4), at the speed of eq. (5): ``v_const`` far away, a quadratic decay
within ``delta_x`` of the target.

The initial guess of each plan is the better of (a) the paper's via-point
path (``path_opt.initial_paths``) and (b) the previous plan's paths,
re-anchored at the current fingertips (receding horizon, while the target is
the same): fewer constraint violations, then shorter. With (b) the few SQP
iterations per plan accumulate over plans toward the optimum of eq. (3)
(evenly spaced knots, the shortest path). Without it the paths restart every
plan, three iterations from a detour leave the knots bunched at the start, the
first direction is dominated by that and consecutive plans can alternate.

The eq. (2) linear attractor ``K (x* - x)`` is the paper's baseline.

Grasp stability, eqs. (6)-(8)
-----------------------------
``g = mean_i angle(x_j - x_i, a_i)``, with ``a_i = R_i a_i^local`` the axis of
fingertip i's contact surface, and ``omega_i = -grad_{R_i} g`` (eq. 7, the
steepest descent of g over rotations of fingertip i, ``x`` held fixed). On the
2F-85 the pads' contact axes are ``+y_tcp`` (left pad) and ``-y_tcp`` (right
pad) and ``x_2 - x_1 = w y_tcp``, so ``g = 0`` in every configuration and the
weight ``alpha(g)`` keeps the term off: the unstable contact of the paper's
Fig. 3 (a finger touching with its side) cannot happen with parallel pads.

Gripper posture, eqs. (9)-(10)
------------------------------
The gripper joint (the width) toward a nominal open posture, weighted by
``beta(h)``, ``h = |x - x*|``, which turns it off near the target.
"""

from __future__ import annotations

import numpy as np

from hrg import path_opt


def tanh_weight(e, eta: float, T: float):
    """Eqs. (8), (10): ``1/2 + 1/2 tanh((e - eta) / T)`` (0 below ``eta``, 1 above)."""
    return 0.5 * (1.0 + np.tanh((e - eta) / T))


def quad_decay_speed(e, v_const: float, delta: float):
    """Eq. (5): ``v_const`` if ``e > delta`` else ``v_const (1 - (1 - e/delta)^2)``."""
    e = np.asarray(e, dtype=float)
    r = np.clip(e / max(delta, 1e-9), 0.0, 1.0)
    return v_const * (1.0 - (1.0 - r) ** 2)


class FingertipPathField:
    """Eqs. (3)-(5) for both fingertips of the parallel jaw.

    ``constraints`` is a list of ``(sdf, eps)`` (``hrg.sdf``): the target
    object (``eps_O``), the table, the obstacles. ``max_separation``: the
    gripper's opening, coupling the two fingertip paths (``path_opt``).
    """

    def __init__(
        self,
        v_const: float = 0.15,
        delta_x: float = 0.05,
        num_knots: int = 100,
        num_via_samples: int = 100,
        via_dist: float = 0.10,
        sqp_iters: int = 3,
        step_bound: float | None = 0.02,
        violation_tol: float = 1e-3,
        max_separation: float | None = 0.085,
    ):
        self.v_const = float(v_const)
        self.delta_x = float(delta_x)
        self.num_knots = int(num_knots)
        self.num_via_samples = int(num_via_samples)
        self.via_dist = float(via_dist)
        self.sqp_iters = int(sqp_iters)
        self.step_bound = None if step_bound is None else float(step_bound)
        self.violation_tol = float(violation_tol)
        self.max_separation = None if max_separation is None else float(max_separation)
        self._prev = None

    def reset(self) -> None:
        """Forget the previous solution (new reach)."""
        self._prev = None

    def plan_paths(self, x: np.ndarray, x_tgt: np.ndarray, constraints, key=None) -> np.ndarray:
        """Paths ``(2, num_knots, 3)`` from ``x`` to ``x_tgt``. ``key``
        identifies the target; the previous solution is a candidate initial
        guess only while it is the same."""

        def clearance(pts):  # worst (d - eps) over all constraints
            return np.min([f.dist(pts, cap=m + 1e-3) - m for f, m in constraints], axis=0)

        def rank(P):  # (violating knots, total length): lower is better
            # An SQP solution lies *on* its active constraints, to the QP
            # solver's tolerance: count only violations deeper than that.
            viol = int(np.sum(clearance(P[:, 1:-1].reshape(-1, 3)) < -self.violation_tol))
            return viol, float(np.linalg.norm(np.diff(P, axis=1), axis=2).sum())

        paths = path_opt.initial_paths(x, x_tgt, self.num_knots, self.num_via_samples, self.via_dist, clearance)
        if key is not None and self._prev is not None and self._prev[0] == key:
            warm = np.stack([path_opt.resample(np.vstack([x[f], self._prev[1][f][1:-1], x_tgt[f]]), self.num_knots)
                             for f in range(len(x))])
            if rank(warm) <= rank(paths):
                paths = warm
        out = path_opt.sqp(paths, constraints, self.sqp_iters, self.max_separation, self.step_bound)
        self._prev = (key, out)
        return out

    def __call__(self, x: np.ndarray, x_tgt: np.ndarray, constraints, key=None):
        """Desired fingertip velocities ``(2, 3)`` (eq. 4) and the paths."""
        paths = self.plan_paths(x, x_tgt, constraints, key)
        c0 = paths[:, 1] - paths[:, 0]
        n = np.linalg.norm(c0, axis=1, keepdims=True)
        direction = np.where(n > 1e-9, c0 / np.maximum(n, 1e-12), 0.0)
        e = np.linalg.norm(x_tgt - x, axis=1)
        return quad_decay_speed(e, self.v_const, self.delta_x)[:, None] * direction, paths


def linear_fingertip_field(x: np.ndarray, x_tgt: np.ndarray, gain: float) -> np.ndarray:
    """Eq. (2), ``K (x* - x)`` with ``K = gain I``."""
    return gain * (x_tgt - x)


def grasp_stability_field(x: np.ndarray, R: np.ndarray, axes_local: np.ndarray):
    """Eqs. (6)-(7) for a two-fingertip grasp.

    ``x`` (2, 3) fingertip positions, ``R`` (2, 3, 3) fingertip orientations,
    ``axes_local`` (2, 3) the axis of each fingertip's contact surface in its
    own frame. Returns ``(omega (2, 3), g)``:

    ``g = (angle(x2 - x1, a1) + angle(x1 - x2, a2)) / 2`` with ``a_i = R_i a_i^local``
    (paper Sec. III), and ``omega_i = -grad_{R_i} g``: for a rotation
    ``R_i <- exp([phi]) R_i`` the angle ``theta_i`` between ``a_i`` and
    ``u_i`` (unit, fixed) changes as ``d theta_i = -(a_i x u_i) . phi / sin theta_i``,
    so ``omega_i = (a_i x u_i) / (2 sin theta_i)``. At ``theta_i = 0`` (contact
    axis already on the line between the fingertips) the angle is minimal and
    ``omega_i = 0``.
    """
    d = x[1] - x[0]
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.zeros((2, 3)), 0.0
    u = np.stack([d / n, -d / n])
    omega = np.zeros((2, 3))
    thetas = []
    for i in range(2):
        a = R[i] @ axes_local[i]
        a = a / np.linalg.norm(a)
        c = np.cross(a, u[i])
        s = float(np.linalg.norm(c))
        thetas.append(float(np.arctan2(s, float(a @ u[i]))))
        if s > 1e-9:
            omega[i] = c / (2.0 * s)
    return omega, float(np.mean(thetas))


def gripper_width_field(w: float, w_nominal: float, v_gr: float, delta_gr: float) -> float:
    """Eq. (9) for the one gripper joint (the width, m/s)."""
    e = w_nominal - w
    return float(np.sign(e) * quad_decay_speed(abs(e), v_gr, delta_gr))
