"""High-level velocity fields (paper Sec. II-A, II-B).

Fingertip linear velocities, eqs. (3)-(5)
-----------------------------------------
Every plan, eq. (3) is solved again from the current fingertips to the
selected targets (``path_opt``); only the paths' initial direction is used
(eq. 4; read over the first ``direction_lookahead`` = 2 cm of the path, see
``path_direction``), at the speed of eq. (5): ``v_const`` far away, a
quadratic decay within ``delta_x`` of the target.

The initial guess of each plan is the better of (a) the paper's via-point
path (``path_opt.initial_paths``) and (b) the previous plan's paths carried
over to the current fingertips and targets (``receding_paths``: the part
travelled cut off, the rest moved with the target; while the target is the
same one): fewer constraint violations, then shorter. With (b) the few SQP
iterations per plan accumulate over plans toward the optimum of eq. (3)
(evenly spaced knots, the shortest path). Without it the paths restart every
plan, three iterations from a detour leave the knots bunched at the start, the
first direction is dominated by that and consecutive plans can alternate.

Straight final approach (the oriented reach; not in the paper): given the
target hand's approach axis ``z*`` (``approach``), the paths of eq. (3) end
on the **approach segment** from the pre-grasp points ``x*_i - approach_dist
z*`` to the targets ``x*_i``, at the depth ``lam`` in [0, 1] set by how well
the fingertips' midpoint is lined up with the segment (``corridor``): inside
a narrow cone around it ``lam = 1`` (the paths run straight to the targets,
so the last ``approach_dist`` is travelled along the approach axis, which
``targets`` checked clear), at twice the cone's radius ``lam = 0`` (the
paths lead to the pre-grasp), linear in between, and never shallower than
the fingertips' own depth along the approach (a goal behind the hand would
send it back; a hand lagging a moving object overshot such a goal and
reversed every plan). The end point is continuous
in the state, so a target that moves (a carried object) slides the goal
along the segment instead of flipping the plan between "to the pre-grasp"
and "to the target" (an on/off corridor with hysteresis did that several
times a second on a moving object, and every flip was a jump of the
commanded direction that rode the acceleration bounds: the arm shook). The
pre-grasp points are spread at the gripper's current opening (until the
straight approach the posture field of eq. (9) sets the opening, not the
targets). The speed is eq. (5) on the distance to the targets throughout.
``h_pre = (1 - lam) |mid - pre| sqrt 2`` (the scale of the paper's stacked
``h``; 0 on the straight approach) is what the posture weight of eq. (10)
fades on.

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


def path_direction(paths: np.ndarray, lookahead: float) -> np.ndarray:
    """The paths' initial direction (eq. 4), (F, 3) unnormalised: from the
    first knot to the point ``lookahead`` along the path (at most half its
    length). The paper takes the first segment; with 100 knots that is
    1-2 mm of a 15 cm path, and the SQP moves the interior knots by a
    fraction of a millimetre from plan to plan, which turned that segment
    by tens of degrees between plans on a moving target (the arm shook).
    Over 2 cm the same wobble is a few degrees, and on a straight path the
    direction is identical."""
    out = np.empty((paths.shape[0], 3))
    for f, P in enumerate(paths):
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s_cum = np.r_[0.0, np.cumsum(seg)]
        s_look = min(float(lookahead), 0.5 * float(s_cum[-1]))
        if s_cum[-1] < 1e-9:
            out[f] = 0.0
            continue
        s_look = max(s_look, float(s_cum[1]))  # at least the first segment
        pt = np.array([np.interp(s_look, s_cum, P[:, k]) for k in range(3)])
        out[f] = pt - P[0]
    return out


def receding_paths(prev: np.ndarray, x: np.ndarray, x_tgt: np.ndarray, num_knots: int) -> np.ndarray:
    """The previous plan's paths ``prev`` (F, T, 3) carried over to the new
    start ``x`` and targets ``x_tgt`` (F, 3): the part already travelled is
    cut off (the fingertips projected onto the paths, one arc-length
    parameter for both, since they are a rigid pair), the remainder is
    shifted so that its start lands on ``x`` and its end on ``x_tgt`` (the
    shift blended linearly along the path: the whole path moves with a
    target that moved) and resampled to ``num_knots``.

    Prepending ``x`` and appending the new target to the old interior knots
    instead left a first knot the fingertips had already passed and a last
    knot behind the moved target: the first segment then pointed anywhere,
    and with a moving object the commanded direction jumped by tens of
    degrees between plans (the arm shook)."""
    F, T, _ = prev.shape
    seg = np.linalg.norm(np.diff(prev, axis=1), axis=2)  # (F, T-1)
    s_cum = np.concatenate([np.zeros((F, 1)), np.cumsum(seg, axis=1)], axis=1)  # (F, T)
    # nearest arc-length parameter of each fingertip on its own path, then the mean
    s0 = []
    for f in range(F):
        a, b = prev[f, :-1], prev[f, 1:]
        ab = b - a
        tt = np.clip(np.einsum("ij,ij->i", x[f] - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(a + tt[:, None] * ab - x[f], axis=1)
        k = int(np.argmin(d))
        s0.append(s_cum[f, k] + tt[k] * seg[f, k])
    out = np.empty((F, num_knots, 3))
    for f in range(F):
        S = s_cum[f, -1]
        s_f = min(s0[f] * (np.mean(s0) / max(np.mean(s0), 1e-12)), S * 0.98)  # the shared parameter, kept in range
        s_new = np.linspace(s_f, S, num_knots)
        P = np.column_stack([np.interp(s_new, s_cum[f], prev[f][:, k]) for k in range(3)])
        u = (s_new - s_f) / max(S - s_f, 1e-12)
        P = P + (1.0 - u)[:, None] * (x[f] - P[0]) + u[:, None] * (x_tgt[f] - P[-1])
        out[f] = path_opt.resample(P, num_knots)
        out[f, 0], out[f, -1] = x[f], x_tgt[f]
    return out


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
        approach_dist: float = 0.0,
        corridor_r0: float = 0.005,
        corridor_slope: float = 0.15,
        direction_lookahead: float = 0.02,
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
        self.approach_dist = float(approach_dist)
        self.corridor_r0 = float(corridor_r0)
        self.corridor_slope = float(corridor_slope)
        self.direction_lookahead = float(direction_lookahead)
        self._prev = None
        self.h_pre = 0.0  # last call: (1 - lam) |mid - pre| sqrt 2, 0 on the straight approach
        self.lam = 0.0  # last call: the depth of the paths' end point on the approach segment

    def reset(self) -> None:
        """Forget the previous solution (new reach)."""
        self._prev = None

    def corridor(self, x: np.ndarray, x_tgt: np.ndarray, approach: np.ndarray) -> float:
        """How well the fingertips' midpoint is lined up with the targets'
        final approach segment, in [0, 1]: with ``s`` its distance back along
        the approach and ``r`` its distance off the axis, the cone's radius is
        ``lim = r0 + slope max(s, 0)``; 1 for ``r <= lim`` (inside), 0 for
        ``r >= 2 lim``, linear between; 0 when ``s`` is outside ``[-1 cm,
        approach_dist + 2 cm]`` (beside or behind the segment), fading over
        the last centimetre of that range."""
        v = x.mean(axis=0) - x_tgt.mean(axis=0)
        s = -float(v @ approach)  # distance back along the approach
        r = float(np.linalg.norm(v + s * approach))
        lim = self.corridor_r0 + self.corridor_slope * max(s, 0.0)
        lateral = float(np.clip(2.0 - r / lim, 0.0, 1.0))
        along = float(np.clip(min(s + 0.01, self.approach_dist + 0.02 - s) / 0.01, 0.0, 1.0))
        return lateral * along

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
            warm = receding_paths(self._prev[1], x, x_tgt, self.num_knots)
            if rank(warm) <= rank(paths):
                paths = warm
        out = path_opt.sqp(paths, constraints, self.sqp_iters, self.max_separation, self.step_bound)
        self._prev = (key, out)
        return out

    def __call__(self, x: np.ndarray, x_tgt: np.ndarray, constraints, key=None, approach=None):
        """Desired fingertip velocities ``(2, 3)`` (eq. 4) and the paths.
        ``approach``: the target hand's approach axis (unit, world): straight
        final approach over ``approach_dist`` (``None``: the paper's paths)."""
        head = None
        self.h_pre = float(np.linalg.norm(x - x_tgt))
        self.lam = 1.0
        if approach is not None and self.approach_dist > 0.0:
            approach = np.asarray(approach, dtype=float)
            # The goal never lies behind the fingertips along the approach: at least as deep
            # as they are (``depth``), else a hand that has passed it (lagging a moving object,
            # hovering off-axis) is sent back to it, overshoots it again, and the commanded
            # direction flips every plan.
            s_back = -float((x.mean(axis=0) - x_tgt.mean(axis=0)) @ approach)
            depth = float(np.clip(1.0 - s_back / self.approach_dist, 0.0, 1.0))
            lam = self.lam = max(self.corridor(x, x_tgt, approach), depth)
            t = np.linspace(0.0, 1.0, self.num_knots).reshape(1, -1, 1)
            # Pre-grasp points: the targets' midpoint backed off along the approach, the
            # fingertips spread along the target closing axis at the CURRENT opening (the
            # posture field sets it until here; the targets set it on the straight approach).
            m_tgt = x_tgt.mean(axis=0)
            m_pre = m_tgt - self.approach_dist * approach
            scale = np.linalg.norm(x[1] - x[0]) / max(float(np.linalg.norm(x_tgt[1] - x_tgt[0])), 1e-9)
            x_pre = m_pre + (x_tgt - m_tgt) * scale
            x_goal = x_pre + lam * (x_tgt - x_pre)  # the paths' end, on the approach segment
            self.h_pre = float((1.0 - lam) * np.sqrt(2.0) * np.linalg.norm(x.mean(axis=0) - m_pre))
            # Always eq. (3) to the goal (the straight line is its own initial guess when that is
            # clear): a straight line taken on trust from off the axis crossed the object, and the
            # pre-shaped fingers stalled on their margin against a lying mug.
            head = self.plan_paths(x, x_goal, constraints, key)
            if lam >= 1.0:
                paths = head
            else:
                tail = (1.0 - t) * x_goal[:, None] + t * x_tgt[:, None]
                paths = np.stack([path_opt.resample(np.vstack([head[f], tail[f][1:]]), self.num_knots)
                                  for f in range(len(x))])
        else:
            paths = self.plan_paths(x, x_tgt, constraints, key)
        first = paths if head is None else head  # the direction is the planned path's
        c0 = path_direction(first, self.direction_lookahead)
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
