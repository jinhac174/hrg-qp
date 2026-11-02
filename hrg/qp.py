"""Weighted velocity-field tracking QP (paper Sec. II-C, eq. 11).

Decision variable ``z = (qdot (7), wdot)``. Cost: a weighted sum of tracking
errors ``sum_k weight_k |v_k - J_k z|^2`` plus a small regulariser. Every
constraint is a distance linearised over the planning horizon ``H``::

    d + (dd/dz) z H >= eps    <=>    (dd/dz) z >= (eps - d) / H

(robot vs environment / itself: FCL; hand spheres vs object / obstacles /
table: ``hrg.sdf``), plus joint / width position limits over ``H``, velocity
limits and acceleration limits (``lb``, ``ub``: the box on ``z``). When a pair
is already inside its margin, the demanded separation rate ``(eps - d)/H`` is
capped at ``sep_rate_max`` while it recovers.

The box on ``z`` is hard (it is what the arm can do). The distance rows are
hard too whenever that problem is feasible (the paper's QP); when it is not,
they are relaxed through slack variables ``s >= 0`` with the linear penalty
``rho sum s`` (``slack_weight``): ``a z + s >= lower``. A linear penalty is exact: with
``rho`` above the tracking cost's gradient (at most ``2 w |v|``, ~2 for a
1 m/s field) every row the box allows holds with ``s = 0``, the same solution
as the hard-constrained QP; when the box does not allow it -- inside a margin,
the arm at rest and the acceleration limit between it and the demanded
separation rate -- the QP returns the fastest recovery the box allows instead
of failing. A small quadratic term ``|s|^2`` is added so OSQP's ADMM
converges (a purely linear slack cost left it iterating); it does not change
which rows hold. (A quadratic penalty alone, large enough to be nearly exact,
also ran out of iterations.)
"""

from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse


class DistanceRows:
    """Accumulates linearised distance constraints ``a z >= lower``."""

    def __init__(self, n: int, horizon: float, sep_rate_max: float):
        self.n = n
        self.H = max(float(horizon), 1e-4)
        self.sep_rate_max = float(sep_rate_max)
        self.A: list[np.ndarray] = []
        self.l: list[float] = []
        self.slack: list[float] = []  # d - eps per row (diagnostics)
        self.labels: list[str] = []

    def add(self, d, grad_z, eps, label="", drift=None) -> None:
        """``d`` (k,), ``grad_z`` = dd/dz (k, n), ``eps`` scalar or (k,);
        ``label``: a name for these rows (one string, or one per row).
        ``drift`` (k,): the rate at which the obstacle's own motion closes
        the distance (``n . v_obstacle`` at the nearest point): the robot
        must separate that much faster, ``d + (dd/dz z - drift) H >= eps``.
        Without it a moving object runs into a hand that the rows believe
        is standing clear."""
        d = np.atleast_1d(np.asarray(d, dtype=float))
        if d.size == 0:
            return
        slack = d - np.broadcast_to(eps, d.shape)
        lower = np.minimum(-slack / self.H, self.sep_rate_max)
        if drift is not None:
            lower = lower + np.broadcast_to(np.asarray(drift, dtype=float), d.shape)
        self.A.extend(np.asarray(grad_z, dtype=float).reshape(-1, self.n))
        self.l.extend(lower.tolist())
        self.slack.extend(slack.tolist())
        self.labels.extend([label] * d.size if isinstance(label, str) else list(label))

    def active(self, tol: float = 1e-3) -> list[tuple[str, float]]:
        """The rows at their margin: ``(label, slack)`` with ``slack < tol``,
        the tightest first. What blocks the motion when the reach stalls."""
        rows = [(lab, s) for lab, s in zip(self.labels, self.slack) if s < tol]
        return sorted(rows, key=lambda r: r[1])

    def __len__(self) -> int:
        return len(self.l)


def solve_tracking_qp(tasks, lb, ub, rows: DistanceRows | None = None, reg: float = 1e-3,
                      slack_weight: float = 100.0, slack_quad: float = 1.0):
    """``tasks``: list of ``(J (k, n), v (k,), weight)``. ``lb``/``ub``: the
    box on ``z`` (hard). ``rows``: distance rows. The rows are first imposed
    as hard constraints (the paper's QP, small and well conditioned); only
    if that problem is infeasible are they relaxed through the penalised
    slacks (module docstring), which has the same optimum whenever the hard
    one is feasible. Returns ``z``, or ``None`` if OSQP fails on both."""
    n = len(lb)
    P = reg * np.eye(n)
    q = np.zeros(n)
    for J, v, wgt in tasks:
        if wgt <= 0.0:
            continue
        J = np.asarray(J, dtype=float).reshape(-1, n)
        P += wgt * J.T @ J
        q -= wgt * J.T @ np.asarray(v, dtype=float).reshape(-1)

    lb, ub = np.asarray(lb, float).copy(), np.asarray(ub, float).copy()
    bad = lb > ub
    if np.any(bad):  # inconsistent box (state outside a limit): squeeze to a point
        lb[bad] = ub[bad]
    m = 0 if rows is None else len(rows)
    if m == 0:
        return _osqp(P, q, np.eye(n), lb, ub)

    A_rows = np.asarray(rows.A)
    A = np.vstack([np.eye(n), A_rows])
    z = _osqp(P, q, A, np.r_[lb, np.asarray(rows.l)], np.r_[ub, np.full(m, np.inf)])
    if z is not None:
        return z
    # variables (z, s): box on z, a z + s >= lower, s >= 0, cost + rho sum(s) + rho2 |s|^2
    Pf = np.zeros((n + m, n + m))
    Pf[:n, :n] = P
    Pf[n:, n:] = slack_quad * np.eye(m)
    qf = np.r_[q, np.full(m, 0.5 * slack_weight)]  # (the objective below is 2 Pf, 2 qf)
    Af = np.zeros((n + 2 * m, n + m))
    Af[:n, :n] = np.eye(n)
    Af[n:n + m, :n] = A_rows
    Af[n:n + m, n:] = np.eye(m)
    Af[n + m:, n:] = np.eye(m)
    z = _osqp(Pf, qf, Af, np.r_[lb, np.asarray(rows.l), np.zeros(m)], np.r_[ub, np.full(2 * m, np.inf)],
              max_iter=20000)
    return None if z is None else z[:n]


def _osqp(P, q, A, l, u, max_iter: int = 4000):
    """``min 1/2 x' (2P) x + (2q)' x  s.t. l <= A x <= u``; ``None`` unless solved."""
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(2.0 * P), q=2.0 * q, A=sparse.csc_matrix(A), l=l, u=u,
        verbose=False, polish=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=max_iter,
    )
    res = solver.solve()
    if res.info.status_val in (osqp.constant("OSQP_SOLVED"), osqp.constant("OSQP_SOLVED_INACCURATE")):
        return np.asarray(res.x, dtype=float)
    return None
