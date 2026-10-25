"""Weighted velocity-field tracking QP (paper Sec. II-C, eq. 11).

Decision variable ``z = (qdot (7), wdot)``. Cost: a weighted sum of tracking
errors ``sum_k weight_k |v_k - J_k z|^2`` plus a small regulariser. Every
constraint is a distance linearised over the planning horizon ``H``::

    d + (dd/dz) z H >= eps    <=>    (dd/dz) z >= (eps - d) / H

(robot vs environment / itself: FCL; hand spheres vs object / obstacles /
table: ``hrg.sdf``), plus joint / width position limits over ``H`` and
velocity limits (``lb``, ``ub``). When a pair is already inside its margin,
the demanded separation rate ``(eps - d)/H`` is capped at ``sep_rate_max`` so
the QP stays feasible while it recovers.
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

    def add(self, d, grad_z, eps, label="") -> None:
        """``d`` (k,), ``grad_z`` = dd/dz (k, n), ``eps`` scalar or (k,);
        ``label``: a name for these rows (one string, or one per row)."""
        d = np.atleast_1d(np.asarray(d, dtype=float))
        if d.size == 0:
            return
        slack = d - np.broadcast_to(eps, d.shape)
        lower = np.minimum(-slack / self.H, self.sep_rate_max)
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


def solve_tracking_qp(tasks, lb, ub, rows: DistanceRows | None = None, reg: float = 1e-3):
    """``tasks``: list of ``(J (k, n), v (k,), weight)``. ``lb``/``ub``: box on
    ``z``. Returns ``z``, or ``None`` if OSQP fails."""
    n = len(lb)
    P = reg * np.eye(n)
    q = np.zeros(n)
    for J, v, wgt in tasks:
        if wgt <= 0.0:
            continue
        J = np.asarray(J, dtype=float).reshape(-1, n)
        P += wgt * J.T @ J
        q -= wgt * J.T @ np.asarray(v, dtype=float).reshape(-1)

    A, l, u = [np.eye(n)], [np.asarray(lb, float)], [np.asarray(ub, float)]
    if rows is not None and len(rows):
        A.append(np.asarray(rows.A))
        l.append(np.asarray(rows.l))
        u.append(np.full(len(rows), np.inf))
    A, l, u = np.vstack(A), np.concatenate(l), np.concatenate(u)
    bad = l > u
    if np.any(bad):  # inconsistent box (state outside a limit): squeeze to a point
        l[bad] = u[bad]

    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(2.0 * P), q=2.0 * q, A=sparse.csc_matrix(A), l=l, u=u,
        verbose=False, polish=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000,
    )
    res = solver.solve()
    if res.info.status_val in (osqp.constant("OSQP_SOLVED"), osqp.constant("OSQP_SOLVED_INACCURATE")):
        return np.asarray(res.x, dtype=float)
    return None
