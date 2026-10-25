"""Fingertip path optimisation, eq. (3) (paper Sec. II-A, II-D).

For the fingertips ``c_f(s)``, ``f = 1, 2``, discretised into ``T`` knots
``c_f[0..T-1]`` with fixed end points (current fingertip, target)::

    min   sum_f sum_k |c_f[k+1] - c_f[k]|^2          (path energy, int |c'|^2 ds)
    s.t.  d(c_f[k], O_j) >= eps_j                    every interior knot, every
                                                     constraint (object, table,
                                                     obstacles)
          |c_1[k] - c_2[k]| <= w_max                 the 2F-85's opening (below)

Initial guess (paper Sec. II-D): the straight line if it violates nothing,
otherwise "two segments through a via point ``via_dist`` from the target with
minimal constraint violation": the via directions are sampled on the sphere
(Fibonacci lattice), and the one with the fewest violating knots wins (ties:
the shortest path). Then a few SQP iterations (paper: ~3): each linearises
the distance constraints at the current knots and solves the QP for the knot
displacement with OSQP.

Parallel jaw (FR3 + 2F-85)
--------------------------
The two fingertips are a rigid pair whose separation is the opening width,
so along the paths ``|c_1[k] - c_2[k]| <= w_max`` (85 mm). The paper's
fingers are independent and need no such coupling. The via search uses one
detour direction for both fingertips, which satisfies the coupling by
construction (the separation then interpolates between the current and the
target one), and the SQP keeps it satisfied (linearised like the distances).
Without it the two paths can pass on opposite sides of the object: fields no
motion of the gripper can follow.

Each SQP iteration moves every knot coordinate at most ``step_bound`` (a trust
region), so the linearised constraints stay valid.
"""

from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


def _two_segments(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, n: int) -> np.ndarray:
    """``n`` points equally spaced in arc length along ``x0 -> x1 -> x2``."""
    l1 = np.linalg.norm(x1 - x0)
    l2 = np.linalg.norm(x2 - x1)
    total = l1 + l2
    if total < 1e-12:
        return np.repeat(x0[None, :], n, axis=0)
    t = np.linspace(0, 1, n)[:, None]
    if l1 < 1e-12:
        return x1 + (x2 - x1) * t
    if l2 < 1e-12:
        return x0 + (x1 - x0) * t
    t = t[:, 0]
    split = l1 / total
    pts = np.empty((n, 3))
    m1 = t <= split
    pts[m1] = x0 + (x1 - x0) * ((t[m1] * total) / l1)[:, None]
    pts[~m1] = x1 + (x2 - x1) * (((t[~m1] - split) * total) / l2)[:, None]
    return pts


def initial_paths(x0: np.ndarray, xg: np.ndarray, num_knots: int, num_via: int, via_dist: float, clearance):
    """Paper's initial guess, one shared via direction for both fingertips.

    ``x0``, ``xg``: ``(F, 3)`` current fingertips and targets. ``clearance``:
    ``(N, 3) -> (N,)``, the worst ``d - eps`` over all constraints (< 0 is a
    violation). Returns ``(F, num_knots, 3)``.
    """
    F = len(x0)
    t = np.linspace(0, 1, num_knots).reshape(1, -1, 1)
    straight = (1 - t) * x0[:, None, :] + t * xg[:, None, :]
    if not np.any(clearance(straight.reshape(-1, 3)) < 0.0):
        return straight

    u = _fibonacci_sphere(num_via)
    via = xg[None] + u[:, None, :] * via_dist  # (S, F, 3)
    # Candidate 2-segment polylines, half the knots per segment, for counting
    # the violations (the chosen one is resampled to ``num_knots`` below).
    th = np.linspace(0, 1, (num_knots - 2) // 2 + 1).reshape(1, 1, -1, 1)
    a, b, c = x0.reshape(1, F, 1, 3), via[:, :, None, :], xg.reshape(1, F, 1, 3)
    cand = np.concatenate([(1 - th) * a + th * b, (1 - th) * b + th * c], axis=2)
    S, _, Tc, _ = cand.shape
    viol = (clearance(cand.reshape(-1, 3)).reshape(S, F, Tc) < 0.0).sum(axis=2).sum(axis=1)  # (S,)
    best = np.where(viol == viol.min())[0]
    length = (np.linalg.norm(via[best] - x0[None], axis=2) + np.linalg.norm(xg[None] - via[best], axis=2)).sum(axis=1)
    k = best[np.argmin(length)]
    return np.stack([_two_segments(x0[f], via[k, f], xg[f], num_knots) for f in range(F)])


def sqp(paths: np.ndarray, constraints, num_iterations: int, max_separation: float | None, step_bound: float | None):
    """SQP iterations on eq. (3) from ``paths`` ``(F, T, 3)`` (end points fixed).

    ``constraints``: list of ``(sdf, eps)``, ``sdf.dist_grad(X (N, 3)) -> (d, grad)``.
    A failed QP keeps the current iterate.
    """
    paths = np.asarray(paths, dtype=np.float64)
    F, T, _ = paths.shape
    n_int = T - 2
    if n_int < 1:
        return paths.copy()
    trajs = [paths[f].copy() for f in range(F)]

    # Hessian of the path energy over the interior knots: the second-difference
    # matrix [-1 2 -1] per coordinate (D^T D of the first differences).
    if n_int == 1:
        P_1D = sparse.csc_matrix([[2.0]])
    else:
        P_1D = sparse.diags([-np.ones(n_int - 1), 2.0 * np.ones(n_int), -np.ones(n_int - 1)], [-1, 0, 1], format="csc")
    P = sparse.block_diag([sparse.kron(P_1D, sparse.eye(3, format="csc"))] * F, format="csc")

    n_wp = F * n_int
    n_var = 3 * n_wp
    row_idx = np.repeat(np.arange(n_wp), 3)
    col_idx = np.arange(n_var)

    for _ in range(int(num_iterations)):
        # Gradient of the energy at the current knots (end points included).
        g = np.concatenate([(2.0 * xf[1:-1] - xf[:-2] - xf[2:]).reshape(-1) for xf in trajs])
        x_int = np.concatenate([xf[1:-1] for xf in trajs], axis=0)  # (F * n_int, 3)
        A_list, l_list, u_list = [], [], []
        for sdf, eps in constraints:  # d + grad . dx >= eps
            d, grad = sdf.dist_grad(x_int)
            A_list.append(sparse.coo_matrix((np.asarray(grad).reshape(-1), (row_idx, col_idx)), shape=(n_wp, n_var)).tocsc())
            l_list.append(eps - np.asarray(d).reshape(-1))
            u_list.append(np.full(n_wp, 1000.0))
        if max_separation is not None and F == 2:
            # |s + dx1 - dx2| ~ |s| + n . (dx1 - dx2) <= w_max at each knot
            s = trajs[0][1:-1] - trajs[1][1:-1]
            nrm = np.linalg.norm(s, axis=1)
            n_k = (s / np.maximum(nrm, 1e-9)[:, None]).reshape(-1)
            rows = np.repeat(np.arange(n_int), 3)
            cols1 = np.arange(3 * n_int)
            A_list.append(sparse.coo_matrix((np.r_[n_k, -n_k], (np.r_[rows, rows], np.r_[cols1, cols1 + 3 * n_int])),
                                            shape=(n_int, n_var)).tocsc())
            l_list.append(np.full(n_int, -1000.0))
            u_list.append(float(max_separation) - nrm)
        if step_bound is not None:
            A_list.append(sparse.identity(n_var, format="csc"))
            l_list.append(np.full(n_var, -float(step_bound)))
            u_list.append(np.full(n_var, float(step_bound)))

        solver = osqp.OSQP()
        solver.setup(P=P, q=g, A=sparse.vstack(A_list), l=np.hstack(l_list), u=np.hstack(u_list), verbose=False)
        res = solver.solve()
        if res.info.status_val not in (osqp.constant("OSQP_SOLVED"), osqp.constant("OSQP_SOLVED_INACCURATE")):
            continue
        dx = res.x.reshape(F, n_int, 3)
        for f in range(F):
            trajs[f][1:-1] += dx[f]
    return np.stack(trajs, axis=0)


def resample(P: np.ndarray, n: int) -> np.ndarray:
    """``n`` points equally spaced in arc length along the polyline ``P``."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-12:
        return np.repeat(P[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])])
