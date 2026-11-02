"""A smooth pose trajectory for an object carried through the workspace
(``hrg.sim_env``, key ``m``: the object moves like something handed over,
not teleported).

``CarryPath``: the position runs along a C2 cubic spline through the
waypoints (``scipy.interpolate.CubicSpline``, clamped: zero velocity at both
ends) and the yaw along another, both driven by one minimum-jerk time law
``s(t) = 10 r^3 - 15 r^4 + 6 r^5`` (zero velocity and acceleration at the
start and the end), so the object glides through the waypoints without
stopping at them, starts and stops smoothly, and its path and speed are
continuous (C2 in time). The duration is the path length over the cruise
speed, at least ``min_duration``. After the end the pose is held.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


def min_jerk(r: float) -> float:
    r = float(np.clip(r, 0.0, 1.0))
    return 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5


class CarryPath:
    def __init__(self, positions, yaws, R0: np.ndarray, speed: float, min_duration: float = 2.0):
        """``positions`` (K, 3) with the start first, ``yaws`` (K,) rotation
        about the world z applied to the start orientation ``R0`` (3x3),
        ``speed`` m/s along the path."""
        P = np.asarray(positions, dtype=float)
        if len(P) < 2:
            raise ValueError("a carry path needs at least two waypoints")
        u = np.linspace(0.0, 1.0, len(P))
        self._pos = CubicSpline(u, P, axis=0, bc_type="clamped")
        self._yaw = CubicSpline(u, np.asarray(yaws, dtype=float), bc_type="clamped")
        self.R0 = np.asarray(R0, dtype=float)
        uu = np.linspace(0.0, 1.0, 400)
        self.length = float(np.sum(np.linalg.norm(np.diff(self._pos(uu), axis=0), axis=1)))
        self.duration = max(self.length / max(float(speed), 1e-6), float(min_duration))

    def pose(self, t: float):
        """``(position (3,), quaternion wxyz (4,))`` at time ``t`` since the start."""
        s = min_jerk(t / self.duration)
        p = self._pos(s)
        R = Rotation.from_euler("z", float(self._yaw(s))).as_matrix() @ self.R0
        q = Rotation.from_matrix(R).as_quat()  # xyzw
        return np.asarray(p, dtype=float), np.array([q[3], q[0], q[1], q[2]])

    def velocity(self, t: float, h: float = 1e-3) -> np.ndarray:
        """Translational velocity (3,) by central difference."""
        return (self.pose(t + h)[0] - self.pose(t - h)[0]) / (2.0 * h)

    def done(self, t: float) -> bool:
        return t >= self.duration


def random_waypoints(rng: np.random.Generator, start: np.ndarray, box, n: int, yaw0: float, yaw_range: float,
                     min_step: float = 0.12):
    """``n`` waypoints after ``start`` inside ``box`` ``[[x0, x1], [y0, y1],
    [z0, z1]]``, each at least ``min_step`` from the previous one, and yaws
    within ``yaw_range`` of ``yaw0``: ``(positions (n + 1, 3), yaws (n + 1,))``."""
    box = np.asarray(box, dtype=float)
    P, Y = [np.asarray(start, dtype=float)], [float(yaw0)]
    for _ in range(n):
        for _try in range(100):
            p = rng.uniform(box[:, 0], box[:, 1])
            if np.linalg.norm(p - P[-1]) >= min_step:
                break
        P.append(p)
        Y.append(yaw0 + float(rng.uniform(-yaw_range, yaw_range)))
    return np.asarray(P), np.asarray(Y)
