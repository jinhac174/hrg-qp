"""Smooth references for the hand (docs/method.md, "Smoothness").

The fields say where the hand should go; nothing in them says how the velocity
gets there. A saturated field steps from rest to top speed, and the QP, bounded
only by what the arm can barely do, rides those bounds: the hand accelerates at
0.6 g, the torque goes to its limit, and the arm brakes on a corner. These two
give the velocity a shape the arm follows with torque to spare.

``PoseProfile``: the pose reference after the grasp. A minimum-jerk motion
(Hoff & Arbib's quintic, re-planned from the reference's own position, velocity
and acceleration every step) from where the hand is to a goal that may move: zero
velocity and acceleration at both ends, no corner where it starts or stops,
continuous when the goal changes, and it cannot overshoot a goal that stays put.

``TaskAccelerationLimit``: the reach's fingertip velocities. The field may
turn or reverse from one plan to the next (the object moved), so the velocity
asked of the hand moves toward the field's at a bounded acceleration and
jerk, from what the last plan actually asked (nothing winds up while the QP
holds the hand back).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Minimum-jerk motion over distance d in time T: peak speed, acceleration and jerk
_V_PEAK, _A_PEAK, _J_PEAK = 1.875, 5.7735, 60.0


def min_jerk_duration(d: float, v_max: float, a_max: float, j_max: float) -> float:
    """The shortest time for a minimum-jerk motion of length ``d`` within the limits."""
    d = abs(float(d))
    return max(_V_PEAK * d / v_max, np.sqrt(_A_PEAK * d / a_max), np.cbrt(_J_PEAK * d / j_max))


def quintic_step(x: np.ndarray, v: np.ndarray, a: np.ndarray, g: np.ndarray, T: float, dt: float):
    """``(x, v, a)`` after ``dt`` of the quintic from them to rest at ``g`` in ``T``."""
    if T <= dt:
        return g.copy(), np.zeros_like(v), np.zeros_like(a)
    d, vT, aT2 = g - x, v * T, a * T * T
    c3 = 10.0 * d - 6.0 * vT - 1.5 * aT2
    c4 = -15.0 * d + 8.0 * vT + 1.5 * aT2
    c5 = 6.0 * d - 3.0 * vT - 0.5 * aT2
    s = dt / T
    x1 = x + s * (vT + s * (0.5 * aT2 + s * (c3 + s * (c4 + s * c5))))
    v1 = (vT + s * (aT2 + s * (3.0 * c3 + s * (4.0 * c4 + s * 5.0 * c5)))) / T
    a1 = (aT2 + s * (6.0 * c3 + s * (12.0 * c4 + s * 20.0 * c5))) / (T * T)
    return x1, v1, a1


class PoseProfile:
    """The reference ``(T_ref, v_ref, omega_ref)``, world frame, ``step``-ped once
    per plan toward a goal pose. ``T_min``: the shortest re-plan, which is also
    what a small change of the goal (pose noise) is smoothed over."""

    def __init__(self, v_max: float, a_max: float, j_max: float, w_max: float, wd_max: float, wdd_max: float,
                 T_min: float = 0.25):
        self.lim = (float(v_max), float(a_max), float(j_max))
        self.rot_lim = (float(w_max), float(wd_max), float(wdd_max))
        self.T_min = float(T_min)
        self.T = None  # the time left
        self.goal = None  # the goal pose of the last step
        self.p, self.v, self.a = np.zeros(3), np.zeros(3), np.zeros(3)
        self.R, self.om, self.al = np.eye(3), np.zeros(3), np.zeros(3)

    @property
    def started(self) -> bool:
        return self.T is not None

    def start(self, T_pose: np.ndarray, v, omega) -> None:
        """From the hand's pose and velocity."""
        self.p, self.R = T_pose[:3, 3].copy(), T_pose[:3, :3].copy()
        self.v, self.om = np.array(v, dtype=float), np.array(omega, dtype=float)
        self.a, self.al = np.zeros(3), np.zeros(3)
        self.T, self.goal = 0.0, T_pose.copy()

    def step(self, T_goal: np.ndarray, dt: float):
        """One plan toward ``T_goal``: returns ``(T_ref (4, 4), v_ref, omega_ref)``."""
        # Only a move of the goal buys time: the time for the distance left would be
        # longer than the time left all through the slow start, and stretch every motion.
        moved = np.linalg.norm(T_goal[:3, 3] - self.goal[:3, 3])
        turned = np.linalg.norm(Rotation.from_matrix(T_goal[:3, :3] @ self.goal[:3, :3].T).as_rotvec())
        self.goal = T_goal.copy()
        self.T = max(self.T - dt, 0.0)
        if moved > 0.0 or turned > 0.0:  # (a goal that stays put: the time runs out, exactly on it)
            self.T = max(self.T, self.T_min, min_jerk_duration(moved, *self.lim),
                         min_jerk_duration(turned, *self.rot_lim))
        e = Rotation.from_matrix(T_goal[:3, :3] @ self.R.T).as_rotvec()
        self.p, self.v, self.a = quintic_step(self.p, self.v, self.a, T_goal[:3, 3], self.T, dt)
        turn, self.om, self.al = quintic_step(np.zeros(3), self.om, self.al, e, self.T, dt)
        self.R = Rotation.from_rotvec(turn).as_matrix() @ self.R
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.p
        return T, self.v.copy(), self.om.copy()


class TaskAccelerationLimit:
    """Moves a set of task velocities (``(n, 3)``, a rigid pair of fingertips: one
    factor for all, so they keep their relative motion) toward the demanded ones
    within ``a_max`` and ``j_max``: the demand is approached at the rate from which
    the jerk limit can still end the acceleration exactly on it (``sqrt(2 j e)``)."""

    def __init__(self, a_max: float, j_max: float, dt: float):
        self.a_max, self.j_max, self.dt = float(a_max), float(j_max), float(dt)
        self.a = None

    def reset(self) -> None:
        self.a = None

    def __call__(self, v_des: np.ndarray, v_now: np.ndarray) -> np.ndarray:
        """``v_now``: the velocity the last plan asked of these points now."""
        if self.a is None:
            self.a = np.zeros_like(v_des)
        e = v_des - v_now
        n = float(np.max(np.linalg.norm(e, axis=1)))
        step = self.j_max * self.dt
        # sqrt(2 j e), of the discrete steps: the acceleration lands on zero as the error does
        a_lim = min(self.a_max, float(np.sqrt(2.0 * self.j_max * n + 0.25 * step * step)) - 0.5 * step)
        a_des = np.zeros_like(e) if n < 1e-9 else e / n * a_lim
        da = a_des - self.a
        m = float(np.max(np.linalg.norm(da, axis=1)))
        self.a = self.a + (da if m <= step else da * (step / m))
        if n <= float(np.max(np.linalg.norm(self.a, axis=1))) * self.dt:  # there within the step
            self.a = e / self.dt
            return v_des
        return v_now + self.a * self.dt
