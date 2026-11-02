"""The 1 kHz realisation of the planner's velocity (paper Sec. II-D on the FR3).

Plans arrive at ``plan_freq`` (100 Hz) with a velocity ``z = (qdot, wdot)``
that the QP changed by at most ``a plan_dt`` since the previous plan (its
acceleration bounds, ``hrg.planner``), with a torque ``M a + g + C v`` within
the FR3's and its change ``M (a - a_prev)`` within the FR3's torque rate over
the period (``RobotModel.torque_bounds``). The control thread turns that
sequence into a continuous profile and the torque::

    qd_cmd(t)  : linear ramp from the previous plan's value to the new one over
                 plan_dt, so qdd_cmd = (qd_new - qd_prev) / plan_dt is exactly the
                 acceleration the QP decided (piecewise constant)   (VelocityRamp)
    q_des      = q + qd_cmd dt                 the paper: "integrated over one control interval"
    tau        = Kp (q_des - q) + Kd (qd_cmd - qd)                      the paper's PD
               + [M(q) qdd_cmd]                                         inertia feed-forward [FR3]
               + g(q) + C(q, qd) qd + tau_f(qd_cmd) + tau_payload       gravity, Coriolis, friction, load

``[M(q) qdd_cmd]`` moves toward its value at that torque rate (the controller):
a plan's change arrives as a ramp within the period. As a step it reached the
robot rate-limited by libfranka, late, and the Kd term with it: the arm lagged
its plan by 50 ms and joints 1 / 3 shook until a reflex (2026-11-02).

With ``Kp = 0`` (``Kp dt qd_cmd`` is 1 % of the Kd term) the paper's law is a
velocity controller whose loop has the time constant ``M_ii / Kd_i``: 30-60 ms
on the FR3's big joints at the gains that are quiet on the robot. The arm then
reaches a commanded velocity that late, which is the latency the QP horizon
``H`` had to cover, and at speed it is the distance ``v x lag`` by which a
margin is overshot. The inertia feed-forward realises the command's
acceleration by construction and leaves ``Kd`` only the error; the lag falls
to the plan period (measured by ``scripts/diagnose_tracking.py``).

Gripper (``WidthStream``): the 2F-85 is a position servo with a speed setting
(the real one over Modbus, the sim the same way). A width *rate* is realised
as a stream of set-points a little ahead of the integrated width, each with the
**speed set to the commanded rate**, so the fingers move continuously at that
rate instead of racing to each set-point at full speed and waiting for the
next (the stop-and-go that looked like twitching).

Object pose (``PosePredictor``): the perception's pose carried forward at its
estimated velocity between arrivals, so the planner sees a moving object move
smoothly at 100 Hz instead of in 30 Hz steps.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from fr3.robot_platform.gripper import robotiq_2f85 as rq
from hrg.hand import MAX_WIDTH


class VelocityRamp:
    """The commanded velocity between plans: a linear ramp from the value at
    the moment a plan arrived to the plan's value, over ``plan_dt``; held
    once reached (a late plan). ``at(t)`` gives ``(v, a)``."""

    def __init__(self, n: int, plan_dt: float):
        self.plan_dt = max(float(plan_dt), 1e-4)
        self._v_from = np.zeros(n)
        self._v_to = np.zeros(n)
        self._t0 = None

    def new_plan(self, v_to, t: float) -> None:
        self._v_from = self.at(t)[0]
        self._v_to = np.asarray(v_to, dtype=float).copy()
        self._t0 = float(t)

    def at(self, t: float):
        if self._t0 is None:
            return self._v_to.copy(), np.zeros_like(self._v_to)
        s = (float(t) - self._t0) / self.plan_dt
        if s >= 1.0:
            return self._v_to.copy(), np.zeros_like(self._v_to)
        s = max(s, 0.0)
        return self._v_from + s * (self._v_to - self._v_from), (self._v_to - self._v_from) / self.plan_dt

    def stop(self, t: float) -> None:
        """Plans stopped: the command is zero from here (the hold takes over)."""
        self._v_from = np.zeros_like(self._v_to)
        self._v_to = np.zeros_like(self._v_to)
        self._t0 = None


class WidthStream:
    """Width-rate control of the position-servoed gripper.

    Every ``dt`` the reference ``w_ref`` integrates the commanded rate (kept
    within ``max_lead`` of the measured width: no wind-up against a blocked
    finger). Every ``stream_dt`` a set-point goes out: ``w_ref`` plus a lead
    of ``|wd| lead_time`` (at least ``min_lead``) in the direction of motion,
    so the servo is still travelling when the next set-point replaces it, at
    the **speed** ``|wd|`` (the gripper's speed range). When the rate drops
    below ``v_stop`` one final set-point stops the fingers at ``w_ref``.
    ``update`` returns ``(width, speed)`` to send, or ``None``."""

    def __init__(self, stream_dt: float, max_lead: float, lead_time: float | None = None,
                 min_lead: float = 0.001, v_stop: float = 0.002,
                 speed_range=rq.SPEED_RANGE):
        self.stream_dt = float(stream_dt)
        self.max_lead = float(max_lead)
        self.lead_time = 2.0 * self.stream_dt if lead_time is None else float(lead_time)
        self.min_lead = float(min_lead)
        self.v_stop = float(v_stop)
        self.v_min, self.v_max = (float(v) for v in speed_range)
        self.w_ref = None
        self._moving = False
        self._last_t = -np.inf

    def start(self, w_meas: float) -> None:
        """(Re)start from the measured width (nothing streamed yet)."""
        self.w_ref = float(w_meas)
        self._moving = False
        self._last_t = -np.inf

    def update(self, wd: float, w_meas: float, dt: float, t: float):
        if self.w_ref is None:
            self.start(w_meas)
        self.w_ref = float(np.clip(self.w_ref + float(wd) * dt, 0.0, MAX_WIDTH))
        self.w_ref = float(np.clip(self.w_ref, w_meas - self.max_lead, w_meas + self.max_lead))
        if t - self._last_t < self.stream_dt:
            return None
        if abs(wd) >= self.v_stop:
            lead = max(abs(wd) * self.lead_time, self.min_lead)
            target = float(np.clip(self.w_ref + np.sign(wd) * lead, 0.0, MAX_WIDTH))
            self._moving, self._last_t = True, t
            return target, float(np.clip(abs(wd), self.v_min, self.v_max))
        if self._moving:  # came to rest: one set-point where the reference is
            self._moving, self._last_t = False, t
            return self.w_ref, self.v_min
        return None


class PosePredictor:
    """The object pose between measurements: a constant-velocity extrapolation.

    Poses arrive at the perception rate (the sim at 30 Hz, FoundationPose at
    ~24 Hz) and the planner runs at 100 Hz: held between arrivals, a moving
    object's targets jump every 3-4 plans, and each jump is a step in the
    fingertip field that the QP answers at its acceleration bounds (measured:
    the plans rode the bounds twice as often as with the true pose, and the
    arm shook). The translational and angular velocity are estimated from
    consecutive poses (first-order smoothed, ``smoothing``) and the last pose
    is carried forward by them, for at most ``max_extrapolation`` s (then
    held: a lost object does not run away). A resting object has a velocity
    of the pose noise only, i.e. an extrapolation error below that noise."""

    def __init__(self, smoothing: float = 0.5, max_extrapolation: float = 0.1, min_dt: float = 0.005):
        self.smoothing = float(smoothing)
        self.max_extrapolation = float(max_extrapolation)
        self.min_dt = float(min_dt)
        self.T = None
        self.t = -np.inf
        self.v = np.zeros(3)
        self.omega = np.zeros(3)

    def update(self, T: np.ndarray, t: float) -> None:
        T = np.asarray(T, dtype=float)
        if self.T is not None and t - self.t >= self.min_dt:
            dt = t - self.t
            v = (T[:3, 3] - self.T[:3, 3]) / dt
            omega = Rotation.from_matrix(T[:3, :3] @ self.T[:3, :3].T).as_rotvec() / dt
            a = self.smoothing
            self.v = a * v + (1.0 - a) * self.v
            self.omega = a * omega + (1.0 - a) * self.omega
        elif self.T is None:
            self.v[:], self.omega[:] = 0.0, 0.0
        self.T, self.t = T.copy(), float(t)

    def at(self, t: float):
        """The pose extrapolated to ``t`` (``None`` before the first pose)."""
        if self.T is None:
            return None
        dt = float(np.clip(t - self.t, 0.0, self.max_extrapolation))
        T = self.T.copy()
        T[:3, 3] = self.T[:3, 3] + self.v * dt
        T[:3, :3] = Rotation.from_rotvec(self.omega * dt).as_matrix() @ self.T[:3, :3]
        return T

    def reset(self) -> None:
        self.T, self.t = None, -np.inf
        self.v[:], self.omega[:] = 0.0, 0.0
