"""The Franka Research 3's joint limits, as the robot enforces them.

From libfranka 0.17.0 (``include/franka/rate_limiting.h``, the version
panda-py is built on here) and Franka's FCI documentation ("Robot and
interface specifications", FR3).

The joint **velocity** limit depends on the joint's position and the
direction of motion: it is the speed from which the joint can still stop
before its position limit at the deceleration ``ddq_dec``, capped at the
nominal ``DQ_MAX``::

    upper(q) = min(DQ_MAX,  max(0, -dq_offset + sqrt(2 ddq_dec (q_max - q))))
    lower(q) = max(-DQ_MAX, min(0,  dq_offset - sqrt(2 ddq_dec (q - q_min))))

Measured joint velocities outside ``[lower(q), upper(q)]`` stop the robot
(reflex ``joint_velocity_violation``). Near the ends of a joint's range the
limit is far below ``DQ_MAX``: joint 4 at -2.68 rad may move toward its lower
end at only 1.42 rad/s.

In torque control the commanded torque is limited to ``TAU_MAX`` and its rate
to ``TAU_RATE_MAX`` (libfranka rate-limits faster changes, and so does
``run_hardware.py``): a torque step arrives as a ramp of 1 Nm per ms.
"""

from __future__ import annotations

import numpy as np

TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])  # Nm
TAU_RATE_MAX = 1000.0  # Nm/s, every joint (libfranka kMaxTorqueRate): faster torque changes are rate-limited
DQ_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])  # rad/s, nominal
DDQ_MAX = np.full(7, 10.0)  # rad/s^2 (libfranka kMaxJointAcceleration; FCI docs)

# libfranka 0.17.0 computeUpper/LowerLimitsJointVelocity: 2 ddq_dec, q_max / q_min, dq_offset
_TWO_DEC = np.array([12.0, 5.17, 7.0, 8.0, 34.0, 11.0, 34.0])
_Q_MAX = np.array([2.75010, 1.79180, 2.90650, -0.1458, 2.81010, 4.52050, 3.01960])
_Q_MIN = np.array([-2.75010, -1.79180, -2.90650, -3.04810, -2.81010, 0.54092, -3.01960])
_DQ_OFFSET = np.array([0.30, 0.20, 0.20, 0.30, 0.35, 0.35, 0.35])


def joint_velocity_limits(q) -> tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` joint velocity limits (rad/s) at joints ``q`` (7,)."""
    q = np.asarray(q, dtype=float)
    upper = np.minimum(DQ_MAX, np.maximum(0.0, -_DQ_OFFSET + np.sqrt(np.maximum(0.0, _TWO_DEC * (_Q_MAX - q)))))
    lower = np.maximum(-DQ_MAX, np.minimum(0.0, _DQ_OFFSET - np.sqrt(np.maximum(0.0, _TWO_DEC * (q - _Q_MIN)))))
    return lower, upper
