"""Smoothness: the references (``hrg.profile``), the arm's self-motion and the
low level's timing. docs/method.md, "Smoothness"."""

import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hrg.hand import MAX_WIDTH, N_ARM
from hrg.lowlevel import VelocityRamp
from hrg.profile import PoseProfile, TaskAccelerationLimit, min_jerk_duration, quintic_step
from hrg.study import START_POSES

DT = 0.01


@pytest.fixture(scope="module")
def gold():
    sys.path.insert(0, "scripts")
    from rollout import make_planner, placements

    p, o = make_planner("gold_bowl", "path")
    return p, o, placements(1, 1, o)[0]


def _pose(z=0.0, angle=0.0):
    T = np.eye(4)
    T[2, 3] = z
    T[:3, :3] = Rotation.from_euler("x", angle).as_matrix()
    return T


def _profile():
    return PoseProfile(0.5, 3.0, 40.0, 2.0, 8.0, 100.0)


# ---------------------------------------------------------------- the pose reference
def test_quintic_step_reaches_the_goal_at_rest_exactly_on_time():
    x, v, a = np.zeros(3), np.zeros(3), np.zeros(3)
    g = np.array([0.15, 0.0, 0.0])
    T = 0.6
    for k in range(60):
        x, v, a = quintic_step(x, v, a, g, T - k * DT, DT)
    np.testing.assert_allclose(x, g, atol=1e-9)
    np.testing.assert_allclose(v, 0.0, atol=1e-9)


def test_lift_profile_is_bell_shaped_within_its_limits_and_never_overshoots():
    pr = _profile()
    pr.start(_pose(), np.zeros(3), np.zeros(3))
    goal = _pose(0.15)
    z, vz = [], []
    for _ in range(120):
        T, v, _ = pr.step(goal, DT)
        z.append(T[2, 3])
        vz.append(v[2])
    z, vz = np.array(z), np.array(vz)
    az = np.diff(vz) / DT
    assert z[-1] == pytest.approx(0.15, abs=1e-4)
    assert z.max() <= 0.15 + 1e-9  # no overshoot
    assert vz.max() <= 0.5 + 1e-9 and vz.max() > 0.4  # uses the speed it is given
    assert np.abs(az).max() <= 3.0 and np.abs(np.diff(az) / DT).max() <= 40.0 * 1.2
    assert vz[0] < 0.03 and abs(vz[-1]) < 1e-3  # no step at either end
    assert np.argmax(z >= 0.149) * DT < 0.75  # and it is not slow


def test_the_time_is_not_stretched_while_the_motion_starts_slowly():
    """The time left comes from the goal's moves. From the distance left, it would be
    longer than the time left all through the slow start, and the motion would stretch."""
    pr = _profile()
    pr.start(_pose(), np.zeros(3), np.zeros(3))
    goal = _pose(0.15)
    expected = min_jerk_duration(0.15, *pr.lim)
    z = [pr.step(goal, DT)[0][2, 3] for _ in range(int(expected / DT) + 1)]
    assert z[-1] > 0.14


def test_a_goal_that_moves_keeps_the_reference_continuous():
    """The upright turn re-aims the goal in mid-motion: no jump of the velocity."""
    pr = _profile()
    pr.start(_pose(), np.zeros(3), np.zeros(3))
    v_prev = np.zeros(3)
    peak = 0.0
    for k in range(150):
        goal = _pose(0.15 if k < 30 else 0.05, 0.0 if k < 50 else 0.4)
        _, v, _ = pr.step(goal, DT)
        peak = max(peak, float(np.linalg.norm(v - v_prev)))
        v_prev = v
    assert peak <= 3.0 * DT * 1.5  # an acceleration of at most 1.5 a_max


def test_pose_noise_on_the_goal_barely_moves_the_reference():
    rng = np.random.default_rng(0)
    pr = _profile()
    pr.start(_pose(), np.zeros(3), np.zeros(3))
    z = []
    for _ in range(200):
        goal = _pose(rng.normal(0.0, 0.001))  # 1 mm, every plan
        z.append(pr.step(goal, DT)[0][2, 3])
    assert np.std(z) < 0.0005


# ---------------------------------------------------------------- the velocity limit
def test_a_reversed_field_reverses_within_the_acceleration_and_the_jerk():
    lim = TaskAccelerationLimit(6.5, 150.0, DT)
    v = np.array([[0.5, 0.0, 0.0]] * 2)
    want = -v
    out, acc = [], []
    prev = v.copy()
    for _ in range(40):
        v = lim(want, v)
        acc.append((v - prev) / DT)
        prev = v.copy()
        out.append(v[0, 0])
    acc = np.array(acc)[:, 0, 0]
    assert np.abs(acc).max() <= 6.5 + 1e-9
    assert np.abs(np.diff(acc)).max() <= 1.3 * 150.0 * DT  # (the last step lands on the demand)
    assert np.all(np.diff(out) <= 1e-9) and np.all(acc <= 1e-9)  # it turns without ringing
    assert out[-1] == pytest.approx(-0.5, abs=1e-6)


def test_the_limit_does_not_wind_up_while_the_hand_is_held_back():
    """Measured from what the last plan asked: however long the QP held the hand at
    rest, the velocity asked next is one step from rest, not the field's."""
    lim = TaskAccelerationLimit(6.5, 150.0, DT)
    for _ in range(100):
        out = lim(np.array([[0.5, 0.0, 0.0]]), np.zeros((1, 3)))
    assert np.linalg.norm(out) <= 6.5 * DT + 1e-9


def test_a_small_change_is_taken_at_once():
    lim = TaskAccelerationLimit(6.5, 150.0, DT)
    v = np.array([[0.3, 0.0, 0.0]])
    for _ in range(10):
        lim(v + 1e-3, v)
    np.testing.assert_allclose(lim(v + 1e-3, v), v + 1e-3)


# ---------------------------------------------------------------- the self-motion
def _null_dir(p, q):
    d = p.robot.data()
    return np.linalg.svd(p.robot.jacobian(q, d)[:, :N_ARM])[2][-1]


def test_a_straight_lift_leaves_the_arms_self_motion_alone(gold):
    """The plans of a lift have no business in the motion that does not move the hand
    (joints 1 and 3 against each other, the arm's lightest). 2026-11-02: +-0.5 rad/s in
    it on the robot's lifts, and the same on a simulated one, until the task priced it."""
    p, _, T_obj = gold
    q, w = np.array(START_POSES["center_up"]), MAX_WIDTH
    q[1] += 0.5
    p.reset()
    T_goal = p.tcp_pose(q)
    T_goal[2, 3] += 0.15
    worst = 0.0
    for _ in range(100):
        qd = p.lift_step(q, w, T_goal, 0.5, 2.0, 8.0, 5.0, 0.5)
        worst = max(worst, abs(float(_null_dir(p, q) @ qd)))
        q = q + qd * DT
    assert worst < 0.05


def test_the_null_task_prices_self_motion_and_not_the_hand(gold):
    p, _, _ = gold
    q = np.array(START_POSES["left_bottom"])
    d = p.robot.data()
    A, v, wgt = p._null_task(p.robot.jacobian(q, d), np.zeros(N_ARM))
    n = _null_dir(p, q)
    U = np.linalg.svd(p.robot.jacobian(q, d)[:, :N_ARM])[2]
    assert wgt > 0.0 and np.linalg.norm(A[:, :N_ARM] @ n) == pytest.approx(1.0, abs=1e-6)
    # the arm's weakest real direction costs next to nothing: pricing it stalled the lift 1 cm short
    assert np.linalg.norm(A[:, :N_ARM] @ U[5]) < 0.1


def test_a_lift_reaches_its_goal(gold):
    """From a pose whose goal lies along the Jacobian's weak direction."""
    p, _, _ = gold
    q, w = np.array(START_POSES["left_bottom"]), MAX_WIDTH
    p.reset()
    T_goal = p.tcp_pose(q)
    T_goal[2, 3] += 0.15
    for _ in range(100):
        q = q + DT * p.lift_step(q, w, T_goal, 0.5, 2.0, 8.0, 5.0, 0.5)
    assert np.linalg.norm(p.tcp_pose(q)[:3, 3] - T_goal[:3, 3]) < 0.002


# ---------------------------------------------------------------- the low level
def test_the_ramp_follows_the_plans_measured_period():
    """Plans every 13 ms: the ramps last until the next plan, no gap without acceleration
    (2 % of the ticks in motion on an idle machine, more on one running the perception)."""
    ramp = VelocityRamp(1, plan_dt=0.01)
    t, gaps, ticks = 0.0, 0, 0
    for k in range(100):
        ramp.new_plan([0.01 * k], t)
        for _ in range(13):
            if k > 50:
                ticks += 1
                gaps += ramp.at(t)[1][0] == 0.0
            t += 0.001
    assert gaps <= 0.05 * ticks
    assert ramp.period == pytest.approx(0.013, abs=5e-4)


def test_the_period_ignores_a_stall_and_never_falls_below_the_plan_period():
    ramp = VelocityRamp(1, plan_dt=0.01)
    ramp.new_plan([0.0], 0.0)
    ramp.new_plan([0.0], 1.0)  # a stall
    assert ramp.period == pytest.approx(0.01)
    for k in range(50):
        ramp.new_plan([0.0], 1.0 + 0.005 * k)
    assert ramp.period == pytest.approx(0.01)


def test_the_payload_fades_in_and_out():
    from omegaconf import OmegaConf

    from hrg.controller import HRGController

    c = HRGController(OmegaConf.load("configs/controller.yaml"))
    q = np.array(START_POSES["center_up"])
    c._cur_time = 0.0
    c._payload_torque(q)
    c._held = (0.5, np.array([0.0, 0.0, 0.0]))
    tau = []
    for k in range(1, 400):
        c._cur_time = k * c._ctrl_dt
        tau.append(c._payload_torque(q).copy())
    tau = np.array(tau)
    full = tau[-1]
    assert np.abs(full).max() > 0.5
    assert np.abs(np.diff(tau, axis=0)).max() <= np.abs(full).max() * 1.2 * c._ctrl_dt / c._held_ramp
    c._held = None  # released: it fades out, it does not vanish
    c._cur_time += c._ctrl_dt
    assert np.abs(c._payload_torque(q)).max() > 0.9 * np.abs(full).max()
    for k in range(400):
        c._cur_time += c._ctrl_dt
        out = c._payload_torque(q)
    assert np.all(out == 0.0)
