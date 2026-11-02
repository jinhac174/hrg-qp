"""Speed: the acceleration-bounded QP (``hrg.planner``, ``hrg.qp``), the
low-level velocity ramp and width stream (``hrg.lowlevel``), and the carried
object's path (``hrg.carry``). docs/method.md, "Speed"."""

import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hrg.carry import CarryPath, min_jerk, random_waypoints
from hrg.hand import MAX_WIDTH, N_ARM, N_Z
from hrg.lowlevel import VelocityRamp, WidthStream
from hrg.qp import DistanceRows, solve_tracking_qp
from hrg.study import START_POSES


@pytest.fixture(scope="module")
def gold():
    sys.path.insert(0, "scripts")
    from rollout import make_planner, placements

    p, o = make_planner("gold_bowl", "path")
    return p, o, placements(1, 1, o)[0]


# ---------------------------------------------------------------- QP
def test_soft_rows_bind_exactly_when_the_box_allows():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(3, N_Z))
    v = np.array([0.1, -0.05, 0.02])
    lb, ub = -np.ones(N_Z), np.ones(N_Z)
    rows = DistanceRows(N_Z, horizon=0.05, sep_rate_max=0.25)
    rows.add([0.01], J[:1], 0.02)  # 1 cm inside a 2 cm margin: open at >= 0.2 m/s
    z = solve_tracking_qp([(J, v, 1.0)], lb, ub, rows, reg=1e-8)
    assert J[0] @ z >= 0.2 - 1e-4  # the linear penalty is exact: the row holds


def test_soft_rows_give_the_fastest_feasible_recovery_instead_of_failing():
    """A row demanding more than the (acceleration) box allows: the hard QP
    is infeasible; the soft one returns the box limit in the row's direction."""
    a = np.zeros(N_Z)
    a[0] = 1.0
    rows = DistanceRows(N_Z, horizon=0.1, sep_rate_max=0.25)
    rows.add([0.0], a[None], 0.02)  # demands z0 >= 0.2
    lb, ub = -0.05 * np.ones(N_Z), 0.05 * np.ones(N_Z)  # the box allows 0.05
    z = solve_tracking_qp([(np.eye(N_Z), np.zeros(N_Z), 1.0)], lb, ub, rows)
    assert z is not None and z[0] == pytest.approx(0.05, abs=1e-4)


# ---------------------------------------------------------------- planner: acceleration bounds
def test_plans_change_by_at_most_the_acceleration_limit(gold):
    p, _, T_obj = gold
    q, w = np.array(START_POSES["left_bottom"]), MAX_WIDTH
    p.reset()
    p.set_velocity(np.zeros(N_ARM))
    prev = np.zeros(N_Z)
    da = p.z_accel * p.plan_dt
    peak = 0.0
    for k in range(150):
        r = p.step(q, w, T_obj)
        z = np.r_[r.qd, r.wd]
        assert np.all(np.abs(z - prev) <= da + 2e-4), k  # (OSQP's tolerance)
        prev = z
        peak = max(peak, np.linalg.norm(r.qd))
        q = q + r.qd * p.plan_dt
        w = float(np.clip(w + r.wd * p.plan_dt, 0.0, MAX_WIDTH))
    assert peak > 0.5  # it does get up to speed


def test_first_plan_from_rest_is_small_and_a_moving_start_is_honoured(gold):
    p, _, T_obj = gold
    q = np.array(START_POSES["center_up"])
    p.reset()
    r0 = p.step(q, MAX_WIDTH, T_obj)  # reset = at rest
    assert np.all(np.abs(r0.qd) <= p.robot.qdd_limit * p.plan_dt + 2e-4)
    p.reset()
    v = np.full(N_ARM, 0.5)
    p.set_velocity(v)
    r1 = p.step(q, MAX_WIDTH, T_obj)
    assert np.all(np.abs(r1.qd - v) <= p.robot.qdd_limit * p.plan_dt + 2e-4)


def test_fast_kinematic_reach_and_lift(gold):
    """With the fast defaults the gold bowl is reached (within the grasp
    trigger) in about a second of perfect tracking, and the lift of 15 cm
    takes well under a second, the whole time under the acceleration limits."""
    p, o, T_obj = gold
    q, w = np.array(START_POSES["left_bottom"]), MAX_WIDTH
    p.reset()
    p.set_velocity(np.zeros(N_ARM))
    for k in range(300):
        r = p.step(q, w, T_obj)
        q = q + r.qd * p.plan_dt
        w = float(np.clip(w + r.wd * p.plan_dt, 0.0, MAX_WIDTH))
        if np.all(np.linalg.norm(r.x - r.x_tgt, axis=1) < 0.01) and r.info["rot_err"] < np.radians(10):
            break
    assert k * p.plan_dt < 1.5, k * p.plan_dt
    for _ in range(20):  # the close: holding still
        p.home_step(q, w, q, 10.0, vmax=1.0)
    T = p.tcp_pose(q)
    T_goal = T.copy()
    T_goal[2, 3] += 0.15
    T_tcp_obj = np.linalg.inv(T) @ T_obj
    held = np.asarray(o.mesh.vertices)[::50] @ T_tcp_obj[:3, :3].T + T_tcp_obj[:3, 3]
    prev = np.zeros(N_ARM)
    for k in range(100):
        qd = p.lift_step(q, w, T_goal, 0.8, 2.0, 8.0, 5.0, 0.5, held=held, held_eps=0.02)
        assert np.all(np.abs(qd - prev) <= p.robot.qdd_limit * p.plan_dt + 2e-4)
        prev = qd
        q = q + qd * p.plan_dt
        if p.tcp_pose(q)[2, 3] - T[2, 3] > 0.14:
            break
    assert k * p.plan_dt < 0.8, k * p.plan_dt


# ---------------------------------------------------------------- low level
def test_velocity_ramp_reaches_each_plan_within_the_period_with_its_acceleration():
    ramp = VelocityRamp(2, plan_dt=0.01)
    ramp.new_plan([0.1, -0.2], t=0.0)
    v, a = ramp.at(0.005)
    np.testing.assert_allclose(v, [0.05, -0.1])
    np.testing.assert_allclose(a, [10.0, -20.0])
    v, a = ramp.at(0.02)  # late next plan: held, no acceleration
    np.testing.assert_allclose(v, [0.1, -0.2])
    np.testing.assert_allclose(a, 0.0)
    ramp.new_plan([0.2, -0.2], t=0.02)
    v, a = ramp.at(0.025)
    np.testing.assert_allclose(v, [0.15, -0.2])
    np.testing.assert_allclose(a, [10.0, 0.0])
    ramp.stop(0.03)
    np.testing.assert_allclose(ramp.at(0.03)[0], 0.0)


def test_width_stream_moves_the_fingers_continuously_at_the_commanded_rate():
    st = WidthStream(stream_dt=0.02, max_lead=0.01)
    st.start(0.085)
    sent = []
    w, t = 0.085, 0.0
    for k in range(1200):  # 1.2 s of closing at 5 cm/s (6 cm: never fully closed)
        out = st.update(-0.05, w, 0.001, t)
        if out is not None:
            sent.append((t, *out))
            w = min(w, out[0] + 0.001)  # the servo follows (a mm behind the set-point)
        t += 0.001
    sent = np.asarray(sent)
    assert 55 <= len(sent) <= 61  # one set-point per stream period
    assert np.allclose(sent[:, 2], 0.05)  # at the commanded speed, not the gripper's maximum
    assert np.all(np.diff(sent[:, 1]) < 0.0)  # the targets march ahead monotonically
    assert sent[0, 1] < 0.085 - 0.001  # ahead of the reference
    # rate drops to zero: one final set-point at the reference, then silence
    out = [st.update(0.0, w, 0.001, t + k * 0.001) for k in range(100)]
    out = [o for o in out if o is not None]
    assert len(out) == 1 and out[0][0] == pytest.approx(st.w_ref)


# ---------------------------------------------------------------- carry
def test_carry_path_is_smooth_and_through_the_box():
    rng = np.random.default_rng(3)
    box = [[0.35, 0.65], [-0.3, 0.3], [0.05, 0.4]]
    P, Y = random_waypoints(rng, np.array([0.5, 0.0, 0.03]), box, 4, 0.0, 1.0)
    assert P.shape == (5, 3) and np.all(P[1:] >= np.asarray(box)[:, 0]) and np.all(P[1:] <= np.asarray(box)[:, 1])
    path = CarryPath(P, Y, np.eye(3), speed=0.3)
    assert path.duration >= path.length / 0.3 - 1e-9
    t = np.linspace(0.0, path.duration, 2000)
    pos = np.array([path.pose(tt)[0] for tt in t])
    np.testing.assert_allclose(pos[0], P[0], atol=1e-9)
    np.testing.assert_allclose(pos[-1], P[-1], atol=1e-9)
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t)
    a = np.diff(v) / np.diff(t)[1:]
    assert v[0] < 1e-3 and v[-1] < 1e-3  # starts and ends at rest
    assert v.max() < 1.0 and np.abs(a).max() < 5.0  # a gentle hand-over
    yaw = [Rotation.from_quat(np.roll(path.pose(tt)[1], -1)).as_euler("xyz")[2] for tt in (0.0, path.duration)]
    assert yaw[0] == pytest.approx(Y[0], abs=1e-9) and yaw[1] == pytest.approx(Y[-1], abs=1e-9)
    assert min_jerk(0.5) == pytest.approx(0.5) and min_jerk(2.0) == 1.0
    assert path.done(path.duration) and not path.done(0.5 * path.duration)


def test_a_joint_past_its_velocity_box_brakes_toward_it(gold):
    """The previous plan moved a joint at 0.3 rad/s; now that joint is at its
    position limit (the velocity box allows nothing further). The next plan
    must decelerate at the acceleration limit (0.3 - 0.1), not pick the far
    edge of the acceleration box (0.3 + 0.1: the bug that drove the arm
    into its limits and its own links)."""
    p, _, _ = gold
    q = np.array(START_POSES["center_up"])
    q[3] = p.robot.q_lower[3]  # joint 4 at its lower limit
    v = np.zeros(N_ARM)
    v[3] = -0.3  # still moving into it
    p.reset()
    p.set_velocity(v)
    lb, ub = p._bounds(q, MAX_WIDTH)
    assert lb[3] == pytest.approx(-0.3 + p.robot.qdd_limit[3] * p.plan_dt)
    assert ub[3] == pytest.approx(lb[3])


# ---------------------------------------------------------------- moving objects
def test_pose_predictor_extrapolates_a_moving_pose_and_holds_a_lost_one():
    from hrg.lowlevel import PosePredictor

    pp = PosePredictor(smoothing=1.0)
    T = np.eye(4)
    for k in range(4):  # 0.3 m/s along x, 1 rad/s about z, at 30 Hz
        T = np.eye(4)
        T[:3, 3] = [0.5 + 0.3 * k / 30, 0.0, 0.1]
        T[:3, :3] = Rotation.from_euler("z", k / 30).as_matrix()
        pp.update(T, k / 30)
    Tp = pp.at(3 / 30 + 0.02)
    np.testing.assert_allclose(Tp[:3, 3], [0.5 + 0.3 * (3 / 30 + 0.02), 0.0, 0.1], atol=1e-9)
    assert Rotation.from_matrix(Tp[:3, :3]).as_euler("xyz")[2] == pytest.approx(3 / 30 + 0.02, abs=1e-9)
    Tl = pp.at(3 / 30 + 1.0)  # lost: held at most max_extrapolation ahead
    np.testing.assert_allclose(Tl[:3, 3], [0.5 + 0.3 * (3 / 30 + 0.1), 0.0, 0.1], atol=1e-9)
    assert pp.at(-1.0) is not None and PosePredictor().at(0.0) is None


def test_approach_goal_is_continuous_and_never_behind_the_hand():
    """The path's end point slides along the approach segment with the
    hand's alignment (no on/off switch), and a hand deeper than the goal
    is not sent back to it."""
    from hrg.fields import FingertipPathField
    from hrg.sdf import HalfSpaceSDF

    f = FingertipPathField(approach_dist=0.10, num_knots=20)
    table = [(HalfSpaceSDF(0.0), 0.01)]
    z = np.array([0.0, 0.0, -1.0])
    x_tgt = np.array([[0.5, -0.02, 0.05], [0.5, 0.02, 0.05]])
    lams = []
    for off in np.linspace(0.0, 0.06, 13):  # 9 cm up (near the pre-grasp), sliding sideways
        x = x_tgt - 0.09 * z + np.array([off, 0.0, 0.0])
        f(x, x_tgt, table, key=1, approach=z)
        lams.append(f.lam)
    assert lams[0] == 1.0 and lams[-1] == pytest.approx(0.1)  # 9 cm up: the depth floor 1 - 0.09 / 0.10
    assert np.all(np.diff(lams) <= 1e-12) and np.max(np.abs(np.diff(lams))) < 0.35  # monotone, no jump
    # 3 cm above the targets but 6 cm off the axis: the goal is at least that deep
    x = x_tgt - 0.03 * z + np.array([0.06, 0.0, 0.0])
    v, paths = f(x, x_tgt, table, key=1, approach=z)
    assert f.lam >= 0.7 - 1e-9
    assert np.all(v[:, 2] <= 1e-9)  # it does not climb back toward the pre-grasp


def test_path_direction_reads_over_a_lookahead():
    from hrg.fields import path_direction

    P = np.zeros((1, 50, 3))
    P[0, :, 0] = np.linspace(0.0, 0.2, 50)  # straight along x
    P[0, 1, 1] = 0.0005  # a half-millimetre wobble of the first interior knot
    d_first = P[0, 1] - P[0, 0]
    assert np.degrees(np.arctan2(d_first[1], d_first[0])) > 5.0  # the raw first segment is off by > 5 deg
    d = path_direction(P, 0.02)[0]
    assert np.degrees(np.arctan2(d[1], d[0])) < 1.5  # over 2 cm it is not
    assert path_direction(P, 10.0)[0][0] == pytest.approx(0.1, abs=1e-3)  # capped at half the path's length


def test_moving_object_rows_demand_the_extra_separation_and_targets_feed_forward(gold):
    """A row with the object closing in at 0.3 m/s along its normal needs
    0.3 m/s more separation rate than the same row at rest; and with the
    object's twist the fingertip fields carry the targets' velocity."""
    a = np.zeros(N_Z)
    a[0] = 1.0
    rows = DistanceRows(N_Z, horizon=0.1, sep_rate_max=0.25)
    rows.add([0.05], a[None], 0.01)  # 4 cm of slack: may approach at 0.4 m/s
    rows.add([0.05], a[None], 0.01, drift=[0.3])  # the object comes at 0.3 m/s: only 0.1 m/s left
    assert rows.l[0] == pytest.approx(-0.4) and rows.l[1] == pytest.approx(-0.1)
    p, _, T_obj = gold
    q = np.array(START_POSES["center_up"])
    p.reset()
    r0 = p.step(q, MAX_WIDTH, T_obj)
    p.reset()
    v = np.array([0.0, 0.3, 0.0])
    r1 = p.step(q, MAX_WIDTH, T_obj, obj_twist=(v, np.zeros(3)))
    # far from the object the QP tracks the fields: the fingertips gain the object's velocity
    from hrg.hand import fingertip_state
    T = p.robot.pose(q, p.robot.data())
    J = p.robot.jacobian(q, p.robot.data())
    _, Jx = fingertip_state(T, J, MAX_WIDTH)
    dv = np.einsum("fij,j->fi", Jx, np.r_[r1.qd - r0.qd, r1.wd - r0.wd])
    assert np.all(dv[:, 1] > 0.05)  # (the acceleration bound from rest caps the first plan)
