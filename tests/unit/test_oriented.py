"""The oriented reach, the upright lift and the low level's friction feed-forward
(docs/superpowers/specs/2026-10-28-oriented-reach-design.md): hand orientations
in the object frame, the grasp chosen in joint space, the straight final
approach, the upright goal, the held object vs the table, normal-aligned rim
pinches, friction."""

import sys

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from hrg.fields import FingertipPathField
from hrg.friction import fit_friction, friction_torque, sweep_profile
from hrg.hand import N_Z, fingertips_local, pad_dz, to_world
from hrg.qp import DistanceRows
from hrg.targets import SWAP, FingertipTargets, pair_misalignment, rim_pinch_pairs

GOLD = "assets/objects/gold_bowl/"
HOME = np.array([0.0, 0.0, 0.0, -1.5707963, 0.0, 1.5707963, 0.0])


@pytest.fixture(scope="module")
def gold():
    sys.path.insert(0, "scripts")
    from rollout import make_planner

    return make_planner("gold_bowl", "path")


def _angle(R_a, R_b):
    return float(np.linalg.norm(Rotation.from_matrix(R_a @ R_b.T).as_rotvec()))


def _from_above(tg, usable=None):
    """Per pair the (usable) direction closest to the object's -z, -1 if none."""
    ok = tg.approach_ok if usable is None else usable
    score = np.where(ok, -tg.dirs()[:, :, 2], -np.inf)
    return np.where(ok.any(axis=1), np.argmax(score, axis=1), -1)


def _placed(obj, rot=np.eye(3), xy=(0.5, 0.0)):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:2, 3] = xy
    T[2, 3] = obj.resting_height(rot)
    return T


# ---------------------------------------------------------------- target data
def test_hand_orientation_puts_the_fingertips_on_their_targets():
    """``R_grasp`` (object frame, rotated with the object) is a rotation
    whose closing axis is the pair axis and whose approach is the stored
    direction; the TCP there puts fingertip 1 on x1 and 2 on x2 (either
    finger order)."""
    tg = FingertipTargets.load(GOLD + "fingertip_targets.npz")
    T_obj = np.eye(4)
    T_obj[:3, :3] = Rotation.from_rotvec([0.4, -0.7, 1.1]).as_matrix()
    T_obj[:3, 3] = [0.5, 0.1, 0.05]
    k = _from_above(tg)
    R = tg.grasp_rotations(T_obj, np.arange(len(tg)), k)
    w = tg.transformed(T_obj)
    for j in range(0, len(tg), 17):
        assert np.allclose(R[j].T @ R[j], np.eye(3), atol=1e-9) and np.linalg.det(R[j]) > 0
        assert np.allclose(R[j][:, 2], T_obj[:3, :3] @ tg.dirs()[j, k[j]], atol=1e-6)
        for swap, (a, b) in ((False, (w.x1[j], w.x2[j])), (True, (w.x2[j], w.x1[j]))):
            Rs = R[j] @ SWAP if swap else R[j]
            width = float(np.linalg.norm(b - a))
            T = np.eye(4)
            T[:3, :3] = Rs
            T[:3, 3] = 0.5 * (a + b) - Rs[:, 2] * pad_dz(width)
            P, _ = fingertips_local(width)
            assert np.allclose(to_world(T, P), [a, b], atol=1e-6)


def test_upright_bowl_can_be_approached_from_above_along_its_wall():
    """On the gold bowl's slanted wall the rim pinches lie along the wall
    normal (22 deg), so every pair has an admissible approach from above
    along the wall, tilted ~22-30 deg."""
    tg = FingertipTargets.load(GOLD + "fingertip_targets.npz")
    k = _from_above(tg)
    z = tg.dirs()[np.arange(len(tg)), k]
    tilt = np.degrees(np.arccos(-z[:, 2]))
    assert np.all(tilt < 35.0), tilt.max()
    assert np.degrees(tg.misalign).max() < 5.0  # antipodal on the slanted wall


def test_rim_pinches_follow_the_wall_normal_on_a_slanted_wall():
    """A cone-shaped wall (gold bowl, 22 deg from vertical): a horizontal
    pinch is 22 deg off the wall normal, the normal-aligned one a few deg
    (the inner and outer skins of make_cup are not quite parallel)."""
    from scripts.prepare_objects import make_cup

    mesh = make_cup(top=0.14, bottom=0.10, height=0.05, wall=0.002)
    p1, p2 = rim_pinch_pairs(mesh, depths=(0.015,), n_azimuth=12)
    mis = np.degrees(pair_misalignment(mesh, p1, p2))
    assert mis.min() < 5.0 and mis.max() > 18.0  # both kinds, 12 hits each
    assert np.sum(mis < 5.0) == 12


def test_usable_needs_the_swept_approach_clear_of_the_table():
    """A direction is usable only if the hand stays above the table at the
    pre-grasp and swept back along the approach: a bowl lying upside down
    cannot be approached from its own 'above' (through the table)."""
    tg = FingertipTargets.load(GOLD + "fingertip_targets.npz")
    up = np.eye(4)
    flipped = np.eye(4)
    flipped[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    flipped[2, 3] = 0.05
    k_up = _from_above(tg, tg.usable(up, 0.0, 0.005, 0.10))
    k_fl = _from_above(tg, tg.usable(flipped, 0.0, 0.005, 0.10))
    z_fl = np.einsum("ij,nj->ni", flipped[:3, :3], tg.dirs()[np.arange(len(tg)), np.maximum(k_fl, 0)])
    # upright: the deepest rim pinches put the fingertips into the table, the others are usable
    mid_z = 0.5 * (tg.p1[:, 2] + tg.p2[:, 2])
    assert np.any(k_up >= 0) and mid_z[k_up >= 0].min() > mid_z[k_up < 0].max()
    # upside down: whatever is left approaches from above in the world, never from below
    assert np.all(z_fl[k_fl >= 0, 2] < 0.0)


def test_targets_file_round_trip(tmp_path):
    tg = FingertipTargets.load(GOLD + "fingertip_targets.npz", approach_axis=[1.0, 0.0, 0.0])
    assert np.allclose(tg.approach_axis, [1, 0, 0])
    tg.save(str(tmp_path / "t.npz"))
    tg2 = FingertipTargets.load(str(tmp_path / "t.npz"))
    for k in ("x1", "x2", "p1", "p2", "approach_ok", "approach_dirs", "misalign"):
        assert np.allclose(getattr(tg, k), getattr(tg2, k))


# ---------------------------------------------------------------- the grasp, chosen in joint space
def test_the_grasp_is_the_nearest_reachable_configuration(gold):
    """Of every usable (pair, direction, finger order) the one whose IK
    configuration is nearest the current joints (plus the preference cost)
    is taken: here, against the same pair's other finger order, joint 7 turns
    less than half a turn."""
    p, obj = gold
    T = _placed(obj)
    p.reset()
    tg = p.targets.transformed(T)
    T_tcp = p.tcp_pose(HOME)
    j, k, swap, R_star, q_g = p._select_oriented(HOME, T_tcp, p.robot.jacobian(HOME, p.robot.data()), T, tg)
    assert q_g is not None
    assert _angle(p.tcp_pose(q_g)[:3, :3], R_star) < np.radians(5.0)  # the configuration is that grasp
    other = p._ik_cache.get((j, k, not swap))
    if other is not None and other["q"] is not None:
        assert np.linalg.norm(q_g - HOME) <= np.linalg.norm(other["q"] - HOME)
    assert abs(q_g[6] - HOME[6]) <= np.pi / 2


def test_no_reachable_grasp_is_reported_not_faked(gold):
    """An object beyond the arm's reach: no candidate's IK converges, the
    plan is ``None`` (the controller holds), and it stays ``None``."""
    p, obj = gold
    T = _placed(obj, xy=(1.6, 0.0))
    p.reset()
    assert p.step(HOME, 0.085, T) is None
    assert p.step(HOME, 0.085, T) is None


# ---------------------------------------------------------------- final approach
def test_final_approach_is_straight_along_the_approach_axis():
    """Inside the corridor (the fingertips' midpoint near the approach axis,
    whatever the opening) the fingertips head straight for the targets and
    the distance to the pre-grasp is 0; outside, they head for the pre-grasp
    points."""
    from hrg.sdf import HalfSpaceSDF

    f = FingertipPathField(approach_dist=0.10, delta_x=0.05, num_knots=20)
    table = [(HalfSpaceSDF(0.0), 0.01)]
    z = np.array([0.0, 0.0, -1.0])
    x_tgt = np.array([[0.5, -0.02, 0.05], [0.5, 0.02, 0.05]])
    x = x_tgt - 0.06 * z + np.array([[0.0, -0.02, 0.0], [0.0, 0.02, 0.0]])  # 6 cm up, opened wider
    v, _ = f(x, x_tgt, table, key=1, approach=z)
    d = (x_tgt - x) / np.linalg.norm(x_tgt - x, axis=1, keepdims=True)
    assert np.allclose(v / np.linalg.norm(v, axis=1, keepdims=True), d) and f.h_pre == 0.0
    f.reset()
    x_far = x_tgt - 0.10 * z + np.array([0.08, 0.0, 0.05])  # beside and above the pre-grasp
    v, _ = f(x_far, x_tgt, table, key=1, approach=z)
    x_pre = x_tgt - 0.10 * z
    assert f.h_pre == pytest.approx(np.linalg.norm(x_far - x_pre))
    assert np.all(np.einsum("fi,fi->f", v, x_pre - x_far) > 0.0)  # toward the pre-grasp


def test_oriented_reach_comes_from_above_with_the_hand_turned(gold):
    """Kinematic rollout, upright gold bowl, from the ready pose: converges,
    the hand within a few degrees of R*, joint 7 turned at most 90 deg, and
    the grasp chosen once (it may change while the first plans search)."""
    p, obj = gold
    T = _placed(obj)
    p.reset()
    q, w = HOME.copy(), 0.085
    chosen = []
    for _ in range(800):
        r = p.step(q, w, T)
        chosen.append((r.target, r.info["approach"], r.swapped))
        if np.all(np.linalg.norm(r.x - r.x_tgt, axis=1) < 0.01) and r.info["rot_err"] < np.radians(10.0):
            break
        q, w = q + 0.01 * r.qd, float(np.clip(w + 0.01 * r.wd, 0.0, 0.085))
    else:
        pytest.fail("did not converge")
    R_tcp = p.tcp_pose(q)[:3, :3]
    assert np.degrees(_angle(R_tcp, r.R_star)) < 5.0
    assert R_tcp[2, 2] < -0.8  # pointing down
    assert abs(q[6] - HOME[6]) <= np.pi / 2
    assert len(set(chosen[10:])) == 1


# ---------------------------------------------------------------- after the grasp
def test_upright_goal_untilts_the_held_object_and_clears_the_table(gold):
    p, obj = gold
    T_tcp = np.eye(4)
    T_tcp[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()  # pointing down
    T_tcp[:3, 3] = [0.5, 0.0, 0.10]
    tilt = Rotation.from_euler("x", 50, degrees=True).as_matrix()
    T_obj = np.eye(4)
    T_obj[:3, :3] = tilt
    T_obj[:3, 3] = [0.5, 0.05, 0.06]
    T_tcp_obj = np.linalg.inv(T_tcp) @ T_obj
    P = np.asarray(obj.mesh.vertices)
    T_up, tilt = p.upright_goal(T_tcp, T_tcp_obj, P, clearance=0.03)
    assert np.degrees(tilt) == pytest.approx(50.0)
    R_obj_up = T_up[:3, :3] @ T_tcp_obj[:3, :3]
    assert np.isclose(R_obj_up[2, 2], 1.0, atol=1e-9)  # upright
    # the smallest tilt: the object's yaw (its x axis' heading) is not spun
    x0, x1 = T_obj[:3, 0], R_obj_up[:, 0]
    assert np.degrees(np.arccos(np.clip(x0[:2] @ x1[:2] / np.linalg.norm(x0[:2]) / np.linalg.norm(x1[:2]), -1, 1))) < 1.0
    low = (P @ R_obj_up.T + T_up[:3, :3] @ T_tcp_obj[:3, 3] + T_up[:3, 3])[:, 2].min()
    assert low >= 0.03 - 1e-9
    assert p.upright_goal(T_tcp, np.linalg.inv(T_tcp) @ _placed(obj), P, 0.03)[1] < 1e-6  # already upright
    T_fix, _ = p.upright_goal(T_tcp, T_tcp_obj, P, 0.03, position=[0.4, 0.1, 0.5])
    assert np.allclose(T_fix[:3, 3], [0.4, 0.1, 0.5])  # the goal position is kept (high enough)


def test_held_object_rows_are_the_table_distance_gradient(gold):
    """Rows keeping a held object's points above the table: ``d`` and its
    derivative in the joints (finite differences)."""
    p, _ = gold
    q = np.array([0.2, 0.5, -0.1, -2.0, 0.1, 2.4, 0.3])
    P_tcp = np.array([[0.03, 0.0, 0.05], [-0.04, 0.02, 0.08], [0.0, -0.05, 0.02]])
    data = p.robot.data()

    def rows_at(qq):
        rows = DistanceRows(N_Z, 1.0, 1e9)
        p.held_rows(p.robot.pose(qq, data), p.robot.jacobian(qq, data), P_tcp, 0.0, rows, reach=10.0)
        return -np.array(rows.l), np.array(rows.A)[:, :7]  # H = 1, eps = 0: lower = -d

    d0, A = rows_at(q)
    h = 1e-6
    fd = np.column_stack([(rows_at(q + h * np.eye(7)[k])[0] - d0) / h for k in range(7)])
    assert np.allclose(A, fd, atol=1e-5)


# ---------------------------------------------------------------- friction
def test_friction_feedforward_is_zero_at_rest_and_capped():
    c, v = np.array([1.0, 0.5]), np.array([0.2, 0.1])
    assert np.allclose(friction_torque(np.zeros(2), 0.5, c, v, 0.02, [1, 1]), 0.0)
    tau = friction_torque(np.array([0.5, -0.5]), 0.5, c, v, 0.02, [0.4, 1.0])
    assert tau[0] == pytest.approx(0.4) and tau[1] < 0.0  # capped; same sign as the command
    assert np.allclose(friction_torque(np.array([0.5, -0.5]), 0.0, c, v, 0.02, [1, 1]), 0.0)  # off


def test_friction_fit_separates_friction_from_model_error():
    v = np.array([0.05, 0.1, 0.2, 0.3])
    fc, fv, off = 0.8, 0.3, 0.25  # a constant gravity-model error rides on both directions
    fc_, fv_, f, o = fit_friction(v, fc + fv * v + off, -(fc + fv * v) + off)
    assert fc_ == pytest.approx(fc) and fv_ == pytest.approx(fv) and np.allclose(o, off)


def test_friction_sweep_covers_the_range_at_constant_speeds():
    q, qd, lab = sweep_profile(0.3, 0.2, [0.05, 0.2], dt=0.001)
    assert q.min() == pytest.approx(0.1) and q.max() == pytest.approx(0.5) and q[-1] == pytest.approx(0.3)
    assert np.allclose(np.gradient(q, 0.001)[1:-1], qd[1:-1], atol=2e-3)
    for k, v in ((1, 0.05), (-1, -0.05), (2, 0.2), (-2, -0.2)):
        assert np.allclose(qd[lab == k], v)



def test_grav_comp_friction_compensation_stays_below_friction():
    """Hand-guiding friction compensation (measured velocity): Coulomb only,
    scale < 1, so it never exceeds the friction it opposes; scale >= 1 is
    refused."""
    from omegaconf import OmegaConf

    from fr3.controller.GravCompController import GravCompController

    cfg = OmegaConf.load("configs/robot/grav_comp.yaml")
    g = GravCompController(cfg)
    assert np.all(g._fr_scale > 0.0) and np.all(g._fr_scale < 1.0)
    tau = friction_torque(np.full(7, 1.0), g._fr_scale, g._fr_coulomb, np.zeros(7), g._fr_vs, g._fr_max)
    assert np.all(tau <= g._fr_coulomb + 1e-12)  # below the measured Coulomb friction at any speed
    cfg.friction.scale = 1.0
    with pytest.raises(ValueError):
        GravCompController(cfg)
