"""The method (hrg/, paper arXiv 2509.01044): fingertip kinematics with the
width as a joint, distance functions, targets and their selection (eq. 12),
the fields (eqs. 2-10), path optimisation (eq. 3), the QP (eq. 11), Gamma(q)
and one planning step."""

import glob

import mujoco as mj
import numpy as np
import pinocchio as pin
import pytest
import trimesh

from fr3.robot_platform.gripper import robotiq_2f85 as rq
from fr3.utils.pinocchio.getter import get_fk_link_pose, get_link_Jacobian
from hrg import obstacles, path_opt
from hrg.fields import (
    FingertipPathField,
    grasp_stability_field,
    gripper_width_field,
    linear_fingertip_field,
    quad_decay_speed,
)
from hrg.hand import (
    N_Z,
    PAD_CONTACT_AXES,
    fingertip_state,
    hand_spheres,
    hand_spheres_local,
)
from hrg.objects import GraspObject
from hrg.qp import DistanceRows, solve_tracking_qp
from hrg.sdf import BoxSDF, HalfSpaceSDF, PointCloudSDF, sdf_from_config
from hrg.targets import FingertipTargets, select_target, selection_costs

URDF = "assets/robot/urdf/fr3_robotiq_rigid.urdf"
TCP = "fr3_gripper_tcp"
Q = np.array([0.3, -0.4, 0.2, -2.0, 0.1, 1.8, 0.5])
OBJECTS = sorted(glob.glob("configs/objects/*.yaml"))
BOWL = "assets/objects/blue_bowl/"


@pytest.fixture(scope="module")
def arm():
    m = pin.buildModelFromUrdf(URDF)
    return m, m.createData()


def _tcp(arm, q):
    m, d = arm
    return get_fk_link_pose(q, m, d, TCP), get_link_Jacobian(q, m, d, TCP)


# ---------------------------------------------------------------- gripper
@pytest.mark.parametrize("w", [0.085, 0.06, 0.03, 0.01])
def test_fingertips_match_mujoco_pad_sites(arm, w):
    model = mj.MjModel.from_xml_path("assets/robot/scene_sim.xml")
    data = mj.MjData(model)
    for i in range(7):
        data.qpos[model.joint(f"fr3_joint{i + 1}").qposadr[0]] = Q[i]
    rq.mj_set_width(model, data, w)
    mj.mj_forward(model, data)
    x, _ = fingertip_state(*_tcp(arm, Q), w)
    sites = np.array([data.site(n).xpos for n in rq.PAD_SITES])  # (left, right)
    assert np.allclose(x, sites, atol=5e-4)


def test_hand_point_jacobians_match_finite_differences(arm):
    w, h = 0.05, 1e-6
    T, J = _tcp(arm, Q)
    x0, Jx = fingertip_state(T, J, w)
    c0, Jc, _ = hand_spheres(T, J, w)
    for k in range(N_Z):
        q, ww = Q.copy(), w
        if k < 7:
            q[k] += h
        else:
            ww += h
        T1, J1 = _tcp(arm, q)
        x1, _ = fingertip_state(T1, J1, ww)
        c1, _, _ = hand_spheres(T1, J1, ww)
        assert np.allclose((x1 - x0) / h, Jx[:, :, k], atol=2e-3)
        assert np.allclose((c1 - c0) / h, Jc[:, :, k], atol=2e-3)


def test_finger_spheres_stay_behind_the_pad_faces():
    for w in (0.01, 0.04, 0.085):
        P, _, r, _ = hand_spheres_local(w)
        # Nothing pokes into the gap between the pads within the fingers'
        # length (the palm is between the finger roots, 4 cm further up).
        near = P[:, 2] > -0.035
        assert np.all(np.abs(P[near, 1]) - r[near] >= 0.5 * w - 1e-9)


# ---------------------------------------------------------------- distance functions
def test_point_cloud_sdf_is_nearest_point_distance_in_world():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3)) * 0.05
    sdf = PointCloudSDF(pts)
    T = np.eye(4)
    T[:3, :3] = pin.rpy.rpyToMatrix(0.3, -0.2, 1.0)
    T[:3, 3] = [0.5, -0.1, 0.2]
    sdf.set_pose(T)
    x = rng.normal(size=(50, 3)) * 0.1 + T[:3, 3]
    d, g = sdf.dist_grad(x)
    pw = pts @ T[:3, :3].T + T[:3, 3]
    brute = np.min(np.linalg.norm(x[:, None] - pw[None], axis=2), axis=1)
    assert np.allclose(d, brute)
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0)
    assert np.allclose(sdf.dist(x, cap=0.02), np.minimum(brute, 0.02))
    # gradient = direction of steepest increase (a step can switch the
    # nearest sample, so check most points)
    eps = 1e-5
    assert np.mean(np.isclose(sdf.dist(x + eps * g) - d, eps, atol=1e-7)) > 0.9


def test_box_sdf_is_exact_and_table_signs():
    half, c = np.array([0.025, 0.035, 0.05]), np.array([0.0, 0.0, 0.05])
    box = BoxSDF(half, centre=c)
    T = np.eye(4)
    T[:3, :3] = pin.rpy.rpyToMatrix(0.3, -0.2, 0.7)
    T[:3, 3] = [0.5, 0.1, 0.0]
    box.set_pose(T)
    rng = np.random.default_rng(0)
    p_loc = c + rng.uniform(-1.0, 1.0, (4000, 3)) * (half + 0.04)
    x = p_loc @ T[:3, :3].T + T[:3, 3]
    d, g = box.dist_grad(x)
    # exact distance outside: to a dense sampling of the surface
    surf = trimesh.creation.box(extents=2 * half).sample(200000) + c
    out = np.any(np.abs(p_loc - c) > half, axis=1)
    brute = np.min(np.linalg.norm(p_loc[out][:200, None] - surf[None], axis=2), axis=1)
    assert np.allclose(d[out][:200], brute, atol=1.5e-3)
    # inside: minus the depth below the nearest face
    depth = np.min(half - np.abs(p_loc[~out] - c), axis=1)
    assert np.allclose(d[~out], -depth)
    # gradient: unit, and the finite-difference slope
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0)
    eps = 1e-6
    assert np.mean(np.isclose(box.dist(x + eps * g) - d, eps, atol=1e-9)) > 0.99
    t = HalfSpaceSDF(0.02)
    d, g = t.dist_grad(np.array([[0, 0, 0.05]]))
    assert np.isclose(d[0], 0.03) and np.allclose(g, [[0, 0, 1]])


def test_obstacle_message_round_trip():
    T = np.eye(4)
    T[:3, :3] = pin.rpy.rpyToMatrix(0.0, 0.0, 0.7)
    T[:3, 3] = [0.4, 0.2, 0.1]
    rows = obstacles.encode([T], [[0.05, 0.1, 0.1]])
    (o,) = obstacles.decode(rows)
    assert rows.shape == (1, obstacles.ROW)
    assert np.allclose(o.T, T) and np.allclose(o.half, [0.05, 0.1, 0.1])


# ---------------------------------------------------------------- objects and targets
@pytest.mark.parametrize("path", OBJECTS)
def test_object_assets(path):
    obj = GraspObject(path)
    spec = mj.MjSpec()
    obj.add_to_spec(spec)
    model = spec.compile()
    assert np.isclose(model.body_mass[model.body("grasp_object").id], obj.mass)
    sdf = sdf_from_config(obj.cfg["sdf"])
    tg = FingertipTargets.load(str(obj.cfg["fingertip_targets"]))
    assert len(tg) >= 20 or obj.name == "red_cup"  # the cup's thin, flared rim: 23
    # targets satisfy eq. (3)'s end-point constraint, contacts are on the surface
    assert np.all(sdf.dist(np.vstack([tg.x1, tg.x2])) >= 0.01)
    assert np.all(sdf.dist(np.vstack([tg.p1, tg.p2])) < 0.006)
    # the opened hand fits: contact width + 2 offsets within the 85 mm stroke
    assert np.all(np.linalg.norm(tg.x2 - tg.x1, axis=1) <= rq.MAX_WIDTH)


def test_selection_eq12_prefers_close_and_aligned_pairs():
    X1 = np.array([[0.0, -0.02, 0.1], [0.2, -0.02, 0.1], [0.0, 0.02, 0.1]])
    X2 = np.array([[0.0, 0.02, 0.1], [0.2, 0.02, 0.1], [0.0, -0.02, 0.1]])
    x = np.array([[0.0, -0.03, 0.15], [0.0, 0.03, 0.15]])
    j, swap = select_target(x, X1, X2, w_angle=0.05)
    assert j == 0 and not swap
    # pair 2 is pair 0 with the fingers swapped: same cost via the swap
    c, s = selection_costs(x, X1, X2, 0.05)
    assert np.isclose(c[0], c[2]) and s[2]
    assert select_target(x, X1, X2, 0.05, valid=np.array([False, True, False]))[0] == 1


# ---------------------------------------------------------------- fields
def test_speed_profile_eq5():
    assert quad_decay_speed(0.2, 0.15, 0.05) == pytest.approx(0.15)
    assert quad_decay_speed(0.0, 0.15, 0.05) == pytest.approx(0.0)
    assert quad_decay_speed(0.025, 0.15, 0.05) == pytest.approx(0.15 * 0.75)
    assert gripper_width_field(0.03, 0.085, 0.1, 0.02) == pytest.approx(0.1)
    assert gripper_width_field(0.085, 0.085, 0.1, 0.02) == pytest.approx(0.0)
    v = linear_fingertip_field(np.zeros((2, 3)), np.array([[1.0, 0, 0], [0.01, 0, 0]]), 3.0)
    assert np.allclose(v, [[3.0, 0, 0], [0.03, 0, 0]])  # eq. (2): K (x* - x), no cap


def test_grasp_stability_is_zero_on_the_parallel_jaw(arm):
    """Eq. (6) with the 2F-85's pads: x2 - x1 = w y_tcp and the contact axes
    are +-y_tcp, so g = 0 and omega = 0 in any configuration and width."""
    T, J = _tcp(arm, Q)
    for w in (0.01, 0.05, 0.085):
        x, _ = fingertip_state(T, J, w)
        om, g = grasp_stability_field(x, np.stack([T[:3, :3]] * 2), PAD_CONTACT_AXES)
        assert g == pytest.approx(0.0, abs=1e-6) and np.allclose(om, 0.0)


def test_grasp_stability_field_is_steepest_descent():
    """Eq. (7): omega_i = -grad_{R_i} g. A contact axis 30 deg off the line
    between the fingertips: g = mean angle, and a rotation of each fingertip
    along omega_i (x fixed) lowers g at the rate |grad|^2."""
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.05, 0.0]])
    R1 = pin.exp3(np.array([0.0, 0.0, np.deg2rad(30.0)]))  # turns +y toward -x
    R = np.stack([R1, np.eye(3)])
    om, g = grasp_stability_field(x, R, PAD_CONTACT_AXES)
    assert g == pytest.approx(np.deg2rad(15.0))
    assert np.allclose(om[1], 0.0) and np.isclose(np.linalg.norm(om[0]), 0.5)
    h = 1e-5
    R_step = np.stack([pin.exp3(om[0] * h) @ R1, np.eye(3)])
    g1 = grasp_stability_field(x, R_step, PAD_CONTACT_AXES)[1]
    assert (g1 - g) / h == pytest.approx(-np.sum(om ** 2), rel=1e-3)


def test_path_field_goes_around_a_bowl_wall():
    """Fingertips beside a bowl, targets on the far rim: the straight line
    crosses the bowl, the planned paths keep eps_O clearance."""
    d = BOWL
    sp = np.load(d + "surface_points.npy")
    obj = PointCloudSDF(sp[:, :3])
    tg = FingertipTargets.load(d + "fingertip_targets.npz")
    k = int(np.argmax(tg.x1[:, 0]))
    x_tgt = np.stack([tg.x1[k], tg.x2[k]])
    x = np.array([[-0.15, -0.02, 0.03], [-0.15, 0.02, 0.03]])
    assert obj.dist(np.linspace(x[0], x_tgt[0], 50)).min() < 0.01  # blocked
    v, P = FingertipPathField()(x, x_tgt, [(obj, 0.01), (HalfSpaceSDF(0.0), 0.01)])
    assert P.shape == (2, 100, 3)
    for f in range(2):
        assert obj.dist(P[f, 1:-1]).min() > 0.0095
        assert P[f, :, 2].min() > 0.0095
        assert np.isclose(np.linalg.norm(v[f]), 0.15)


class _Far:
    """A constraint that never binds."""

    def dist_grad(self, x):
        return np.ones(len(x)), np.zeros((len(x), 3))


def test_path_sqp_per_constraint_margins():
    P = np.linspace([0, 0, 0.05], [0.2, 0, 0.05], 30)[None]
    out = path_opt.sqp(P, [(_Far(), 0.01), (HalfSpaceSDF(0.0), 0.08)], 3, None, None)
    assert out[0, 1:-1, 2].min() > 0.08 - 1e-3


def test_path_sqp_minimises_the_path_energy():
    """Eq. (3) without active constraints: the optimum of sum |c[k+1] - c[k]|^2
    with fixed end points is the straight line with evenly spaced knots, which
    one unconstrained SQP step reaches exactly from any start."""
    rng = np.random.default_rng(0)
    a, b = np.array([0.0, 0.0, 0.1]), np.array([0.2, 0.1, 0.3])
    P = np.linspace(a, b, 20) + np.r_[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0][:, None] * rng.normal(size=(20, 3)) * 0.02
    out = path_opt.sqp(P[None], [(_Far(), 0.0)], 1, None, None)
    assert np.allclose(out[0], np.linspace(a, b, 20), atol=1e-4)


# ---------------------------------------------------------------- QP
def test_qp_tracks_and_respects_distance_rows():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(3, N_Z))
    v = np.array([0.1, -0.05, 0.02])
    lb, ub = -np.ones(N_Z), np.ones(N_Z)
    z = solve_tracking_qp([(J, v, 1.0)], lb, ub, reg=1e-8)
    assert np.allclose(J @ z, v, atol=1e-4)
    # a distance 1 cm inside its 2 cm margin along J[0]: must open at >= 0.2 m/s
    rows = DistanceRows(N_Z, horizon=0.05, sep_rate_max=0.25)
    rows.add([0.01], J[:1], 0.02)
    z = solve_tracking_qp([(J, v, 1.0)], lb, ub, rows, reg=1e-8)
    assert J[0] @ z >= 0.2 - 1e-4


def test_distance_rows_report_what_binds():
    """The diagnostics of a stalled reach: rows at their margin, tightest
    first, labelled; a comfortable row is not listed."""
    rows = DistanceRows(N_Z, horizon=0.05, sep_rate_max=0.25)
    rows.add([0.0195, 0.05], np.ones((2, N_Z)), 0.02, ["object:sphere0", "object:sphere1"])
    rows.add([0.021], np.ones((1, N_Z)), 0.02, "fr3_link6-fr3_link1")
    rows.add([0.019], np.ones((1, N_Z)), 0.02, "table:sphere3")
    assert rows.active(tol=1e-3) == [("table:sphere3", pytest.approx(-0.001)), ("object:sphere0", pytest.approx(-0.0005))]
    assert rows.active(tol=2e-3)[-1][0] == "fr3_link6-fr3_link1"


def test_shared_via_gives_both_fingertips_the_same_detour():
    """Rigid parallel jaw: both fingertips detour through the same via
    direction (via - goal identical), even when each straight line is blocked
    differently."""
    box = BoxSDF([0.025, 0.035, 0.05], centre=[0.5, 0.0, 0.05])
    x0 = np.array([[0.5, 0.05, 0.15], [0.5, -0.03, 0.15]])
    xg = np.array([[0.5, 0.04, 0.09], [0.5, -0.05, 0.09]])
    P = path_opt.initial_paths(x0, xg, 100, 100, 0.1, lambda p: box.dist(p) - 0.01)
    # the knee of each 2-segment path is its via point
    knee = [P[f][np.argmax(np.linalg.norm(np.diff(P[f], 2, axis=0), axis=1)) + 1] for f in range(2)]
    assert np.allclose(knee[0] - xg[0], knee[1] - xg[1], atol=5e-3)


def test_path_sqp_keeps_fingertips_within_the_opening():
    """The 2F-85 coupling of eq. (3): |c1 - c2| <= 85 mm along the paths, even
    from an initial guess that bulges apart."""
    t = np.linspace(0.0, 1.0, 30)[:, None]
    bulge = np.sin(np.pi * t) * np.array([0.0, 0.08, 0.0])
    P = np.stack([np.array([0.0, -0.03, 0.1]) * (1 - t) + np.array([0.2, -0.03, 0.1]) * t - bulge,
                  np.array([0.0, 0.03, 0.1]) * (1 - t) + np.array([0.2, 0.03, 0.1]) * t + bulge])
    out = path_opt.sqp(P, [(_Far(), 0.01)], 3, 0.085, None)
    assert np.linalg.norm(out[0] - out[1], axis=1).max() <= 0.085 + 2e-3


def test_target_validity_follows_the_object_pose():
    """Rim pinches on the upright bowl clear the table; turned upside down
    (rim on the table) they cannot."""
    tg = FingertipTargets.load(BOWL + "fingertip_targets.npz")
    up = tg.valid(np.eye(4), table_z=0.0, eps_table=0.005)
    T = np.eye(4)
    T[:3, :3] = pin.rpy.rpyToMatrix(np.pi, 0.0, 0.0)
    T[2, 3] = 0.076  # bowl height: rim down on the table
    down = tg.valid(T, table_z=0.0, eps_table=0.005)
    assert up.mean() > 0.5 and down.mean() < up.mean()


def test_warm_started_paths_converge_to_the_eq3_optimum():
    """Repeated plans for the same target (same state): the few SQP iterations
    per plan accumulate, so the knots end up evenly spaced (the discrete
    optimum of the path energy) and the path no longer than the first one."""
    d = BOWL
    obj = PointCloudSDF(np.load(d + "surface_points.npy")[:, :3])
    tg = FingertipTargets.load(d + "fingertip_targets.npz")
    k = int(np.argmax(tg.x1[:, 0]))
    x_tgt = np.stack([tg.x1[k], tg.x2[k]])
    x = np.array([[-0.15, -0.02, 0.03], [-0.15, 0.02, 0.03]])
    field = FingertipPathField(max_separation=0.085)
    cons = [(obj, 0.01), (HalfSpaceSDF(0.0), 0.01)]
    first = field.plan_paths(x, x_tgt, cons, key=(k, False))
    for _ in range(15):
        P = field.plan_paths(x, x_tgt, cons, key=(k, False))
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    assert np.all(seg.std(axis=1) / seg.mean(axis=1) < 0.15)
    assert np.all(seg.sum(axis=1) <= np.linalg.norm(np.diff(first, axis=1), axis=2).sum(axis=1) + 1e-3)
    assert min(obj.dist(P[f, 1:-1]).min() for f in range(2)) > 0.0095


# ---------------------------------------------------------------- robot model, planner
@pytest.fixture(scope="module")
def planner():
    import sys

    sys.path.insert(0, "scripts")
    from rollout import make_planner

    return make_planner("blue_bowl", "path")


def test_params_reject_unknown_keys():
    from hrg.planner import HRGParams

    with pytest.raises(ValueError):
        HRGParams.from_config({"v_const": 0.1, "v_constant": 0.2})
    with pytest.raises(ValueError):
        HRGParams.from_config({"field": "straight"})


def test_gamma_rows_match_finite_differences(planner):
    """Gamma(q) of eq. (11): each row is the exact gradient of a distance
    (robot vs walls, robot vs itself), as the QP linearises it."""
    from hrg.qp import DistanceRows

    p, _ = planner
    robot, data = p.robot, p.robot.data()
    margins = np.r_[[robot.env_margin] * len(robot.env_pairs), [robot.self_margin] * len(robot.self_pairs)]

    def dist_rows(q):
        rows = DistanceRows(N_Z, 1.0, 1e9)  # H = 1, no cap: lower = margin - d
        robot.gamma_rows(q, data, rows)
        return margins - np.array(rows.l), np.array(rows.A)[:, :7]

    rng = np.random.default_rng(0)
    q0 = np.array([0.4, 0.6, -0.3, -1.4, 0.5, 1.2, 0.3])
    for q in [q0] + [q0 + rng.normal(size=7) * 0.3 for _ in range(4)]:
        d0, A = dist_rows(q)
        h = 1e-6
        fd = np.column_stack([(dist_rows(q + h * np.eye(7)[k])[0] - d0) / h for k in range(7)])
        assert np.allclose(A, fd, atol=1e-4), np.abs(A - fd).max()


def test_reach_step_is_a_function_of_the_state(planner):
    """One plan (eqs. 2-12) at a given state: valid target, the fingertip
    tasks tracked as far as the constraints allow, all hand spheres moving no
    closer to the object than the QP's linearised margin permits."""
    p, obj = planner
    T_obj = np.eye(4)
    T_obj[:3, 3] = [0.5, 0.0, obj.resting_height(np.eye(3))]
    q = np.array([0.0, 0.0, 0.0, -1.5707963, 0.0, 1.5707963, 0.0])
    p.reset()
    r = p.step(q, 0.085, T_obj)
    assert r is not None and r.qp_ok
    assert r.paths.shape == (2, 100, 3)
    assert np.all(np.isfinite(r.qd)) and np.isfinite(r.wd)
    # far from the target the fingertips move at v_const along the paths
    T = p.robot.pose(q, p.robot.data())
    J = p.robot.jacobian(q, p.robot.data())
    x, Jx = fingertip_state(T, J, 0.085)
    v = np.einsum("fij,j->fi", Jx, np.r_[r.qd, r.wd])
    d = (r.paths[:, 1] - r.paths[:, 0])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    assert np.all(np.einsum("fi,fi->f", v, d) > 0.1)
    # every plan reports what binds (nothing, at the ready pose far from the object)
    assert r.info["binding"] == dict(rows=[], joint_limits=[])
    # the same state gives the same plan (from a reset)
    p.reset()
    r2 = p.step(q, 0.085, T_obj)
    assert np.allclose(r.qd, r2.qd) and r.target == r2.target
