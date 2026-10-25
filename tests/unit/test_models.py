"""Robot models: MuJoCo MJCF vs URDF (Pinocchio, Viser), the Robotiq 2F-85
tables, and the plant's gravity compensation (the FR3 firmware behaviour the
whole repo relies on)."""

import mujoco as mj
import numpy as np
import pinocchio as pin
import pytest
import yourdfpy

from fr3.robot_platform.gripper import robotiq_2f85 as rq
from fr3.robot_platform.sim.franka_fr3.SimFrankaFR3 import (
    ARM_JOINT_NAMES,
    FR3_HOME_Q,
    GRIPPER_JOINT_NAME,
    SimFrankaFR3,
)

SCENE = "assets/robot/scene_sim.xml"
URDF = "assets/robot/urdf/fr3_robotiq_rigid.urdf"  # controllers
URDF_VIS = "assets/robot/urdf/fr3_robotiq.urdf"  # Viser


def build(gravity_comp=True):
    """Compile the scene the way the env does and return (platform, model, data)."""
    plat = SimFrankaFR3(gravity_comp=gravity_comp)
    spec = mj.MjSpec.from_file(SCENE)
    plat.modify_mjspec(spec)
    model = spec.compile()
    data = mj.MjData(model)
    plat.set_mj_data_name_idx(data)
    plat.reset(data)
    mj.mj_forward(model, data)
    return plat, model, data


def test_mjcf_has_arm_and_gripper():
    model = mj.MjModel.from_xml_path(SCENE)
    for n in ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]:
        assert model.joint(n) is not None
    assert model.site("fr3_gripper_tcp") is not None
    b = model.body("fr3_gripper").id
    mass = sum(model.body_mass[i] for i in range(model.nbody) if _in_subtree(model, i, b))
    assert mass == pytest.approx(rq.MASS, abs=0.01)


def _in_subtree(model, i, root):
    while i > 0:
        if i == root:
            return True
        i = model.body_parentid[i]
    return False


def test_urdf_joints():
    """Controllers: 7 arm joints (gripper rigid). Viser: + the driver angle."""
    assert pin.buildModelFromUrdf(URDF).nq == 7
    u = yourdfpy.URDF.load(URDF_VIS, load_meshes=False)
    assert u.actuated_joint_names == ARM_JOINT_NAMES + ["robotiq_driver_joint"]


def _settle_gripper(model, data, angle, steps=3000):
    """Drive the gripper to ``angle`` with the arm pinned at home, no gravity."""
    act = model.actuator("actuator_gripper").id
    arm = [model.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
    data.ctrl[act] = angle
    for _ in range(steps):
        data.qpos[arm] = FR3_HOME_Q
        data.qvel[:7] = 0.0
        mj.mj_step(model, data)


@pytest.mark.parametrize("angle", [0.0, 0.3, 0.6])
def test_gripper_linkage_mjcf_matches_urdf_and_tables(angle):
    """At a given driver angle: the URDF's <mimic> linkage puts the pads where
    MuJoCo's closed linkage does, and robotiq_2f85's width / pad-height tables
    match the pad faces."""
    model = mj.MjModel.from_xml_path(SCENE)
    model.opt.gravity[:] = 0.0
    data = mj.MjData(model)
    _settle_gripper(model, data, angle)
    drv = data.joint(GRIPPER_JOINT_NAME).qpos[0]
    g = data.body("fr3_gripper")
    R, p = g.xmat.reshape(3, 3), g.xpos
    u = yourdfpy.URDF.load(URDF_VIS, load_meshes=False)
    u.update_cfg(np.r_[FR3_HOME_Q, drv])
    for s in ("left", "right"):
        p_mj = R.T @ (data.body(f"robotiq_{s}_pad").xpos - p)
        p_urdf = u.get_transform(f"robotiq_{s}_pad", "fr3_gripper")[:3, 3]
        assert np.allclose(p_mj, p_urdf, atol=5e-4)
    faces = [R.T @ (data.site(f"robotiq_{s}_pad_face").xpos - p) for s in ("left", "right")]
    width = abs(faces[0][1] - faces[1][1])
    assert width == pytest.approx(float(rq.width_from_angle(drv)), abs=1e-3)
    assert 0.5 * (faces[0][2] + faces[1][2]) == pytest.approx(float(rq.pad_z(width)), abs=1e-3)


def test_urdf_gravity_matches_mjcf():
    """g(q) from the controllers' URDF (gripper rigid, open) equals MuJoCo's:
    what controllers with plant_gravity_comp: false rely on."""
    model = mj.MjModel.from_xml_path(SCENE)
    data = mj.MjData(model)
    pm = pin.buildModelFromUrdf(URDF)
    pd = pm.createData()
    rng = np.random.default_rng(0)
    for _ in range(3):
        q = rng.uniform(pm.lowerPositionLimit, pm.upperPositionLimit)
        data.qpos[:] = 0.0
        data.qpos[[model.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]] = q
        data.qvel[:] = 0.0
        mj.mj_forward(model, data)
        dof = [model.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
        g_pin = pin.computeGeneralizedGravity(pm, pd, q)
        # The linkage's internal equality forces are not gravity; with the
        # gripper open and at rest they are negligible on the arm joints.
        assert np.allclose(data.qfrc_bias[dof], g_pin, atol=0.05), (data.qfrc_bias[dof], g_pin)


@pytest.mark.parametrize("seed", range(5))
def test_fk_mjcf_matches_urdf(seed):
    """TCP pose from MuJoCo and Pinocchio agree at random arm poses."""
    _, model, data = build()
    pm = pin.buildModelFromUrdf(URDF)
    pd = pm.createData()
    rng = np.random.default_rng(seed)
    lo, hi = pm.lowerPositionLimit[:7], pm.upperPositionLimit[:7]
    q = rng.uniform(lo, hi)

    for i, n in enumerate(ARM_JOINT_NAMES):
        data.qpos[model.joint(n).qposadr[0]] = q[i]
    mj.mj_kinematics(model, data)
    base = data.body("base")
    T_mj = np.eye(4)
    site = data.site("fr3_gripper_tcp")
    T_mj[:3, :3], T_mj[:3, 3] = site.xmat.reshape(3, 3), site.xpos
    T_base = np.eye(4)
    T_base[:3, :3], T_base[:3, 3] = base.xmat.reshape(3, 3), base.xpos
    T_mj = np.linalg.inv(T_base) @ T_mj

    pin.forwardKinematics(pm, pd, q)
    pin.updateFramePlacements(pm, pd)
    T_pin = pd.oMf[pm.getFrameId("fr3_gripper_tcp")].homogeneous
    assert np.allclose(T_mj[:3, 3], T_pin[:3, 3], atol=1e-3)
    assert np.allclose(T_mj[:3, :3], T_pin[:3, :3], atol=1e-3)


def zero_torque_drift(gravity_comp, seconds=1.0):
    plat, model, data = build(gravity_comp)
    idx = plat._arm_qpos_idx
    q0 = data.qpos[idx].copy()
    for _ in range(int(seconds / model.opt.timestep)):
        data.ctrl[plat._arm_act_idx] = 0.0
        mj.mj_step(model, data)
    return np.max(np.abs(data.qpos[idx] - q0))


def test_plant_gravity_comp_holds_arm():
    """gravity_comp: true = the FR3 firmware: zero torque, the arm floats."""
    assert zero_torque_drift(True) < 0.01


def test_without_gravity_comp_arm_falls():
    assert zero_torque_drift(False) > 0.1


def test_link_side_torque_at_rest_is_gravity_load():
    """tau_J (what the FR3 torque sensors measure) carries the gravity load,
    which is what makes payload-mass estimation possible."""
    plat, model, data = build(True)
    kp = np.array([600, 600, 600, 600, 250, 150, 50.0])
    kd = np.array([50, 50, 50, 20, 20, 20, 10.0])
    q_des = np.array(FR3_HOME_Q)
    for _ in range(1000):
        q = data.qpos[plat._arm_qpos_idx]
        qd = data.qvel[plat._arm_dof_idx]
        data.ctrl[plat._arm_act_idx] = kp * (q_des - q) - kd * qd
        mj.mj_step(model, data)
    mj.mj_forward(model, data)
    tau_J = plat._link_side_torque(data)
    g = data.qfrc_bias[plat._arm_dof_idx]  # at rest, bias = gravity
    assert np.allclose(tau_J, g, atol=0.3), (tau_J, g)
