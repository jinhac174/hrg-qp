"""HardwareFrankaFR3 command law + safety layer, against the fake backend.

``_step`` is called directly (no LCM, no terminal), with a recording backend
so we see exactly the torque that would go to the robot."""

import time
from queue import Queue

import numpy as np
import pytest
from omegaconf import OmegaConf

from fr3.data_type.basic_types.GripperCmdData import GripperCmdData
from fr3.data_type.basic_types.JointCtrlData import JointCtrlData
from fr3.robot_platform.hardware.franka_fr3.HardwareFrankaFR3 import FakeBackend, HardwareFrankaFR3

CFG = "configs/robot/hardware.yaml"
Q0 = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, 0])


class FrozenBackend(FakeBackend):
    """Robot that does not move (state fixed), records the torques."""

    def __init__(self, dq=None):
        super().__init__(Q0)
        self.dq = np.zeros(7) if dq is None else np.asarray(dq, float)
        self.sent = []

    def state(self):
        return Q0.copy(), self.dq.copy(), np.zeros(7)

    def set_torque(self, tau):
        self.sent.append(np.asarray(tau).copy())


@pytest.fixture
def hw():
    cfg = OmegaConf.load(CFG)
    cfg.robot.backend = "fake"
    h = HardwareFrankaFR3(cfg)
    h.set_sub_que_dict({"ctrl_sub_que": Queue(), "intr_sub_que_dict": {"franka_gripper_cmd": Queue()}})
    pubs = {c: Queue() for c in ("franka_arm_joint_meas", "franka_gripper_joint_meas", "franka_gripper_state")}
    h.set_pub_que_dict({"intr_pub_que_dict": pubs})
    h.initialize()
    h._backend = FrozenBackend()
    h._hold_q = Q0.copy()
    # Steps are called back to back but wall-clock time still passes (a lot on
    # a loaded machine): no command timeout unless a test enables it.
    h._ctrl_timeout = 1e9
    h._gravity_model = "robot"  # tests check the command law alone
    yield h
    h.shutdown()


def send(h, q_des, kp, kd=None, tau_ff=None, qd_des=None):
    c = JointCtrlData(num_joints=7)
    z = np.zeros(7)
    c.set_data(time.time(), np.asarray(q_des, float), z if qd_des is None else qd_des,
               z if tau_ff is None else np.asarray(tau_ff, float), np.asarray(kp, float),
               z if kd is None else np.asarray(kd, float))
    h.ctrl_sub_que.put(c)


def step(h, n=1):
    for _ in range(n):
        h._last_step_t = 0.0  # bypass the rate limiter
        h._step()
    return h._backend.sent[-1]


def test_command_law_pd_plus_feedforward(hw):
    """tau = kp (q_des - q) + kd (qd_des - qd) + tau_ff (small: no limits hit)."""
    dq = np.full(7, 0.01)
    hw._backend.dq = dq
    send(hw, Q0 + 0.001, kp=np.full(7, 100.0), kd=np.full(7, 10.0), tau_ff=np.full(7, 0.05))
    tau = step(hw)
    assert np.allclose(tau, 100 * 0.001 - 10 * 0.01 + 0.05, atol=1e-9)


def test_joint_error_clamp(hw):
    """A set-point 1 rad away only produces kp * max_joint_error."""
    send(hw, Q0 + 1.0, kp=np.full(7, 10.0))
    tau = step(hw, 200)  # past the torque-rate ramp
    assert np.allclose(tau, 10.0 * hw._max_joint_error)


def test_torque_saturation(hw):
    send(hw, Q0, kp=np.zeros(7), tau_ff=np.full(7, 500.0))
    tau = step(hw, 1000)
    assert np.allclose(tau, hw._tau_max)


def test_torque_rate_limit(hw):
    send(hw, Q0, kp=np.zeros(7), tau_ff=np.full(7, 10.0))
    tau = step(hw)
    assert np.all(np.abs(tau) <= hw._tau_rate_max * hw._update_dt + 1e-12)


def test_timeout_falls_back_to_hold(hw):
    send(hw, Q0 + 0.05, kp=np.full(7, 100.0))
    step(hw, 50)
    hw._ctrl_timeout = 0.1
    hw._last_cmd_t -= 1.0  # command is now stale
    hw._hold_q = Q0 + 0.0  # hold where we are -> zero spring force
    tau = step(hw, 50)
    assert np.all(np.abs(tau) < np.abs(hw._backend.sent[49]) + 1e-9)
    assert np.all(np.abs(tau) < 3.0)  # ramping down to the hold torque (0)


def test_non_finite_command_is_ignored(hw):
    send(hw, Q0 + 0.001, kp=np.full(7, 100.0))
    step(hw)
    good = hw._last_cmd
    send(hw, np.full(7, np.nan), kp=np.full(7, 100.0))
    tau = step(hw)
    assert hw._last_cmd is good and np.all(np.isfinite(tau))
    hw._ctrl_timeout = 0.1
    hw._last_cmd_t -= 1.0  # only NaN arriving from now on -> timeout -> hold
    send(hw, Q0, kp=np.full(7, 100.0), tau_ff=np.full(7, np.inf))
    assert np.all(np.isfinite(step(hw, 20)))


def test_velocity_brake_damps_excess_only(hw):
    hw._vel_mode = "brake"
    hw._backend.dq = np.r_[3.0, np.zeros(6)]  # 1 rad/s above max_joint_vel[0] = 2
    send(hw, Q0, kp=np.zeros(7))
    tau = step(hw, 50)
    assert not hw._vel_stopped  # keeps following the controller
    assert tau[0] < -1.0 and np.allclose(tau[1:], 0.0, atol=1e-9)


def test_velocity_stop_and_release(hw):
    hw._vel_mode = "stop"
    hw._backend.dq = np.r_[3.0, np.zeros(6)]  # above max_joint_vel[0] = 2
    send(hw, Q0 + 0.05, kp=np.full(7, 100.0))
    step(hw)
    assert hw._vel_stopped
    hw._backend.dq = np.zeros(7)
    send(hw, Q0 + 0.05, kp=np.full(7, 100.0))
    tau = step(hw, 50)
    assert np.allclose(tau, 0.0, atol=1e-9)  # holding, command ignored
    hw._handle_key("h")
    assert not hw._vel_stopped


def test_arm_meas_published(hw):
    step(hw)
    m = hw.intr_pub_que_dict["franka_arm_joint_meas"].get_nowait()
    assert np.allclose(m.get_q(), Q0)


def test_gripper_command_roundtrip(hw):
    """Commands are de-duplicated by seq, run in the action thread, and the
    result comes back in gripper_state (seq, success)."""
    hw._backend = FakeBackend(Q0)
    c = GripperCmdData()
    c.set_data(time.time(), 7, GripperCmdData.MOVE, width=0.07, speed=0.5)
    for _ in range(3):  # LCM resends the same command
        hw._gripper_cmd_que.put(c)
    hw._dispatch_gripper_cmds()
    assert hw._gripper_actions.qsize() == 1
    t0 = time.time()
    while hw._gripper_last[0] != 7 and time.time() - t0 < 2.0:
        time.sleep(0.01)
    assert hw._gripper_last == (7, True)
    st = hw.intr_pub_que_dict["franka_gripper_state"]
    t0 = time.time()
    last = None
    while time.time() - t0 < 1.0:
        while not st.empty():
            last = st.get_nowait()
        if last is not None and last.last_seq == 7:
            break
        time.sleep(0.02)
    # the 2F-85 positions in 1/255 steps of its stroke (~0.4 mm)
    assert last is not None and last.last_seq == 7 and abs(last.width - 0.07) < 1e-3


def test_streamed_width_setpoints_follow_the_newest(hw):
    """Reactive grasping streams MOVE set-points at ~50 Hz: a new MOVE
    preempts the running one, superseded ones are skipped, and the gripper
    ends at the last set-point well before the moves could have run one by
    one."""
    hw._backend = FakeBackend(Q0)
    widths = np.linspace(0.085, 0.03, 15)
    t0 = time.time()
    for k, w in enumerate(widths):
        c = GripperCmdData()
        c.set_data(time.time(), 100 + k, GripperCmdData.MOVE, width=float(w), speed=0.15)
        hw._gripper_cmd_que.put(c)
        hw._dispatch_gripper_cmds()
        time.sleep(0.02)
    last_seq = 100 + len(widths) - 1
    while hw._gripper_last[0] != last_seq and time.time() - t0 < 3.0:
        time.sleep(0.01)
    assert hw._gripper_last == (last_seq, True)
    assert time.time() - t0 < 1.5
    assert abs(hw._gripper.state()[0] - widths[-1]) < 1e-3


def test_shutdown_is_idempotent(hw):
    hw.shutdown()
    hw.shutdown()


def test_controller_gravity_mode_refuses_command_without_gravity(hw, monkeypatch):
    import pinocchio as pin

    model = pin.buildModelFromUrdf("assets/robot/urdf/fr3_robotiq_rigid.urdf")
    hw._pin = (pin, model, model.createData())
    hw._gravity_model = "controller"
    g = hw._urdf_gravity(Q0)
    monkeypatch.setattr(hw._backend, "robot_gravity", lambda q: g, raising=False)
    send(hw, Q0, kp=np.zeros(7))  # plant_gravity_comp: true style: no g(q)
    step(hw)
    assert hw._grav_rejected
    hw._grav_rejected = None  # as after that controller stopped
    send(hw, Q0, kp=np.zeros(7), tau_ff=g)  # carries gravity
    step(hw)
    assert hw._grav_rejected is False
