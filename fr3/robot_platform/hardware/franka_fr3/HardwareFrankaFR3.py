"""Real Franka Research 3 (panda-py) + Robotiq 2F-85 (Modbus over USB): the
DROID setup.

Same LCM contract as the simulator (``SimFrankaFR3``), so every controller runs
unchanged on the robot (with ``env_mode: real``):

* in : ``joint_ctrl_t`` (7 arm joints) -> ``tau = kp (q_des - q) + kd (qd_des - qd) + tau_ff``
       ``gripper_cmd_t``               -> move / grasp / stop / homing (robotiq.py)
* out: ``joint_meas_t`` arm (q, dq, tau_J), ``joint_meas_t`` gripper (driver
       angle, for Viser), ``gripper_state_t``

The FR3 compensates gravity itself in torque control, exactly like the sim
with ``gravity_comp: true``: the commanded torque goes on top of it. Set the
end-effector load in Desk to the gripper (0.925 kg, centre of mass 5.2 cm
beyond the flange on its axis), otherwise the arm sags or drifts up.

panda-py runs libfranka's 1 kHz torque loop in C++ (``AppliedTorque``) and
always applies the latest torque this process set (zero-order hold). This
process only has to keep up approximately.

Safety layer (on top of the robot's own reflexes):

* ``max_joint_error``: ``q_des`` is clamped to ``q +/- max_joint_error`` so a
  controller that jumps its set-point far away only
  produces a bounded spring force.
* torque saturation (FR3 limits) and a torque-rate limit (``tau_rate_max``,
  Nm/s): no torque steps, which libfranka would reject (discontinuity reflex)
  and which jerk the arm.
* ``ctrl_timeout``: without fresh commands (controller stopped / crashed / not
  started yet, or sending NaN / inf) the arm holds where it is with a soft PD.
* ``max_joint_vel``: above it, ``velocity_limit: brake`` (default) adds
  damping on the excess speed of that joint only (``vel_brake_kd``), the
  controller keeps control; ``velocity_limit: stop`` switches to hold and
  stays there until key ``h`` releases.

Keys (hardware terminal): ``q`` quit, ``h`` clear a velocity stop,
``r`` recover from a robot error (reflex) and restart torque control.
"""

from __future__ import annotations

import copy
import queue
import threading
import time

import numpy as np
from omegaconf import DictConfig

from fr3.data_type.basic_types.GripperCmdData import GripperCmdData
from fr3.data_type.basic_types.GripperStateData import GripperStateData
from fr3.data_type.basic_types.JointCtrlData import JointCtrlData
from fr3.data_type.basic_types.JointMeasData import JointMeasData
from fr3.robot_platform.gripper import robotiq_2f85 as rq
from fr3.robot_platform.hardware.BaseHardwarePlatform import BaseHardwarePlatform
from fr3.robot_platform.hardware.franka_fr3.robotiq import FakeRobotiqTransport, RobotiqGripper

N = 7
FR3_TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
FR3_DQ_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
# Default pose for move_to_start: [0, 0, 0, -90°, 0, 90°, 0] (joint 7 = 0, DROID).
FR3_HOME_Q = [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0]
GRIPPER_MASS = 0.925  # kg: Robotiq 2F-85 + mount (CoM 5.2 cm along flange z)
# panda-py's set_default_behavior(): contact / reflex thresholds on the
# estimated EXTERNAL torque (Nm, per joint) and wrench (N x3, Nm x3).
PANDAPY_DEFAULT_TORQUE_THR = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
PANDAPY_DEFAULT_FORCE_THR = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class PandaPyBackend:
    """The real robot. Needs a panda-py build for your robot's libfranka
    version (the PyPI wheel bundles libfranka 0.9.2 = Panda only; FR3 wheels
    are on the panda-py GitHub releases page)."""

    def __init__(self, hostname: str, collision_thresholds=None, load=None):
        import panda_py
        from panda_py import controllers

        self._panda_py, self._controllers = panda_py, controllers
        self._panda = panda_py.Panda(hostname)
        self._ctrl = None
        self._model = None
        if collision_thresholds:
            self.set_collision_thresholds(**collision_thresholds)
        if load:
            # Mass the Desk payload is missing (wrist camera + mount), from
            # scripts/identify_payload.py. Point mass: tiny inertia.
            com = [float(x) for x in load["com"]]
            self._panda.get_robot().set_load(
                float(load["mass"]), com, (np.eye(3) * 1e-4).flatten().tolist()
            )
        self.report_payload()

    def report_payload(self):
        """The FR3 compensates gravity with the end-effector mass set in Desk.
        A wrong value is a constant torque error: the arm sags (or rises)
        whenever friction lets go."""
        s = self._panda.get_state()
        m = s.m_ee + s.m_load
        print(f"[FR3] Payload in the robot's gravity model: m_ee {s.m_ee:.3f} kg at "
              f"{np.round(s.F_x_Cee, 3).tolist()} m, m_load {s.m_load:.3f} kg at "
              f"{np.round(s.F_x_Cload, 3).tolist()} m")
        if s.m_ee < GRIPPER_MASS - 0.1 or m > GRIPPER_MASS + 0.5:
            print(f"[FR3] WARNING: {m:.3f} kg != the Robotiq gripper's {GRIPPER_MASS} kg: gravity "
                  "compensation is off, the arm will drift. Desk > Settings > End effector.")

    def set_collision_thresholds(self, torque, force):
        """Reflex thresholds on the external torque / wrench (the robot stops,
        red lights, above them). Contact is still flagged at panda-py's
        defaults. Must be called while the robot is not moving."""
        torque = [float(x) for x in torque]
        force = [float(x) for x in force]
        lo_t = list(np.minimum(PANDAPY_DEFAULT_TORQUE_THR, torque))
        lo_f = list(np.minimum(PANDAPY_DEFAULT_FORCE_THR, force))
        self._panda.get_robot().set_collision_behavior(lo_t, torque, lo_f, force)
        print(f"[FR3] Collision reflex thresholds: torque {torque} Nm, force {force} N / Nm")

    def start_torque(self):
        # No extra damping / filtering inside panda-py: our command already has it.
        self._ctrl = self._controllers.AppliedTorque(damping=np.zeros(N), filter_coeff=1.0)
        self._panda.start_controller(self._ctrl)

    def stop_torque(self):
        self._panda.stop_controller()

    def recover(self):
        self._panda.recover()

    def move_to_start(self):
        # Not panda-py's move_to_start(): its pose has joint 7 = pi/4 (Franka Hand).
        if not self._panda.move_to_joint_position(FR3_HOME_Q, speed_factor=0.2):
            print("[FR3] Did not reach the start pose; torque control starts where the arm is.")

    def state(self):
        s = self._panda.get_state()
        return np.array(s.q), np.array(s.dq), np.array(s.tau_J)

    def robot_gravity(self, q):
        """Gravity torque the FR3 itself compensates at q (its arm model + the
        Desk payload + set_load)."""
        if self._model is None:
            self._model = self._panda.get_model()
        s = self._panda.get_state()
        return np.array(self._model.gravity(list(q), s.m_total, list(s.F_x_Ctotal)))

    def set_torque(self, tau):
        self._ctrl.set_control(np.asarray(tau, dtype=np.float64))


class FakeBackend:
    """No robot: a crude 7-joint plant (gravity already compensated, like the
    FR3). For testing this process and the safety layer on a machine without
    a robot (the gripper is faked separately: ``gripper.backend: fake``)."""

    def __init__(self, q0=None, gravity_fn=None):
        # Gravity the fake "FR3" compensates itself (the URDF's; none if None).
        self._gravity_fn = gravity_fn
        self._q = np.array(q0 if q0 is not None else [0, 0, 0, -np.pi / 2, 0, np.pi / 2, 0], dtype=float)
        self._dq = np.zeros(N)
        self._tau = np.zeros(N)
        self._inertia = np.array([0.76, 1.83, 1.22, 1.2, 0.12, 0.13, 0.08])
        self._t = time.time()
        self._lock = threading.Lock()

    def start_torque(self):
        self._t = time.time()

    def stop_torque(self):
        self._tau[:] = 0.0

    def recover(self):
        pass

    def move_to_start(self):
        pass

    def robot_gravity(self, q):
        return np.zeros(N) if self._gravity_fn is None else self._gravity_fn(q)

    def _integrate(self):
        t = time.time()
        dt = min(t - self._t, 0.01)
        self._t = t
        ddq = (self._tau - 0.5 * self._dq) / self._inertia
        self._dq += ddq * dt
        self._q += self._dq * dt

    def state(self):
        with self._lock:
            self._integrate()
            return self._q.copy(), self._dq.copy(), self._tau.copy()

    def set_torque(self, tau):
        with self._lock:
            self._integrate()
            self._tau = np.asarray(tau, dtype=float).copy()


# ---------------------------------------------------------------------------
class HardwareFrankaFR3(BaseHardwarePlatform):
    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        hw = config["robot"]
        self._backend_name = str(hw.get("backend", "pandapy"))
        self._hostname = str(hw.get("hostname", ""))
        self._move_to_start = bool(hw.get("move_to_start", False))
        ct = hw.get("collision_thresholds", None)
        self._collision_thresholds = (
            {"torque": list(ct["torque"]), "force": list(ct["force"])} if ct else None
        )
        # robot = the FR3's own gravity compensation (default); urdf = cancel
        # it and use the Pinocchio model of gravity_urdf instead; controller =
        # cancel it, the controller sends g(q) itself (plant_gravity_comp:
        # false); the hold (no fresh command) still uses gravity_urdf.
        self._gravity_model = str(hw.get("gravity_model", "robot"))
        if self._gravity_model not in ("robot", "urdf", "controller"):
            raise ValueError("robot.gravity_model must be 'robot', 'urdf' or 'controller', "
                             f"got {self._gravity_model!r}")
        self._gravity_urdf = str(hw.get("gravity_urdf", "assets/robot/urdf/fr3_robotiq_rigid.urdf"))
        self._pin = None
        self._grav_rejected = None  # gravity_model: controller check, per controller start
        ld = hw.get("load", None)
        self._load = {"mass": float(ld["mass"]), "com": list(ld["com"])} if ld else None
        gc = config.get("gripper", {}) or {}
        self._gripper_backend = str(gc.get("backend", "auto"))
        if self._gripper_backend == "auto":
            self._gripper_backend = "fake" if self._backend_name == "fake" else "robotiq"
        self._gripper_port = str(gc.get("port", "/dev/ttyUSB0"))
        self._gripper_activate = bool(gc.get("activate_on_start", True))

        s = config["safety"]
        self._tau_max = np.minimum(np.asarray(s.get("tau_max", FR3_TAU_MAX), dtype=float), FR3_TAU_MAX)
        self._tau_rate_max = float(s.get("tau_rate_max", 1000.0))
        self._max_joint_error = float(s.get("max_joint_error", 0.1))
        self._dq_stop = np.asarray(s.get("max_joint_vel", 0.9 * FR3_DQ_MAX), dtype=float)
        self._vel_mode = str(s.get("velocity_limit", "brake"))
        if self._vel_mode not in ("brake", "stop"):
            raise ValueError(f"safety.velocity_limit must be 'brake' or 'stop', got {self._vel_mode!r}")
        self._vel_brake_kd = np.asarray(
            s.get("vel_brake_kd", [20, 20, 20, 20, 5, 5, 2]), dtype=float
        )
        self._ctrl_timeout = float(s.get("ctrl_timeout", 0.1))
        self._hold_kp = np.asarray(s.get("hold_kp", [300, 300, 300, 300, 125, 75, 25]), dtype=float)
        self._hold_kd = np.asarray(s.get("hold_kd", [35, 35, 35, 15, 15, 15, 7]), dtype=float)

        pm = config["pub_manager"]
        self._arm_ch = pm["arm_joint_meas_channel"]
        self._gripper_joint_ch = pm["gripper_joint_meas_channel"]
        self._gstate_ch = pm.get("gripper_state_channel", "")
        self._gripper_cmd_ch = config["sub_manager"].get("gripper_cmd_channel", "")
        self._gripper_rate = float(config.get("gripper_state_freq", 30.0))

        self._backend = None
        self._gripper = None
        self._gripper_cmd_que = None
        self._last_cmd: JointCtrlData | None = None
        self._last_cmd_t = 0.0
        self._hold_q = None
        self._vel_stopped = False
        self._error = False
        self._tau_prev = np.zeros(N)
        self._last_step_t = 0.0
        self._arm_meas = JointMeasData(num_joints=N)

        self._gripper_actions: queue.Queue = queue.Queue()
        self._gripper_busy = False
        self._gripper_running = None  # mode of the action in progress
        self._gripper_last = (-1, True)
        self._gripper_done_seq = -1
        self._stop_event = threading.Event()
        self._threads = []
        # Collision geometry of the cell (walls / floor + robot primitives) for
        # the collision-aware controllers and Viser, like the sim env does.
        self._scene_cfg = config.get("scene_info", None)
        self._scene_pub = None

    # ------------------------------------------------------------ wiring
    def set_sub_que_dict(self, sub_data_queue_dict):
        self.ctrl_sub_que = sub_data_queue_dict["ctrl_sub_que"]
        if self._gripper_cmd_ch:
            self._gripper_cmd_que = sub_data_queue_dict["intr_sub_que_dict"].get(self._gripper_cmd_ch)

    def set_pub_que_dict(self, pub_data_queue_dict):
        self.intr_pub_que_dict = pub_data_queue_dict["intr_pub_que_dict"]
        self.time_pub_que = pub_data_queue_dict.get("time_pub_que")

    # ------------------------------------------------------------ lifecycle
    def initialize(self):
        import pinocchio as pin

        model = pin.buildModelFromUrdf(self._gravity_urdf)
        self._pin = (pin, model, model.createData())
        if self._backend_name == "fake":
            print("[FR3] FAKE backend: no robot connected.")
            self._backend = FakeBackend(gravity_fn=self._urdf_gravity)
        else:
            print(f"[FR3] Connecting to {self._hostname} ...")
            self._backend = PandaPyBackend(self._hostname, self._collision_thresholds, self._load)
        if self._move_to_start:
            print(f"[FR3] Moving to the start pose {np.round(FR3_HOME_Q, 3).tolist()} ...")
            self._backend.move_to_start()
        if self._gripper_backend == "fake":
            self._gripper = RobotiqGripper(transport=FakeRobotiqTransport())
        else:
            print(f"[FR3] Robotiq 2F-85 on {self._gripper_port} ...")
            self._gripper = RobotiqGripper(self._gripper_port)
        if self._gripper_activate:
            # Activation makes the 2F-85 close and open once (it calibrates):
            # keep the fingers clear.
            print("[FR3] Activating the gripper (it closes and opens once) ...")
            self._gripper.homing()
        q, _, _ = self._backend.state()
        self._hold_q = q.copy()
        self._tau_prev = np.zeros(N)
        if self._backend_name != "fake":
            diff = self._backend.robot_gravity(q) - self._urdf_gravity(q)
            print(f"[FR3] Gravity: robot model - URDF model at this pose = "
                  f"{np.round(diff, 2).tolist()} Nm")
            if self._gravity_model == "urdf":
                print("[FR3] gravity_model: urdf -- the FR3's gravity compensation is cancelled, "
                      f"{self._gravity_urdf} holds the arm.")
            elif self._gravity_model == "controller":
                print("[FR3] gravity_model: controller -- the FR3's gravity compensation is "
                      "cancelled. The CONTROLLER must send g(q): run it with "
                      "--plant_gravity_comp false, or the arm FALLS. (Hold uses the URDF.)")
        self._backend.start_torque()
        for target in (self._gripper_action_loop, self._gripper_state_loop):
            th = threading.Thread(target=target, daemon=True)
            th.start()
            self._threads.append(th)
        if self._scene_cfg:
            from fr3.robot_platform.hardware.franka_fr3.scene_info import SceneInfoPublisher

            sc = self._scene_cfg
            self._scene_pub = SceneInfoPublisher(
                sc["scene_xml_path"], sc["static_col_info_channel"],
                sc["robot_col_info_channel"], sc.get("rate", 10.0),
            )
            self._scene_pub.start()
            print(f"[FR3] Publishing the cell's collision geometry ({sc['scene_xml_path']}): "
                  f"{self._scene_pub.describe()}")
        self._hardware_ready = True
        print("[FR3] Torque control running. Holding until a controller sends commands. "
              "Keys: q quit, h clear velocity stop, r recover from a robot error.")

    def shutdown(self):
        if self._stop_event.is_set():
            return  # already shut down
        self._stop_event.set()
        if self._scene_pub is not None:
            self._scene_pub.stop()
        if self._backend is not None:
            try:
                self._backend.stop_torque()
            except Exception as e:  # noqa: BLE001 - shutting down anyway
                print(f"[FR3] stop_torque: {e}")
        print("[FR3] Shut down.")

    def estop(self):
        self.shutdown()

    def _handle_key(self, key):
        if key == "q":
            self._finished = True
            self.shutdown()
        elif key == "h":
            if self._vel_stopped:
                print("[FR3] Velocity stop cleared: following commands again.")
            self._vel_stopped = False
        elif key == "r":
            self._recover()

    def _recover(self):
        print("[FR3] Recovering from robot error ...")
        try:
            self._backend.recover()
            q, _, _ = self._backend.state()
            self._hold_q, self._tau_prev = q.copy(), np.zeros(N)
            self._backend.start_torque()
            self._error = False
            print("[FR3] Recovered; holding.")
        except Exception as e:  # noqa: BLE001 - report and stay in error
            print(f"[FR3] Recovery failed: {e}")

    # ------------------------------------------------------------ 1 kHz-ish step
    def _step(self):
        now = time.time()
        if now - self._last_step_t < self._update_dt or self._error:
            return
        self._last_step_t = now
        try:
            q, dq, tau_J = self._backend.state()
        except Exception as e:  # noqa: BLE001 - libfranka reflexes surface here
            return self._on_error(e)

        cmd = None
        while self.ctrl_sub_que is not None and not self.ctrl_sub_que.empty():
            cmd = self.ctrl_sub_que.get_nowait()
        if isinstance(cmd, JointCtrlData) and cmd.get_valid() is not False:
            if all(np.all(np.isfinite(x)) for x in cmd.get_data()[1:]):
                self._last_cmd, self._last_cmd_t = cmd, now
            else:  # NaN / inf would pass np.clip: drop it (-> timeout hold)
                print("[FR3] Non-finite command ignored.")

        excess = np.maximum(np.abs(dq) - self._dq_stop, 0.0)
        if self._vel_mode == "stop" and np.any(excess > 0) and not self._vel_stopped:
            self._vel_stopped = True
            self._hold_q = q.copy()
            print(f"[FR3] Joint velocity limit exceeded ({np.round(np.abs(dq), 2)}): "
                  "holding. Press h to resume commands.")

        fresh = self._last_cmd is not None and now - self._last_cmd_t <= self._ctrl_timeout
        if not fresh:
            self._grav_rejected = None  # controller gone: check the next one again
        if fresh and not self._vel_stopped and self._gravity_model == "controller" and self._pin is not None:
            if self._grav_rejected is None:  # first command of this controller
                self._grav_rejected = self._lacks_gravity(q, dq)
        use_cmd = fresh and not self._vel_stopped and not self._grav_rejected
        if use_cmd:
            _, q_des, qd_des, tau_ff, kp, kd = self._last_cmd.get_data()
            q_des = np.clip(q_des, q - self._max_joint_error, q + self._max_joint_error)
            tau = kp * (q_des - q) + kd * (qd_des - dq) + tau_ff
            self._hold_q = q.copy()
        else:
            # No (fresh) command: hold the last commanded / measured pose.
            tau = self._hold_kp * (self._hold_q - q) - self._hold_kd * dq
        if self._vel_mode == "brake":
            # Damp only the speed above the limit, only on that joint.
            tau = tau - self._vel_brake_kd * excess * np.sign(dq)
        if self._gravity_model != "robot" and self._pin is not None:
            tau = tau - self._backend.robot_gravity(q)
            if self._gravity_model == "urdf" or not use_cmd:
                tau = tau + self._urdf_gravity(q)  # hold never relies on a controller

        tau = np.clip(tau, -self._tau_max, self._tau_max)
        step = self._tau_rate_max * self._update_dt
        tau = self._tau_prev + np.clip(tau - self._tau_prev, -step, step)
        self._tau_prev = tau
        try:
            self._backend.set_torque(tau)
        except Exception as e:  # noqa: BLE001
            return self._on_error(e)

        self._arm_meas.set_data(now, q, dq, tau_J)
        que = self.intr_pub_que_dict.get(self._arm_ch)
        if que is not None:
            que.put(copy.deepcopy(self._arm_meas))
        self._dispatch_gripper_cmds()

    def _lacks_gravity(self, q, dq):
        """gravity_model: controller -- does the controller's command carry
        g(q)? One started with plant_gravity_comp: true sends ~0 on the
        shoulder / elbow and the arm would fall: refuse it (keep holding)."""
        _, q_des, qd_des, tau_ff, kp, kd = self._last_cmd.get_data()
        q_des = np.clip(q_des, q - self._max_joint_error, q + self._max_joint_error)
        tau = kp * (q_des - q) + kd * (qd_des - dq) + tau_ff
        g = self._urdf_gravity(q)
        big = np.abs(g) > 5.0
        bad = big & (np.abs(tau - g) > 0.5 * np.abs(g))
        if np.any(bad):
            print(f"[FR3] Controller REFUSED: its torque {np.round(tau, 1).tolist()} does not carry "
                  f"gravity {np.round(g, 1).tolist()} (gravity_model: controller). Start it with "
                  "plant_gravity_comp: false. Holding.")
            return True
        print("[FR3] Controller sends gravity: following it.")
        return False

    def _urdf_gravity(self, q):
        pin, model, data = self._pin
        return pin.computeGeneralizedGravity(model, data, np.asarray(q, dtype=float))

    def _on_error(self, e):
        self._error = True
        print(f"[FR3] Robot error: {e}\n[FR3] Torque control stopped. Fix the cause, then press r.")

    # ------------------------------------------------------------ gripper
    def _dispatch_gripper_cmds(self):
        """Hand new gripper commands (by seq) to the action thread; STOP is
        executed right away so it can interrupt a running move / grasp. A MOVE
        arriving while a MOVE runs preempts it (streamed width set-points,
        reactive grasping): the newest set-point wins."""
        if self._gripper_cmd_que is None:
            return
        while not self._gripper_cmd_que.empty():
            cmd = self._gripper_cmd_que.get_nowait()
            if not isinstance(cmd, GripperCmdData) or cmd.seq == self._gripper_done_seq:
                continue
            self._gripper_done_seq = cmd.seq
            if cmd.mode == GripperCmdData.STOP:
                ok = self._gripper.stop()
                self._gripper_last = (cmd.seq, bool(ok))
            else:
                if cmd.mode == GripperCmdData.MOVE and self._gripper_running == GripperCmdData.MOVE:
                    self._gripper.preempt()
                self._gripper_actions.put(cmd)

    def _gripper_action_loop(self):
        while not self._stop_event.is_set():
            try:
                cmd = self._gripper_actions.get(timeout=0.1)
            except queue.Empty:
                continue
            # A MOVE followed by newer commands is superseded: skip it.
            while cmd.mode == GripperCmdData.MOVE and not self._gripper_actions.empty():
                self._gripper_last = (cmd.seq, False)
                cmd = self._gripper_actions.get_nowait()
            self._gripper_busy = True
            self._gripper_running = cmd.mode
            try:
                ok = self._gripper.action(
                    cmd.mode, cmd.width, cmd.speed, cmd.force, cmd.epsilon_inner, cmd.epsilon_outer
                )
            except Exception as e:  # noqa: BLE001 - a failed action is reported, not fatal
                print(f"[FR3] Gripper action failed: {e}")
                ok = False
            self._gripper_last = (cmd.seq, bool(ok))
            self._gripper_running = None
            self._gripper_busy = False

    def _gripper_state_loop(self):
        joint = JointMeasData(num_joints=1)
        state = GripperStateData()
        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                width, max_width, grasped = self._gripper.state()
            except Exception as e:  # noqa: BLE001
                print(f"[FR3] Gripper read failed: {e}")
                time.sleep(1.0)
                continue
            angle = float(rq.angle_from_width(width))  # Viser draws the linkage from it
            joint.set_data(t0, np.array([angle]), np.zeros(1), np.zeros(1))
            seq, ok = self._gripper_last
            state.set_data(t0, width, max_width, grasped, self._gripper_busy, seq, ok)
            for ch, data in ((self._gripper_joint_ch, joint), (self._gstate_ch, state)):
                que = self.intr_pub_que_dict.get(ch) if ch else None
                if que is not None:
                    que.put(copy.deepcopy(data))
            time.sleep(max(0.0, 1.0 / self._gripper_rate - (time.time() - t0)))
