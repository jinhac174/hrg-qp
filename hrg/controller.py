"""The reactive grasp controller on the FR3 + 2F-85 (LCM process, ``run_controller.py``).

Plan thread (``plan_freq``: the paper's 100 Hz)
-----------------------------------------------
At the *measured* joints ``q`` and width ``w`` (the feedback of the method):

* ``reach``: ``hrg.planner.ReachPlanner.step`` (eqs. 2-12) gives ``qdot_des``,
  ``wdot_des``. When both fingertips are within ``grasp_trigger_dist`` of
  their targets (paper: 1 cm in sim, 1.5 cm on the robot) the gripper closes
  (``close``).
* ``close``: wait for the gripper. Fingers stopped on something: ``lift``.
  Nothing between them: open again and reach again (``reopen``).
* ``lift``: the TCP goes ``lift_height`` straight up (pose field through the
  same QP), then ``hold``. The object lost (``is_grasped`` false) in ``lift``
  or ``hold``: open and reach again.
* ``idle`` / ``hold``: zero velocity (the joint reference stays where it is).
* ``home`` (key ``h``): joint-space field to ``default_q_values`` through the
  same QP.

Control thread (``ctrl_freq``, 1 kHz)
------------------------------------
Paper Sec. II-D: ``qdot_des`` is "integrated over one control interval to
obtain the desired joint position" and tracked by "PD control with gravity
compensation": every control tick

    q_des <- q_des + qdot_des dt
    tau    = Kp (q_des - q) + Kd (qdot_des - qdot) + g(q) + C(q, qdot) qdot

(sent as ``joint_ctrl_t``; the sim / ``run_hardware.py`` evaluates the PD at
1 kHz with the latest measurement). The desired position is the running
integral of the commands. Restarting it from the measured ``q`` every tick
would leave ``Kp (q_des - q) = Kp qdot_des dt`` (~0.02 Nm at 1 kHz), a pure
damper that the FR3's joint friction (0.2-1 Nm) stops at low commanded speeds.
``q_des`` stays within ``max_ref_err`` of ``q`` (no wind-up when the arm is
blocked or pushed); since the QP is solved at the measured state, a push or a
lag is fed back at the next plan. ``qdot_des`` passes a first-order filter
(``qd_filter``: the plan changes in steps at 100 Hz, the FR3 rejects torque
steps) and drops to zero if no plan arrived for ``max_plan_age`` s.

The gripper is position-servoed (the Robotiq, and its sim): ``wdot_des`` is
realised the same way, as a set-point ``w_ref += wdot_des dt`` kept within
``width_max_lead`` of the measured width and streamed as ``move`` commands at
``width_stream_freq``.

Keyboard (controller terminal)
------------------------------
``v`` reach and grasp (again)   ``x`` open and stop   ``h`` home   ``p`` pause   ``q`` quit
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Empty, Full

import numpy as np
from scipy.spatial.transform import Rotation

from fr3.controller.BaseController import BaseController
from fr3.data_type.basic_types.ColInfoData import ColInfoData
from fr3.data_type.basic_types.GripperCmdData import GripperCmdData
from fr3.data_type.basic_types.GripperStateData import GripperStateData
from fr3.data_type.basic_types.JointCtrlData import JointCtrlData
from fr3.data_type.basic_types.JointMeasData import JointMeasData
from fr3.data_type.basic_types.NamedVecListData import NamedVecListData
from fr3.data_type.basic_types.SE3PoseData import SE3PoseData
from hrg import obstacles as obstacle_msg
from hrg.hand import MAX_WIDTH, MAX_WIDTH_RATE, N_ARM
from hrg.objects import GraspObject
from hrg.paths import resolve
from hrg.planner import HRGParams, ReachPlanner, ReachStep, binding_text
from hrg.robot import RobotModel
from hrg.sdf import sdf_from_config
from hrg.targets import FingertipTargets


@dataclass
class Plan:
    """What the plan thread hands to the control thread."""

    qd: np.ndarray  # (7,)
    wd: float = 0.0
    reach: ReachStep | None = None


class HRGController(BaseController):
    def __init__(self, config):
        super().__init__(config)
        sm, pm, g = config["sub_manager"], config["pub_manager"], config["grasp"]

        self._robot = RobotModel(
            config["urdf_path"], config["joint_velocity_limit"],
            env_pairs=config["col_pairs"], self_pairs=config["self_col_pairs"],
            env_margin=float(config["env_col_margin"]), self_margin=float(config["self_col_margin"]),
        )
        self._obj = GraspObject(config["object_config"])
        self._params = HRGParams.from_config(config["hrg"])
        self._planner = ReachPlanner(
            self._params, self._robot, FingertipTargets.load(resolve(self._obj.cfg["fingertip_targets"])),
            sdf_from_config(self._obj.cfg["sdf"]), table_z=float(g["table_z"]),
        )
        self._q_home = np.asarray(config["default_q_values"], dtype=float)
        self._plan_dt = 1.0 / float(config["plan_freq"])

        # Grasp / lift / home
        self._trigger = float(g["grasp_trigger_dist"])
        self._grasp_force = float(g["grasp_force"])
        self._grasp_speed = float(g["grasp_speed"])
        self._gripper_timeout = float(g["gripper_timeout"])
        self._lift_height = float(g["lift_height"])
        self._lift_timeout = float(g["lift_timeout"])
        self._lift_tol = float(g["lift_tol"])
        self._lift_cfg = dict(v_max=float(g["lift_speed"]), w_max=float(g["lift_angular_speed"]),
                              pos_gain=float(g["lift_pos_gain"]), rot_gain=float(g["lift_rot_gain"]),
                              rot_weight=float(g["lift_rot_weight"]))
        self._home_gain = float(config["home_gain"])

        # Low level
        self._kp = np.asarray(config["kp"], dtype=float)
        self._kd = np.asarray(config["kd"], dtype=float)
        self._qd_filter = float(config["qd_filter"])
        self._max_ref_err = float(config["max_ref_err"])
        self._max_plan_age = float(config["max_plan_age"])
        self._w_stream_dt = 1.0 / float(config["width_stream_freq"])
        self._w_lead = float(config["width_max_lead"])

        # Channels
        self._arm_meas_channel = sm["arm_joint_meas_channel"]
        self._gripper_state_channel = sm["gripper_state_channel"]
        self._robot_col_channel = sm["robot_col_info_channel"]
        self._static_col_channel = sm["static_col_info_channel"]
        self._object_pose_channel = sm["object_pose_channel"]
        self._obstacles_channel = sm.get("obstacles_channel", "")
        self._gripper_cmd_channel = pm["gripper_cmd_channel"]
        self._fingertip_channel = pm.get("fingertip_channel", "")
        self._paths_channel = pm.get("fingertip_paths_channel", "")

        # Measurements (written by the control thread, read by the plan thread)
        self._lock = threading.Lock()
        self._q = self._q_home.copy()
        self._qd = np.zeros(N_ARM)
        self._have_q = False
        self._T_obj = None
        self._obstacles: list = []
        self._gripper_state: GripperStateData | None = None

        # State machine (plan thread + keyboard, under _fsm_lock)
        self._fsm_lock = threading.Lock()
        self._state = "idle"
        self._t_start = time.time()
        self._T_lift = None
        self._lift_t0 = 0.0
        self._wait_seq = None
        self._wait_t0 = 0.0
        self._plan_times: list[float] = []

        # Gripper commands come from both threads (width stream: control;
        # grasp / open: plan + keyboard). One lock, and the stream re-checks
        # _streaming under it, so no move can follow (and cancel) a grasp.
        self._gripper_lock = threading.RLock()
        self._gripper_seq = 0
        self._streaming = False

        # Control thread
        self._plan_que = None  # set in initialize (queue.Queue(maxsize=1))
        self._qd_des = np.zeros(N_ARM)
        self._wd_des = 0.0
        self._qd_filt = np.zeros(N_ARM)
        self._q_ref = None
        self._plan_stamp = -np.inf
        self._w_ref = None
        self._w_last_sent = None
        self._w_last_sent_t = -np.inf
        self._pin_ctrl = self._robot.data()

        self._stop = threading.Event()
        self._plan_thread = threading.Thread(target=self._plan_loop, name="hrg_plan", daemon=False)
        print(f"[HRG] field={self._params.field}, {len(self._planner.targets)} fingertip target pairs for "
              f"{self._obj.name}. Keys: v = reach + grasp, x = open + stop, h = home, p / q.")

    # ------------------------------------------------------------ lifecycle
    def initialize(self) -> None:
        from queue import Queue

        self._plan_que = Queue(maxsize=1)
        robot_col = self._wait_col_info(self.intr_sub_que_dict[self._robot_col_channel], "robot")
        static_col = self._wait_col_info(self.extr_sub_que_dict[self._static_col_channel], "static")
        self._robot.set_collision_geometry(robot_col, static_col)
        print("[HRG] Collision geometry received.")

    @staticmethod
    def _wait_col_info(que, label: str, timeout: float = 60.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not que.empty():
                msg = que.get()
                if not isinstance(msg, ColInfoData):
                    raise ValueError(f"Expected ColInfoData for {label}, got {type(msg)}")
                return msg.get_data()[1]
            time.sleep(0.001)
        raise TimeoutError(f"No {label} collision info (sim: run_env.py; robot: run_hardware.py).")

    def _sub_thread_start(self) -> None:
        self._plan_thread.start()

    def _sub_thread_stop(self) -> None:
        self._stop.set()
        if self._plan_thread.is_alive():
            self._plan_thread.join(timeout=2.0)

    # ------------------------------------------------------------ inputs (control thread)
    @staticmethod
    def _latest(que_dict, channel):
        if not channel or channel not in que_dict:
            return None
        que, item = que_dict[channel], None
        while not que.empty():
            item = que.get()
        return item

    def _check_and_get_data_from_que(self) -> None:
        meas = self._latest(self.intr_sub_que_dict, self._arm_meas_channel)
        gs = self._latest(self.intr_sub_que_dict, self._gripper_state_channel)
        pose = self._latest(self.extr_sub_que_dict, self._object_pose_channel)
        obs = self._latest(self.extr_sub_que_dict, self._obstacles_channel)
        with self._lock:
            if isinstance(meas, JointMeasData):
                _, q, qd, _ = meas.get_data()
                self._q[:], self._qd[:] = q[:N_ARM], qd[:N_ARM]
                self._have_q = True
            if isinstance(gs, GripperStateData):
                self._gripper_state = gs
            if isinstance(pose, SE3PoseData):
                _, pos, wxyz = pose.get_data()
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat(np.roll(wxyz, -1)).as_matrix()
                T[:3, 3] = pos
                self._T_obj = T
            if isinstance(obs, NamedVecListData):
                self._obstacles = obstacle_msg.decode(obs.get_data()[2]) if obs.num_vecs else []

    def _snapshot(self):
        with self._lock:
            gs = self._gripper_state
            return (self._q.copy(), None if gs is None else float(gs.width),
                    None if self._T_obj is None else self._T_obj.copy(), list(self._obstacles), gs)

    # ------------------------------------------------------------ keyboard
    def _handle_key(self, key: str) -> None:
        if key == "v":
            if self._snapshot()[2] is None:
                print("[HRG] No object pose yet.")
                return
            with self._fsm_lock:
                self._planner.reset()
                self._t_start = time.time()
                self._enter("reach")
            self._streaming = True
        elif key == "x":
            self._streaming = False
            self._gripper(GripperCmdData.MOVE, MAX_WIDTH)
            with self._fsm_lock:
                self._enter("idle")
        elif key == "h":
            self._streaming = False
            with self._fsm_lock:
                self._enter("home")
        else:
            super()._handle_key(key)

    def _enter(self, state: str) -> None:
        if state != self._state:
            print(f"[HRG] {self._state} -> {state}")
        self._state = state

    # ------------------------------------------------------------ gripper
    def _send_gripper(self, mode: int, width: float, speed: float, force: float, eps_in: float, eps_out: float) -> int:
        with self._gripper_lock:
            self._gripper_seq += 1
            cmd = GripperCmdData()
            cmd.set_data(self._cur_time, self._gripper_seq, mode, width, speed, force, eps_in, eps_out)
            self.pub_que_dict[self._gripper_cmd_channel].put(cmd)
            return self._gripper_seq

    def _gripper(self, mode: int, width: float) -> int:
        return self._send_gripper(mode, width, self._grasp_speed, 0.0, 0.01, 0.01)

    def _grasp(self, contact_width: float, pregrasp_width: float) -> int:
        """Close until contact and squeeze with ``grasp_force``. Success = the
        fingers stopped on something: between half the contact width (an
        empty close ends near 0) and just inside the pre-grasp opening (the
        2F-85's flat pads meet a slanted wall, a bowl or glass rim, with their
        edges, so they stop well above the wall's thickness)."""
        lo = 0.5 * contact_width
        hi = max(pregrasp_width - 0.002, contact_width + 0.002)
        return self._send_gripper(GripperCmdData.GRASP, contact_width, self._grasp_speed, self._grasp_force,
                                  contact_width - lo, hi - contact_width)

    def _wait_for(self, seq: int) -> None:
        self._wait_seq, self._wait_t0 = seq, time.time()

    def _gripper_done(self, gs):
        """``None`` while the awaited command runs, else its success flag."""
        if gs is None or self._wait_seq is None:
            return None
        if gs.last_seq == self._wait_seq and not gs.busy:
            return bool(gs.last_success)
        if time.time() - self._wait_t0 > self._gripper_timeout:
            print("[HRG] Gripper command timed out.")
            return False
        return None

    def _reopen(self) -> None:
        with self._gripper_lock:
            self._streaming = False
        self._wait_for(self._gripper(GripperCmdData.MOVE, MAX_WIDTH))
        self._enter("reopen")

    # ------------------------------------------------------------ plan thread
    def _plan_loop(self) -> None:
        last = -np.inf
        while not self._stop.is_set():
            if self._have_q and self._cur_time - last >= self._plan_dt:
                last = self._cur_time
                plan = self._plan()
                try:
                    self._plan_que.get_nowait()
                except Empty:
                    pass
                try:
                    self._plan_que.put_nowait(plan)
                except Full:
                    pass
            time.sleep(0.0001)

    def _plan(self) -> Plan:
        q, w, T_obj, obstacles, gs = self._snapshot()
        w = MAX_WIDTH if w is None else w
        hold = Plan(np.zeros(N_ARM))
        with self._fsm_lock:
            state = self._state
        if state == "reach":  # planned outside the lock (the keyboard takes it)
            return hold if T_obj is None else self._reach(q, w, T_obj, obstacles)
        with self._fsm_lock:
            if state != self._state:  # changed by a key meanwhile
                return hold
            if state == "close":
                done = self._gripper_done(gs)
                if done:
                    self._T_lift = self._planner.tcp_pose(q)
                    self._T_lift[2, 3] += self._lift_height
                    self._lift_t0 = time.time()
                    print(f"[HRG] Grasped after {time.time() - self._t_start:.1f} s; lifting.")
                    self._enter("lift")
                elif done is False:
                    print("[HRG] Grasp failed (nothing between the fingers): opening, reaching again.")
                    self._reopen()
                return hold
            if state in ("lift", "hold"):
                if gs is not None and not gs.is_grasped:
                    print("[HRG] Object lost: opening, reaching again.")
                    self._reopen()
                    return hold
                if state == "hold":
                    return hold
                T_tcp = self._planner.tcp_pose(q)
                if (np.linalg.norm(T_tcp[:3, 3] - self._T_lift[:3, 3]) <= self._lift_tol
                        or time.time() - self._lift_t0 > self._lift_timeout):
                    print(f"[HRG] Lifted ({time.time() - self._t_start:.1f} s). Press x to release.")
                    self._enter("hold")
                    return hold
                return Plan(self._planner.lift_step(q, w, self._T_lift, **self._lift_cfg))
            if state == "reopen":
                if self._gripper_done(gs) is not None:
                    self._planner.reset()
                    self._enter("reach")
                    self._streaming = True
                return hold
            if state == "home":
                return Plan(self._planner.home_step(q, w, self._q_home, self._home_gain))
        return hold

    def _reach(self, q, w, T_obj, obstacles) -> Plan:
        t0 = time.perf_counter()
        r = self._planner.step(q, w, T_obj, obstacles)
        if r is None:
            print("[HRG] No reachable fingertip target for this object pose.")
            return Plan(np.zeros(N_ARM))
        if not r.qp_ok:
            print("[HRG] QP failed: holding this plan.")
        err = np.linalg.norm(r.x - r.x_tgt, axis=1)
        if np.all(err <= self._trigger):
            with self._fsm_lock, self._gripper_lock:
                if self._state != "reach":  # a key meanwhile
                    return Plan(np.zeros(N_ARM))
                self._streaming = False
                self._wait_for(self._grasp(r.contact_width, float(np.linalg.norm(r.x_tgt[1] - r.x_tgt[0]))))
                self._enter("close")
            ts = np.asarray(self._plan_times or [np.nan]) * 1e3
            print(f"[HRG] Fingertips at the targets (err {err[0] * 1e3:.1f} / {err[1] * 1e3:.1f} mm): "
                  f"grasping at {r.contact_width * 1e3:.1f} mm, {self._grasp_force:.0f} N")
            print(f"[HRG] plan time {np.median(ts):.1f} ms median, {np.max(ts):.1f} ms max over the reach")
            self._plan_times.clear()
            return Plan(np.zeros(N_ARM), reach=r)
        self._plan_times.append(time.perf_counter() - t0)
        if len(self._plan_times) >= 300:
            ts = np.asarray(self._plan_times) * 1e3
            print(f"[HRG] plan time {np.median(ts):.1f} ms median, {ts.max():.1f} ms max; "
                  f"h = {r.info['h'] * 100:.1f} cm, alpha {r.info['alpha']:.2f}, beta {r.info['beta']:.2f}, "
                  f"{r.info['n_rows']} constraint rows{binding_text(r.info['binding'])}")
            self._plan_times.clear()
        return Plan(r.qd, r.wd, reach=r)

    # ------------------------------------------------------------ control thread
    def _update(self) -> None:
        if not self._have_q:  # nothing is commanded before the first measurement
            return
        try:
            plan = self._plan_que.get_nowait()
        except Empty:
            plan = None
        if plan is not None:
            self._qd_des, self._wd_des = plan.qd, plan.wd
            self._plan_stamp = self._cur_time
            if plan.reach is not None:
                self._publish_reach(plan.reach)
        with self._lock:
            q, qd = self._q.copy(), self._qd.copy()
            gs = self._gripper_state
        fresh = self._cur_time - self._plan_stamp <= self._max_plan_age

        # Arm: q_des += qdot_des dt; PD + gravity / Coriolis compensation.
        a = float(np.clip(self._qd_filter, 0.0, 1.0))
        self._qd_filt = a * (self._qd_des if fresh else 0.0) + (1.0 - a) * self._qd_filt
        if self._q_ref is None:
            self._q_ref = q.copy()
        self._q_ref = np.clip(self._q_ref + self._qd_filt * self._ctrl_dt, q - self._max_ref_err, q + self._max_ref_err)
        cmd = JointCtrlData(num_joints=N_ARM)
        cmd.set_data(self._cur_time, self._q_ref.copy(), self._qd_filt.copy(),
                     self._robot.gravity_coriolis(q, qd, self._pin_ctrl), self._kp, self._kd)
        self.ctrl_pub_que.put(cmd)

        # Gripper: w_ref += wdot_des dt, streamed as move set-points.
        if gs is None:
            return
        w_meas = float(gs.width)
        if self._w_ref is None or not self._streaming:
            self._w_ref = w_meas
            return
        wd = self._wd_des if fresh else 0.0
        self._w_ref = float(np.clip(self._w_ref + wd * self._ctrl_dt, 0.0, MAX_WIDTH))
        self._w_ref = float(np.clip(self._w_ref, w_meas - self._w_lead, w_meas + self._w_lead))
        if (self._cur_time - self._w_last_sent_t >= self._w_stream_dt
                and (self._w_last_sent is None or abs(self._w_ref - self._w_last_sent) > 5e-4)):
            with self._gripper_lock:
                if not self._streaming:
                    return
                self._send_gripper(GripperCmdData.MOVE, self._w_ref, MAX_WIDTH_RATE, 0.0, 0.01, 0.01)
            self._w_last_sent, self._w_last_sent_t = self._w_ref, self._cur_time

    # ------------------------------------------------------------ viz
    # Fingertip names: thumb = fingertip 1 (left pad), index = fingertip 2.
    _FT = ("thumb", "index")

    def _put_latest(self, channel, item) -> None:
        que = self.pub_que_dict.get(channel)
        if que is None:
            return
        try:
            que.get_nowait()
        except Empty:
            pass
        try:
            que.put_nowait(item)
        except Full:
            pass

    def _publish_reach(self, r: ReachStep) -> None:
        if self._fingertip_channel:
            msg = NamedVecListData(name=self._fingertip_channel)
            msg.set_data(self._cur_time, [f"{f}_{k}" for k in ("cur", "tgt") for f in self._FT],
                         np.vstack([r.x, r.x_tgt]))
            self._put_latest(self._fingertip_channel, msg)
        if self._paths_channel and r.paths is not None:
            P = r.paths[:, :: max(1, r.paths.shape[1] // 25)]  # ~25 points each
            msg = NamedVecListData(name=self._paths_channel)
            msg.set_data(self._cur_time, [f"{self._FT[f]}_{k}" for f in range(P.shape[0]) for k in range(P.shape[1])],
                         P.reshape(-1, 3))
            self._put_latest(self._paths_channel, msg)
