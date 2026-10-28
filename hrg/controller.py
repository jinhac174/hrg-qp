"""The reactive grasp controller on the FR3 + 2F-85 (LCM process, ``run_controller.py``).

Plan thread (``plan_freq``: the paper's 100 Hz)
-----------------------------------------------
At the *measured* joints ``q`` and width ``w`` (the feedback of the method):

* ``reach``: ``hrg.planner.ReachPlanner.step`` (eqs. 2-12) gives ``qdot_des``,
  ``wdot_des``. When both fingertips are within ``grasp_trigger_dist`` of
  their targets (paper: 1 cm in sim, 1.5 cm on the robot) the gripper closes
  (``close``).
  With the oriented reach (``hrg.orient``) the hand must also be within
  ``grasp_trigger_rot`` of its target orientation.
* ``close``: wait for the gripper. Fingers stopped on something: ``lift``.
  Nothing between them: open again and reach again (``reopen``).
* ``lift``: the TCP goes ``lift_height`` straight up (pose field through the
  same QP), then ``upright`` (``grasp.upright``) or ``hold``.
* ``upright``: the hand turns the held object back to its upright pose
  (``ReachPlanner.upright_goal``: the smallest tilt, raised to clear the
  table) until it is within ``upright_tol``, then ``hold``. During ``lift`` and ``upright`` the held object's points stay
  ``held_eps`` above the table (QP rows), and its weight (``mass`` of its
  config at its centre of mass) is added to ``g(q)`` (``payload_comp``). The
  object's pose in the hand is taken when the gripper reports the grasp and
  refreshed from the tracker whenever plausible (the grasped point still
  between the pads); the turn is a feedback loop on the object's tilt, its
  goal recomputed every plan, so an object that pivots in the fingers is
  turned until it is upright. The object
  lost (``is_grasped`` false) in ``lift``, ``upright`` or ``hold``: open and
  reach again.
* ``idle`` / ``close`` / ``reopen`` / ``hold``: hold the joints where the
  state began (``hold_gain (q_hold - q)``, through the planner like every
  other motion).
* ``home`` (key ``h``): joint-space field to ``default_q_values`` through the
  same QP.

Control thread (``ctrl_freq``, 1 kHz)
------------------------------------
Paper Sec. II-D: ``qdot_des`` is "integrated over one control interval to
obtain the desired joint position" and tracked by "PD control with gravity
compensation": every control tick (``ref_mode: step``, the paper)

    q_des  = q + qdot_des dt
    tau    = Kp (q_des - q) + Kd (qdot_des - qdot) + g(q) + C(q, qdot) qdot + tau_f(qdot_des)

(sent as ``joint_ctrl_t``; the sim / ``run_hardware.py`` evaluates the PD at
1 kHz with the latest measurement). ``Kp (q_des - q) = Kp dt qdot_des`` then
only adds ~1 % to the Kd term, so ``kp`` is 0: a damping (velocity)
controller, and position is fed back by the planner (the QP is solved at the
measured state every 10 ms; the holding states run a joint field).
``ref_mode: integral`` is the earlier variant, ``q_des <- q_des + qdot_des
dt`` within ``max_ref_err`` of ``q`` (with ``kp`` > 0 an integral of the
velocity error). ``tau_f`` is the friction feed-forward on the *commanded*
velocity, ``scale (coulomb tanh(qdot_des / v_smooth) + viscous qdot_des)``,
capped at ``max`` (``friction:``; on the robot from
``scripts/robot/identify_friction.py``). ``qdot_des`` passes a first-order
filter (``qd_filter``: the plan changes in steps at 100 Hz, the FR3 rejects
torque steps). If no plan arrived for ``max_plan_age`` s the joints are held
where they were (``stale_hold_kp`` / ``kd``) until plans come again.

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
from fr3.utils.lie.so3 import skew
from hrg.friction import friction_torque
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
            self._params, self._robot,
            FingertipTargets.load(resolve(self._obj.cfg["fingertip_targets"]), self._obj.cfg.get("approach_axis")),
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
        self._trigger_rot = np.radians(float(g.get("grasp_trigger_rot", 10.0)))
        self._upright = bool(g.get("upright", True))
        self._upright_clearance = float(g.get("upright_clearance", 0.03))
        self._upright_tol = np.radians(float(g.get("upright_tol", 3.0)))
        self._upright_timeout = float(g.get("upright_timeout", 8.0))
        self._held_eps = float(g.get("held_eps", 0.02))
        self._held_pts = self._held_points(int(g.get("held_points", 64)))
        self._payload_comp = bool(g.get("payload_comp", True))
        self._obj_com = self._object_com()
        self._held = None  # (mass, centre of mass in the TCP frame) while holding: added to g(q)
        self._home_gain = float(config["home_gain"])
        self._hold_gain = float(config.get("hold_gain", 5.0))
        self._hold_vmax = float(config.get("hold_vmax", 0.3))

        # Low level
        self._ref_mode = str(config.get("ref_mode", "step"))
        if self._ref_mode not in ("step", "integral"):
            raise ValueError(f"ref_mode must be 'step' or 'integral', got {self._ref_mode!r}")
        self._kp = np.asarray(config["kp"], dtype=float)
        self._kd = np.asarray(config["kd"], dtype=float)
        self._stale_kp = np.asarray(config.get("stale_hold_kp", [300, 300, 300, 300, 125, 75, 25]), dtype=float)
        self._stale_kd = np.asarray(config.get("stale_hold_kd", [35, 35, 35, 15, 15, 15, 7]), dtype=float)
        fr = config.get("friction", None) or {}
        self._fr_scale = float(fr.get("scale", 0.0))
        self._fr_coulomb = np.asarray(fr.get("coulomb", [0.0] * N_ARM), dtype=float)
        self._fr_viscous = np.asarray(fr.get("viscous", [0.0] * N_ARM), dtype=float)
        self._fr_vs = float(fr.get("v_smooth", 0.02))
        self._fr_max = np.asarray(fr.get("max", [1.0] * N_ARM), dtype=float)
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
        self._T_tcp_obj = None  # the object in the hand, recorded when the gripper closes
        self._grasp_point = np.zeros(3)  # the grasped contact pair's midpoint, object frame
        self._no_target = False  # reported that no target is reachable (once)
        self._q_hold = None  # joints the holding states keep
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
        self._q_stale = None
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
                self._held = None  # the reach opens the gripper
                self._enter("reach")
            self._streaming = True
        elif key == "x":
            self._streaming = False
            self._held = None
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
            self._q_hold = None  # a holding state holds where it begins
        self._state = state

    def _held_points(self, n: int) -> np.ndarray:
        """``n`` points of the object's surface (object frame, farthest-point
        sampled): what is kept above the table while it is held."""
        sdf = self._obj.cfg.get("sdf", {})
        if str(sdf.get("type", "")) == "point_cloud":
            P = np.load(resolve(sdf["path"]))[:, :3]
        else:
            P = np.asarray(self._obj.mesh.vertices, dtype=float)
        if len(P) <= n:
            return P.copy()
        idx = [int(np.argmin(P[:, 2]))]
        d = np.linalg.norm(P - P[idx[0]], axis=1)
        for _ in range(n - 1):
            idx.append(int(np.argmax(d)))
            d = np.minimum(d, np.linalg.norm(P - P[idx[-1]], axis=1))
        return P[idx]

    def _object_com(self) -> np.ndarray:
        """The object's centre of mass, object frame (the mean of its evenly
        sampled surface: a thin-walled object's)."""
        sdf = self._obj.cfg.get("sdf", {})
        if str(sdf.get("type", "")) == "point_cloud":
            return np.load(resolve(sdf["path"]))[:, :3].mean(axis=0)
        return np.asarray(self._obj.mesh.center_mass, dtype=float)

    def _set_held(self) -> None:
        """Holding the object: its weight goes into the gravity feed-forward."""
        if self._payload_comp and self._T_tcp_obj is not None:
            self._held = (self._obj.mass, self._T_tcp_obj[:3, :3] @ self._obj_com + self._T_tcp_obj[:3, 3])

    def _payload_torque(self, q) -> np.ndarray:
        """``J_p(c)^T m g`` for the held object at ``c`` (TCP frame): what
        holds it up (``None`` held: zeros)."""
        held = self._held
        if held is None:
            return np.zeros(N_ARM)
        m, c = held
        T = self._robot.pose(q, self._pin_ctrl)
        J = self._robot.jacobian(q, self._pin_ctrl)
        Jp = J[3:] - skew(T[:3, :3] @ c) @ J[:3]
        return Jp.T @ np.array([0.0, 0.0, m * 9.81])

    def _hold_plan(self, q) -> Plan:
        """Keep the joints where the holding state began (through the
        planner: with ``kp = 0`` nothing else holds the arm)."""
        if self._q_hold is None:
            self._q_hold = q.copy()
        qd = np.clip(self._hold_gain * (self._q_hold - q), -self._hold_vmax, self._hold_vmax)
        return Plan(qd)

    def _held_tcp(self):
        """The held object's points in the TCP frame (None before a grasp)."""
        if self._T_tcp_obj is None:
            return None
        return self._held_pts @ self._T_tcp_obj[:3, :3].T + self._T_tcp_obj[:3, 3]

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
        self._held = None
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
        with self._fsm_lock:
            state = self._state
        if state == "reach":  # planned outside the lock (the keyboard takes it)
            return self._hold_plan(q) if T_obj is None else self._reach(q, w, T_obj, obstacles)
        with self._fsm_lock:
            if state != self._state:  # changed by a key meanwhile
                return self._hold_plan(q)
            if state == "close":
                done = self._gripper_done(gs)
                if done:
                    self._T_lift = self._planner.tcp_pose(q)
                    if T_obj is not None:  # the object as it sits in the closed fingers
                        self._T_tcp_obj = np.linalg.inv(self._T_lift) @ T_obj
                    self._set_held()
                    self._T_lift[2, 3] += self._lift_height
                    self._lift_t0 = time.time()
                    print(f"[HRG] Grasped after {time.time() - self._t_start:.1f} s; lifting.")
                    self._enter("lift")
                elif done is False:
                    print("[HRG] Grasp failed (nothing between the fingers): opening, reaching again.")
                    self._reopen()
                return self._hold_plan(q)
            if state in ("lift", "upright", "hold"):
                if gs is not None and not gs.is_grasped:
                    print("[HRG] Object lost: opening, reaching again.")
                    self._reopen()
                    return self._hold_plan(q)
                if state == "hold":
                    return self._hold_plan(q)
                T_tcp = self._planner.tcp_pose(q)
                if state == "lift":
                    if (np.linalg.norm(T_tcp[:3, 3] - self._T_lift[:3, 3]) <= self._lift_tol
                            or time.time() - self._lift_t0 > self._lift_timeout):
                        return self._after_lift(q, T_tcp, T_obj)
                    return Plan(self._planner.lift_step(q, w, self._T_lift, **self._lift_cfg,
                                                        held=self._held_tcp(), held_eps=self._held_eps))
                # upright: a feedback loop on the object's tilt in the hand (refreshed by the
                # tracker when plausible), the goal recomputed every plan
                self._refresh_in_hand(T_tcp, T_obj)
                T_up, tilt = self._planner.upright_goal(T_tcp, self._T_tcp_obj, self._held_pts,
                                                        self._upright_clearance, self._T_lift[:3, 3])
                if ((tilt <= self._upright_tol and np.linalg.norm(T_tcp[:3, 3] - T_up[:3, 3]) <= self._lift_tol)
                        or time.time() - self._lift_t0 > self._upright_timeout):
                    print(f"[HRG] Upright ({np.degrees(tilt):.1f} deg, {time.time() - self._t_start:.1f} s). "
                          "Press x to release.")
                    self._enter("hold")
                    return self._hold_plan(q)
                return Plan(self._planner.lift_step(q, w, T_up, **self._lift_cfg,
                                                    held=self._held_tcp(), held_eps=self._held_eps))
            if state == "reopen":
                if self._gripper_done(gs) is not None:
                    self._planner.reset()
                    self._enter("reach")
                    self._streaming = True
                return self._hold_plan(q)
            if state == "home":
                return Plan(self._planner.home_step(q, w, self._q_home, self._home_gain))
        return self._hold_plan(q)

    def _refresh_in_hand(self, T_tcp, T_obj) -> bool:
        """Take the object's pose in the hand from the tracker if it is
        plausible: the grasped point (between the pads) within 2 cm of where
        it was, the object turned by at most 45 deg (it may pivot about the
        grasp; the tracker may also lose an object held in the fingers, the
        hand cannot). True if refreshed."""
        if T_obj is None or self._T_tcp_obj is None:
            return False
        T_live = np.linalg.inv(T_tcp) @ T_obj
        d_rot = np.linalg.norm(Rotation.from_matrix(T_live[:3, :3] @ self._T_tcp_obj[:3, :3].T).as_rotvec())
        g = self._grasp_point
        d_pos = np.linalg.norm((T_live[:3, :3] - self._T_tcp_obj[:3, :3]) @ g + T_live[:3, 3] - self._T_tcp_obj[:3, 3])
        if d_rot > np.radians(45.0) or d_pos > 0.02:
            return False
        self._T_tcp_obj = T_live
        self._set_held()
        return True

    def _after_lift(self, q, T_tcp, T_obj) -> Plan:
        """Lifted: turn the object upright (``upright``) or hold."""
        self._refresh_in_hand(T_tcp, T_obj)
        tilt = 0.0
        if self._upright and self._T_tcp_obj is not None:
            T_up, tilt = self._planner.upright_goal(T_tcp, self._T_tcp_obj, self._held_pts, self._upright_clearance)
        if tilt <= self._upright_tol:
            print(f"[HRG] Lifted ({time.time() - self._t_start:.1f} s). Press x to release.")
            self._enter("hold")
            return self._hold_plan(q)
        print(f"[HRG] Lifted ({time.time() - self._t_start:.1f} s); turning the object upright "
              f"({np.degrees(tilt):.0f} deg).")
        self._T_lift, self._lift_t0 = T_up, time.time()  # its position stays the goal's
        self._enter("upright")
        return self._hold_plan(q)

    def _reach(self, q, w, T_obj, obstacles) -> Plan:
        t0 = time.perf_counter()
        r = self._planner.step(q, w, T_obj, obstacles)
        if r is None:
            if not self._no_target:
                print("[HRG] No reachable fingertip target for this object pose (holding).")
            self._no_target = True
            return self._hold_plan(q)
        self._no_target = False
        if not r.qp_ok:
            print("[HRG] QP failed: holding this plan.")
        err = np.linalg.norm(r.x - r.x_tgt, axis=1)
        aligned = r.R_star is None or r.info["rot_err"] <= self._trigger_rot
        if np.all(err <= self._trigger) and aligned:
            with self._fsm_lock, self._gripper_lock:
                if self._state != "reach":  # a key meanwhile
                    return self._hold_plan(q)
                self._streaming = False
                # The object in the hand, for the lift (live tracking is
                # unreliable once the gripper hides it).
                self._T_tcp_obj = np.linalg.inv(self._planner.tcp_pose(q)) @ T_obj
                tg = self._planner.targets
                self._grasp_point = 0.5 * (tg.p1[r.target] + tg.p2[r.target])  # object frame
                self._wait_for(self._grasp(r.contact_width, float(np.linalg.norm(r.x_tgt[1] - r.x_tgt[0]))))
                self._enter("close")
            ts = np.asarray(self._plan_times or [np.nan]) * 1e3
            rot = "" if r.R_star is None else f", hand {np.degrees(r.info['rot_err']):.1f} deg off"
            print(f"[HRG] Fingertips at the targets (err {err[0] * 1e3:.1f} / {err[1] * 1e3:.1f} mm{rot}): "
                  f"grasping at {r.contact_width * 1e3:.1f} mm, {self._grasp_force:.0f} N")
            print(f"[HRG] plan time {np.median(ts):.1f} ms median, {np.max(ts):.1f} ms max over the reach")
            self._plan_times.clear()
            plan = self._hold_plan(q)  # (the state is close now: holds from here)
            plan.reach = r
            return plan
        self._plan_times.append(time.perf_counter() - t0)
        if len(self._plan_times) >= 300:
            ts = np.asarray(self._plan_times) * 1e3
            print(f"[HRG] plan time {np.median(ts):.1f} ms median, {ts.max():.1f} ms max; "
                  f"h = {r.info['h'] * 100:.1f} cm, alpha {r.info['alpha']:.2f}, beta {r.info['beta']:.2f}, "
                  f"{r.info['n_rows']} constraint rows{binding_text(r.info['binding'])}")
            self._plan_times.clear()
        self._q_hold = None  # moving: a later hold starts where it is then
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

        # Arm: q_des from qdot_des (ref_mode); PD + gravity / Coriolis + friction.
        a = float(np.clip(self._qd_filter, 0.0, 1.0))
        self._qd_filt = a * (self._qd_des if fresh else 0.0) + (1.0 - a) * self._qd_filt
        tau_ff = (self._robot.gravity_coriolis(q, qd, self._pin_ctrl) + self.friction(self._qd_filt)
                  + self._payload_torque(q))
        kp, kd = self._kp, self._kd
        if not fresh:  # no plan: hold the joints where the plans stopped
            if self._q_stale is None:
                self._q_stale = q.copy()
            self._q_ref = self._q_stale.copy()
            kp, kd = self._stale_kp, self._stale_kd
        elif self._ref_mode == "step":  # the paper: one control interval from the measured q
            self._q_stale = None
            self._q_ref = q + self._qd_filt * self._ctrl_dt
        else:  # running integral of the commands
            self._q_stale = None
            if self._q_ref is None:
                self._q_ref = q.copy()
            self._q_ref = np.clip(self._q_ref + self._qd_filt * self._ctrl_dt,
                                  q - self._max_ref_err, q + self._max_ref_err)
        cmd = JointCtrlData(num_joints=N_ARM)
        cmd.set_data(self._cur_time, self._q_ref.copy(), self._qd_filt.copy(), tau_ff, kp, kd)
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

    def friction(self, qd_cmd: np.ndarray) -> np.ndarray:
        """Friction feed-forward on the commanded joint velocities
        (``hrg.friction``): zero at a zero command."""
        return friction_torque(qd_cmd, self._fr_scale, self._fr_coulomb, self._fr_viscous, self._fr_vs, self._fr_max)

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
