"""MuJoCo env for the simulation studies (paper Sec. III-A: Table I, Fig. 7, 8).

The FR3 + 2F-85 in the scene of ``scene_xml_path`` (``fr3.env.MujocoColEnv``:
it also publishes the robot / scene collision primitives for ``Gamma(q)``)
plus:

* one free object (``object_config``, ``hrg.objects.GraspObject``), placed at
  a random x / y / yaw within ``obj_spawn_xy_half`` / ``obj_spawn_yaw_half``
  of ``obj_spawn_xy_center``, rotated by ``obj_spawn_rpy`` (paper Table I: the
  mug and wine glass also lie on their side), never within
  ``place_clearance`` of the hand. Its pose is published on
  ``object_pose_channel`` at ``1 / object_pose_update_dt`` (on the robot:
  FoundationPose).
* ``obstacles``: boxes moving on the table (mocap bodies, sinusoidal in x / y),
  published on ``obstacles_channel`` (``hrg.obstacles``). Robot-obstacle
  contacts are counted and reported (the paper counts them as failures).

Keys (terminal or viewer): ``k`` push the hand with ``push_force`` N,
horizontally in a random direction, for ``push_duration`` s (the paper's
"physical pushes"); ``o`` move the object anywhere within ``jump_xy_half``
(a "sudden change of the target"); ``m`` **carry** the object: it is welded
to a kinematic carrier (a mocap body) that glides along a smooth random path
through ``carry.box`` (mid-air included, ``hrg.carry``) at ``carry.speed``
and then holds it there, like a hand holding it out, until both finger pads
touch it, when the carrier lets go; ``r`` reset.

Config::

    obstacles:
      - {half: [0.03, 0.06, 0.06], centre: [0.45, 0.2, 0.06], amp: [0.0, 0.08], freq: 0.15, phase: 0.0}
    carry: {box: [[0.35, 0.65], [-0.3, 0.3], [0.05, 0.4]], waypoints: 4, speed: 0.3, yaw_range: 1.0, seed: 0}
"""

from __future__ import annotations

import copy

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from fr3.data_type.basic_types.NamedVecListData import NamedVecListData
from fr3.data_type.basic_types.SE3PoseData import SE3PoseData
from fr3.env.MujocoColEnv import MujocoColEnv
from hrg import obstacles as obstacle_msg
from hrg.carry import CarryPath, random_waypoints
from hrg.objects import GraspObject

CARRY_EQ = "carry"  # the weld between the carrier and the object
PAD_BODIES = ("robotiq_left_pad", "robotiq_right_pad")


class HRGSimEnv(MujocoColEnv):
    def __init__(self, config, *args, **kwargs):
        # Set before super().__init__: it calls _modify_mjspec() and reset().
        self.obj = GraspObject(config["object_config"])
        if config.get("obj_spawn_rpy", None) is not None:
            self.obj.R_spawn = Rotation.from_euler("xyz", list(config["obj_spawn_rpy"])).as_matrix()
        pm = config["pub_manager"]
        self._object_pose_channel = pm.get("object_pose_channel", "")
        self._obstacles_channel = pm.get("obstacles_channel", "")
        self._object_pose_dt = float(config.get("object_pose_update_dt", 1.0 / 30.0))
        self._last_object_pose_t = -1e9
        self._spawn_xy_center = np.asarray(config.get("obj_spawn_xy_center", [0.5, 0.0]), dtype=float)
        self._spawn_xy_half = np.asarray(config.get("obj_spawn_xy_half", [0.05, 0.08]), dtype=float)
        self._spawn_yaw_half = float(config.get("obj_spawn_yaw_half", np.pi))
        self._jump_xy_half = np.asarray(config.get("jump_xy_half", self._spawn_xy_half), dtype=float)
        self._place_clearance = float(config.get("place_clearance", 0.03))
        seed = int(config.get("obj_spawn_seed", 0))
        self._rng = np.random.default_rng(seed or None)
        self._obs_cfg = [dict(o) for o in (config.get("obstacles", []) or [])]
        self._push_force = float(config.get("push_force", 80.0))
        self._push_duration = float(config.get("push_duration", 0.3))
        self._push_rng = np.random.default_rng(seed + 100)
        self._push_until = -1.0
        self._push_vec = np.zeros(3)
        self._obs_contact_steps = 0
        self._object_body_id = -1
        self._carry_cfg = dict(config.get("carry", None) or {})
        self._carry_rng = np.random.default_rng(int(self._carry_cfg.get("seed", seed)) + 200)
        self._carry = None  # (CarryPath, t_start) while the object is carried / held out
        self._carry_touch_steps = 0
        super().__init__(config, *args, **kwargs)
        self._pad_bodies = {self.mj_model.body(n).id for n in PAD_BODIES}
        self._carry_eq = self.mj_model.equality(CARRY_EQ).id
        self._carrier = self.mj_model.body("carrier").mocapid[0]
        self._robot_bodies = {b for b in range(self.mj_model.nbody)
                              if self.mj_model.body(b).name.startswith(("fr3_", "robotiq_"))}
        print(f"[HRGSimEnv] object={self.obj.name} ({self.obj.mass} kg), {len(self._obs_cfg)} moving obstacles. "
              "Keys: k = push the hand, o = move the object, r = reset.")

    # ------------------------------------------------------------------ scene
    def _modify_mjspec(self) -> None:
        self.obj.add_to_spec(self.mj_spec)
        # Grasping-friendly contacts: elliptic friction cones and the noslip
        # solver, so a held object does not creep out of the fingers.
        opt = self.mj_spec.option
        opt.cone = mj.mjtCone.mjCONE_ELLIPTIC
        opt.impratio = 10.0
        opt.noslip_iterations = 5
        for i, o in enumerate(self._obs_cfg):
            body = self.mj_spec.worldbody.add_body(name=f"obstacle_{i}", mocap=True, pos=list(o["centre"]))
            body.add_geom(name=f"obs_box_{i}", type=mj.mjtGeom.mjGEOM_BOX, size=list(o["half"]),
                          rgba=[0.9, 0.5, 0.2, 0.8])
        # The carrier (key m): a kinematic body welded to the object while it is carried.
        self.mj_spec.worldbody.add_body(name="carrier", mocap=True, pos=[0.5, 0.0, 0.5])
        eq = self.mj_spec.add_equality()
        eq.name, eq.type, eq.objtype = CARRY_EQ, mj.mjtEq.mjEQ_WELD, mj.mjtObj.mjOBJ_BODY
        eq.name1, eq.name2, eq.active = "carrier", "grasp_object", False
        eq.solref = [0.005, 1.0]  # stiff: the object follows the carrier to the mm

    def reset(self) -> None:
        super().reset()
        self._object_body_id = self.mj_model.body("grasp_object").id
        self._carry = None
        if self.mj_model is not None and hasattr(self, "_carry_eq"):
            self.mj_data.eq_active[self._carry_eq] = 0
        self._place_object(self._spawn_xy_half)

    def _place_object(self, xy_half: np.ndarray) -> None:
        """Random x / y / yaw at rest on the table, redrawn while the object's
        footprint would come within ``place_clearance`` of the hand."""
        v = self.obj.mesh.vertices @ self.obj.R_spawn.T
        radius = float(np.linalg.norm(v[:, :2], axis=1).max())
        top = float(v[:, 2].max() - v[:, 2].min())
        hand = [self.mj_model.body(n).id for n in ("fr3_gripper", "robotiq_left_pad", "robotiq_right_pad")]
        for _ in range(50):
            xy = self._spawn_xy_center + self._rng.uniform(-xy_half, xy_half)
            yaw = float(self._rng.uniform(-self._spawn_yaw_half, self._spawn_yaw_half))
            self._set_object_pose(xy, Rotation.from_euler("z", yaw).as_matrix() @ self.obj.R_spawn)
            near = [np.linalg.norm(self.mj_data.xpos[b][:2] - xy) < radius + self._place_clearance
                    and self.mj_data.xpos[b][2] < top + self._place_clearance for b in hand]
            if not any(near):
                break
        else:
            print("[HRGSimEnv] No placement clear of the hand found; keeping the last one.")
        self._last_object_pose_t = -1e9  # publish the new pose right away
        print(f"[HRGSimEnv] object at xy=({xy[0]:.3f}, {xy[1]:.3f}), yaw={yaw:.2f} rad")

    def _set_object_pose(self, xy, R) -> None:
        jid = self.mj_model.body_jntadr[self._object_body_id]
        qadr, dadr = int(self.mj_model.jnt_qposadr[jid]), int(self.mj_model.jnt_dofadr[jid])
        q = Rotation.from_matrix(R).as_quat()  # xyzw

        def apply():
            self.mj_data.qpos[qadr:qadr + 3] = [xy[0], xy[1], self.obj.resting_height(R) + 0.001]
            self.mj_data.qpos[qadr + 3:qadr + 7] = [q[3], q[0], q[1], q[2]]
            self.mj_data.qvel[dadr:dadr + 6] = 0.0
            mj.mj_forward(self.mj_model, self.mj_data)

        viewer = getattr(self, "mj_viewer", None)
        if viewer is not None and viewer.is_running():
            with viewer.lock():
                apply()
        else:
            apply()

    def _obstacle_pose(self, i: int, t: float) -> np.ndarray:
        o = self._obs_cfg[i]
        amp = np.r_[np.asarray(o.get("amp", [0.0, 0.0]), dtype=float), 0.0]
        s = np.sin(2.0 * np.pi * float(o.get("freq", 0.0)) * t + float(o.get("phase", 0.0)))
        return np.asarray(o["centre"], dtype=float) + amp * s

    # ------------------------------------------------------------------ keys
    def _handle_key(self, key: str) -> None:
        if key in ("k", "K"):
            self._push()
        elif key in ("o", "O"):
            self._place_object(self._jump_xy_half)
        elif key in ("m", "M"):
            self._start_carry()
        else:
            super()._handle_key(key)

    def _key_callback(self, keycode) -> None:
        if keycode in (ord("k"), ord("K")):
            self._push()
        elif keycode in (ord("o"), ord("O")):
            self._place_object(self._jump_xy_half)
        elif keycode in (ord("m"), ord("M")):
            self._start_carry()
        else:
            super()._key_callback(keycode)

    # ------------------------------------------------------------------ carry
    def _start_carry(self) -> None:
        """Weld the object to the carrier where it is and send the carrier
        along a smooth random path (``hrg.carry``); at the end it holds the
        object out until the gripper's pads take it."""
        c = self._carry_cfg
        b = self._object_body_id
        R0 = self.mj_data.xmat[b].reshape(3, 3).copy()
        p0 = self.mj_data.xpos[b].copy()
        P, Y = random_waypoints(self._carry_rng, p0, c.get("box", [[0.35, 0.65], [-0.3, 0.3], [0.05, 0.4]]),
                                int(c.get("waypoints", 4)), 0.0, float(c.get("yaw_range", 1.0)))
        path = CarryPath(P, Y, R0, float(c.get("speed", 0.3)), float(c.get("min_duration", 2.0)))
        self.mj_data.mocap_pos[self._carrier] = p0
        self.mj_data.mocap_quat[self._carrier] = self.mj_data.xquat[b]
        self.mj_model.eq_data[self._carry_eq, :10] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # anchor 0, relpose = identity
        self.mj_data.eq_active[self._carry_eq] = 1
        jid = self.mj_model.body_jntadr[b]
        dadr = int(self.mj_model.jnt_dofadr[jid])
        self.mj_data.qvel[dadr:dadr + 6] = 0.0
        self._carry = (path, float(self.mj_data.time))
        self._carry_touch_steps = 0
        print(f"[HRGSimEnv] Carry: {len(P) - 1} waypoints, {path.length:.2f} m in {path.duration:.1f} s, "
              f"to ({P[-1][0]:.2f}, {P[-1][1]:.2f}, {P[-1][2]:.2f}); released when both pads touch the object.")

    def _carry_step(self) -> None:
        path, t0 = self._carry
        t = float(self.mj_data.time) - t0
        p, quat = path.pose(t)
        self.mj_data.mocap_pos[self._carrier] = p
        self.mj_data.mocap_quat[self._carrier] = quat
        # Let go as soon as both pads hold the object, moving or not: a kinematic carrier that
        # keeps pulling a grasped object drags the whole arm along with it.
        self._carry_touch_steps = self._carry_touch_steps + 1 if self._pads_touch_object() else 0
        if self._carry_touch_steps * self.mj_model.opt.timestep >= 0.02:
            self.mj_data.eq_active[self._carry_eq] = 0
            self._carry = None
            print(f"[HRGSimEnv] Carry: both pads on the object, released it ({t:.1f} s after the start"
                  f"{'' if path.done(t) else ', still moving'}).")

    def _pads_touch_object(self) -> bool:
        """Both finger pads in contact with the object. Any other robot body
        touching the carried object is reported (the hand ran into it)."""
        m, d = self.mj_model, self.mj_data
        touching = set()
        for c in d.contact[: d.ncon]:
            bodies = (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
            if self._object_body_id in bodies:
                touching.update(b for b in bodies if b in self._pad_bodies)
                other = [b for b in bodies if b in self._robot_bodies and b not in self._pad_bodies]
                if other:
                    self._carry_hits = getattr(self, "_carry_hits", 0) + 1
                    if self._carry_hits % 100 == 1:
                        print(f"[HRGSimEnv] Carry: {m.body(other[0]).name} hit the carried object "
                              f"(t = {d.time:.2f} s, {self._carry_hits} contact steps so far)")
        return len(touching) == len(self._pad_bodies)

    @property
    def carrying(self) -> bool:
        return self._carry is not None

    def _push(self) -> None:
        th = self._push_rng.uniform(0.0, 2.0 * np.pi)
        self._push_vec = self._push_force * np.array([np.cos(th), np.sin(th), 0.0])
        self._push_until = float(self.mj_data.time) + self._push_duration
        print(f"[HRGSimEnv] Push: {np.round(self._push_vec, 1)} N for {self._push_duration} s")

    # ------------------------------------------------------------------ step
    def _step_simulation(self) -> None:
        t = float(self.mj_data.time)
        for i in range(len(self._obs_cfg)):
            self.mj_data.mocap_pos[self.mj_model.body(f"obstacle_{i}").mocapid[0]] = self._obstacle_pose(i, t)
        if self._carry is not None:
            self._carry_step()
        self.mj_data.xfrc_applied[self.mj_model.body("fr3_gripper").id, :3] = (
            self._push_vec if t < self._push_until else 0.0)
        super()._step_simulation()
        if self._obs_cfg:
            self._count_obstacle_contacts()

    def _count_obstacle_contacts(self) -> None:
        m, d = self.mj_model, self.mj_data
        for c in d.contact[: d.ncon]:
            names = (m.geom(c.geom1).name, m.geom(c.geom2).name)
            bodies = (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
            if any(n.startswith("obs_box_") for n in names) and any(b in self._robot_bodies for b in bodies):
                self._obs_contact_steps += 1
                if self._obs_contact_steps % 200 == 1:
                    print(f"[HRGSimEnv] ROBOT-OBSTACLE CONTACT ({self._obs_contact_steps} steps so far)")
                return

    # ------------------------------------------------------------------ publish
    @staticmethod
    def _put_latest(que, item) -> None:
        while not que.empty():
            try:
                que.get_nowait()
            except Exception:
                break
        que.put(item)

    def _sync_data_from_sim(self, intr_pub_que_dict, extr_pub_que_dict, time_pub_que):
        super()._sync_data_from_sim(intr_pub_que_dict, extr_pub_que_dict, time_pub_que)
        t = float(self.mj_data.time)
        if t - self._last_object_pose_t < self._object_pose_dt:
            return
        self._last_object_pose_t = t
        if self._object_pose_channel:
            b = self._object_body_id
            T = np.eye(4)
            T[:3, :3] = self.mj_data.xmat[b].reshape(3, 3)
            T[:3, 3] = self.mj_data.xpos[b]
            msg = SE3PoseData(name=self._object_pose_channel)
            msg.set_from_matrix(t, T)
            self._put_latest(extr_pub_que_dict[self._object_pose_channel], msg)
        if self._obstacles_channel and self._obs_cfg:
            T_list = []
            for i in range(len(self._obs_cfg)):
                T = np.eye(4)
                T[:3, 3] = self.mj_data.mocap_pos[self.mj_model.body(f"obstacle_{i}").mocapid[0]]
                T_list.append(T)
            rows = obstacle_msg.encode(T_list, [o["half"] for o in self._obs_cfg])
            msg = NamedVecListData(name=self._obstacles_channel)
            msg.set_data(t, [f"obstacle_{i}" for i in range(len(rows))], rows)
            self._put_latest(extr_pub_que_dict[self._obstacles_channel], copy.deepcopy(msg))
