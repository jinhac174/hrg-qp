"""Offscreen MuJoCo rendering of what is on the LCM bus, with the method's
quantities drawn in: for videos of sim runs and, from the same channels, of
the real robot.

The scene (``assets/robot/scene_sim.xml`` + the object + obstacle boxes) is
posed kinematically from the messages -- arm joints, gripper width, object
pose, obstacles -- and the controller's fingertip channels add

* the current fingertips (left pad: red, right pad: blue),
* their selected targets (eq. 12; the larger, lighter spheres),
* the planned fingertip paths (eq. 3; the red / blue curves).
"""

from __future__ import annotations

import mujoco as mj
import numpy as np

from fr3.robot_platform.gripper import robotiq_2f85 as rq
from hrg import obstacles as obstacle_msg
from hrg.objects import GraspObject
from hrg.paths import resolve

FT_RGBA = (np.array([0.92, 0.25, 0.25, 1.0]), np.array([0.25, 0.5, 0.95, 1.0]))
TGT_RGBA = (np.array([1.0, 0.65, 0.65, 0.55]), np.array([0.65, 0.78, 1.0, 0.55]))


class SceneRenderer:
    def __init__(self, object_config: str, obstacle_halves=(), width: int = 1280, height: int = 720,
                 scene_xml: str = "assets/robot/scene_sim.xml", spawn_rpy=None):
        spec = mj.MjSpec.from_file(resolve(scene_xml))
        self.obj = GraspObject(object_config)
        self.obj.add_to_spec(spec)
        for i, h in enumerate(obstacle_halves):
            body = spec.worldbody.add_body(name=f"obstacle_{i}", mocap=True, pos=[0.0, 0.0, -1.0])
            body.add_geom(name=f"obs_box_{i}", type=mj.mjtGeom.mjGEOM_BOX, size=list(h), rgba=[0.9, 0.5, 0.2, 0.8])
        spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
        self.model = spec.compile()
        for g in range(self.model.ngeom):  # the virtual walls are constraints, not scenery
            if self.model.geom(g).name.startswith("wall"):
                self.model.geom_rgba[g, 3] = 0.0
        self.model.mat_reflectance[:] = 0.0  # no mirror floor
        self.data = mj.MjData(self.model)
        self._arm = [self.model.joint(f"fr3_joint{i + 1}").qposadr[0] for i in range(7)]
        jid = self.model.body("grasp_object").jntadr[0]
        self._obj_q = int(self.model.jnt_qposadr[jid])
        self._n_obs = len(obstacle_halves)
        self.renderer = mj.Renderer(self.model, height=height, width=width)
        self.opt = mj.MjvOption()
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.set_view()
        # Close-up: follows the object (``render(closeup=True)``).
        self.closeup = mj.MjvCamera()
        self.closeup.type = mj.mjtCamera.mjCAMERA_FREE
        self.closeup.distance, self.closeup.azimuth, self.closeup.elevation = 0.42, 225.0, -28.0
        rq.mj_set_width(self.model, self.data, rq.MAX_WIDTH)
        self.data.qpos[self._obj_q:self._obj_q + 3] = [0.5, 0.0, -1.0]  # hidden until a pose arrives
        self.data.qpos[self._obj_q + 3] = 1.0

    def set_view(self, lookat=(0.5, 0.04, 0.17), distance: float = 0.9, azimuth: float = 145.0,
                 elevation: float = -16.0) -> None:
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation

    # ------------------------------------------------------------ state
    def set_state(self, q_arm=None, width=None, obj_pos=None, obj_quat_wxyz=None, obstacle_rows=None) -> None:
        if q_arm is not None:
            self.data.qpos[self._arm] = np.asarray(q_arm, dtype=float)[:7]
        if width is not None:
            rq.mj_set_width(self.model, self.data, float(width))
        if obj_pos is not None:
            self.data.qpos[self._obj_q:self._obj_q + 3] = obj_pos
            self.data.qpos[self._obj_q + 3:self._obj_q + 7] = obj_quat_wxyz
        if obstacle_rows is not None and self._n_obs:
            rows = np.asarray(obstacle_rows, dtype=float).reshape(-1, obstacle_msg.ROW)
            for i, r in enumerate(rows[: self._n_obs]):
                mid = self.model.body(f"obstacle_{i}").mocapid[0]
                self.data.mocap_pos[mid] = r[:3]
                self.data.mocap_quat[mid] = r[3:7]
        mj.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ drawing
    def _add(self, gtype, size, pos, mat, rgba) -> None:
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return
        mj.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.asarray(mat, float).reshape(-1), np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def _sphere(self, p, r, rgba) -> None:
        self._add(mj.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], p, np.eye(3), rgba)

    def _segment(self, a, b, r, rgba) -> None:
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                        np.asarray(rgba, np.float32))
        mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, r, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1

    def render(self, fingertips=None, targets=None, paths=None, closeup: bool = False) -> np.ndarray:
        """RGB frame (H, W, 3) uint8 of the posed scene with the overlays,
        from the main view or the close-up on the object."""
        cam = self.cam
        if closeup:
            cam = self.closeup
            cam.lookat[:] = self.data.qpos[self._obj_q:self._obj_q + 3] + np.array([0.0, 0.0, 0.06])
        self.renderer.update_scene(self.data, cam, self.opt)
        if paths is not None:
            for f, P in enumerate(paths):
                for a, b in zip(P[:-1], P[1:]):
                    self._segment(a, b, 0.004, FT_RGBA[f])
        if targets is not None:
            for f, p in enumerate(targets):
                self._sphere(p, 0.014, TGT_RGBA[f])
        if fingertips is not None:
            for f, p in enumerate(fingertips):
                self._sphere(p, 0.009, FT_RGBA[f])
        return self.renderer.render()
