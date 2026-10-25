"""Live 3-D view of the reactive grasp in the browser (Viser), from LCM.

Shows the FR3 + 2F-85 (measured joints and gripper), the object at its pose
(sim or FoundationPose), the moving obstacles, the fingertips and their
selected targets (eq. 12), and the planned fingertip paths (eq. 3). Sim and
robot publish the same channels, so it works for both.

    python run_visualizer.py --config configs/visualizer.yaml --object_mesh assets/objects/blue_bowl/visual.obj
"""

from __future__ import annotations

import threading
import time

import lcm
import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF

from fr3.lcm_type.joint.joint_meas_t import joint_meas_t
from fr3.lcm_type.pose.se3_pose_t import se3_pose_t
from fr3.lcm_type.vector.vec_list_t import vec_list_t
from hrg import obstacles as obstacle_msg
from hrg.paths import resolve

# thumb = fingertip 1 (left pad), index = fingertip 2 (right pad)
_FT = ("thumb", "index")
_FT_COLOR = ((235, 70, 70), (70, 140, 235))
_TGT_COLOR = ((255, 170, 170), (170, 200, 255))


class HRGViewer:
    def __init__(self, cfg):
        ch = cfg["channels"]
        self._latest: dict = {}
        self._lock = threading.Lock()
        self._lc = lcm.LCM()
        decoders = {
            ch["arm_joint_meas"]: joint_meas_t.decode,
            ch["gripper_joint_meas"]: joint_meas_t.decode,
            ch["object_pose"]: se3_pose_t.decode,
            ch["obstacles"]: vec_list_t.decode,
            ch["fingertips"]: vec_list_t.decode,
            ch["fingertip_paths"]: vec_list_t.decode,
        }
        for name, dec in decoders.items():
            self._lc.subscribe(name, self._handler(name, dec))
        self._ch = ch
        self._vis_dt = 1.0 / float(cfg.get("vis_freq", 30))

        self.server = viser.ViserServer(host=str(cfg.get("host", "127.0.0.1")), port=int(cfg.get("port", 8080)))
        print(f"[Viewer] http://localhost:{self.server.get_port()}")
        urdf = URDF.load(resolve(cfg["urdf_path"]), load_meshes=True, build_scene_graph=True,
                         load_collision_meshes=False, build_collision_scene_graph=False)
        self._robot = ViserUrdf(self.server, urdf_or_path=urdf, load_meshes=True, load_collision_meshes=False)
        self._q = np.zeros(8)  # 7 arm joints + the 2F-85 driver angle (the rest mimics it)

        self._obj = None
        if cfg.get("object_mesh", ""):
            mesh = trimesh.load(resolve(cfg["object_mesh"]), force="mesh")
            self._obj = self.server.scene.add_frame("/object", show_axes=True, axes_length=0.05,
                                                    axes_radius=0.002, visible=False)
            self.server.scene.add_mesh_trimesh("/object/mesh", mesh)
        self._ft, self._tgt = [], []
        for i, f in enumerate(_FT):
            self._ft.append(self.server.scene.add_icosphere(f"/hrg/{f}", radius=0.007, color=_FT_COLOR[i], visible=False))
            self._tgt.append(self.server.scene.add_icosphere(f"/hrg/{f}_target", radius=0.009, color=_TGT_COLOR[i],
                                                             visible=False))
        self._paths = [self.server.scene.add_line_segments(f"/hrg/{f}_path", points=np.zeros((1, 2, 3), np.float32),
                                                           colors=_FT_COLOR[i], line_width=4.0, visible=False)
                       for i, f in enumerate(_FT)]
        self._boxes: list = []

    def _handler(self, name, decode):
        def f(_channel, data):
            msg = decode(data)
            with self._lock:
                self._latest[name] = msg
        return f

    def _take(self, key):
        with self._lock:
            return self._latest.pop(self._ch[key], None)

    def _update(self) -> None:
        arm, grip = self._take("arm_joint_meas"), self._take("gripper_joint_meas")
        if arm is not None:
            self._q[:7] = arm.q[:7]
        if grip is not None:
            self._q[7] = grip.q[0]
        self._robot.update_cfg(self._q)

        pose = self._take("object_pose")
        if pose is not None and self._obj is not None:
            self._obj.position, self._obj.wxyz, self._obj.visible = tuple(pose.position), tuple(pose.quat_wxyz), True

        obs = self._take("obstacles")
        if obs is not None:
            rows = np.asarray(obs.vec_list, dtype=float).reshape(-1, obstacle_msg.ROW)
            while len(self._boxes) < len(rows):  # viser box sizes are fixed at creation
                k = len(self._boxes)
                self._boxes.append(self.server.scene.add_box(f"/obstacles/{k}", color=(230, 130, 50),
                                                             dimensions=tuple(2.0 * rows[k, 7:10]), opacity=0.6))
            for h, r in zip(self._boxes, rows):
                h.position, h.wxyz, h.visible = tuple(r[:3]), tuple(r[3:7]), True
            for h in self._boxes[len(rows):]:
                h.visible = False

        ft = self._take("fingertips")
        if ft is not None:
            p = {n: np.asarray(v, dtype=float) for n, v in zip(ft.name_list, ft.vec_list)}
            for i, f in enumerate(_FT):
                if f"{f}_cur" in p and f"{f}_tgt" in p:
                    self._ft[i].position, self._ft[i].visible = tuple(p[f"{f}_cur"]), True
                    self._tgt[i].position, self._tgt[i].visible = tuple(p[f"{f}_tgt"]), True

        paths = self._take("fingertip_paths")
        if paths is not None:
            for i, f in enumerate(_FT):
                pts = np.asarray([v for n, v in zip(paths.name_list, paths.vec_list) if n.startswith(f + "_")],
                                 dtype=np.float32)
                if len(pts) >= 2:
                    self._paths[i].points = np.stack([pts[:-1], pts[1:]], axis=1)
                    self._paths[i].visible = True

    def run(self) -> None:
        stop = threading.Event()

        def spin():
            while not stop.is_set():
                self._lc.handle_timeout(10)

        t = threading.Thread(target=spin, daemon=True)
        t.start()
        try:
            while True:
                t0 = time.time()
                self._update()
                time.sleep(max(0.0, self._vis_dt - (time.time() - t0)))
        except KeyboardInterrupt:
            stop.set()
