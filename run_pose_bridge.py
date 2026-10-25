"""Robot: FoundationPose's pose -> the object pose the controller uses.

FoundationPose (``run_perception.py``) publishes, as ``vec_list_t``
``[x, y, z, R (row-major 3x3)]``, the pose of the object's *oriented bounding
box* frame in the camera's "world" frame (``inv(extrinsic)`` of the camera
info). The controller wants the pose of the object's *mesh* frame (the frame
the fingertip targets are defined in) in the *robot base* frame, as
``se3_pose_t``::

    T_base_mesh = T_base_world @ T_world_bb @ to_origin

* ``to_origin`` = ``trimesh.bounds.oriented_bounds(mesh)`` of the SAME mesh
  FoundationPose loads (``mesh_file``): the object's ``visual.obj``.
* ``T_base_world`` = identity when the camera extrinsics are calibrated
  against the robot base; otherwise set ``base_from_world``.

    python run_pose_bridge.py --config configs/robot/pose_bridge.yaml --mesh_file assets/objects/blue_bowl/visual.obj
"""

import time

import lcm
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from fr3.lcm_type.pose.se3_pose_t import se3_pose_t
from fr3.lcm_type.vector.vec_list_t import vec_list_t
from hrg.cli import load_config


def bb_pose_to_mesh_pose(vec12, to_origin, T_base_world=None):
    """``[p, R row-major]`` of the bounding-box frame -> 4x4 mesh-frame pose."""
    T_world_bb = np.eye(4)
    T_world_bb[:3, 3] = vec12[:3]
    T_world_bb[:3, :3] = np.asarray(vec12[3:12]).reshape(3, 3)
    T_base_world = np.eye(4) if T_base_world is None else T_base_world
    return T_base_world @ T_world_bb @ to_origin


def main() -> None:
    cfg = load_config(__doc__)
    to_origin, _ = trimesh.bounds.oriented_bounds(trimesh.load(cfg["mesh_file"], force="mesh"))
    T_base_world = np.asarray(cfg["base_from_world"], dtype=float)
    lc = lcm.LCM()
    out_channel = cfg["object_pose_channel"]
    n = [0]

    def on_pose(_channel, data):
        msg = vec_list_t.decode(data)
        if msg.num_vecs < 1:
            return
        T = bb_pose_to_mesh_pose(np.asarray(msg.vec_list[0], dtype=float), to_origin, T_base_world)
        out = se3_pose_t()
        out.timestamp = msg.timestamp
        out.position = T[:3, 3].tolist()
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        out.quat_wxyz = [w, x, y, z]
        lc.publish(out_channel, out.encode())
        n[0] += 1

    lc.subscribe(cfg["fp_pose_channel"], on_pose)
    print(f"[PoseBridge] {cfg['fp_pose_channel']} (bounding box, camera world) -> {out_channel} (mesh frame, robot base)")
    t_report = time.time()
    try:
        while True:
            lc.handle_timeout(100)
            if time.time() - t_report > 5.0:
                print(f"[PoseBridge] {n[0] / (time.time() - t_report):.1f} poses/s")
                n[0], t_report = 0, time.time()
    except KeyboardInterrupt:
        print("[PoseBridge] stopped.")


if __name__ == "__main__":
    main()
