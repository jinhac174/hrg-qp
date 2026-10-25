"""run_pose_bridge.py: FoundationPose's bounding-box pose -> the mesh frame."""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from run_pose_bridge import bb_pose_to_mesh_pose


def test_bounding_box_pose_maps_back_to_the_mesh_frame():
    mesh = trimesh.load("assets/objects/blue_mug/visual.obj", force="mesh")
    to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
    # the object's true pose in the camera world frame
    T_world_mesh = np.eye(4)
    T_world_mesh[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    T_world_mesh[:3, 3] = [0.4, -0.1, 0.05]
    # FoundationPose reports the oriented bounding box frame: T_world_bb = T_world_mesh @ inv(to_origin)
    T_world_bb = T_world_mesh @ np.linalg.inv(to_origin)
    vec12 = np.r_[T_world_bb[:3, 3], T_world_bb[:3, :3].reshape(-1)]
    assert np.allclose(bb_pose_to_mesh_pose(vec12, to_origin), T_world_mesh, atol=1e-9)
    # and with a base <- world transform
    T_base_world = np.eye(4)
    T_base_world[:3, 3] = [0.1, 0.0, -0.2]
    assert np.allclose(bb_pose_to_mesh_pose(vec12, to_origin, T_base_world), T_base_world @ T_world_mesh, atol=1e-9)
