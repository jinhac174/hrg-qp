"""Graspable objects (``configs/objects/<name>.yaml``), shared by the sim env
and the controller so both agree on mesh, mass and frame.

Config::

    name: blue_bowl
    mesh_path: assets/objects/blue_bowl/visual.obj      # object frame = mesh frame
    collision_dir: assets/objects/blue_bowl/collision   # convex parts (concave objects)
    mass: 0.25
    rgba: [...]
    friction: [...]
    spawn_rpy: [0, 0, 0]                                # upright on the table
    sdf: {type: point_cloud, path: ...}                 # hrg.sdf
    fingertip_targets: assets/objects/blue_bowl/fingertip_targets.npz

(written by ``scripts/prepare_objects.py``).
"""

from __future__ import annotations

import glob
import os

import mujoco as mj
import numpy as np
import trimesh
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

from hrg.paths import resolve


class GraspObject:
    def __init__(self, cfg):
        if isinstance(cfg, str):
            cfg = OmegaConf.load(resolve(cfg))
        self.cfg: DictConfig = cfg
        self.name = str(cfg["name"])
        self.mesh_path = resolve(cfg["mesh_path"])
        self.mass = float(cfg["mass"])
        self.rgba = list(cfg.get("rgba", [0.6, 0.6, 0.6, 1.0]))
        self.friction = list(cfg.get("friction", [0.5, 0.005, 0.0001]))
        self.R_spawn = Rotation.from_euler("xyz", list(cfg.get("spawn_rpy", [0.0, 0.0, 0.0]))).as_matrix()
        self.mesh: trimesh.Trimesh = trimesh.load(self.mesh_path, force="mesh")

    def resting_height(self, R_world_obj: np.ndarray, floor_z: float = 0.0) -> float:
        """Object-frame origin height at which the object rests on the floor."""
        z = (self.mesh.vertices @ R_world_obj.T)[:, 2]
        return floor_z - float(z.min())

    def add_to_spec(self, spec: mj.MjSpec, body_name: str = "grasp_object", pos=(0.5, 0.0, 0.1)):
        """Add a free body with this object to ``spec`` (body frame = object frame).

        MuJoCo collides a mesh as its convex hull. With ``collision_dir`` the
        mesh is visual only and the object collides through its convex parts
        (a bowl's hull would fill the bowl); the mass is split by part volume.
        """
        mesh_name = f"{body_name}_mesh"
        mesh = spec.add_mesh(name=mesh_name, file=self.mesh_path)
        body = spec.worldbody.add_body(name=body_name, pos=list(pos))
        body.add_freejoint(name=f"{body_name}_free")
        collision_dir = self.cfg.get("collision_dir", None)
        if not collision_dir:
            geom = body.add_geom(name=f"{body_name}_geom", type=mj.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
                                 rgba=self.rgba, friction=self.friction)
            geom.mass = self.mass
            return body
        # Visual: the (open, scanned) mesh, no contacts, no mass.
        mesh.inertia = mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
        vis = body.add_geom(name=f"{body_name}_geom", type=mj.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
                            rgba=self.rgba, contype=0, conaffinity=0, group=1)
        vis.density = 0.0
        files = sorted(glob.glob(os.path.join(resolve(collision_dir), "*.obj")))
        if not files:
            raise FileNotFoundError(f"No convex parts in {collision_dir}")
        vols = np.maximum([trimesh.load(f, force="mesh").volume for f in files], 1e-9)
        for k, (f, v) in enumerate(zip(files, vols)):
            name = f"{body_name}_col{k:03d}"
            spec.add_mesh(name=name, file=f)
            g = body.add_geom(name=name, type=mj.mjtGeom.mjGEOM_MESH, meshname=name,
                              rgba=[0.5, 0.5, 0.5, 0.0], friction=self.friction, group=3)
            g.mass = float(self.mass * v / vols.sum())
        return body
