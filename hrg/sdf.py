"""Signed distance functions ``d(x, O)`` (paper Sec. II-A, II-D).

The paper requires ``d(x, O)`` to be the signed *Euclidean* distance from a
point to the surface of ``O`` (negative inside); for convex objects it uses
superquadrics, for concave objects with thin surfaces the nearest-point
distance to a surface point cloud ("given the thinness of the surface,
ignoring the sign hardly matters"). Here:

* ``PointCloudSDF``: thin concave objects (bowl, mug, wine glass, plate),
  as in the paper.
* ``BoxSDF``: the box object and the obstacles. Their exact signed distance,
  i.e. the e -> 0 superquadric. (A superquadric's implicit function is not a
  Euclidean distance: with box-like exponents it overestimates the clearance
  near edges and corners by up to ~2 cm, so a 1 cm margin would not hold.)
* ``HalfSpaceSDF``: the table, ``z >= z0``.

All share one interface. The shape lives in its own frame; ``set_pose`` places
it in the world (the object pose from the sim or FoundationPose), and
``dist_grad`` takes and returns world quantities::

    d, g = sdf.dist_grad(x)      # x (N, 3) -> d (N,), g = dd/dx (N, 3)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from hrg.paths import resolve


class PosedSDF:
    """Base: a distance function in a local frame, placed by ``set_pose``."""

    def __init__(self):
        self.T = np.eye(4)

    def set_pose(self, T_world_local: np.ndarray) -> None:
        self.T = np.asarray(T_world_local, dtype=float).copy()

    def dist_grad(self, x: np.ndarray):
        R, p = self.T[:3, :3], self.T[:3, 3]
        x_loc = (np.asarray(x, dtype=float).reshape(-1, 3) - p) @ R
        d, g_loc = self._local(x_loc)
        return d, g_loc @ R.T

    def dist(self, x: np.ndarray, cap: float | None = None) -> np.ndarray:
        """Distances only; with ``cap``, values above it may be returned as
        ``cap`` (enough for counting violations, and much cheaper for a
        point cloud)."""
        R, p = self.T[:3, :3], self.T[:3, 3]
        x_loc = (np.asarray(x, dtype=float).reshape(-1, 3) - p) @ R
        return self._local_dist(x_loc, cap)

    def _local(self, x: np.ndarray):
        raise NotImplementedError

    def _local_dist(self, x: np.ndarray, cap):
        return self._local(x)[0]


class PointCloudSDF(PosedSDF):
    """Nearest-point distance to a surface point cloud (local frame).

    The gradient is the unit vector from the nearest point to the query. A
    query closer than ``1e-9`` (on a sample) gets the stored normal if one is
    given, else +z.
    """

    def __init__(self, points: np.ndarray, normals: np.ndarray | None = None):
        super().__init__()
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.normals = None if normals is None else np.asarray(normals, dtype=float).reshape(-1, 3)
        self._tree = cKDTree(self.points)

    def _local_dist(self, x, cap):
        if cap is None:
            return self._tree.query(x, k=1)[0]
        d, _ = self._tree.query(x, k=1, distance_upper_bound=float(cap))
        return np.minimum(d, cap)

    def _local(self, x):
        d, idx = self._tree.query(x, k=1)
        diff = x - self.points[idx]
        g = np.zeros_like(x)
        ok = d > 1e-9
        g[ok] = diff[ok] / d[ok, None]
        if np.any(~ok):
            g[~ok] = self.normals[idx[~ok]] if self.normals is not None else (0.0, 0.0, 1.0)
        return d, g


class BoxSDF(PosedSDF):
    """Exact signed distance to a box with half extents ``half``, centred at
    ``centre`` in the local frame. Outside: Euclidean distance to the nearest
    face, edge or corner, gradient along that offset. Inside: minus the depth
    below the nearest face, gradient = that face's normal."""

    def __init__(self, half, centre=(0.0, 0.0, 0.0)):
        super().__init__()
        self.half = np.asarray(half, dtype=float)
        self.centre = np.asarray(centre, dtype=float)

    def _local(self, x):
        p = x - self.centre
        s = np.where(p >= 0.0, 1.0, -1.0)
        q = np.abs(p) - self.half
        out = np.maximum(q, 0.0)
        d_out = np.linalg.norm(out, axis=1)
        k = np.argmax(q, axis=1)
        d = np.where(d_out > 0.0, d_out, q[np.arange(len(q)), k])
        g = np.zeros_like(p)
        o = d_out > 0.0
        g[o] = s[o] * out[o] / d_out[o, None]
        i = np.where(~o)[0]
        g[i, k[i]] = s[i, k[i]]
        return d, g


class HalfSpaceSDF(PosedSDF):
    """The table: ``d = z - z0`` (world frame; ``set_pose`` unused)."""

    def __init__(self, z0: float = 0.0):
        super().__init__()
        self.z0 = float(z0)

    def dist(self, x, cap=None):
        return np.asarray(x, dtype=float).reshape(-1, 3)[:, 2] - self.z0

    def dist_grad(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 3)
        g = np.zeros_like(x)
        g[:, 2] = 1.0
        return x[:, 2] - self.z0, g


def sdf_from_config(cfg) -> PosedSDF:
    """An object's distance function from its config ``sdf:`` entry::

        sdf: {type: point_cloud, path: assets/objects/<name>/surface_points.npy}
        sdf: {type: box, half: [hx, hy, hz], centre: [cx, cy, cz]}
    """
    kind = str(cfg["type"])
    if kind == "point_cloud":
        data = np.load(resolve(cfg["path"]))
        return PointCloudSDF(data[:, :3], data[:, 3:6] if data.shape[1] >= 6 else None)
    if kind == "box":
        return BoxSDF(cfg["half"], cfg.get("centre", (0.0, 0.0, 0.0)))
    raise ValueError(f"Unknown sdf type {kind!r}")
