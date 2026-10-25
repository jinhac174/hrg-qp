"""Non-target obstacles as oriented boxes (paper: tissue boxes, Fig. 8, 11).

Wire format (``vec_list_t`` on the ``obstacles`` channel), one row per
obstacle, world (robot base) frame::

    [x, y, z, qw, qx, qy, qz, hx, hy, hz]      (10 values: pose, half extents)

In the sim the env publishes its moving boxes; on the robot a perception node
(FoundationPose on each obstacle, or a fixed list) publishes the same.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from hrg.sdf import BoxSDF

ROW = 10


def encode(T_list, half_list) -> np.ndarray:
    rows = []
    for T, h in zip(T_list, half_list):
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        rows.append(np.r_[T[:3, 3], q[3], q[:3], h])
    return np.asarray(rows, dtype=float).reshape(-1, ROW)


def decode(rows: np.ndarray) -> list[BoxSDF]:
    out = []
    for r in np.asarray(rows, dtype=float).reshape(-1, ROW):
        sdf = BoxSDF(r[7:10])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(np.r_[r[4:7], r[3]]).as_matrix()
        T[:3, 3] = r[:3]
        sdf.set_pose(T)
        out.append(sdf)
    return out
