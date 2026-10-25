"""Fingertip target pairs and their selection (paper Sec. II, eq. 12; Fig. 4).

The paper assumes a set of candidate fingertip target pairs ``x*_j``,
predefined in the object frame (it takes them from T2SQNet), and selects one
every plan. Here they are generated once per object
(``scripts/prepare_objects.py``) and selected online.

Generation (object frame, upright)
----------------------------------
1. Contact pairs ``(p1, p2)`` from two sources:

   * rim pinches, as T2SQNet places them on tableware (paper Fig. 4): at a
     few depths below the top, horizontal rays toward the vertical axis; the
     outer and the inner skin they cross form a pair (a mug's handle gives
     handle pinches the same way);
   * antipodal pairs anywhere on the surface (``sample_antipodal_grasps``).
2. Pre-grasp fingertip targets: each pad backs off ``offset`` along the pair
   axis, ``x1 = p1 - offset a``, ``x2 = p2 + offset a`` (``a = (p2 - p1)/|.|``),
   so the hand opens to ``|p2 - p1| + 2 offset`` (within the 85 mm stroke).
   The targets themselves must satisfy the path constraint of eq. (3),
   ``d(x*, O) >= eps_O``: they are the paths' end points.
3. Approach directions: the hand may be turned about the pair axis; for each
   of ``num_approach`` such directions the open hand at the pre-grasp must be
   admissible under the QP's constraints (eq. 11): every hand sphere
   (``hand.hand_spheres_local``) at least ``qp_eps_obj`` from the object (with
   the object's own distance function, the one the QP uses), and the gripper
   boxes clear of the actual mesh (what the simulation collides with). A pair
   with no such direction is dropped. The admissible directions are stored
   with their sphere centres (object frame).
4. Thin the set out (farthest-point sampling on the pair centres).

Validity online
---------------
The table depends on how the object lies: a pair is valid if one of its
admissible directions keeps every hand sphere ``qp_eps_table`` above the table
at the current object pose (``FingertipTargets.valid``).

Selection (eq. 12): the valid pair (either finger order) minimising
``|x1 - x1*| + |x2 - x2*| + w * angle(x2 - x1, x2* - x1*)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from hrg.hand import MAX_WIDTH, HandCollisionChecker, hand_spheres_local, pad_dz


@dataclass
class FingertipTargets:
    """Candidate pairs in the object frame (one row per pair)."""

    x1: np.ndarray  # (N, 3) pre-grasp target of fingertip 1 (left pad)
    x2: np.ndarray  # (N, 3) fingertip 2 (right pad)
    p1: np.ndarray  # (N, 3) contact points the pads close onto
    p2: np.ndarray
    # Admissible approach directions: (N, K) mask and the hand sphere centres
    # (N, K, S, 3) at the pre-grasp for each, object frame; radii (S,).
    approach_ok: np.ndarray | None = None
    spheres: np.ndarray | None = None
    sphere_r: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.x1)

    @property
    def contact_width(self) -> np.ndarray:
        return np.linalg.norm(self.p2 - self.p1, axis=1)

    def save(self, path: str) -> None:
        np.savez(path, x1=self.x1, x2=self.x2, p1=self.p1, p2=self.p2,
                 approach_ok=self.approach_ok, spheres=self.spheres, sphere_r=self.sphere_r)

    @classmethod
    def load(cls, path: str) -> "FingertipTargets":
        d = np.load(path)
        extra = {k: d[k] for k in ("approach_ok", "spheres", "sphere_r") if k in d.files}
        return cls(d["x1"], d["x2"], d["p1"], d["p2"], **extra)

    def transformed(self, T: np.ndarray) -> "FingertipTargets":
        """The target points in the world (the approach data stays object frame)."""
        R, p = T[:3, :3], T[:3, 3]
        f = lambda X: X @ R.T + p  # noqa: E731
        return FingertipTargets(f(self.x1), f(self.x2), f(self.p1), f(self.p2),
                                self.approach_ok, self.spheres, self.sphere_r)

    def valid(self, T_obj: np.ndarray, table_z: float, eps_table: float) -> np.ndarray:
        """(N,) pairs with an admissible approach whose hand spheres all stay
        ``eps_table`` above the table with the object at ``T_obj``."""
        if self.spheres is None:
            return np.ones(len(self), dtype=bool)
        z = np.einsum("j,nksj->nks", T_obj[2, :3], self.spheres) + T_obj[2, 3]
        clear = np.min(z - self.sphere_r, axis=2) >= table_z + eps_table  # (N, K)
        return np.any(clear & self.approach_ok, axis=1)


# ---------------------------------------------------------------- contact pairs
@dataclass
class AntipodalPairs:
    """Contact pairs in the object frame (one row per pair)."""

    p1: np.ndarray  # (N, 3) contact points
    p2: np.ndarray

    def __len__(self) -> int:
        return len(self.p1)


def sample_antipodal_grasps(
    mesh: trimesh.Trimesh,
    num_samples: int = 3000,
    mu: float = 0.8,
    cone_frac: float = 0.6,
    max_width: float = MAX_WIDTH - 0.01,
    min_width: float = 0.01,
    min_separation: float = 0.008,
    seed: int = 0,
) -> AntipodalPairs:
    """Two-finger contact pairs that pass the friction-cone test.

    For a random surface point ``p1`` with outward normal ``n1``, shoot a ray
    into the object along ``-n1``; where it leaves the surface is ``p2``. The
    pair is *antipodal* if the line ``p1 -> p2`` lies inside both friction
    cones: the angle between the squeeze direction and each inward normal is at
    most ``cone_frac * atan(mu)``. Pairs whose midpoints are closer than
    ``min_separation`` (with a similar axis) are de-duplicated, best-aligned
    first.
    """
    rng = np.random.default_rng(seed)
    pts, face_idx = trimesh.sample.sample_surface(mesh, num_samples, seed=rng)
    normals = mesh.face_normals[face_idx]

    # Ray from just inside p1 along the inward normal.
    locs, ray_idx, tri_idx = mesh.ray.intersects_location(pts - normals * 1e-4, -normals, multiple_hits=True)

    cos_max = np.cos(cone_frac * np.arctan(mu))
    # For each ray keep the farthest exit point within max_width (thick parts
    # are skipped automatically).
    best = {}
    for loc, r, t in zip(locs, ray_idx, tri_idx):
        d = float(np.linalg.norm(loc - pts[r]))
        if d < min_width or d > max_width:
            continue
        if r not in best or d > best[r][0]:
            best[r] = (d, loc, mesh.face_normals[t])

    rows = []
    for r, (d, p2, n2) in best.items():
        p1, n1 = pts[r], normals[r]
        axis = (p2 - p1) / d
        c1 = float(-n1 @ axis)  # finger at p1 pushes along +axis
        c2 = float(n2 @ axis)  # finger at p2 pushes along -axis
        if c1 < cos_max or c2 < cos_max:
            continue
        rows.append((p1, p2, d, float(np.arccos(np.clip(min(c1, c2), -1.0, 1.0)))))

    rows.sort(key=lambda row: row[3])
    kept, kept_rows = [], []
    for row in rows:
        c = 0.5 * (row[0] + row[1])
        a = (row[1] - row[0]) / row[2]
        if any(np.linalg.norm(c - kc) < min_separation and abs(a @ ka) > 0.95 for kc, ka in kept):
            continue
        kept.append((c, a))
        kept_rows.append(row)

    if not kept_rows:
        return AntipodalPairs(np.zeros((0, 3)), np.zeros((0, 3)))
    return AntipodalPairs(np.asarray([r[0] for r in kept_rows]), np.asarray([r[1] for r in kept_rows]))


def rim_pinch_pairs(mesh: trimesh.Trimesh, depths=(0.008, 0.015, 0.025), n_azimuth: int = 72,
                    max_width: float = 0.05):
    """Horizontal pinches across the wall below the top of an upright object:
    ``(p1 (N, 3) outer, p2 (N, 3) inner)``. The object is centred in x / y."""
    top = float(mesh.bounds[1][2])
    R = float(np.linalg.norm(mesh.bounds[1][:2] - mesh.bounds[0][:2])) + 0.05
    th = 2.0 * np.pi * np.arange(n_azimuth) / n_azimuth
    dirs_xy = np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
    P1, P2 = [], []
    for depth in depths:
        z = top - depth
        origins = dirs_xy * R + np.array([0.0, 0.0, z])
        locs, ray_idx, _ = mesh.ray.intersects_location(origins, -dirs_xy, multiple_hits=True)
        for r in range(n_azimuth):
            hits = locs[ray_idx == r]
            if len(hits) < 2:
                continue
            hits = hits[np.argsort(np.linalg.norm(hits - origins[r], axis=1))]
            # first entry / exit, skipping duplicate hits on the same skin
            p1 = hits[0]
            rest = hits[np.linalg.norm(hits - p1, axis=1) > 1e-3]
            if len(rest) == 0:
                continue
            p2 = rest[0]
            if np.linalg.norm(p2 - p1) <= max_width:
                P1.append(p1)
                P2.append(p2)
    return np.asarray(P1).reshape(-1, 3), np.asarray(P2).reshape(-1, 3)


# ---------------------------------------------------------------- generation
def _hand_pose(c: np.ndarray, y: np.ndarray, z: np.ndarray, width: float) -> np.ndarray:
    """TCP pose with the pads centred on ``c`` at opening ``width``."""
    T = np.eye(4)
    T[:3, :3] = np.column_stack([np.cross(y, z), y, z])
    T[:3, 3] = c - pad_dz(width) * z
    return T


def _approach_dirs(y: np.ndarray, k: int) -> np.ndarray:
    """``k`` unit vectors perpendicular to ``y``."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    z0 = np.cross(y, np.cross(ref, y))
    z0 /= np.linalg.norm(z0)
    th = 2.0 * np.pi * np.arange(k) / k
    return np.cos(th)[:, None] * z0 + np.sin(th)[:, None] * np.cross(y, z0)


def generate_fingertip_targets(
    mesh: trimesh.Trimesh,
    sdf,
    offset: float = 0.015,
    eps_path: float = 0.01,
    qp_eps_obj: float = 0.01,
    num_samples: int = 4000,
    num_approach: int = 12,
    max_targets: int = 150,
    min_width: float = 0.002,
    mu: float = 0.8,
    box_margin: float = 0.003,
    rim_depths=(0.008, 0.015, 0.025),
    seed: int = 0,
) -> FingertipTargets:
    """See the module docstring. ``sdf``: the object's distance function in
    its own frame (identity pose), the same one the controller uses."""
    max_contact = MAX_WIDTH - 2.0 * offset - 0.003
    pairs = sample_antipodal_grasps(mesh, num_samples=num_samples, mu=mu, max_width=max_contact,
                                    min_width=min_width, seed=seed)
    rp1, rp2 = rim_pinch_pairs(mesh, rim_depths, max_width=max_contact)
    # Rim rays have no spurious hits: keep walls down to 0.5 mm (the scanned
    # wine glass is 1 mm thick). min_width only guards the antipodal sampler.
    P1 = np.vstack([rp1, pairs.p1]) if len(pairs) else rp1
    P2 = np.vstack([rp2, pairs.p2]) if len(pairs) else rp2
    chk = HandCollisionChecker(mesh)

    keep = []
    for p1, p2 in zip(P1, P2):
        w = float(np.linalg.norm(p2 - p1))
        if w < 5e-4:
            continue
        a = (p2 - p1) / w
        x1, x2 = p1 - offset * a, p2 + offset * a
        if np.min(sdf.dist(np.stack([x1, x2]))) < eps_path:
            continue
        open_w = min(w + 2.0 * offset, MAX_WIDTH)
        S_loc, _, r, _ = hand_spheres_local(open_w)
        c = 0.5 * (p1 + p2)
        ok, centres = [], []
        for z in _approach_dirs(a, num_approach):
            T = _hand_pose(c, a, z, open_w)
            C = S_loc @ T[:3, :3].T + T[:3, 3]
            centres.append(C)
            ok.append(
                bool(np.min(sdf.dist(C) - r) >= qp_eps_obj)
                and not chk.collides(T, open_w, margin=box_margin)
            )
        if any(ok):
            keep.append((x1, x2, p1, p2, np.asarray(ok), np.asarray(centres), r))

    if not keep:
        e = np.zeros((0, 3))
        return FingertipTargets(e, e, e, e)
    x1, x2, p1, p2, ok, centres, radii = (np.asarray(v) for v in zip(*keep))

    # Farthest-point thinning on the pair centres (starting from the first).
    sel = np.arange(len(x1))
    if len(x1) > max_targets:
        c = 0.5 * (x1 + x2)
        idx = [0]
        dmin = np.linalg.norm(c - c[0], axis=1)
        for _ in range(max_targets - 1):
            j = int(np.argmax(dmin))
            idx.append(j)
            dmin = np.minimum(dmin, np.linalg.norm(c - c[j], axis=1))
        sel = np.asarray(idx)
    return FingertipTargets(x1[sel], x2[sel], p1[sel], p2[sel], ok[sel], centres[sel], radii[0])


# ---------------------------------------------------------------- selection (eq. 12)
def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    c = np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-12)
    return np.arccos(np.clip(c, -1.0, 1.0))


def selection_costs(x: np.ndarray, X1: np.ndarray, X2: np.ndarray, w_angle: float):
    """Eq. (12) for every candidate, in both finger orders.

    ``x``: current fingertips ``(2, 3)``; ``X1``, ``X2``: ``(N, 3)`` world
    targets. Returns ``(cost (N,), swapped (N,) bool)``: the cheaper order.
    """
    d = x[1] - x[0]
    c_keep = (np.linalg.norm(X1 - x[0], axis=1) + np.linalg.norm(X2 - x[1], axis=1)
              + w_angle * _angle(d[None], X2 - X1))
    c_swap = (np.linalg.norm(X2 - x[0], axis=1) + np.linalg.norm(X1 - x[1], axis=1)
              + w_angle * _angle(d[None], X1 - X2))
    swapped = c_swap < c_keep
    return np.where(swapped, c_swap, c_keep), swapped


def select_target(x: np.ndarray, X1: np.ndarray, X2: np.ndarray, w_angle: float, valid=None):
    """Index and finger order of the closest target pair (eq. 12), or
    ``(None, False)`` if no candidate is valid."""
    cost, swapped = selection_costs(x, X1, X2, w_angle)
    if valid is not None:
        cost = np.where(valid, cost, np.inf)
    if not np.isfinite(cost).any():
        return None, False
    j = int(np.argmin(cost))
    return j, bool(swapped[j])
