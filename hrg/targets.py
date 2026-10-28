"""Fingertip target pairs and their selection (paper Sec. II, eq. 12; Fig. 4).

The paper assumes a set of candidate fingertip target pairs ``x*_j``,
predefined in the object frame (it takes them from T2SQNet), and selects one
every plan. Here they are generated once per object
(``scripts/prepare_objects.py``) and selected online.

Generation (object frame, upright)
----------------------------------
1. Contact pairs ``(p1, p2)`` from two sources:

   * rim pinches, as T2SQNet places them on tableware (paper Fig. 4): at a
     few depths below the top, horizontal rays toward the vertical axis find
     the outer skin; from there a ray along the wall's inward normal finds the
     inner skin, so the pair is antipodal on a slanted wall too (a mug's
     handle gives handle pinches the same way);
   * antipodal pairs anywhere on the surface (``sample_antipodal_grasps``).
2. Pre-grasp fingertip targets: each pad backs off ``offset`` along the pair
   axis, ``x1 = p1 - offset a``, ``x2 = p2 + offset a`` (``a = (p2 - p1)/|.|``),
   so the hand opens to ``|p2 - p1| + 2 offset`` (within the 85 mm stroke).
   The targets themselves must satisfy the path constraint of eq. (3),
   ``d(x*, O) >= eps_O``: they are the paths' end points.
3. Approach directions: the hand may be turned about the pair axis; for each
   of ``num_approach`` such directions (the TCP ``z``, ``approach_dirs``) the
   open hand must be admissible under the QP's constraints (eq. 11) at the
   pre-grasp *and* swept back along the approach over ``approach_dist`` (the
   straight final approach, ``fields.FingertipPathField``): every hand sphere
   (``hand.hand_spheres_local``) at least ``qp_eps_obj`` from the object (with
   the object's own distance function, the one the QP uses), and the gripper
   boxes clear of the actual mesh (what the simulation collides with). A pair
   with no such direction is dropped. The admissible directions are stored
   with their sphere centres at the pre-grasp (object frame).
4. Of pairs at the same place (centres within 3 mm, axes within 30 deg; a
   rim hit's horizontal and normal pinch) keep the one closest to antipodal
   (``misalign``, the pair axis vs the surface normals), then thin the set
   out (farthest-point sampling on the pair centres).

Validity online
---------------
The table depends on how the object lies: a direction is usable if every
hand sphere stays ``qp_eps_table`` above the table at the pre-grasp and at
the far end of the approach (``FingertipTargets.usable``); a pair is valid if
one of its admissible directions is usable.

Hand orientation (the oriented reach)
-------------------------------------
Each direction gives the pair a hand orientation in the object frame,
``R_grasp = [y x z, y, z]`` with ``y = a`` (fingertip 1 on ``x1``) and ``z``
the approach (``grasp_rotations``); online ``R* = R_obj R_grasp``, and the
other finger order is ``R* Rz(pi)``. Which (pair, direction, order) to take
is decided in joint space by the planner (``ReachPlanner._select_oriented``).

Selection (eq. 12, the paper's reach): the valid pair (either finger order)
minimising ``|x1 - x1*| + |x2 - x2*| + w * angle(x2 - x1, x2* - x1*)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from hrg.hand import MAX_WIDTH, HandCollisionChecker, hand_spheres_local, pad_dz

SWAP = np.diag([-1.0, -1.0, 1.0])  # R* -> the other finger order (Rz(pi))


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
    # TCP z of each approach direction (N, K, 3), object frame. Older files
    # lack it: recomputed from the pair axes.
    approach_dirs: np.ndarray | None = None
    # (N,) how far the pair axis is from the surface normals at p1 / p2 (rad,
    # the worse of the two): 0 = antipodal, the pads flat on the surface.
    misalign: np.ndarray | None = None
    # Preferred approach (TCP z) in the object frame: the object config's
    # ``approach_axis``; from above by default.
    approach_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))

    def __len__(self) -> int:
        return len(self.x1)

    @property
    def contact_width(self) -> np.ndarray:
        return np.linalg.norm(self.p2 - self.p1, axis=1)

    def save(self, path: str) -> None:
        arrays = dict(x1=self.x1, x2=self.x2, p1=self.p1, p2=self.p2, approach_ok=self.approach_ok,
                      spheres=self.spheres, sphere_r=self.sphere_r, approach_dirs=self.dirs(),
                      misalign=self.misalign)
        np.savez(path, **{k: v for k, v in arrays.items() if v is not None})

    @classmethod
    def load(cls, path: str, approach_axis=None) -> "FingertipTargets":
        d = np.load(path)
        extra = {k: d[k] for k in ("approach_ok", "spheres", "sphere_r", "approach_dirs", "misalign") if k in d.files}
        tg = cls(d["x1"], d["x2"], d["p1"], d["p2"], **extra)
        if approach_axis is not None:
            a = np.asarray(approach_axis, dtype=float)
            tg.approach_axis = a / np.linalg.norm(a)
        return tg

    def transformed(self, T: np.ndarray) -> "FingertipTargets":
        """The target points in the world (the approach data stays object frame)."""
        R, p = T[:3, :3], T[:3, 3]
        f = lambda X: X @ R.T + p  # noqa: E731
        return FingertipTargets(f(self.x1), f(self.x2), f(self.p1), f(self.p2),
                                self.approach_ok, self.spheres, self.sphere_r,
                                self.dirs(), self.misalign, self.approach_axis)

    # ------------------------------------------------------------ approach directions
    def dirs(self) -> np.ndarray | None:
        """(N, K, 3) approach directions, object frame."""
        if self.approach_dirs is None and self.approach_ok is not None:
            a = self.p2 - self.p1
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            k = self.approach_ok.shape[1]
            self.approach_dirs = np.stack([_approach_dirs(ai, k) for ai in a]) if len(a) else np.zeros((0, k, 3))
        return self.approach_dirs

    def usable(self, T_obj: np.ndarray, table_z: float, eps_table: float, approach_dist: float = 0.0) -> np.ndarray:
        """(N, K) admissible directions whose hand spheres all stay
        ``eps_table`` above the table with the object at ``T_obj``, at the
        pre-grasp and ``approach_dist`` back along the approach (the sweep
        is a translation: its lowest point is at one of the two ends)."""
        if self.spheres is None:
            return np.ones((len(self), 1), dtype=bool)
        up = T_obj[2, :3]
        z = np.einsum("j,nksj->nks", up, self.spheres) + T_obj[2, 3]
        clear = np.min(z - self.sphere_r, axis=2) >= table_z + eps_table  # (N, K)
        if approach_dist > 0.0 and self.dirs() is not None:
            back = z - approach_dist * np.einsum("j,nkj->nk", up, self.dirs())[:, :, None]
            clear &= np.min(back - self.sphere_r, axis=2) >= table_z + eps_table
        return clear & self.approach_ok

    def valid(self, T_obj: np.ndarray, table_z: float, eps_table: float, approach_dist: float = 0.0) -> np.ndarray:
        """(N,) pairs with a usable approach direction (``usable``)."""
        return np.any(self.usable(T_obj, table_z, eps_table, approach_dist), axis=1)

    def grasp_rotations(self, T_obj: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        """(M, 3, 3) world hand orientations ``R_obj R_grasp`` of pairs ``j``
        approached along their directions ``k`` (fingertip 1 on ``x1``)."""
        a = self.p2[j] - self.p1[j]
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        z = self.dirs()[j, k]
        z = z - np.sum(z * a, axis=1, keepdims=True) * a  # exactly perpendicular
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        R = np.stack([np.cross(a, z), a, z], axis=2)  # columns x, y, z
        return np.einsum("ij,njk->nik", T_obj[:3, :3], R)


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
    """Pinches across the wall below the top of an upright object:
    ``(p1 (N, 3) outer, p2 (N, 3) inner)``. The object is centred in x / y.

    A horizontal ray toward the axis finds the outer skin ``p1``. Two pairs
    per hit: the horizontal ray's own exit (T2SQNet's pinch), and where a ray
    from ``p1`` along the wall's inward normal leaves the wall, antipodal on
    a slanted wall (22 deg on the gold bowl), when it differs. Which of them
    the hand can approach depends on the wall: along a straight slanted wall
    only the normal one sweeps clear, on a curved one (the wine glass) the
    tilted hand meets the curve and the horizontal one remains (the
    admissibility check in ``generate_fingertip_targets`` decides). Where the
    outer normal is nearly vertical (a flat top) only the horizontal one."""
    top = float(mesh.bounds[1][2])
    R = float(np.linalg.norm(mesh.bounds[1][:2] - mesh.bounds[0][:2])) + 0.05
    th = 2.0 * np.pi * np.arange(n_azimuth) / n_azimuth
    dirs_xy = np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
    P1, N1, P2_ray = [], [], []
    for depth in depths:
        z = top - depth
        origins = dirs_xy * R + np.array([0.0, 0.0, z])
        locs, ray_idx, tri_idx = mesh.ray.intersects_location(origins, -dirs_xy, multiple_hits=True)
        for r in range(n_azimuth):
            sel = ray_idx == r
            hits, tris = locs[sel], tri_idx[sel]
            if len(hits) < 2:
                continue
            order = np.argsort(np.linalg.norm(hits - origins[r], axis=1))
            hits, tris = hits[order], tris[order]
            # first entry / exit, skipping duplicate hits on the same skin
            p1 = hits[0]
            rest = hits[np.linalg.norm(hits - p1, axis=1) > 1e-3]
            if len(rest) == 0:
                continue
            n1 = mesh.face_normals[tris[0]]
            if n1 @ dirs_xy[r] < 0.0:  # outward = toward where the ray came from
                n1 = -n1
            P1.append(p1)
            N1.append(n1)
            P2_ray.append(rest[0])
    if not P1:
        return np.zeros((0, 3)), np.zeros((0, 3))
    P1, N1, P2_ray = np.asarray(P1), np.asarray(N1), np.asarray(P2_ray)

    # Along the inward normal: the nearest exit beyond the start, just inside p1.
    out1, out2 = [P1], [P2_ray]
    idx = np.flatnonzero(np.abs(N1[:, 2]) < 0.7)
    if len(idx):
        locs, ray_idx, _ = mesh.ray.intersects_location(P1[idx] - 1e-4 * N1[idx], -N1[idx], multiple_hits=True)
        for i, r in enumerate(idx):
            hits = locs[ray_idx == i]
            d = np.linalg.norm(hits - P1[r], axis=1)
            d = d[d > 3e-4]
            if not len(d):
                continue
            p2 = P1[r] - d.min() * N1[r]
            a_ray = (P2_ray[r] - P1[r]) / np.linalg.norm(P2_ray[r] - P1[r])
            if a_ray @ -N1[r] < np.cos(np.radians(5.0)):  # a different pair
                out1.append(P1[r:r + 1])
                out2.append(p2[None])
    P1, P2 = np.vstack(out1), np.vstack(out2)
    keep = np.linalg.norm(P2 - P1, axis=1) <= max_width
    return P1[keep], P2[keep]


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
    num_approach: int = 24,
    max_targets: int = 150,
    min_width: float = 0.002,
    mu: float = 0.8,
    box_margin: float = 0.003,
    rim_depths=(0.008, 0.015, 0.025),
    approach_dist: float = 0.10,
    sweep_steps: int = 4,
    seed: int = 0,
) -> FingertipTargets:
    """See the module docstring. ``sdf``: the object's distance function in
    its own frame (identity pose), the same one the controller uses.
    ``approach_dist``: the straight final approach checked behind each
    pre-grasp (``hrg.approach_dist``), in ``sweep_steps`` steps."""
    max_contact = MAX_WIDTH - 2.0 * offset - 0.003
    pairs = sample_antipodal_grasps(mesh, num_samples=num_samples, mu=mu, max_width=max_contact,
                                    min_width=min_width, seed=seed)
    rp1, rp2 = rim_pinch_pairs(mesh, rim_depths, max_width=max_contact)
    # Rim rays have no spurious hits: keep walls down to 0.5 mm (the scanned
    # wine glass is 1 mm thick). min_width only guards the antipodal sampler.
    P1 = np.vstack([rp1, pairs.p1]) if len(pairs) else rp1
    P2 = np.vstack([rp2, pairs.p2]) if len(pairs) else rp2
    chk = HandCollisionChecker(mesh)
    steps = np.linspace(0.0, approach_dist, max(int(sweep_steps), 1) + 1) if approach_dist > 0 else np.zeros(1)

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
        dirs = _approach_dirs(a, num_approach)
        for z in dirs:
            T = _hand_pose(c, a, z, open_w)
            C = S_loc @ T[:3, :3].T + T[:3, 3]
            centres.append(C)
            good = True
            for back in steps:  # the hand swept back along the approach
                Ts = T.copy()
                Ts[:3, 3] -= back * z
                if (np.min(sdf.dist(C - back * z) - r) < qp_eps_obj
                        or chk.collides(Ts, open_w, margin=box_margin)):
                    good = False
                    break
            ok.append(good)
        if any(ok):
            keep.append((x1, x2, p1, p2, np.asarray(ok), np.asarray(centres), r, dirs))

    if not keep:
        e = np.zeros((0, 3))
        return FingertipTargets(e, e, e, e)
    x1, x2, p1, p2, ok, centres, radii, dirs = (np.asarray(v) for v in zip(*keep))
    mis = pair_misalignment(mesh, p1, p2)
    # Same place, similar axis: the most antipodal one.
    a = (p2 - p1) / np.linalg.norm(p2 - p1, axis=1, keepdims=True)
    mid = 0.5 * (p1 + p2)
    kept: list[int] = []
    for i in np.argsort(mis, kind="stable"):
        if not any(np.linalg.norm(mid[i] - mid[k]) < 0.003 and abs(a[i] @ a[k]) > np.cos(np.radians(30.0))
                   for k in kept):
            kept.append(i)
    kept = np.sort(np.asarray(kept))
    x1, x2, p1, p2, ok, centres, dirs, mis = (v[kept] for v in (x1, x2, p1, p2, ok, centres, dirs, mis))

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
    return FingertipTargets(x1[sel], x2[sel], p1[sel], p2[sel], ok[sel], centres[sel], radii[0], dirs[sel], mis[sel])


def pair_misalignment(mesh: trimesh.Trimesh, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """(N,) the worse angle between a pad's pressing direction and the
    surface's inward normal at its contact (rad): the pad at ``p1`` presses
    along ``a = (p2 - p1)/|.|``, the one at ``p2`` along ``-a``."""
    a = (p2 - p1) / np.maximum(np.linalg.norm(p2 - p1, axis=1, keepdims=True), 1e-12)
    _, _, f1 = trimesh.proximity.closest_point(mesh, p1)
    _, _, f2 = trimesh.proximity.closest_point(mesh, p2)
    n1, n2 = mesh.face_normals[f1], mesh.face_normals[f2]
    # outward normal at p1 faces away from p2 (and at p2 away from p1), whatever the mesh's winding
    n1 = np.where(np.sum(n1 * a, axis=1, keepdims=True) > 0, -n1, n1)
    n2 = np.where(np.sum(n2 * a, axis=1, keepdims=True) < 0, -n2, n2)
    c1 = np.sum(a * -n1, axis=1)
    c2 = np.sum(-a * -n2, axis=1)
    return np.arccos(np.clip(np.minimum(c1, c2), -1.0, 1.0))


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

