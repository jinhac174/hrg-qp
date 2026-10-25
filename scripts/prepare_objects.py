"""Prepare the objects (paper Table I / real experiments).

For each object (raw BundleSDF scan in ``assets/objects/<name>/scan``, in the
camera frame it was scanned in) this writes ``assets/objects/<name>/``:

* ``visual.obj``           cleaned, upright mesh: z up, resting on z = 0,
                           centred in x / y (the object frame from here on)
* ``collision/part_*.obj`` convex decomposition of the shell. MuJoCo collides
                           a mesh as its convex hull, which would fill a bowl.
* ``surface_points.npy``   (N, 6) surface points + normals: the point-cloud
                           distance function of the paper
* ``fingertip_targets.npz`` candidate pre-grasp fingertip pairs
                           (``hrg/targets.py``)

and ``configs/objects/<name>.yaml``.

The upright rotations were found once with ``trimesh.poses.compute_stable_poses``
and checked (open side up: a vertical ray at the centre hits the inside of the
bottom); they are fixed here so the output is reproducible.

    python scripts/prepare_objects.py                 # all objects
    python scripts/prepare_objects.py --only blue_bowl
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrg.paths import REPO as ROOT  # noqa: E402
from hrg.sdf import sdf_from_config  # noqa: E402
from hrg.targets import generate_fingertip_targets  # noqa: E402

MESH_DIR = "assets/objects"
CFG_DIR = "configs/objects"

# name: raw mesh, rotation (rotvec) to upright, mass (kg), colour.
# "box" is synthetic (T2SQNet-sized): the scanned mouse box is 12 cm at its
# narrowest, wider than the 2F-85 can take with the pre-grasp offsets.
OBJECTS = {
    "wine_glass": dict(rotvec=[np.pi, 0.0, 0.0], mass=0.20, rgba=[0.85, 0.9, 0.95, 0.6]),
    "blue_bowl": dict(rotvec=[-2.156449, -0.041221, -0.022115], mass=0.25, rgba=[0.2, 0.35, 0.8, 1.0]),
    "blue_mug": dict(rotvec=[-2.033391, 0.153188, 0.094314], mass=0.30, rgba=[0.25, 0.4, 0.85, 1.0]),
    "white_plate": dict(rotvec=[-2.892381, 0.422862, 0.046793], mass=0.35, rgba=[0.92, 0.92, 0.9, 1.0]),
    "red_cup": dict(rotvec=[-np.pi / 2, 0.0, 0.0], mass=0.05, rgba=[0.85, 0.15, 0.15, 1.0]),
    "tape": dict(rotvec=[np.pi / 2, 0.0, 0.0], mass=0.10, rgba=[0.8, 0.75, 0.6, 1.0]),
    "box": dict(extents=[0.05, 0.07, 0.10], mass=0.15, rgba=[0.75, 0.55, 0.3, 1.0]),
}


def clean_upright(mesh: trimesh.Trimesh, rotvec) -> trimesh.Trimesh:
    """Merge texture-seam vertices, drop scan debris, rotate upright, put the
    base on z = 0 and the x / y bounding-box centre at the origin."""
    mesh = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=True)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.remove_unreferenced_vertices()
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        areas = np.array([p.area for p in parts])
        mesh = trimesh.util.concatenate([p for p, a in zip(parts, areas) if a > 0.02 * areas.max()])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    mesh.apply_transform(T)
    lo, hi = mesh.bounds
    mesh.apply_translation([-0.5 * (lo[0] + hi[0]), -0.5 * (lo[1] + hi[1]), -lo[2]])
    return mesh


def shell_convex_parts(mesh: trimesh.Trimesh, cell: float, overlap: float = 0.002,
                       n_samples: int = 150000, min_thick: float = 0.0015):
    """Convex parts of a thin-walled object: the hull of the surface samples
    in each ``cell``-sized cube (grown by ``overlap`` so neighbours overlap).
    A cell holding both skins of a wall gives a solid piece of wall; one
    holding a single skin gets ``min_thick`` of thickness inward.

    For these scans this is tighter than CoACD, whose voxel remeshing thickens
    thin walls: across the bowl's rim pinches the wall the fingers meet is
    +1.0 mm (median) thicker than the real one, vs +2.7 mm for CoACD.
    """
    pts, fidx = trimesh.sample.sample_surface(mesh, n_samples, seed=0)
    nrm = mesh.face_normals[fidx]
    keys = np.floor(pts / cell).astype(int)
    parts = []
    for key in np.unique(keys, axis=0):
        m = np.all((pts >= key * cell - overlap) & (pts <= (key + 1) * cell + overlap), axis=1)
        if m.sum() < 8:
            continue
        hull = trimesh.Trimesh(pts[m]).convex_hull
        if hull.volume < 0.5 * cell**2 * min_thick:
            hull = trimesh.Trimesh(np.vstack([pts[m], pts[m] - min_thick * nrm[m]])).convex_hull
        parts.append(hull)
    return parts


def prepare(name: str, spec: dict, args) -> None:
    t0 = time.time()
    out = os.path.join(ROOT, MESH_DIR, name)
    os.makedirs(os.path.join(out, "collision"), exist_ok=True)

    if "extents" in spec:
        ext = np.asarray(spec["extents"], dtype=float)
        mesh = trimesh.creation.box(extents=ext)
        mesh.apply_translation([0.0, 0.0, 0.5 * ext[2]])
        parts = [mesh]
        sdf = dict(type="box", half=(0.5 * ext).tolist(), centre=[0.0, 0.0, float(0.5 * ext[2])])
    else:
        raw = trimesh.load(os.path.join(ROOT, MESH_DIR, name, "scan", "textured_mesh.obj"), force="mesh")
        mesh = clean_upright(raw, spec["rotvec"])
        parts = shell_convex_parts(mesh, args.cell)
        sdf = dict(type="point_cloud", path=f"{MESH_DIR}/{name}/surface_points.npy")

    mesh.export(os.path.join(out, "visual.obj"))
    for f in os.listdir(os.path.join(out, "collision")):
        os.remove(os.path.join(out, "collision", f))
    for k, p in enumerate(parts):
        p.export(os.path.join(out, "collision", f"part_{k:03d}.obj"))

    pts, fidx = trimesh.sample.sample_surface_even(mesh, args.num_points, seed=0)
    np.save(os.path.join(out, "surface_points.npy"), np.hstack([pts, mesh.face_normals[fidx]]))

    # Targets are checked with the distance function the controller will use.
    tg = generate_fingertip_targets(mesh, sdf_from_config(sdf), max_targets=args.max_targets)
    tg.save(os.path.join(out, "fingertip_targets.npz"))

    lo, hi = mesh.bounds
    cfg_path = os.path.join(ROOT, CFG_DIR, f"{name}.yaml")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        f.write(
            f"# Generated by scripts/prepare_objects.py. Object frame: upright, base on\n"
            f"# z = 0, centred in x / y. Size {np.round(hi - lo, 3).tolist()} m.\n"
            f"name: {name}\n"
            f"mesh_path: {MESH_DIR}/{name}/visual.obj\n"
            f"collision_dir: {MESH_DIR}/{name}/collision\n"
            f"mass: {spec['mass']}\n"
            f"rgba: {spec['rgba']}\n"
            f"friction: [0.8, 0.005, 0.0001]\n"
            f"spawn_rpy: [0.0, 0.0, 0.0]\n"
            f"sdf: {sdf}\n"
            f"fingertip_targets: {MESH_DIR}/{name}/fingertip_targets.npz\n"
        )
    print(f"[{name}] size {np.round(hi - lo, 3)} m, {len(parts)} convex parts, "
          f"{len(pts)} surface points, {len(tg)} fingertip targets ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--num_points", type=int, default=6000)
    ap.add_argument("--max_targets", type=int, default=150)
    ap.add_argument("--cell", type=float, default=0.012, help="convex part size (m)")
    args = ap.parse_args()
    for name, spec in OBJECTS.items():
        if args.only and name not in args.only:
            continue
        prepare(name, spec, args)


if __name__ == "__main__":
    main()
