"""Kinematic rollouts of the reach (no physics, no LCM, no timing).

``hrg.planner.ReachPlanner.step`` -- exactly what the controller runs every
plan -- in a closed loop with perfect tracking (``q += qdot dt``,
``w += wdot dt`` at the planner rate) over many object placements and start
poses. Deterministic and independent of CPU load: it separates the method
(local minima, deadlocks, constraint violations) from execution (dynamics,
friction, timing) in the real-time simulation.

Per rollout: converged (both fingertips within ``--tol`` of their targets
within ``--time`` s), time, final fingertip error, and the smallest clearance
of the hand spheres to the object and the table along the way; with the
oriented reach (``hrg.orient``) also the hand's final angle to its target
orientation and how far joint 7 turned. A rollout
that does not converge prints what binds at the end (collision rows at their
margin, joints at their limits). ``scripts/dissect.py`` does the same for a
recorded simulation trial.

    python scripts/rollout.py --objects box blue_bowl --placements 20
    python scripts/rollout.py --fields path linear --orientations upright rx+90
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrg.hand import MAX_WIDTH, hand_spheres  # noqa: E402
from hrg.objects import GraspObject  # noqa: E402
from hrg.paths import resolve  # noqa: E402
from hrg.planner import HRGParams, ReachPlanner, binding_text  # noqa: E402
from hrg.robot import RobotModel, scene_collision_info  # noqa: E402
from hrg.sdf import sdf_from_config  # noqa: E402
from hrg.study import ORIENTATIONS, START_POSES, TABLE1_OBJECTS, orientations_for  # noqa: E402
from hrg.targets import FingertipTargets  # noqa: E402

CTRL_CFG = "configs/controller.yaml"
ENV_CFG = "configs/sim/env.yaml"


def make_planner(obj: str, field: str, overrides=()) -> tuple[ReachPlanner, GraspObject]:
    cfg = OmegaConf.merge(OmegaConf.load(resolve(CTRL_CFG)), OmegaConf.from_dotlist(list(overrides)))
    cfg.hrg.field = field
    o = GraspObject(f"configs/objects/{obj}.yaml")
    robot = RobotModel(cfg.urdf_path, cfg.joint_velocity_limit, cfg.col_pairs, cfg.self_col_pairs,
                       cfg.env_col_margin, cfg.self_col_margin)
    robot.set_collision_geometry(*scene_collision_info(OmegaConf.load(resolve(ENV_CFG)).scene_xml_path))
    planner = ReachPlanner(HRGParams.from_config(cfg.hrg), robot,
                           FingertipTargets.load(resolve(o.cfg.fingertip_targets), o.cfg.get("approach_axis")),
                           sdf_from_config(o.cfg.sdf), table_z=float(cfg.grasp.table_z))
    return planner, o


def placements(n: int, seed: int, obj: GraspObject, rpy=(0.0, 0.0, 0.0)):
    """Random x / y / yaw in the env's spawn region, the object resting on
    the table in orientation ``rpy``."""
    env = OmegaConf.load(resolve(ENV_CFG))
    rng = np.random.default_rng(seed)
    c, h = np.asarray(env.obj_spawn_xy_center), np.asarray(env.obj_spawn_xy_half)
    R0 = Rotation.from_euler("xyz", list(rpy)).as_matrix()
    out = []
    for _ in range(n):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_matrix() @ R0
        T[:2, 3] = c + rng.uniform(-h, h)
        T[2, 3] = obj.resting_height(T[:3, :3])
        out.append(T)
    return out


def rollout(planner: ReachPlanner, q0, T_obj, horizon: float, dt: float, tol: float,
            rot_tol: float = np.radians(10.0)) -> dict:
    """``rot_tol``: with the oriented reach, the hand must also be this close
    to its target orientation (the controller's ``grasp_trigger_rot``)."""
    planner.reset()
    q0 = np.asarray(q0, dtype=float)
    q, w = q0.copy(), MAX_WIDTH
    data = planner.robot.data()

    def clearances(q, w):
        J = planner.robot.jacobian(q, data)
        C, _, r = hand_spheres(planner.robot.pose(q, data), J, w)
        return float(np.min(planner.obj_sdf.dist(C) - r)), float(np.min(C[:, 2] - r - planner.table.z0))

    planner.obj_sdf.set_pose(T_obj)
    clear_obj, clear_table = clearances(q, w)
    if clear_obj < planner.p.qp_eps_obj:  # the env redraws such placements
        return dict(converged=False, t=np.nan, err=np.nan, clear_obj=np.inf, clear_table=np.inf,
                    note="invalid start: hand within the margin of the object")
    err, r = np.full(2, np.inf), None
    for k in range(int(horizon / dt)):
        r = planner.step(q, w, T_obj)
        if r is None:
            return dict(converged=False, t=np.nan, err=np.nan, clear_obj=clear_obj, clear_table=clear_table,
                        note="no target")
        err = np.linalg.norm(r.x - r.x_tgt, axis=1)
        co, ct = clearances(q, w)
        clear_obj, clear_table = min(clear_obj, co), min(clear_table, ct)
        if np.all(err < tol) and not r.info.get("rot_err", 0.0) > rot_tol:
            return dict(converged=True, t=k * dt, err=float(err.max()), clear_obj=clear_obj,
                        clear_table=clear_table, note="", rot=float(r.info.get("rot_err", np.nan)),
                        dq7=float(abs(q[6] - q0[6])))
        q = q + r.qd * dt
        w = float(np.clip(w + r.wd * dt, 0.0, MAX_WIDTH))
    return dict(converged=False, t=np.nan, err=float(err.max()), clear_obj=clear_obj, clear_table=clear_table,
                note=binding_text(r.info["binding"]).lstrip("; "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", default=TABLE1_OBJECTS)
    ap.add_argument("--poses", nargs="*", default=list(START_POSES))
    ap.add_argument("--fields", nargs="*", default=["path"])
    ap.add_argument("--placements", type=int, default=10)
    ap.add_argument("--orientations", nargs="*", default=["upright"], help="upright / rx+90 / ... or all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.01, help="planner period (s)")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--set", nargs="*", default=[], help="controller config overrides, e.g. hrg.sqp_iters=5")
    args = ap.parse_args()

    for field in args.fields:
        for obj in args.objects:
            planner, o = make_planner(obj, field, args.set)
            for ori, pose in [(a, b) for a in orientations_for(obj, args.orientations) for b in args.poses]:
                res = []
                for i, T in enumerate(placements(args.placements, args.seed, o, ORIENTATIONS[ori])):
                    t0 = time.time()
                    r = rollout(planner, START_POSES[pose], T, args.time, args.dt, args.tol)
                    res.append(r)
                    if not r["converged"]:
                        yaw = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")[2]
                        print(f"  STUCK {field} {obj} {ori} {pose} #{i} xy=({T[0, 3]:.3f},{T[1, 3]:.3f}) "
                              f"yaw={yaw:.2f}: err {r['err'] * 100:.1f} cm {r['note']} ({time.time() - t0:.0f} s)",
                              flush=True)
                res = [r for r in res if not r["note"].startswith("invalid start")]
                ts = [r["t"] for r in res if r["converged"]]
                rot = [np.degrees(r["rot"]) for r in res if r["converged"] and np.isfinite(r["rot"])]
                dq7 = [np.degrees(r["dq7"]) for r in res if r["converged"]]
                ori_txt = (f", hand {np.max(rot):.1f} deg off at most, joint 7 turned {np.max(dq7):.0f} deg at most"
                           if rot else "")
                print(f"{field:6s} {obj:11s} {ori:7s} {pose:12s} {sum(r['converged'] for r in res):3d}/{len(res)} "
                      f"converged, t {np.median(ts) if ts else np.nan:.1f} s median, min clearance obj "
                      f"{min(r['clear_obj'] for r in res) * 100:.1f} cm, table "
                      f"{min(r['clear_table'] for r in res) * 100:.1f} cm{ori_txt}", flush=True)


if __name__ == "__main__":
    main()
