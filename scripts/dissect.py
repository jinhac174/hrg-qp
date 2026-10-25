"""Replay a recorded simulation trial kinematically and show what binds.

``scripts/eval_sim.py`` saves the measured state of every trial
(``trace.npz``: ``state`` = time, q, width, object position, quaternion).
This script starts the planner from the recorded state at ``--t_start``,
follows the recorded object pose from there with perfect tracking
(``scripts/rollout.py``'s loop) and prints, at the end, the collision rows at
their margin and the joints at their limits: the paper's Fig. 12 question,
"why did this reach stop?", answered deterministically and off-line.

    python scripts/dissect.py outputs/eval/fig7/path_blue_mug_upright_center_up_2/trace.npz --t_start 4.5
    python scripts/dissect.py <trial>/trace.npz --t_start 29 --time 0      # the recorded end state only
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrg.hand import MAX_WIDTH  # noqa: E402
from hrg.planner import binding_text  # noqa: E402
from hrg.study import START_POSES  # noqa: E402
from scripts.rollout import make_planner  # noqa: E402


def recorded_pose(rec: np.ndarray, t: float) -> np.ndarray:
    i = int(np.argmin(np.abs(rec[:, 0] - t)))
    T = np.eye(4)
    T[:3, 3] = rec[i, 9:12]
    qw = rec[i, 12:16]
    T[:3, :3] = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--t_start", type=float, default=0.0, help="start from the recorded state at this time")
    ap.add_argument("--time", type=float, default=20.0, help="replay duration (0: the recorded state only)")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--tol", type=float, default=0.01, help="the grasp trigger distance")
    ap.add_argument("--set", nargs="*", default=[], help="controller config overrides")
    args = ap.parse_args()

    rec = np.load(args.trace)["state"]
    trial = os.path.basename(os.path.dirname(os.path.abspath(args.trace)))  # <field>_<object>_<orientation>_<pose>_<seed>
    field, rest = trial.split("_", 1)
    rest = rest.rsplit("_", 1)[0]  # drop the seed
    pose = next(p for p in START_POSES if rest.endswith("_" + p))
    obj, orientation = rest[: -len(pose) - 1].rsplit("_", 1)
    planner, _ = make_planner(obj, field, args.set)
    i0 = int(np.argmin(np.abs(rec[:, 0] - args.t_start)))
    q, w = rec[i0, 1:8].copy(), float(rec[i0, 8])
    print(f"{trial}: from the recorded state at t = {rec[i0, 0]:.2f} s, q = {np.round(q, 3).tolist()}, "
          f"w = {w * 100:.1f} cm, object at {np.round(rec[i0, 9:12], 3).tolist()}")

    r, last = None, -np.inf
    for k in range(max(1, int(args.time / args.dt))):
        t = args.t_start + k * args.dt
        r = planner.step(q, w, recorded_pose(rec, t))
        if r is None:
            print("no valid target at this object pose")
            return
        err = np.linalg.norm(r.x - r.x_tgt, axis=1)
        if t - last >= 1.0 or np.all(err <= args.tol):
            print(f"  t = {t:5.2f} s  fingertip errors {err[0] * 100:.2f} / {err[1] * 100:.2f} cm  "
                  f"target {r.target}{'s' if r.swapped else ''}  |qdot| {np.linalg.norm(r.qd):.3f}  w {w * 100:.1f} cm")
            last = t
        if np.all(err <= args.tol):
            print(f"converged at t = {t:.2f} s")
            break
        q = q + r.qd * args.dt
        w = float(np.clip(w + r.wd * args.dt, 0.0, MAX_WIDTH))
    else:
        print("not converged")
    print("at the end:", binding_text(r.info["binding"]).lstrip("; ") or "nothing at its margin")
    for i in range(2):
        print(f"  fingertip {i}: x - x* = {np.round((r.x[i] - r.x_tgt[i]) * 1e3, 1).tolist()} mm")


if __name__ == "__main__":
    main()
