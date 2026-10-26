"""Simulation study (paper Sec. III-A: Table I, Fig. 6, Fig. 7, Fig. 8).

For every object x orientation x start pose x placement (x field), start the
sim env and the controller headless, press ``v`` and record:

* success: the object is lifted (> 5 cm) and the gripper holds it within
  ``--timeout`` s (paper: "lifts the object within 20 seconds");
* the fingertip-centre error at the end (Fig. 6), from the controller's
  fingertip channel;
* time to grasp, time to lift, plan time, robot-obstacle contact (Fig. 8:
  a failure), the rates at which commands / measurements reached this
  process (bus health).

Perturbations (Fig. 7): ``--push_at`` / ``--jump_at`` press the env's ``k``
(push the hand) / ``o`` (move the object) at those times after ``v``. Each
trial saves ``trace.npz``: ``trace`` (time, fingertip centre, target centre,
object position) for ``scripts/plot_trace.py`` and ``state`` (time, q, width,
object position, quaternion wxyz) to replay the trial kinematically.

    python scripts/eval_sim.py --fields path linear --orientations all        # Table I
    python scripts/eval_sim.py --objects blue_bowl --poses center_up --seeds 1 \
        --push_at 1.5 4.0 --jump_at 3.0 --timeout 30                        # Fig. 7
    python scripts/eval_sim.py --env configs/sim/env_obstacles.yaml \
        --objects wine_glass blue_bowl --seeds 3 --timeout 30               # Fig. 8

Results: ``outputs/eval/<stamp>/results.csv``, one log folder per trial.

Run it alone: every trial is two busy real-time processes, and other sims or
tests on the machine starve them. Trials broken by the machine (LCM down after
the Mac slept, a wall-clock jump, a process that did not start) are marked
``infra`` and retried.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Own LCM bus, so a sim you have open is not disturbed (child processes inherit it).
os.environ.setdefault("LCM_DEFAULT_URL", f"udpm://239.255.76.67:{random.randint(30000, 39999)}?ttl=0")

from fr3.lcm_type.gripper.gripper_state_t import gripper_state_t  # noqa: E402
from fr3.lcm_type.joint.joint_ctrl_t import joint_ctrl_t  # noqa: E402
from fr3.lcm_type.joint.joint_meas_t import joint_meas_t  # noqa: E402
from fr3.lcm_type.pose.se3_pose_t import se3_pose_t  # noqa: E402
from fr3.lcm_type.vector.vec_list_t import vec_list_t  # noqa: E402
from hrg.paths import REPO  # noqa: E402
from hrg.processes import Listener, Procs, lcm_ok  # noqa: E402
from hrg.study import ORIENTATIONS, START_POSES, TABLE1_OBJECTS, orientations_for  # noqa: E402

ENV_CFG = "configs/sim/env.yaml"
CTRL_CFG = "configs/controller.yaml"
CHANNELS = {
    "franka_arm_joint_meas": joint_meas_t.decode,
    "franka_arm_joint_ctrl": joint_ctrl_t.decode,
    "franka_gripper_state": gripper_state_t.decode,
    "franka_object_pose": se3_pose_t.decode,
    "franka_hrg_fingertips": vec_list_t.decode,
}


def wait_for_lcm() -> None:
    warned = False
    while not lcm_ok():
        if not warned:
            print("  LCM self-test failing (multicast route gone? the Mac slept / network changed); waiting...",
                  flush=True)
            warned = True
        time.sleep(5.0)


def _vec(q) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in q) + "]"


def fingertip_centres(msg):
    """(current, target) fingertip centres from the controller's fingertip message."""
    p = {n: np.asarray(v, dtype=float) for n, v in zip(msg.name_list, msg.vec_list)}
    return 0.5 * (p["thumb_cur"] + p["index_cur"]), 0.5 * (p["thumb_tgt"] + p["index_tgt"])


def env_args(env_cfgs, obj: str, pose: str, seed: int, orientation: str):
    return ["run_env.py", "--config", *env_cfgs, "--vis_mode", "vis_off",
            "--object_config", f"configs/objects/{obj}.yaml", "--obj_spawn_seed", str(seed),
            "--platform.default_q_values", _vec(START_POSES[pose]),
            "--obj_spawn_rpy", _vec(ORIENTATIONS[orientation])]


def ctrl_args(obj: str, field: str, extra=()):
    return ["run_controller.py", "--config", CTRL_CFG, "--object_config", f"configs/objects/{obj}.yaml",
            "--hrg.field", field, *extra]


def run_trial(out: Path, obj: str, pose: str, seed: int, field: str, timeout: float, env_cfgs=(ENV_CFG,),
              push_at=(), jump_at=(), orientation: str = "upright", ctrl_extra=()) -> dict:
    row = dict(object=obj, orientation=orientation, pose=pose, seed=seed, field=field, success=False,
               t_grasp=np.nan, t_lift=np.nan, ft_err=np.nan, infra=False)
    with Listener(CHANNELS) as lst, Procs(out) as procs:
        env = procs.start("env", *env_args(env_cfgs, obj, pose, seed, orientation))
        if not lst.wait_for(lambda: "franka_object_pose" in lst.last, 60):
            row["note"], row["infra"] = "env did not start", True
            return row
        lst.spin(1.0)
        z0 = float(lst.last["franka_object_pose"].position[2])
        ctrl = procs.start("ctrl", *ctrl_args(obj, field, ctrl_extra))
        if not lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, 90):
            row["note"], row["infra"] = "controller did not start", True
            return row
        lst.spin(1.0)
        ctrl.key("v")
        lst.reset_counts()
        t0 = t_prev = time.time()
        err_hist, trace, state = [], [], []
        events = sorted([(t, "k") for t in push_at] + [(t, "o") for t in jump_at])
        while time.time() - t0 < timeout:
            lst.spin(0.05)
            if time.time() - t_prev > 2.0:  # the machine slept / froze
                row["note"], row["infra"] = "wall-clock jump (sleep?)", True
                return row
            t_prev = time.time()
            while events and time.time() - t0 >= events[0][0]:
                env.key(events.pop(0)[1])
            ft = lst.last.get("franka_hrg_fingertips")
            if ft is not None and ft.num_vecs == 4:
                c, c_tgt = fingertip_centres(ft)
                err_hist.append(np.linalg.norm(c - c_tgt))
                trace.append(np.r_[time.time() - t0, c, c_tgt, lst.last["franka_object_pose"].position])
                jm, gs, op = (lst.last.get(k) for k in ("franka_arm_joint_meas", "franka_gripper_state",
                                                         "franka_object_pose"))
                if jm is not None and gs is not None:  # the full state, to replay a trial kinematically
                    state.append(np.r_[time.time() - t0, jm.q[:7], gs.width, op.position, op.quat_wxyz])
            if np.isnan(row["t_grasp"]) and "Grasped after" in ctrl.log():
                row["t_grasp"] = time.time() - t0
            gs = lst.last.get("franka_gripper_state")
            held = gs is not None and gs.is_grasped
            if float(lst.last["franka_object_pose"].position[2]) - z0 > 0.05 and held and not events:
                row["success"], row["t_lift"] = True, time.time() - t0
                break
        dt = time.time() - t0
        row["ctrl_hz"] = lst.count["franka_arm_joint_ctrl"] / dt
        row["meas_hz"] = lst.count["franka_arm_joint_meas"] / dt
        if row["meas_hz"] < 500.0 or row["ctrl_hz"] < 300.0:  # the bus or a process died (expected 1000 / ~700 Hz)
            row["note"], row["infra"] = f"streams dropped ({row['meas_hz']:.0f} / {row['ctrl_hz']:.0f} Hz)", True
            return row
        if err_hist:
            row["ft_err"] = float(err_hist[-1])
            np.savez(out / "trace.npz", trace=np.asarray(trace), state=np.asarray(state),
                     push_at=np.asarray(push_at, float), jump_at=np.asarray(jump_at, float))
        row["obstacle_contact"] = "ROBOT-OBSTACLE CONTACT" in env.log()
        log = ctrl.log()
        row["n_grasp_fail"] = log.count("Grasp failed") + log.count("Object lost")
        plan = [ln for ln in log.splitlines() if "plan time" in ln]
        row["plan_ms"] = plan[-1].split("plan time ")[1].split(" ms")[0] if plan else ""
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", default=TABLE1_OBJECTS)
    ap.add_argument("--poses", nargs="*", default=list(START_POSES))
    ap.add_argument("--seeds", type=int, default=5, help="object placements per pose")
    ap.add_argument("--orientations", nargs="*", default=["upright"],
                    help="upright / rx+90 / rx-90 / ry+90 / ry-90, or 'all' (rotations for the mug and wine glass)")
    ap.add_argument("--fields", nargs="*", default=["path"])
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--env", nargs="*", default=[], help="extra env configs on top of configs/sim/env.yaml")
    ap.add_argument("--ctrl", nargs="*", default=[], metavar="KEY=VALUE",
                    help="controller config overrides, e.g. --ctrl hrg.horizon=0.15 grasp.lift_height=0.2")
    ap.add_argument("--push_at", type=float, nargs="*", default=[])
    ap.add_argument("--jump_at", type=float, nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or os.path.join(REPO, "outputs", "eval", time.strftime("%Y%m%d_%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)
    env_cfgs = [ENV_CFG, *args.env]
    ctrl_extra = [a for kv in args.ctrl for a in ("--" + kv.split("=", 1)[0], kv.split("=", 1)[1])]
    rows = []
    for field in args.fields:
        for obj in args.objects:
            for ori, pose in [(o, p) for o in orientations_for(obj, args.orientations) for p in args.poses]:
                for seed in range(1, args.seeds + 1):
                    for _ in range(3):
                        wait_for_lcm()
                        r = run_trial(out / f"{field}_{obj}_{ori}_{pose}_{seed}", obj, pose, seed, field,
                                      args.timeout, env_cfgs, args.push_at, args.jump_at, ori, ctrl_extra)
                        if not r["infra"]:
                            break
                        print(f"  infra problem ({r.get('note')}), retrying", flush=True)
                    rows.append(r)
                    print(f"{field:6s} {obj:12s} {ori:7s} {pose:12s} seed {seed}: "
                          f"{'OK ' if r['success'] else 'FAIL'} t_lift={r['t_lift']:.1f}s "
                          f"ft_err={r['ft_err'] * 100:.1f}cm plan={r.get('plan_ms', '')}ms "
                          f"{'OBSTACLE CONTACT ' if r.get('obstacle_contact') else ''}{r.get('note', '')}", flush=True)
                    with open(out / "results.csv", "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=sorted({k for x in rows for k in x}))
                        w.writeheader()
                        w.writerows(rows)
    print("\nSuccess rates (infra trials excluded):")
    for field in args.fields:
        for obj in args.objects:
            s = [r["success"] for r in rows if r["field"] == field and r["object"] == obj and not r["infra"]]
            if s:
                print(f"  {field:6s} {obj:12s} {100.0 * np.mean(s):5.1f}%  ({sum(s)}/{len(s)})")
    print(f"Results: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
