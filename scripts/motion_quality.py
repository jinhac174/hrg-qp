"""How smooth is the motion? One grasp in the real-time simulation, scored.

``scripts/diagnose_tracking.py`` answers "how fast does the arm follow" and
stops as soon as the controller enters ``hold``; this one keeps recording
through the settling, which is where the lift's overshoot and shaking are, and
scores what is seen and what stops the robot:

* lift: height overshoot past where the hand comes to rest, settling time
  (to 2 mm and 1 cm/s), the height's peak speed and acceleration;
* shake: RMS of the joint velocities above ``--band`` Hz, the arm's own
  oscillation (the plans have no content there), per phase, and the joint 1 / 3
  counter-motion;
* the hand's peak acceleration and jerk (20 ms smoothed);
* torque: what the low level sends, ``kd (qd_des - qd) + tau_ff``, as a fraction of the
  FR3's limit per joint, and its rate before anything limits it.

``--scenario static``: a resting object. ``hand``: the object moved around
while the arm chases it (``configs/sim/env_hand.yaml``: speed, direction changes
and perception noise like a bowl moved by hand), caught as it comes to rest.

    python scripts/motion_quality.py --object gold_bowl --pose center_up
    python scripts/motion_quality.py --scenario hand --ctrl grasp.lift_speed=0.3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LCM_DEFAULT_URL", f"udpm://239.255.76.67:{random.randint(30000, 39999)}?ttl=0")
os.environ["HRG_LOG"] = "0"

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from fr3.lcm_type.gripper.gripper_state_t import gripper_state_t  # noqa: E402
from fr3.lcm_type.joint.joint_ctrl_t import joint_ctrl_t  # noqa: E402
from fr3.lcm_type.joint.joint_meas_t import joint_meas_t  # noqa: E402
from fr3.lcm_type.pose.se3_pose_t import se3_pose_t  # noqa: E402
from fr3.robot_platform import fr3_limits  # noqa: E402
from hrg.paths import REPO, resolve  # noqa: E402
from hrg.processes import Procs  # noqa: E402
from hrg.study import ORIENTATIONS, START_POSES  # noqa: E402
from scripts.diagnose_tracking import History, _vec  # noqa: E402

TCP = "fr3_gripper_tcp"


def _smooth(x: np.ndarray, k: int = 20) -> np.ndarray:
    w = np.ones(k) / k
    return np.array([np.convolve(x[:, j], w, mode="same") for j in range(x.shape[1])]).T


def fk(q: np.ndarray, urdf: str) -> np.ndarray:
    model = pin.buildModelFromUrdf(resolve(urdf))
    data, fid = model.createData(), model.getFrameId(TCP)
    out = np.empty((len(q), 3))
    for i, qi in enumerate(q):
        pin.framesForwardKinematics(model, data, qi)
        out[i] = data.oMf[fid].translation
    return out


def resample(t: np.ndarray, x: np.ndarray, tt: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(tt, t, x[:, j]) for j in range(x.shape[1])])


def score(hist: dict, urdf: str, band: float, kd=None) -> dict:
    meas, cmd = hist["franka_arm_joint_meas"], hist["franka_arm_joint_ctrl"]
    tm = np.array([m.timestamp for m in meas])
    tm, idx = np.unique(tm, return_index=True)
    q, qd = np.array([meas[i].q for i in idx]), np.array([meas[i].qd for i in idx])
    tc = np.array([m.timestamp for m in cmd])
    tc, idc = np.unique(tc, return_index=True)
    qd_des = np.array([cmd[i].qd_des for i in idc])
    tau_ff = np.array([cmd[i].tau_ff for i in idc])
    kd_c = np.array([cmd[i].kd for i in idc])
    t = np.arange(max(tm[0], tc[0]), min(tm[-1], tc[-1]), 1e-3)
    q, qd = resample(tm, q, t), resample(tm, qd, t)
    qd_des, tau_ff, kd_c = resample(tc, qd_des, t), resample(tc, tau_ff, t), resample(tc, kd_c, t)
    tau = kd_c * (qd_des - qd) + tau_ff
    x = fk(q, urdf)
    v = np.gradient(x, t, axis=0)
    a = np.gradient(_smooth(v), t, axis=0)
    j = np.gradient(_smooth(a), t, axis=0)
    sos = butter(3, band, btype="high", fs=1000.0, output="sos")
    hp = sosfiltfilt(sos, qd, axis=0)
    # smoothed over the whole run, then cut into phases: smoothing a cut signal tapers its ends,
    # which reads as a huge acceleration when a phase begins or ends in motion
    qdd = np.gradient(_smooth(qd), axis=0) * 1e3
    cmd_a = np.gradient(_smooth(qd_des), axis=0) * 1e3
    cmd_j = np.gradient(_smooth(cmd_a), axis=0) * 1e3

    gs = hist["franka_gripper_state"]
    tg = np.array([m.timestamp for m in gs])
    grasped = np.array([bool(m.is_grasped) for m in gs])
    busy = np.array([bool(m.busy) for m in gs])
    t_lift = float(tg[np.argmax(grasped & ~busy)]) if np.any(grasped & ~busy) else np.nan
    res: dict = dict(t_total=float(t[-1] - t[0]))
    res["_trace"] = dict(t=t - t[0], q=q, qd=qd, qd_des=qd_des, tau=tau, tau_ff=tau_ff, x=x, v=v, a=a)
    phases = {"all": np.ones_like(t, dtype=bool)}
    if np.isfinite(t_lift):
        res["t_grasp"] = float(t_lift - t[0])
        phases["reach"] = t < t_lift
        phases["lift"] = t >= t_lift
        lift = t >= t_lift
        z = x[lift, 2]
        tl = t[lift] - t_lift
        z_end = float(np.median(z[-300:]))
        speed = np.linalg.norm(v[lift], axis=1)
        rest = (np.abs(z - z_end) > 0.002) | (speed > 0.01)
        res.update(lift_height=float(z_end - z[0]), lift_over_mm=float((z.max() - z_end) * 1e3),
                   lift_settle_s=float(tl[np.flatnonzero(rest)[-1]]) if rest.any() else 0.0,
                   lift_vz=float(v[lift, 2].max()), lift_az=float(np.abs(a[lift, 2]).max()))
    for name, sel in phases.items():
        if sel.sum() < 50:
            continue
        res[f"{name}_shake"] = np.round(np.sqrt(np.mean(hp[sel] ** 2, axis=0)), 4).tolist()
        res[f"{name}_shake_max"] = float(np.sqrt(np.mean(hp[sel] ** 2, axis=0)).max())
        res[f"{name}_c13"] = float(np.abs(hp[sel, 0] - hp[sel, 2]).max())
        res[f"{name}_a"] = float(np.linalg.norm(a[sel], axis=1).max())
        res[f"{name}_j"] = float(np.linalg.norm(j[sel], axis=1).max())
        res[f"{name}_tau"] = float((np.abs(tau[sel]) / fr3_limits.TAU_MAX).max())
        res[f"{name}_tau_j"] = np.round((np.abs(tau[sel]) / fr3_limits.TAU_MAX).max(axis=0), 2).tolist()
        dtau = np.abs(np.diff(tau[sel], axis=0)) * 1e3
        res[f"{name}_dtau_sat"] = float(np.mean(dtau > fr3_limits.TAU_RATE_MAX))
        res[f"{name}_dtau"] = float(np.percentile(dtau, 99.9))
        res[f"{name}_v"] = float(np.linalg.norm(v[sel], axis=1).max())
        res[f"{name}_qdd"] = float(np.abs(qdd[sel]).max())
        # what was asked, unlike the above blind to contact (a carried object hitting a finger)
        res[f"{name}_cmd_qdd"] = float(np.abs(cmd_a[sel]).max())
        res[f"{name}_cmd_jrk"] = float(np.percentile(np.abs(cmd_j[sel]), 99.9))
        res[f"{name}_ff"] = float((np.abs(tau_ff[sel]) / fr3_limits.TAU_MAX).max())
    return res


def run(args) -> dict:
    extra = [a for kv in args.ctrl for a in ("--" + kv.split("=", 1)[0], kv.split("=", 1)[1])]
    channels = {"franka_arm_joint_ctrl": joint_ctrl_t.decode, "franka_arm_joint_meas": joint_meas_t.decode,
                "franka_gripper_state": gripper_state_t.decode, "franka_object_pose": se3_pose_t.decode}
    env_cfg = list(args.env)
    if args.scenario == "hand":
        env_cfg = ["configs/sim/env_hand.yaml", *env_cfg]
    with History(channels, tuple(channels)) as lst, Procs(args.out) as procs:
        obj = f"configs/objects/{args.object}.yaml"
        env = procs.start("env", "run_env.py", "--config", "configs/sim/env.yaml", *env_cfg, "--vis_mode", "vis_off",
                          "--object_config", obj, "--obj_spawn_seed", str(args.seed),
                          "--platform.default_q_values", _vec(START_POSES[args.pose]),
                          "--obj_spawn_rpy", _vec(ORIENTATIONS["upright"]))
        if not lst.wait_for(lambda: "franka_object_pose" in lst.last, 60):
            sys.exit("env did not start")
        lst.spin(1.0)
        ctrl = procs.start("ctrl", "run_controller.py", "--config", "configs/controller.yaml", "--object_config", obj, *extra)
        if not lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, 90):
            sys.exit("controller did not start")
        for h in lst.hist.values():
            h.clear()
        if args.scenario == "hand":
            env.key("m")
            lst.spin(0.3)
        ctrl.key("v")
        t0 = time.time()
        t_done = None
        while time.time() - t0 < args.timeout:
            lst.spin(0.05)
            log = ctrl.log()
            if t_done is None and "Press x to release" in log:
                t_done = time.time()
            if t_done is not None and time.time() - t_done > args.settle:
                break
        hist, log = lst.hist, ctrl.log()
        res = score(hist, OmegaConf.load(resolve("configs/controller.yaml")).urdf_path, args.band)
        res["held"] = t_done is not None
        res["qp_failed"] = log.count("QP failed")
        res["reselect"] = log.count("no longer reachable")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="gold_bowl")
    ap.add_argument("--pose", default="center_up", choices=list(START_POSES))
    ap.add_argument("--scenario", default="static", choices=["static", "hand"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--settle", type=float, default=2.5, help="s recorded after hold begins")
    ap.add_argument("--band", type=float, default=5.0, help="Hz: joint velocity above this counts as shake")
    ap.add_argument("--ctrl", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--env", nargs="*", default=[], help="extra env configs")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "quality"))
    ap.add_argument("--json", default=None, help="append the result to this file")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    res = run(args)
    np.savez(os.path.join(args.out, "trace.npz"), **res.pop("_trace"))
    res.update(tag=args.tag, object=args.object, pose=args.pose, scenario=args.scenario, seed=args.seed)
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
