"""How fast does the arm follow the planner, and how smooth is the gripper?

Runs one reach in the real-time simulation (like ``scripts/eval_sim.py``) and
records every message on the bus: the controller's joint command (``qd_des``,
``tau_ff``) and the arm's measurement (``qd``) at 1 kHz, the gripper's width
at 100 Hz and the controller's gripper commands. Then it reports, per joint,

* the **lag** between the commanded and the measured joint velocity (the
  shift that maximises their cross-correlation): the closed-loop latency the
  QP horizon ``H`` must cover;
* the velocity **tracking error** (RMS of ``qd - qd_des`` while moving);
* the **peak** joint speed and acceleration actually reached;

and for the gripper the number of set-points sent, the width-rate roughness
(RMS of the width acceleration: a set-point stream that stops and goes between
set-points shows up here) and the peak width rate. ``--ctrl KEY=VALUE``
overrides controller parameters (as in eval_sim). ``trace.npz`` is saved.

    python scripts/diagnose_tracking.py --object gold_bowl --pose left_bottom
    python scripts/diagnose_tracking.py --object blue_mug --ctrl hrg.v_const=0.8 kd=[75,75,75,30,30,30,15]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LCM_DEFAULT_URL", f"udpm://239.255.76.67:{random.randint(30000, 39999)}?ttl=0")
os.environ["HRG_LOG"] = "0"

import numpy as np  # noqa: E402

from fr3.lcm_type.gripper.gripper_cmd_t import gripper_cmd_t  # noqa: E402
from fr3.lcm_type.gripper.gripper_state_t import gripper_state_t  # noqa: E402
from fr3.lcm_type.joint.joint_ctrl_t import joint_ctrl_t  # noqa: E402
from fr3.lcm_type.joint.joint_meas_t import joint_meas_t  # noqa: E402
from fr3.lcm_type.pose.se3_pose_t import se3_pose_t  # noqa: E402
from hrg.paths import REPO  # noqa: E402
from hrg.processes import Listener, Procs  # noqa: E402
from hrg.study import ORIENTATIONS, START_POSES  # noqa: E402


class History(Listener):
    """A ``Listener`` that also keeps every message of some channels."""

    def __init__(self, channels: dict, keep: tuple):
        super().__init__(channels)
        self.hist = {c: [] for c in keep}

    def _cb(self, ch, decode):
        f = super()._cb(ch, decode)

        def g(_c, data):
            f(_c, data)
            if ch in self.hist:
                self.hist[ch].append(self.last[ch])
        return g


def _vec(q) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in q) + "]"


def lag_seconds(t_cmd, u, t_meas, y, max_lag: float = 0.2):
    """Shift of ``y(t)`` relative to ``u(t)`` (s, > 0: y lags) maximising the
    cross-correlation of the two signals resampled on a 1 ms grid."""
    t = np.arange(max(t_cmd[0], t_meas[0]), min(t_cmd[-1], t_meas[-1]), 1e-3)
    if len(t) < 100:
        return np.nan
    ui, yi = np.interp(t, t_cmd, u), np.interp(t, t_meas, y)
    ui, yi = ui - ui.mean(), yi - yi.mean()
    if np.std(ui) < 1e-3 or np.std(yi) < 1e-3:
        return np.nan
    lags = np.arange(0, int(max_lag * 1e3))
    c = [np.dot(ui[: len(ui) - k], yi[k:]) / (len(ui) - k) for k in lags]
    return float(lags[int(np.argmax(c))] * 1e-3)


def analyse(hist: dict) -> dict:
    cmd, meas = hist["franka_arm_joint_ctrl"], hist["franka_arm_joint_meas"]
    gs, gc = hist["franka_gripper_state"], hist["franka_gripper_cmd"]
    tc = np.array([m.timestamp for m in cmd])
    qd_des = np.array([m.qd_des for m in cmd])
    tau_ff = np.array([m.tau_ff for m in cmd])
    tm = np.array([m.timestamp for m in meas])
    qd = np.array([m.qd for m in meas])
    out = dict(t_cmd=tc, qd_des=qd_des, tau_ff=tau_ff, t_meas=tm, qd=qd)
    n = qd.shape[1]
    out["lag"] = np.array([lag_seconds(tc, qd_des[:, j], tm, qd[:, j]) for j in range(n)])
    qd_des_i = np.column_stack([np.interp(tm, tc, qd_des[:, j]) for j in range(n)])
    moving = np.abs(qd_des_i).max(axis=1) > 0.05
    err = qd - qd_des_i
    out["rms_err"] = np.sqrt(np.mean(err[moving] ** 2, axis=0)) if moving.any() else np.full(n, np.nan)
    out["peak_qd"] = np.abs(qd).max(axis=0)
    dt = np.diff(tm)
    ok = dt > 1e-6
    qdd = np.diff(qd, axis=0)[ok] / dt[ok][:, None]
    # 20 ms smoothing before the peak: the raw 1 kHz difference is noise
    k = 20
    qdd_s = np.array([np.convolve(qdd[:, j], np.ones(k) / k, mode="valid") for j in range(n)]).T
    out["peak_qdd"] = np.abs(qdd_s).max(axis=0) if len(qdd_s) else np.full(n, np.nan)
    # Gripper
    tg = np.array([m.timestamp for m in gs])
    w = np.array([m.width for m in gs])
    out["t_w"], out["w"] = tg, w
    op = hist.get("franka_object_pose", [])
    if op:
        out["t_obj"] = np.array([m.timestamp for m in op])
        out["obj_pos"] = np.array([m.position for m in op])
    out["n_gripper_cmds"] = len(gc)
    out["gripper_cmds"] = np.array([(m.timestamp, m.mode, m.width, m.speed) for m in gc]) if gc else np.zeros((0, 4))
    if len(w) > 10:
        dtg = np.diff(tg)
        okg = dtg > 1e-6
        wd = np.diff(w)[okg] / dtg[okg]
        wdd = np.diff(wd) / dtg[okg][1:]
        out["peak_wd"] = float(np.abs(wd).max())
        out["rms_wdd"] = float(np.sqrt(np.mean(wdd ** 2)))
        # the time the fingers spend stopped between moves while the set-point stream runs
        moves = out["gripper_cmds"][out["gripper_cmds"][:, 1] == 0] if len(gc) else np.zeros((0, 4))  # MOVE = 0
        if len(moves) > 2:
            t0, t1 = moves[0, 0], moves[-1, 0]
            sel = (tg[1:] > t0) & (tg[1:] < t1)
            seg = wd[sel[okg]] if sel.any() else wd
            out["stream_stopped_frac"] = float(np.mean(np.abs(seg) < 0.005)) if len(seg) else np.nan
    return out


def report(a: dict) -> None:
    np.set_printoptions(precision=3, suppress=True)
    print("\nper joint                 1      2      3      4      5      6      7")
    print(f"lag cmd -> meas (ms)  {np.round(a['lag'] * 1e3).astype(int)}")
    print(f"rms velocity err (rad/s) {a['rms_err']}")
    print(f"peak |qd| (rad/s)        {a['peak_qd']}")
    print(f"peak |qdd| (rad/s^2)     {a['peak_qdd']}")
    print(f"\ngripper: {a['n_gripper_cmds']} commands, peak width rate {a.get('peak_wd', np.nan) * 1e3:.0f} mm/s, "
          f"width acceleration RMS {a.get('rms_wdd', np.nan):.2f} m/s^2, "
          f"fingers stopped {100 * a.get('stream_stopped_frac', np.nan):.0f} % of the set-point stream")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="gold_bowl")
    ap.add_argument("--pose", default="left_bottom", choices=list(START_POSES))
    ap.add_argument("--orientation", default="upright", choices=list(ORIENTATIONS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--ctrl", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--env", nargs="*", default=[], help="extra env configs")
    ap.add_argument("--carry_at", type=float, default=None, help="s after v (negative: before) to carry the object (env key m)")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "diag"))
    args = ap.parse_args()
    extra = [a for kv in args.ctrl for a in ("--" + kv.split("=", 1)[0], kv.split("=", 1)[1])]
    os.makedirs(args.out, exist_ok=True)
    channels = {"franka_arm_joint_ctrl": joint_ctrl_t.decode, "franka_arm_joint_meas": joint_meas_t.decode,
                "franka_gripper_state": gripper_state_t.decode, "franka_gripper_cmd": gripper_cmd_t.decode,
                "franka_object_pose": se3_pose_t.decode}
    keep = tuple(channels)
    with History(channels, keep) as lst, Procs(args.out) as procs:
        obj = f"configs/objects/{args.object}.yaml"
        env = procs.start("env", "run_env.py", "--config", "configs/sim/env.yaml", *args.env, "--vis_mode", "vis_off",
                    "--object_config", obj, "--obj_spawn_seed", str(args.seed),
                    "--platform.default_q_values", _vec(START_POSES[args.pose]),
                    "--obj_spawn_rpy", _vec(ORIENTATIONS[args.orientation]))
        if not lst.wait_for(lambda: "franka_object_pose" in lst.last, 60):
            sys.exit("env did not start")
        lst.spin(1.0)
        z0 = float(lst.last["franka_object_pose"].position[2])
        ctrl = procs.start("ctrl", "run_controller.py", "--config", "configs/controller.yaml",
                           "--object_config", obj, *extra)
        if not lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, 90):
            sys.exit("controller did not start")
        for h in lst.hist.values():
            h.clear()
        carried = args.carry_at is None
        if not carried and args.carry_at < 0:  # the object is already on the move when the reach starts
            env.key("m")
            carried = True
            lst.spin(-args.carry_at)
        ctrl.key("v")
        t0 = time.time()
        while time.time() - t0 < args.timeout:
            lst.spin(0.1)
            if not carried and time.time() - t0 >= args.carry_at:
                env.key("m")
                carried = True
            gs = lst.last.get("franka_gripper_state")
            if (float(lst.last["franka_object_pose"].position[2]) - z0 > 0.05 and gs is not None and gs.is_grasped
                    and "Press x to release" in ctrl.log()):
                break
        print(f"trial over after {time.time() - t0:.1f} s; {len(lst.hist['franka_arm_joint_ctrl'])} commands, "
              f"{len(lst.hist['franka_arm_joint_meas'])} measurements recorded")
        a = analyse(lst.hist)
        log = ctrl.log()
    report(a)
    path = os.path.join(args.out, f"trace_{args.object}_{args.pose}_{int(time.time())}.npz")
    np.savez(path, **{k: v for k, v in a.items() if isinstance(v, np.ndarray)}, log=log)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
