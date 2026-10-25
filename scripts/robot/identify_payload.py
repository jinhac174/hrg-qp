"""Measure the mass the FR3's gravity model is missing (DROID: the wrist ZED
camera + its mount, cables) and where it sits on the flange.

The FR3 compensates gravity with its own model plus the end-effector payload
set in Desk. Unknown mass (off the gripper axis) leaves a torque that friction
hides in some poses only: the arm sinks there. At rest, the link-side joint
torque sensors read the true gravity torque, so

    tau_J - g_robot(q) = (missing load gravity)(q) + sensor offset

is linear in [m, m*cx, m*cy, m*cz] (point mass, flange frame) plus a
per-joint offset, and a least-squares fit over a few poses gives it.

Run with run_hardware.py STOPPED (this connects to the robot itself):

    python scripts/robot/identify_payload.py

The arm is in Franka's teaching mode (gravity compensated, free to move).
Put it in a pose, LET GO, wait for the beep line; repeat in ~10 different
poses, wrist turned different ways (the camera's offset only shows when the
wrist orientation changes). Then press Enter: the result is printed as the
`load:` block for configs/robot/hardware.yaml.
"""

import argparse
import select
import sys
import time

import numpy as np


def load_basis(model, q):
    """Columns: gravity torque of the load per [m, m cx, m cy, m cz]."""
    z = [0.0, 0.0, 0.0]
    g0 = np.array(model.gravity(q, 0.0, z))
    g1 = np.array(model.gravity(q, 1.0, z))
    cols = [g1 - g0]
    for e in np.eye(3):
        cols.append(np.array(model.gravity(q, 1.0, e.tolist())) - g1)
    return np.stack(cols, axis=1)  # 7 x 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hostname", default="192.168.0.113")
    ap.add_argument("--still_time", type=float, default=1.5, help="s at rest per sample")
    ap.add_argument("--min_pose_change", type=float, default=0.3, help="rad between samples")
    args = ap.parse_args()

    import panda_py

    panda = panda_py.Panda(args.hostname)
    model = panda.get_model()
    s = panda.get_state()
    print(f"Robot payload now: m_ee {s.m_ee:.3f} kg at {np.round(s.F_x_Cee, 3).tolist()}, "
          f"m_load {s.m_load:.3f} kg at {np.round(s.F_x_Cload, 3).tolist()}")
    print("Teaching mode on: move the arm to a pose, let go, keep hands off until the sample "
          "is taken. ~10 poses, turn the wrist. Press Enter to finish.\n")
    panda.teaching_mode(True)

    samples = []  # (q, residual)
    still_since, buf = None, []
    try:
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break
            s = panda.get_state()
            q, dq = np.array(s.q), np.array(s.dq)
            if np.max(np.abs(dq)) < 0.005:
                still_since = still_since or time.time()
                buf.append((q, np.array(s.tau_J), np.array(model.gravity(s))))
                if time.time() - still_since > args.still_time:
                    qm = np.mean([b[0] for b in buf], axis=0)
                    if all(np.max(np.abs(qm - p)) > args.min_pose_change for p, _ in samples):
                        r = np.mean([b[1] - b[2] for b in buf], axis=0)
                        samples.append((qm, r))
                        print(f"  sample {len(samples):2d}: residual {np.round(r, 2).tolist()} Nm")
                    still_since, buf = None, []
            else:
                still_since, buf = None, []
            time.sleep(0.01)
    finally:
        panda.teaching_mode(False)

    if len(samples) < 6:
        print(f"Only {len(samples)} poses: need at least 6. Nothing fitted.")
        return
    # Unknowns: [m, m cx, m cy, m cz] + 7 joint offsets.
    A = np.vstack([np.hstack([load_basis(model, q), np.eye(7)]) for q, _ in samples])
    b = np.concatenate([r for _, r in samples])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    m, mc, bias = x[0], x[1:4], x[4:]
    fit = A @ x
    print(f"\nResidual RMS before {np.sqrt(np.mean(b**2)):.3f} Nm, after fit "
          f"{np.sqrt(np.mean((b - fit)**2)):.3f} Nm")
    print(f"Joint torque sensor offsets (not applied): {np.round(bias, 2).tolist()} Nm")
    if m < 0.02:
        print(f"Missing mass {m:.3f} kg: nothing worth adding.")
        return
    c = mc / m
    print(f"Missing load: {m:.3f} kg at {np.round(c, 3).tolist()} m (flange frame)\n")
    print("Put this under robot: in configs/robot/hardware.yaml:\n")
    print("  load:")
    print(f"    mass: {m:.3f}")
    print(f"    com: [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")


if __name__ == "__main__":
    main()
