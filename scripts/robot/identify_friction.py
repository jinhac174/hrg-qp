"""Measure the joint friction of the FR3 (or the sim) for the friction
feed-forward (``hrg.friction``, ``friction:`` in configs/robot/controller.yaml).

One joint at a time moves back and forth around where the arm is (the others
hold) at a few constant speeds, tracked by a PD on top of the gravity model
(the controllers' URDF). On every constant-speed pass the PD supplies what the
model lacks; the friction is the part that changes sign with the direction
(``hrg.friction.fit_friction``). Prints the ``friction:`` block (start it at
``scale: 0.5``) and saves the raw data to ``outputs/friction/``.

With run_hardware.py running (and nothing else commanding the arm), from a
pose with ``amplitude`` rad of room on every joint (the default pose is fine):

    python scripts/robot/identify_friction.py --config configs/robot/identify_friction.yaml
    python scripts/robot/identify_friction.py --config configs/robot/identify_friction.yaml --joints [5,6,7]

In the sim (validates it: it must recover the MuJoCo model's frictionloss):
``run_env.py`` with configs/sim/env.yaml, then this with ``--env_mode sim_real_time``.

Keys: ``q`` stop (the hardware holds the arm), ``p`` pause. Each joint takes
about 40 s; it moves at most ``amplitude`` rad at up to the largest speed.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fr3.communication.lcm.publisher.pub_manager.controller.JointCtrlGraspPubManager import (  # noqa: E402
    JointCtrlGraspPubManager,
)
from fr3.communication.lcm.subscriber.sub_manager.controller.FrankaSubManager import FrankaSubManager  # noqa: E402
from fr3.controller.BaseController import BaseController  # noqa: E402
from fr3.data_type.basic_types.JointCtrlData import JointCtrlData  # noqa: E402
from fr3.data_type.basic_types.JointMeasData import JointMeasData  # noqa: E402
from hrg.cli import load_config  # noqa: E402
from hrg.friction import fit_friction, sweep_profile  # noqa: E402
from hrg.paths import resolve  # noqa: E402

N_ARM = 7


class FrictionIdController(BaseController):
    def __init__(self, config):
        super().__init__(config)
        self._model = pin.buildModelFromUrdf(resolve(config["urdf_path"]))
        self._data = self._model.createData()
        self._arm_channel = config["sub_manager"]["arm_joint_meas_channel"]
        self._kp = np.asarray(config["kp"], dtype=float)
        self._kd = np.asarray(config["kd"], dtype=float)
        self._joints = [int(j) - 1 for j in config["joints"]]
        self._speeds = [float(v) for v in config["speeds"]]
        self._amp = float(config["amplitude"])
        self._limit_margin = float(config.get("limit_margin", 0.15))
        self._out_dir = resolve(config.get("output_dir", "outputs/friction"))
        self._q = np.zeros(N_ARM)
        self._qd = np.zeros(N_ARM)
        self._have_q = False
        self._q0 = None
        self._plan = []  # [(joint, q_ref (T,), qd_ref (T,), label (T,))]
        self._seg = 0
        self._k = 0
        self._log: dict[int, list] = {}  # joint -> [(label, tau_pd, qd)]
        self.result = None

    def initialize(self) -> None:
        print(f"[FrictionId] Joints {[j + 1 for j in self._joints]}, speeds {self._speeds} rad/s, "
              f"+-{self._amp} rad around the current pose. Keys: q stop, p pause.")

    def _check_and_get_data_from_que(self) -> None:
        que = self.intr_sub_que_dict[self._arm_channel]
        meas = None
        while not que.empty():
            meas = que.get()
        if isinstance(meas, JointMeasData):
            _, q, qd, _ = meas.get_data()
            self._q[:], self._qd[:] = q[:N_ARM], qd[:N_ARM]
            self._have_q = True

    def _build(self) -> None:
        lo, hi = self._model.lowerPositionLimit, self._model.upperPositionLimit
        for j in self._joints:
            room = min(self._q0[j] - lo[j], hi[j] - self._q0[j]) - self._limit_margin
            amp = min(self._amp, room)
            if amp < 0.05:
                print(f"[FrictionId] Joint {j + 1}: only {room:.2f} rad of room to its limits, skipped.")
                continue
            q, qd, lab = sweep_profile(float(self._q0[j]), amp, self._speeds, dt=self._ctrl_dt)
            self._plan.append((j, q, qd, lab))
            self._log[j] = []
        total = sum(len(p[1]) for p in self._plan) * self._ctrl_dt
        print(f"[FrictionId] {len(self._plan)} joints, {total:.0f} s.")

    def _update(self) -> None:
        if not self._have_q:
            return
        if self._q0 is None:
            self._q0 = self._q.copy()
            self._build()
        q_ref, qd_ref = self._q0.copy(), np.zeros(N_ARM)
        if self._seg < len(self._plan):
            j, qj, qdj, lab = self._plan[self._seg]
            if self._k == 0:
                print(f"[FrictionId] Joint {j + 1} ...")
            q_ref[j], qd_ref[j] = qj[self._k], qdj[self._k]
            tau_pd = self._kp * (q_ref - self._q) + self._kd * (qd_ref - self._qd)
            if lab[self._k]:
                self._log[j].append((int(lab[self._k]), float(tau_pd[j]), float(self._qd[j])))
            self._k += 1
            if self._k >= len(qj):
                self._seg, self._k = self._seg + 1, 0
        elif self.result is None:
            self.result = self._fit()
            self._finished = True
        tau_ff = pin.computeGeneralizedGravity(self._model, self._data, self._q)
        pin.computeCoriolisMatrix(self._model, self._data, self._q, self._qd)
        tau_ff = tau_ff + self._data.C @ self._qd
        cmd = JointCtrlData(num_joints=N_ARM)
        cmd.set_data(self._cur_time, q_ref, qd_ref, tau_ff, self._kp, self._kd)
        self.ctrl_pub_que.put(cmd)

    def _fit(self) -> dict:
        coulomb, viscous = np.zeros(N_ARM), np.zeros(N_ARM)
        raw = {}
        print("\n[FrictionId] joint  speed rad/s   friction Nm (+/- mean)   model error Nm")
        for j, rows in self._log.items():
            rows = np.asarray(rows)
            if not len(rows):
                continue
            raw[f"joint{j + 1}"] = rows
            tp, tn, sp = [], [], []
            for k, v in enumerate(self._speeds):
                up, dn = rows[rows[:, 0] == k + 1], rows[rows[:, 0] == -(k + 1)]
                if len(up) and len(dn):
                    sp.append(v)
                    tp.append(up[:, 1].mean())
                    tn.append(dn[:, 1].mean())
            if not sp:
                continue
            fc, fv, f, off = fit_friction(sp, tp, tn)
            coulomb[j], viscous[j] = fc, fv
            for v, fi, oi in zip(sp, f, off):
                print(f"[FrictionId]   {j + 1}     {v:5.2f}        {fi:6.3f}                  {oi:+6.3f}")
            print(f"[FrictionId]   {j + 1}  -> coulomb {fc:.3f} Nm, viscous {fv:.3f} Nm s/rad")
        os.makedirs(self._out_dir, exist_ok=True)
        path = os.path.join(self._out_dir, time.strftime("friction_%Y%m%d_%H%M%S.npz"))
        np.savez(path, coulomb=coulomb, viscous=viscous, speeds=self._speeds, **raw)
        cap = np.maximum(np.round(coulomb, 2), 0.05)
        print(f"\n[FrictionId] Saved {path}. For configs/robot/controller.yaml (joints not measured: 0):\n")
        print("friction:\n  scale: 0.5")
        print(f"  coulomb: [{', '.join(f'{c:.3f}' for c in coulomb)}]")
        print(f"  viscous: [{', '.join(f'{c:.3f}' for c in viscous)}]")
        print("  v_smooth: 0.02")
        print(f"  max: [{', '.join(f'{c:.2f}' for c in cap)}]\n")
        return dict(coulomb=coulomb, viscous=viscous, path=path)


def main() -> None:
    cfg = load_config(__doc__)
    controller = FrictionIdController(cfg)
    sub = FrankaSubManager(**cfg["sub_manager"])
    pub = JointCtrlGraspPubManager(**cfg["pub_manager"])
    controller.set_sub_que_dict(sub.get_sub_que_dict())
    controller.set_ctrl_pub_que(pub.get_ctrl_pub_que())
    controller.set_pub_que_dict(pub.get_pub_que_dict())
    sub.start()
    pub.start()
    try:
        controller.initialize()
        controller.start()
    finally:
        sub.stop()
        pub.stop()
        controller.stop()


if __name__ == "__main__":
    main()
