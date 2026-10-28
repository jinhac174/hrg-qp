"""Gravity compensation: the arm floats and can be moved by hand (from
franka_visual_servoing). For posing the arm during the camera calibration.

Every control tick it sends

    tau_ff = g(q) - arm_damping * qdot,   kp = kd = 0,   q_des = q, qdot_des = 0

with g(q) from ``urdf_path`` (the Robotiq as a rigid payload, the same model
``run_hardware.py`` holds the arm with). On the FR3 ``gravity_model:
controller`` cancels the robot's own compensation, so this g(q) is the only
one. Stopping it (``q``) leaves ``run_hardware.py`` holding the last pose.

``friction:`` (``scale`` > 0, one or per joint) adds Coulomb friction compensation on the
*measured* velocity, so the arm is lighter to move by hand:
``scale coulomb tanh(qdot / v_smooth)``, capped at ``max``. Only Coulomb, at
``scale`` < 1 of the identified value (``scripts/robot/identify_friction.py``):
the compensation then stays below the friction it opposes, the arm still
loses energy and cannot drive itself; the viscous part is not compensated
and ``arm_damping`` stays. ``v_smooth`` well above the velocity noise, so the
arm does not buzz at rest. ``v_fade``: the compensation fades at speed (full
when starting and moving slowly, little once a thrown arm coasts).
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from fr3.controller.BaseController import BaseController
from fr3.data_type.basic_types.JointCtrlData import JointCtrlData
from fr3.data_type.basic_types.JointMeasData import JointMeasData
from hrg.friction import friction_torque

N_ARM = 7


class GravCompController(BaseController):
    def __init__(self, config):
        super().__init__(config)
        self._model = pin.buildModelFromUrdf(str(config["urdf_path"]))
        if self._model.nq != N_ARM:
            raise ValueError(f"{config['urdf_path']}: expected {N_ARM} joints, got {self._model.nq}")
        self._data = self._model.createData()
        self._damping = float(config.get("arm_damping", 0.5))  # Nm per rad/s
        fr = config.get("friction", None) or {}
        self._fr_scale = np.broadcast_to(np.asarray(fr.get("scale", 0.0), dtype=float), (N_ARM,)).copy()
        if np.any(self._fr_scale >= 1.0):
            raise ValueError("friction.scale must stay below 1 (above it the arm drives itself)")
        self._fr_coulomb = np.asarray(fr.get("coulomb", [0.0] * N_ARM), dtype=float)
        self._fr_vs = float(fr.get("v_smooth", 0.05))
        self._fr_fade = float(fr.get("v_fade", 0.0))
        self._fr_max = np.asarray(fr.get("max", [1.0] * N_ARM), dtype=float)
        self._arm_channel = config["sub_manager"]["arm_joint_meas_channel"]
        self._q = np.zeros(N_ARM)
        self._qd = np.zeros(N_ARM)
        self._have_q = False

    def initialize(self) -> None:
        fr = (f", friction compensation {np.round(self._fr_scale, 2).tolist()} x "
              f"{np.round(self._fr_coulomb, 2).tolist()} Nm" if np.any(self._fr_scale > 0.0) else "")
        print(f"[GravComp] Floating the arm (damping {self._damping} Nm s/rad{fr}). Keys: p pause, q quit.")

    def _check_and_get_data_from_que(self) -> None:
        que = self.intr_sub_que_dict[self._arm_channel]
        meas = None
        while not que.empty():
            meas = que.get()
        if isinstance(meas, JointMeasData):
            _, q, qd, _ = meas.get_data()
            self._q[:], self._qd[:] = q[:N_ARM], qd[:N_ARM]
            self._have_q = True

    def _update(self) -> None:
        if not self._have_q:  # nothing is commanded before the first measurement
            return
        tau = pin.computeGeneralizedGravity(self._model, self._data, self._q) - self._damping * self._qd
        tau = tau + friction_torque(self._qd, self._fr_scale, self._fr_coulomb, np.zeros(N_ARM), self._fr_vs,
                                    self._fr_max, self._fr_fade)
        cmd = JointCtrlData(num_joints=N_ARM)
        cmd.set_data(self._cur_time, self._q.copy(), np.zeros(N_ARM), tau, np.zeros(N_ARM), np.zeros(N_ARM))
        self.ctrl_pub_que.put(cmd)
