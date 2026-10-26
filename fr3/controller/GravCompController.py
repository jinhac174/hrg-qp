"""Gravity compensation: the arm floats and can be moved by hand (from
franka_visual_servoing). For posing the arm during the camera calibration.

Every control tick it sends

    tau_ff = g(q) - arm_damping * qdot,   kp = kd = 0,   q_des = q, qdot_des = 0

with g(q) from ``urdf_path`` (the Robotiq as a rigid payload, the same model
``run_hardware.py`` holds the arm with). On the FR3 ``gravity_model:
controller`` cancels the robot's own compensation, so this g(q) is the only
one. Stopping it (``q``) leaves ``run_hardware.py`` holding the last pose.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from fr3.controller.BaseController import BaseController
from fr3.data_type.basic_types.JointCtrlData import JointCtrlData
from fr3.data_type.basic_types.JointMeasData import JointMeasData

N_ARM = 7


class GravCompController(BaseController):
    def __init__(self, config):
        super().__init__(config)
        self._model = pin.buildModelFromUrdf(str(config["urdf_path"]))
        if self._model.nq != N_ARM:
            raise ValueError(f"{config['urdf_path']}: expected {N_ARM} joints, got {self._model.nq}")
        self._data = self._model.createData()
        self._damping = float(config.get("arm_damping", 0.5))  # Nm per rad/s
        self._arm_channel = config["sub_manager"]["arm_joint_meas_channel"]
        self._q = np.zeros(N_ARM)
        self._qd = np.zeros(N_ARM)
        self._have_q = False

    def initialize(self) -> None:
        print(f"[GravComp] Floating the arm (damping {self._damping} Nm s/rad). Keys: p pause, q quit.")

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
        cmd = JointCtrlData(num_joints=N_ARM)
        cmd.set_data(self._cur_time, self._q.copy(), np.zeros(N_ARM), tau, np.zeros(N_ARM), np.zeros(N_ARM))
        self.ctrl_pub_que.put(cmd)
