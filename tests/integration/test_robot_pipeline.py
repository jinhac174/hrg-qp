"""The robot-side processes without the robot: ``run_hardware.py`` with its
fake FR3 / Robotiq backends (the same safety layer, scene info and channels as
on the robot) and the controller in real mode on the real cell's
configuration (configs/robot/controller.yaml), fed an object pose as the pose
bridge would publish it."""

import threading
import time

import lcm
import numpy as np

from fr3.lcm_type.collision.col_info_t import col_info_t
from fr3.lcm_type.gripper.gripper_state_t import gripper_state_t
from fr3.lcm_type.joint.joint_ctrl_t import joint_ctrl_t
from fr3.lcm_type.joint.joint_meas_t import joint_meas_t
from fr3.lcm_type.pose.se3_pose_t import se3_pose_t
from fr3.lcm_type.vector.vec_list_t import vec_list_t
from hrg.objects import GraspObject
from hrg.processes import Listener

HW = ("run_hardware.py", "--config", "configs/robot/hardware.yaml", "--robot.backend", "fake")
CTRL = ("run_controller.py", "--config", "configs/controller.yaml", "configs/robot/controller.yaml",
        "--object_config", "configs/objects/blue_bowl.yaml")
HOME_Q = np.array([0.0, 0.0, 0.0, -1.5707963, 0.0, 1.5707963, 0.0])
STARTUP = 120
CHANNELS = {
    "franka_arm_joint_meas": joint_meas_t.decode,
    "franka_arm_joint_ctrl": joint_ctrl_t.decode,
    "franka_gripper_state": gripper_state_t.decode,
    "franka_static_col_info": col_info_t.decode,
    "franka_robot_col_info": col_info_t.decode,
    "franka_hrg_fingertips": vec_list_t.decode,
}


def _publish_object_pose(stop: threading.Event) -> None:
    """The bowl upright at (0.5, 0) on the table, at 30 Hz (like the pose bridge)."""
    lc = lcm.LCM()
    msg = se3_pose_t()
    msg.position = [0.5, 0.0, GraspObject("configs/objects/blue_bowl.yaml").resting_height(np.eye(3))]
    msg.quat_wxyz = [1.0, 0.0, 0.0, 0.0]
    while not stop.is_set():
        msg.timestamp = time.time()
        lc.publish("franka_object_pose", msg.encode())
        time.sleep(1 / 30)


def _fingertip_error(lst) -> float:
    m = lst.last["franka_hrg_fingertips"]
    p = {n: np.asarray(v) for n, v in zip(m.name_list, m.vec_list)}
    return max(np.linalg.norm(p["thumb_cur"] - p["thumb_tgt"]), np.linalg.norm(p["index_cur"] - p["index_tgt"]))


def test_hardware_and_controller_in_real_mode(procs):
    with Listener(CHANNELS) as lst:
        procs.start("hw", *HW)
        assert lst.wait_for(lambda: "franka_arm_joint_meas" in lst.last, STARTUP), "hardware did not publish"
        assert lst.wait_for(lambda: "franka_static_col_info" in lst.last and "franka_robot_col_info" in lst.last,
                            STARTUP), "no scene info from run_hardware"
        assert np.max(np.abs(np.asarray(lst.last["franka_arm_joint_meas"].q) - HOME_Q)) < 0.02

        stop = threading.Event()
        pub = threading.Thread(target=_publish_object_pose, args=(stop,), daemon=True)
        pub.start()
        try:
            ctrl = procs.start("ctrl", *CTRL)
            assert lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, STARTUP), ctrl.log()[-2000:]
            # idle: the arm holds still
            lst.reset_counts()
            lst.spin(2.0)
            assert lst.count["franka_arm_joint_ctrl"] / 2.0 > 500  # ~1 kHz in real mode
            assert np.max(np.abs(np.asarray(lst.last["franka_arm_joint_meas"].q) - HOME_Q)) < 0.02
            # reach: the fingertips close in on the selected rim pinch
            ctrl.key("v")
            assert lst.wait_for(lambda: "franka_hrg_fingertips" in lst.last, 10), ctrl.log()[-2000:]
            e0 = _fingertip_error(lst)
            lst.spin(8.0)
            log = ctrl.log()
            reached = "reach -> close" in log
            assert reached or _fingertip_error(lst) < 0.5 * e0, (e0, log[-2000:])
            assert "QP failed" not in log and "No reachable" not in log
        finally:
            stop.set()
