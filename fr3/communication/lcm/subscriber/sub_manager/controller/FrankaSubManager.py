from queue import Queue

from fr3.communication.lcm.subscriber.BaseSubManager import BaseControllerSubManager
from fr3.communication.lcm.subscriber.data_subscriber.ClockSubscriber import ClockSubscriber
from fr3.communication.lcm.subscriber.data_subscriber.ColInfoSubscriber import ColInfoSubscriber
from fr3.communication.lcm.subscriber.data_subscriber.GripperStateSubscriber import GripperStateSubscriber
from fr3.communication.lcm.subscriber.data_subscriber.JointMeasSubscriber import JointMeasSubscriber
from fr3.communication.lcm.subscriber.data_subscriber.NamedVecListSubscriber import NamedVecListSubscriber
from fr3.communication.lcm.subscriber.data_subscriber.SE3PoseSubscriber import SE3PoseSubscriber


class FrankaSubManager(BaseControllerSubManager):
    """Controller-side subscriber for the FR3 + Robotiq 2F-85.

    intr: ``arm_joint_meas_channel`` (7, latest only), ``gripper_state_channel``,
    ``robot_col_info_channel``.
    extr: ``static_col_info_channel``, ``object_pose_channel`` (se3_pose_t),
    ``obstacles_channel`` (vec_list_t, ``hrg.obstacles``).
    Also the sim clock (``sim_clock``) for the sim_real_time mode.
    """

    def __init__(
        self,
        arm_joint_meas_channel: str = "franka_arm_joint_meas",
        gripper_state_channel: str = "",
        robot_col_info_channel: str = "",
        static_col_info_channel: str = "",
        object_pose_channel: str = "",
        obstacles_channel: str = "",
        num_arm_joints: int = 7,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        intr = self.data_queue_dict["intr_sub_que_dict"]
        extr = self.data_queue_dict["extr_sub_que_dict"]
        # Latest wins: the controller reads one sample per loop iteration, so
        # an unbounded FIFO would build up stale state.
        intr[arm_joint_meas_channel] = Queue(maxsize=1)
        JointMeasSubscriber(self._lcm_instance, intr[arm_joint_meas_channel],
                            num_joints=num_arm_joints).subscribe(arm_joint_meas_channel)
        for que_dict, ch, cls in (
            (intr, gripper_state_channel, GripperStateSubscriber),
            (intr, robot_col_info_channel, ColInfoSubscriber),
            (extr, static_col_info_channel, ColInfoSubscriber),
            (extr, object_pose_channel, SE3PoseSubscriber),
            (extr, obstacles_channel, NamedVecListSubscriber),
        ):
            if ch:
                que_dict[ch] = Queue()
                cls(self._lcm_instance, que_dict[ch]).subscribe(ch)
        ClockSubscriber(self._lcm_instance, self.data_queue_dict["time_sub_que"]).subscribe("sim_clock")
