from queue import Queue

from fr3.communication.lcm.publisher.BasePubManager import BaseEnvPubManager
from fr3.communication.lcm.publisher.data_publisher.ColInfoPublisher import ColInfoPublisher
from fr3.communication.lcm.publisher.data_publisher.GripperStatePublisher import GripperStatePublisher
from fr3.communication.lcm.publisher.data_publisher.JointMeasPublisher import JointMeasPublisher
from fr3.communication.lcm.publisher.data_publisher.NamedVecListPublisher import NamedVecListPublisher
from fr3.communication.lcm.publisher.data_publisher.SE3PosePublisher import SE3PosePublisher


class FrankaPubManager(BaseEnvPubManager):
    """Sim env / hardware side publisher for the FR3 + Robotiq 2F-85.

    intr: ``arm_joint_meas_channel`` (7), ``gripper_joint_meas_channel`` (the
    driver angle), ``gripper_state_channel``, ``robot_col_info_channel``
    (link-local collision primitives).
    extr: ``static_col_info_channel`` (world-fixed scene geoms),
    ``object_pose_channel`` (se3_pose_t), ``obstacles_channel`` (vec_list_t,
    ``hrg.obstacles``). An empty channel name is not published.
    """

    def __init__(
        self,
        *args,
        arm_joint_meas_channel: str = "franka_arm_joint_meas",
        gripper_joint_meas_channel: str = "franka_gripper_joint_meas",
        gripper_state_channel: str = "franka_gripper_state",
        robot_col_info_channel: str = "",
        static_col_info_channel: str = "",
        object_pose_channel: str = "",
        obstacles_channel: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        intr = self.pub_que_dict["intr_pub_que_dict"]
        extr = self.pub_que_dict["extr_pub_que_dict"]
        for que_dict, ch, cls in (
            (intr, arm_joint_meas_channel, JointMeasPublisher),
            (intr, gripper_joint_meas_channel, JointMeasPublisher),
            (intr, gripper_state_channel, GripperStatePublisher),
            (intr, robot_col_info_channel, ColInfoPublisher),
            (extr, static_col_info_channel, ColInfoPublisher),
            (extr, object_pose_channel, SE3PosePublisher),
            (extr, obstacles_channel, NamedVecListPublisher),
        ):
            if ch:
                que_dict[ch] = Queue()
                self.publisher_dict[ch] = cls(self._lcm_instance, ch, que_dict[ch])
