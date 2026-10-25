from queue import Queue

from fr3.communication.lcm.subscriber.BaseSubManager import BaseEnvSubManeager
from fr3.communication.lcm.subscriber.data_subscriber.GripperCmdSubscriber import (
    GripperCmdSubscriber,
)
from fr3.communication.lcm.subscriber.data_subscriber.JointCtrlSubsriber import (
    JointCtrlSubscriber,
)


class FrankaSubManager(BaseEnvSubManeager):
    """Env-side subscriber for the FR3 + Robotiq 2F-85.

    * ``ctrl_channel``        joint_ctrl_t (7 arm joints) -> ``ctrl_sub_que``
    * ``gripper_cmd_channel`` gripper_cmd_t -> ``intr_sub_que_dict[channel]``
      (picked up by the sim platform via ``set_sub_que_dict``)
    """

    def __init__(
        self,
        ctrl_channel: str = "franka_arm_joint_ctrl",
        gripper_cmd_channel: str = "franka_gripper_cmd",
        num_joints: int = 7,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        JointCtrlSubscriber(
            self._lcm_instance,
            self.data_queue_dict["ctrl_sub_que"],
            num_joints=num_joints,
        ).subscribe(ctrl_channel)

        if gripper_cmd_channel:
            que = Queue()
            self.data_queue_dict["intr_sub_que_dict"][gripper_cmd_channel] = que
            GripperCmdSubscriber(self._lcm_instance, que).subscribe(
                gripper_cmd_channel
            )
