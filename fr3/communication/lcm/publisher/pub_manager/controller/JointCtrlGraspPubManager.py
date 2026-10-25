from queue import Queue

from fr3.communication.lcm.publisher.BasePubManager import BaseCtrlPubManager
from fr3.communication.lcm.publisher.data_publisher.JointCtrlPublisher import JointCtrlPublisher
from fr3.communication.lcm.publisher.data_publisher.NamedVecListPublisher import NamedVecListPublisher


class JointCtrlGraspPubManager(BaseCtrlPubManager):
    """Controller-side publisher: ``ctrl_channel`` (joint_ctrl_t), the gripper
    (``gripper_cmd_channel``, base class) and the reactive grasp's
    visualisation: ``fingertip_channel`` (fingertips + targets) and
    ``fingertip_paths_channel`` (the eq. (3) paths), vec_list_t."""

    def __init__(self, ctrl_channel: str = "franka_arm_joint_ctrl", fingertip_channel: str = "",
                 fingertip_paths_channel: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.publisher_dict[ctrl_channel] = JointCtrlPublisher(self._lcm_instance, ctrl_channel, self.ctrl_pub_que)
        for ch in (fingertip_channel, fingertip_paths_channel):
            if ch:
                self.pub_que_dict[ch] = Queue(maxsize=1)
                self.publisher_dict[ch] = NamedVecListPublisher(self._lcm_instance, ch, self.pub_que_dict[ch])
