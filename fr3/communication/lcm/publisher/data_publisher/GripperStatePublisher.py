from queue import Queue

import lcm

from fr3.communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from fr3.data_type.basic_types.GripperStateData import GripperStateData
from fr3.lcm_type.gripper.gripper_state_t import gripper_state_t


class GripperStatePublisher(BaseDataPublisher):
    """Publish GripperStateData as gripper_state_t."""

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)

    def check_que_and_publish(self):
        if not self._data_que.empty():
            data = self._data_que.get()
            if isinstance(data, GripperStateData):
                t, width, max_width, is_grasped, busy, last_seq, last_success = (
                    data.get_data()
                )
                msg = gripper_state_t()
                msg.timestamp = t
                msg.width = width
                msg.max_width = max_width
                msg.is_grasped = is_grasped
                msg.busy = busy
                msg.last_seq = last_seq
                msg.last_success = last_success
                self._lcm_instance.publish(self._channel, msg.encode())
