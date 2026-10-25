from queue import Queue

import lcm

from fr3.communication.lcm.subscriber.BaseDataSubscriber import BaseDataSubscriber
from fr3.data_type.basic_types.GripperStateData import GripperStateData
from fr3.lcm_type.gripper.gripper_state_t import gripper_state_t


class GripperStateSubscriber(BaseDataSubscriber):
    """Decode gripper_state_t into GripperStateData."""

    def __init__(self, lcm_instance: lcm.LCM, data_queue: Queue):
        super().__init__(lcm_instance)
        self.data_queue = data_queue

    def _callback(self, channel: str, data):
        msg = gripper_state_t.decode(data)
        state = GripperStateData()
        state.set_data(
            msg.timestamp,
            msg.width,
            msg.max_width,
            msg.is_grasped,
            msg.busy,
            msg.last_seq,
            msg.last_success,
        )
        self.data_queue.put(state)
