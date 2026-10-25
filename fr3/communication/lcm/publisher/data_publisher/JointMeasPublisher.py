from queue import Queue
import time
import lcm

from fr3.communication.lcm.publisher.BaseDataPublisher import BaseDataPublisher
from fr3.data_type.basic_types.JointMeasData import JointMeasData
from fr3.lcm_type.joint.joint_meas_t import joint_meas_t


class JointMeasPublisher(BaseDataPublisher):
    """
    JointMeasPublisher is a class that inherits from BaseDataPublisher.
    It is responsible for publishing joint measurement data on a specific LCM channel.
    """

    def __init__(
        self, lcm_instance: lcm.LCM, channel: str, data_que: Queue, *args, **kwargs
    ):
        super().__init__(lcm_instance, channel, data_que)
        self._exe_t = time.time()

    def check_que_and_publish(self):
        """
        Check if the queue is not empty and publish the data to the LCM channel.
        """

        # Latest wins: controllers need the current state, not a backlog. If
        # this loop falls behind the producer, drop the stale samples instead
        # of delivering them late.
        data = None
        while not self._data_que.empty():
            data = self._data_que.get()
        if data is not None:
            self._exe_t = time.time()
            if isinstance(data, JointMeasData):
                t, q, qd, tau = data.get_data()
                msg = joint_meas_t()
                msg.timestamp = t
                msg.q = q.tolist()
                msg.qd = qd.tolist()
                msg.tau = tau.tolist()
                msg.num_joints = len(q)
                self._lcm_instance.publish(self._channel, msg.encode())
