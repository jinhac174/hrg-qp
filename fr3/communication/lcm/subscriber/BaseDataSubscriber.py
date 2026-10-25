from abc import ABC, abstractmethod
from queue import Empty, Full, Queue

import lcm


class BaseDataSubscriber(ABC):
    """
    Base class for LCM data subscribers.
    """

    def __init__(self, lcm_instance: lcm.LCM):
        """
        Initialize the BaseDataSubscriber with an LCM instance.

        @param[in]: lcm_instance (lcm.LCM): The LCM instance to use for subscribing to messages.
        """
        self._lcm_instance = lcm_instance

    def subscribe(self, channel: str):
        """
        Subscribe to the given channel.
        """
        self._lcm_instance.subscribe(channel, self._callback)

    @abstractmethod
    def _callback(self, channel: str, data):
        """
        Callback method to handle incoming messages.

        This method should be implemented by subclasses to process the received messages.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def put_latest(que: Queue, data) -> None:
        """Put ``data``; on a bounded queue (``Queue(maxsize=1)``) that is
        full, drop the oldest item first: latest wins.

        State streams (joint measurements) must be consumed as the newest
        sample. With an unbounded FIFO, a consumer that reads one item per loop
        and falls behind the 1 kHz producer for a moment (GIL held by a planning
        thread, CPU load) works on ever older data: seconds of latency, and a
        feedback controller acting on it goes unstable.
        """
        while True:
            try:
                que.put_nowait(data)
                return
            except Full:
                try:
                    que.get_nowait()
                except Empty:
                    pass

