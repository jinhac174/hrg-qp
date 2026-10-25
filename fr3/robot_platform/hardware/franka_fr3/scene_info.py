"""Collision geometry of the real cell on LCM, for the collision-aware
controllers and Viser. In simulation the env publishes it; on the robot
run_hardware.py does (``scene_info:`` in its config).

Static obstacles come from a scene XML (``floor``, ``wall_*``, ``table*``,
``obs_*`` geoms); the robot's planning primitives from fr3_col_primitives().
"""

import threading
import time
from queue import Queue

import lcm
import mujoco as mj

from fr3.communication.lcm.publisher.data_publisher.ColInfoPublisher import ColInfoPublisher
from fr3.data_type.basic_types.ColInfoData import ColInfoData
from fr3.robot_platform.sim.franka_fr3.SimFrankaFR3 import fr3_col_primitives
from fr3.utils.collision.wall_geoms import extract_static_col_geoms


class SceneInfoPublisher:
    def __init__(self, scene_xml_path, static_col_info_channel, robot_col_info_channel, rate=10.0):
        model = mj.MjModel.from_xml_path(str(scene_xml_path))
        self._static = ColInfoData(name=static_col_info_channel)
        for g in extract_static_col_geoms(model):
            self._static.add_col_geom(**g)
        self._robot = ColInfoData(name=robot_col_info_channel)
        for g in fr3_col_primitives():
            self._robot.add_col_geom(**g)
        self._period = 1.0 / float(rate)
        self._stop = threading.Event()
        self._thread = None

    def describe(self) -> str:
        return (f"static {sorted(self._static.dict_col_geoms)}; "
                f"robot {len(self._robot.dict_col_geoms)} primitives")

    def run(self):
        """Publish both at ``rate`` until stop()."""
        lc = lcm.LCM()
        pubs = []
        for data in (self._static, self._robot):
            que = Queue()
            pubs.append((ColInfoPublisher(lc, data.name, que), que, data))
        while not self._stop.is_set():
            t = time.time()
            for pub, que, data in pubs:
                data.set_time(t)
                que.put(data)
                pub.check_que_and_publish()
            self._stop.wait(self._period)

    def start(self):
        self._thread = threading.Thread(target=self.run, name="scene_info", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
