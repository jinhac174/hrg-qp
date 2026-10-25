"""Robot: the ZED cameras -> RGB-D + camera info on LCM (for FoundationPose).

    python run_sensors.py --config configs/robot/zed.yaml
"""

from fr3.communication.lcm.publisher.pub_manager.sensor.RealSensePubManager import RealSensePubManager
from fr3.communication.lcm.subscriber.BaseSubManager import BaseSubManager
from fr3.sensor.sensor_manager.ZEDManager import ZEDManager
from hrg.cli import load_config


def main() -> None:
    cfg = load_config(__doc__)
    sensors = ZEDManager(cfg)
    sensors.initialize_sensors_and_sensor_info()
    sub = BaseSubManager(**cfg["sub_manager"])
    pm = dict(cfg["pub_manager"])
    pub = RealSensePubManager(realsense=pm.pop("zed"), **pm)  # same messages per camera
    sensors.set_sub_que_dict(sub.get_sub_que_dict())
    sensors.set_pub_que_dict(pub.get_pub_que_dict())
    sub.start()
    pub.start()
    try:
        sensors.start()
    finally:
        sub.stop()
        pub.stop()
        sensors.stop()


if __name__ == "__main__":
    main()
