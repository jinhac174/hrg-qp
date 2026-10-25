"""The real FR3 + Robotiq 2F-85 (libfranka via panda-py, Robotiq over RS-485).

Replaces run_env.py on the robot: same channels, real measurements; it also
publishes the cell's collision geometry for Gamma(q).

    python run_hardware.py --config configs/robot/hardware.yaml
    python run_hardware.py --config configs/robot/hardware.yaml --robot.backend fake   # no robot
"""

from fr3.communication.lcm.publisher.pub_manager.env.franka_fr3.FrankaPubManager import FrankaPubManager
from fr3.communication.lcm.subscriber.sub_manager.env.FrankaSubManager import FrankaSubManager
from fr3.robot_platform.hardware.franka_fr3.HardwareFrankaFR3 import HardwareFrankaFR3
from hrg.cli import load_config


def main() -> None:
    cfg = load_config(__doc__)
    hardware = HardwareFrankaFR3(cfg)
    sub = FrankaSubManager(**cfg["sub_manager"])
    pub = FrankaPubManager(**cfg["pub_manager"])
    hardware.set_sub_que_dict(sub.get_sub_que_dict())
    hardware.set_pub_que_dict(pub.get_pub_que_dict())
    sub.start()
    pub.start()
    try:
        hardware.initialize()
        hardware.start()
    finally:
        # Stop torque control even if something above raised.
        hardware.shutdown()
        pub.stop()
        sub.stop()


if __name__ == "__main__":
    main()
