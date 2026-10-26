"""Robot: gravity compensation, the arm moved by hand (for the camera calibration).

    python run_grav_comp.py --config configs/robot/grav_comp.yaml
"""

from fr3.communication.lcm.publisher.pub_manager.controller.JointCtrlGraspPubManager import JointCtrlGraspPubManager
from fr3.communication.lcm.subscriber.sub_manager.controller.FrankaSubManager import FrankaSubManager
from fr3.controller.GravCompController import GravCompController
from hrg.cli import load_config


def main() -> None:
    cfg = load_config(__doc__)
    controller = GravCompController(cfg)
    sub = FrankaSubManager(**cfg["sub_manager"])
    pub = JointCtrlGraspPubManager(**cfg["pub_manager"])
    controller.set_sub_que_dict(sub.get_sub_que_dict())
    controller.set_ctrl_pub_que(pub.get_ctrl_pub_que())
    controller.set_pub_que_dict(pub.get_pub_que_dict())
    sub.start()
    pub.start()
    try:
        controller.initialize()
        controller.start()
    finally:
        sub.stop()
        pub.stop()
        controller.stop()


if __name__ == "__main__":
    main()
