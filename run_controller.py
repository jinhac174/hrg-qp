"""The reactive grasp controller (``hrg.controller``), sim or robot.

    python run_controller.py --config configs/controller.yaml --object_config configs/objects/blue_bowl.yaml
    python run_controller.py --config configs/controller.yaml --env_mode real --hrg.field linear
"""

from fr3.communication.lcm.publisher.pub_manager.controller.JointCtrlGraspPubManager import JointCtrlGraspPubManager
from fr3.communication.lcm.subscriber.sub_manager.controller.FrankaSubManager import FrankaSubManager
from hrg.cli import load_config
from hrg.controller import HRGController


def main() -> None:
    cfg = load_config(__doc__)
    controller = HRGController(cfg)
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
