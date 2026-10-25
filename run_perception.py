"""Robot: FoundationPose + SAM2 tracking of the object (GPU workstation).

Needs ``third_party/FoundationPose`` (see docs/robot.md).

    python run_perception.py --config configs/robot/foundation_pose.yaml --mesh_file assets/objects/blue_bowl/visual.obj
"""

from fr3.communication.lcm.publisher.pub_manager.perception.NamedVecPubManager import NamedVecPubManager
from fr3.communication.lcm.subscriber.sub_manager.perception.SingleRGBDSubManager import SingleRGBDSubManager
from hrg.cli import load_config


def main() -> None:
    cfg = load_config(__doc__)
    from fr3.perception.foundation_pose.FoundationPoseTrackingSAM2 import FoundationPoseTrackingSAM2

    perception = FoundationPoseTrackingSAM2(cfg)
    sub = SingleRGBDSubManager(**cfg["sub_manager"])
    pub = NamedVecPubManager(**cfg["pub_manager"])
    perception.set_sub_que_dict(sub.get_sub_que_dict())
    perception.set_percep_pub_que_dict(pub.get_pub_que_dict())
    sub.start()
    pub.start()
    try:
        perception.initialize()
        perception.start()
    finally:
        sub.stop()
        pub.stop()
        perception.stop()


if __name__ == "__main__":
    main()
