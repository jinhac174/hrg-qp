"""Simulation: the FR3 + 2F-85, the object and the obstacles in MuJoCo (``hrg.sim_env``).

    python run_env.py --config configs/sim/env.yaml --object_config configs/objects/blue_bowl.yaml
"""

from fr3.communication.lcm.publisher.pub_manager.env.franka_fr3.FrankaPubManager import FrankaPubManager
from fr3.communication.lcm.subscriber.sub_manager.env.FrankaSubManager import FrankaSubManager
from hrg.cli import load_config
from hrg.sim_env import HRGSimEnv


def main() -> None:
    cfg = load_config(__doc__)
    env = HRGSimEnv(cfg)
    sub = FrankaSubManager(**cfg["sub_manager"])
    pub = FrankaPubManager(**cfg["pub_manager"])
    env.set_sub_que_dict(sub.get_sub_que_dict())
    env.set_pub_que_dict(pub.get_pub_que_dict())
    sub.start()
    pub.start()
    try:
        env.initialize()
        env.start()
    finally:
        pub.stop()
        sub.stop()
        env.stop()


if __name__ == "__main__":
    main()
