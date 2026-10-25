"""Robot: publish a fixed list of box obstacles (measured by hand) on the
obstacles channel, in ``hrg.obstacles``' format, at 30 Hz.

The paper tracks its obstacles (tissue boxes) with FoundationPose; a tracker
per obstacle can publish the same rows instead of this script.

    python scripts/robot/publish_obstacles.py --config configs/robot/obstacles.yaml
"""

import os
import sys
import time

import lcm
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fr3.lcm_type.vector.vec_list_t import vec_list_t  # noqa: E402
from hrg import obstacles as obstacle_msg  # noqa: E402
from hrg.cli import load_config  # noqa: E402


def main() -> None:
    cfg = load_config(__doc__)
    T_list, halves = [], []
    for o in cfg["obstacles"]:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("z", float(o.get("yaw", 0.0))).as_matrix()
        T[:3, 3] = o["centre"]
        T_list.append(T)
        halves.append(o["half"])
    rows = obstacle_msg.encode(T_list, halves)
    msg = vec_list_t()
    msg.num_vecs, msg.vec_dim = len(rows), obstacle_msg.ROW
    msg.name_list = [f"obstacle_{i}" for i in range(len(rows))]
    msg.vec_list = rows.tolist()
    lc = lcm.LCM()
    print(f"[Obstacles] {len(rows)} boxes on {cfg['channel']} at 30 Hz. Ctrl-C to stop.")
    try:
        while True:
            msg.timestamp = time.time()
            lc.publish(cfg["channel"], msg.encode())
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
