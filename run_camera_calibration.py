"""Robot: base -> camera extrinsic calibration (AprilTag guess + differentiable
rendering of the robot's outline; from franka_visual_servoing, tutorial 14).

Needs run_hardware.py, run_sensors.py and run_grav_comp.py running (docs/robot.md):

    python run_camera_calibration.py --config configs/robot/calib_zed_ext1.yaml
    python run_camera_calibration.py --config configs/robot/calib_zed_ext1.yaml --init config   # refine the current one
"""

import sys

from fr3.utils.calibration.diff_base_to_cam import CalibInterrupted, DiffBaseToCamCalibrator
from hrg.cli import load_config


def main() -> None:
    cfg = load_config(__doc__)
    try:
        DiffBaseToCamCalibrator(cfg).run()
    except CalibInterrupted:
        print("[Calib] Stopped by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
