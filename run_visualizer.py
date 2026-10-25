"""Live 3-D view in the browser (``hrg.viz``): robot, object, obstacles,
fingertips, targets and the planned fingertip paths.

    python run_visualizer.py --config configs/visualizer.yaml --object_mesh assets/objects/blue_bowl/visual.obj
"""

from hrg.cli import load_config
from hrg.viz import HRGViewer


if __name__ == "__main__":
    HRGViewer(load_config(__doc__)).run()
