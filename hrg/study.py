"""The simulation study's conditions (paper Sec. III-A, Table I)."""

import numpy as np

# "gripper at left-bottom, center-up, right-bottom": center_up is the FR3
# ready pose; the other two point the gripper down 27 cm to either side of the
# object with the fingertips at about rim height (TCP 8 cm above the table).
# From there the straight line to most targets crosses the object, which is
# where the eq. (2) linear field gets stuck.
START_POSES = {
    "left_bottom": [0.5587, 0.2044, 0.0121, -2.4036, -0.0048, 2.6079, 0.5747],
    "center_up": [0.0, 0.0, 0.0, -1.5707963, 0.0, 1.5707963, 0.0],
    "right_bottom": [-0.5587, 0.2044, -0.0121, -2.4036, 0.0048, 2.6079, -0.5747],
}
# The mug and the wine glass are also tested rotated 90 deg about each
# horizontal axis (``all``: these five for them, upright for the others).
ORIENTATIONS = {
    "upright": [0.0, 0.0, 0.0],
    "rx+90": [np.pi / 2, 0.0, 0.0],
    "rx-90": [-np.pi / 2, 0.0, 0.0],
    "ry+90": [0.0, np.pi / 2, 0.0],
    "ry-90": [0.0, -np.pi / 2, 0.0],
}
ROTATED_OBJECTS = ("blue_mug", "wine_glass")
# Table I objects available for the 2F-85 (the dish has no target it can
# reach under the paper's margins; see docs/results.md).
TABLE1_OBJECTS = ["box", "blue_bowl", "blue_mug", "wine_glass"]


def orientations_for(obj: str, requested) -> list[str]:
    if list(requested) == ["all"]:
        return list(ORIENTATIONS) if obj in ROTATED_OBJECTS else ["upright"]
    return list(requested)
