import fcl
import numpy as np

# A robot collision primitive is named after the link frame it is attached to.
# A link carrying several primitives names the extra ones "<frame>__<k>"
# (e.g. "fr3_link5__2"); col_frame_name() recovers the frame for FK.
COL_FRAME_SEP = "__"


def col_frame_name(name: str) -> str:
    """Link frame a collision primitive is attached to."""
    return name.split(COL_FRAME_SEP, 1)[0]


def get_fcl_geom(type: str, size: np.ndarray, *args, **kwargs):
    """
    Get the FCL geometry based on the type and size.
    @param[in] type: The type of the geometry.
    @param[in] size: The size of the geometry.
    @return: The FCL geometry.
    """
    if type == "box":
        return fcl.Box(size[0], size[1], size[2])
    elif type == "cylinder":
        return fcl.Cylinder(size[0], size[1])
    elif type == "sphere":
        return fcl.Sphere(size[0])
    elif type == "capsule":
        # size = (radius, length of the straight segment); total = length + 2r
        return fcl.Capsule(size[0], size[1])
    else:
        raise ValueError(f"Unknown geometry type: {type}")
