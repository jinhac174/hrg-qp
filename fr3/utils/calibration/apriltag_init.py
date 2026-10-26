"""Initial camera extrinsic from one AprilTag lying on the desk.

The detector finds the tag's 4 corners in the image (pixels). The tag's real
size gives the same 4 corners in the tag's own frame (metres). With the
camera intrinsics K, PnP ("Perspective-n-Point") solves for the pose that
maps the metre corners onto the pixel corners:

    T_cam_tag   pose of the tag in the camera frame        (from the image)
    T_base_tag  pose of the tag in the robot base frame    (you measure it)

    T_base_cam = T_base_tag @ inv(T_cam_tag)                (camera in the base)
    T_world2cam = inv(T_base_cam)                           (what configs store)

Tag frame (OpenCV ArUco convention): origin at the centre of the black
square, x to the right and y to the top as you read the tag, z out of the
printed face. A tag lying face-up on the desk has z = base z.

Uses OpenCV's ArUco module (it includes the AprilTag families); no extra
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

FAMILIES = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


@dataclass
class TagDetection:
    T_cam_tag: np.ndarray  # 4x4, tag pose in the camera frame
    corners: np.ndarray  # (4, 2) pixels: top-left, top-right, bottom-right, bottom-left
    reproj_err_px: float  # RMS distance between detected and re-projected corners


def tag_corners_3d(size: float) -> np.ndarray:
    """Corners in the tag frame, in the detector's order (TL, TR, BR, BL).

    This is the point order solvePnP's IPPE_SQUARE method expects."""
    h = size / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def make_detector(family: str = "tag36h11") -> cv2.aruco.ArucoDetector:
    if family not in FAMILIES:
        raise ValueError(f"tag family must be one of {sorted(FAMILIES)}, got {family!r}")
    params = cv2.aruco.DetectorParameters()
    # AprilTag's own corner refinement (sub-pixel line fits on the border).
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(FAMILIES[family]), params)


def detect_tag(
    rgb: np.ndarray,
    K: np.ndarray,
    tag_id: int,
    size: float,
    detector: cv2.aruco.ArucoDetector,
) -> TagDetection | None:
    """Find tag ``tag_id`` and solve its pose. None if it is not seen.

    The image is assumed undistorted (the ZED SDK and MuJoCo both give
    rectified images), so no distortion coefficients are used."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    hits = [i for i, x in enumerate(ids.flatten()) if int(x) == int(tag_id)]
    if not hits:
        return None
    px = corners[hits[0]].reshape(4, 2).astype(np.float64)
    obj = tag_corners_3d(size)
    K = np.asarray(K, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, px, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - px) ** 2, axis=1))))
    return TagDetection(T_cam_tag=T, corners=px, reproj_err_px=err)


def tag_in_base(cfg) -> np.ndarray:
    """T_base_tag from the config's ``tag_in_base``.

    ``position``: tag centre in the base frame (m). Orientation, either
    ``yaw_deg`` (tag face-up on the desk; angle of the tag's x axis from the
    base x axis, about base z) or ``quaternion`` [x, y, z, w]."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(cfg["position"], dtype=np.float64)
    if cfg.get("quaternion", None) is not None:
        T[:3, :3] = Rotation.from_quat(np.asarray(cfg["quaternion"], dtype=np.float64)).as_matrix()
    else:
        T[:3, :3] = Rotation.from_euler("z", float(cfg.get("yaw_deg", 0.0)), degrees=True).as_matrix()
    return T


def average_poses(Ts) -> np.ndarray:
    """Mean of several 4x4 poses: mean translation, rotation = the mean
    rotation matrix projected back onto SO(3) (fine for nearby poses)."""
    Ts = np.asarray(Ts, dtype=np.float64)
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].mean(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = Ts[:, :3, 3].mean(axis=0)
    return T


def cam_in_base_from_tag(T_base_tag: np.ndarray, T_cam_tag: np.ndarray) -> np.ndarray:
    """T_base_cam (the camera's pose in the base, OpenCV axes)."""
    return T_base_tag @ np.linalg.inv(T_cam_tag)
