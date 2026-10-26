"""The camera calibration (fr3/utils/calibration, run_camera_calibration.py):
the AprilTag initial guess on a synthetic image of a tag at a known pose, the
silhouette metrics and losses, and (GPU) the full refinement recovering a known
camera pose from rendered masks."""

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from fr3.utils.calibration import apriltag_init as tag
from fr3.utils.calibration import silhouette_metrics as sm

TAG_SIZE = 0.148
TAG = {"position": [0.363, 0.0, 0.0], "yaw_deg": 90.0}  # configs/robot/calib_zed_ext1.yaml
K = np.array([[520.0, 0.0, 480.0], [0.0, 520.0, 360.0], [0.0, 0.0, 1.0]])  # 960x720: the whole tag in view


def _pose_err(A, B):
    dR = A[:3, :3] @ B[:3, :3].T
    return np.linalg.norm(A[:3, 3] - B[:3, 3]), np.rad2deg(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec()))


def _true_cam_in_base():
    """The ext1 camera of configs/robot/zed.yaml (its pose in the base)."""
    c = OmegaConf.load("configs/robot/zed.yaml").zed.ext1.tf_link_to_cam
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = Rotation.from_quat(list(c.quaternion)).as_matrix()
    T_w2c[:3, 3] = list(c.translation)
    return np.linalg.inv(T_w2c)


def _tag_image(T_cam_tag, shape=(720, 960), cell_px=40):
    """The tag (black square = TAG_SIZE, a white cell of margin) seen by K at T_cam_tag."""
    black = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 8 * cell_px)
    tex = cv2.copyMakeBorder(black, cell_px, cell_px, cell_px, cell_px, cv2.BORDER_CONSTANT, value=255)
    c, s = tex.shape[0] / 2.0, 8 * cell_px / TAG_SIZE  # texture px -> tag frame: x right, y up
    A = np.array([[1 / s, 0, -c / s], [0, -1 / s, c / s], [0, 0, 1]])
    H = K @ np.c_[T_cam_tag[:3, 0], T_cam_tag[:3, 1], T_cam_tag[:3, 3]] @ A
    img = cv2.warpPerspective(tex, H, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue=120)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def test_camera_from_tag_on_a_synthetic_image():
    T_base_cam = _true_cam_in_base()
    T_base_tag = tag.tag_in_base(TAG)
    T_cam_tag = np.linalg.inv(T_base_cam) @ T_base_tag
    rgb = _tag_image(T_cam_tag)
    det = tag.detect_tag(rgb, K, 0, TAG_SIZE, tag.make_detector("tag36h11"))
    assert det is not None and det.reproj_err_px < 0.5
    dt, dr = _pose_err(det.T_cam_tag, T_cam_tag)
    assert dt < 0.005 and dr < 1.0
    dt, dr = _pose_err(tag.cam_in_base_from_tag(T_base_tag, det.T_cam_tag), T_base_cam)
    assert dt < 0.01 and dr < 1.0
    assert tag.detect_tag(rgb, K, 5, TAG_SIZE, tag.make_detector("tag36h11")) is None  # wrong id


def test_tag_in_base_yaw_and_quaternion_agree():
    a = tag.tag_in_base({"position": [0.1, 0.2, 0.0], "yaw_deg": 90.0})
    b = tag.tag_in_base({"position": [0.1, 0.2, 0.0], "quaternion": [0, 0, np.sqrt(0.5), np.sqrt(0.5)]})
    np.testing.assert_allclose(a, b, atol=1e-9)
    np.testing.assert_allclose(a[:3, 0], [0, 1, 0], atol=1e-9)  # tag x = base y


def test_average_poses():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    T[:3, 3] = [1, 2, 3]
    Ts = []
    for s in (+1, -1):
        dT = np.eye(4)
        dT[:3, :3] = Rotation.from_rotvec([0, 0, s * 0.01]).as_matrix()
        dT[:3, 3] = [s * 0.01, 0, 0]
        Ts.append(dT @ T)
    dt, dr = _pose_err(tag.average_poses(Ts), T)
    assert dt < 1e-3 and dr < 1e-3


def _disk(c, r=30, shape=(120, 160)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r


def test_metrics_iou_and_outline_error():
    a = _disk((60, 60))
    assert sm.iou(a, a) == 1.0
    assert sm.boundary_error(a, a)["mean"] == 0.0
    b = _disk((65, 60))  # shifted 5 px
    e = sm.boundary_error(a, b)
    assert 1.5 < e["mean"] < 5.0 and 4.0 <= e["p95"] <= 6.0
    assert 0.7 < sm.iou(a, b) < 0.9


def test_outline_skips_image_border():
    # Robot cut off by the left image edge: that edge is not an outline.
    m = np.zeros((50, 50), bool)
    m[10:40, 0:20] = True
    edge = sm.outline(m)
    assert not edge[:, 0].any()
    assert edge[10:40, 19].all()


def test_distance_maps():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    d_out, d_in = sm.distance_maps(m)
    assert d_out[20, 20] == 0 and d_in[5, 5] == 0
    assert abs(d_out[20, 5] - 5) < 0.1  # 5 px left of the mask
    assert abs(d_in[20, 12] - 3) < 0.1  # 3 px inside the mask edge


def test_boundary_loss_pulls_from_afar():
    """A soft disk moved by gradient descent on the boundary loss reaches the
    observed disk, also from a start where the two do not overlap (the area
    loss has no gradient there)."""
    torch = pytest.importorskip("torch")
    obs = _disk((100, 60), r=15)
    d_out, d_in = (torch.as_tensor(x) for x in sm.distance_maps(obs))
    yy, xx = torch.meshgrid(torch.arange(120.0), torch.arange(160.0), indexing="ij")

    def render(c):  # disk with a 1 px soft edge (like nvdiffrast's antialiasing)
        return (15.5 - torch.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + 1e-6)).clamp(0.0, 1.0)

    for loss_fn, should_reach in ((lambda r: sm.boundary_loss(r, d_out, d_in), True),
                                  (lambda r: sm.area_loss(r, torch.as_tensor(obs, dtype=torch.float32)), False)):
        c = torch.tensor([40.0, 60.0], requires_grad=True)  # 60 px away: no overlap
        opt = torch.optim.Adam([c], lr=1.0)
        for _ in range(300):
            opt.zero_grad()
            loss_fn(render(c)).backward()
            opt.step()
        reached = float(torch.linalg.norm(c.detach() - torch.tensor([100.0, 60.0]))) < 1.0
        assert reached == should_reach


def test_refinement_recovers_the_camera(tmp_path):
    """The whole calibrator offline: masks rendered from a known camera in six
    arm poses, the start 2 cm / 1.5 deg off, the result within a few mm."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU (nvdiffrast)")
    from fr3.utils.calibration.diff_base_to_cam import DiffBaseToCamCalibrator
    from fr3.utils.calibration.diff_silhouette import DiffSilhouetteRenderer

    cfg = OmegaConf.load("configs/robot/calib_zed_ext1.yaml")
    cfg.merge_with({"init": "config", "show": False, "n_iters": 800, "output_dir": str(tmp_path), "num_holdout": 2})
    cal = DiffBaseToCamCalibrator(cfg, connect=False)
    H, W = 360, 640
    Kc = np.array([[265.0, 0.0, 320.0], [0.0, 265.0, 180.0], [0.0, 0.0, 1.0]])
    T_true = _true_cam_in_base()
    rend = DiffSilhouetteRenderer(Kc, H, W, gray=0.7, device="cuda")
    views = []
    for q in ([0, 0, 0, -1.57, 0, 1.57, 0], [0.4, 0.2, 0, -1.9, 0, 2.0, 0.5], [-0.4, 0.3, 0.2, -1.6, 0.3, 1.9, -0.5],
              [0.2, -0.3, 0, -2.2, 0, 1.9, 1.0], [-0.2, 0.5, -0.3, -1.4, -0.4, 1.8, 0.2], [0.6, 0.0, 0.2, -2.0, 0.2, 2.2, 0.8]):
        m = cal._fk.mesh_at_q(np.r_[q, 0.0])
        v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)
        rend.set_mesh(v, f)
        with torch.no_grad():
            mask = rend.render(torch.as_tensor(T_true, dtype=torch.float32, device="cuda")).cpu().numpy() > 0.35
        views.append({"rgb": np.zeros((H, W, 3), np.uint8), "q": np.r_[q, 0.0], "mask": mask, "tag": None,
                      "vertices": v, "faces": f})
    dT = np.eye(4)
    dT[:3, :3] = Rotation.from_rotvec(np.deg2rad(1.5) * np.array([0.6, 0.0, 0.8])).as_matrix()
    dT[:3, 3] = [0.012, -0.012, 0.008]  # 1.9 cm
    cal._K, cal._H, cal._W = Kc, H, W
    cal._T_info_c2b = dT @ T_true
    T_final, _ = cal.calibrate(views)
    dt0, dr0 = _pose_err(cal._T_info_c2b, T_true)
    dt, dr = _pose_err(T_final, T_true)
    assert dt0 > 0.015 and dr0 > 1.0
    assert dt < 0.004 and dr < 0.3, (dt, dr)
