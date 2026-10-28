"""Base-camera extrinsic calibration: AprilTag guess, then differentiable rendering.

Pipeline (tutorial 14):

1. Capture. Pose the arm (gravity compensation, by hand); SPACE stores the
   RGB image, the joint angles q, the AprilTag detection (if the tag is
   visible) and the URDF mesh at q in the base frame. For each capture SAM2
   segments the robot: where the tag is seen, it starts from the robot's
   outline projected with the tag's camera guess (points inside it, the
   gripper included, and around it); you add clicks where that mask is
   wrong and accept. ``capture_from: <run>/views.npz`` re-uses the images and
   joint angles of an earlier run (same camera pose): only the masks again.
2. Initial guess (``init``). ``apriltag``: camera pose from the tag on the
   desk (``tag_in_base``, measured by you) and the tag's pose in the image.
   ``config``: the extrinsic the camera info already carries (a previous
   calibration). ``gt_noise``: sim only, the true pose plus noise.
3. Refinement. Adam moves the camera pose so the rendered robot silhouette
   matches the SAM2 masks of the fit views (``loss``: boundary / area).
4. Evaluation. For the initial guess and the result: per view IoU and
   outline error (px), on the fit views AND on held-out views not used in
   step 3 (``num_holdout``); the tag's re-projection; in sim, the error vs
   the true pose. Overlays and a result file go to ``output_dir``.

Conventions (OpenCV camera axes: x right, y down, z forward):
``T_cam2base`` = the camera's pose in the base ("cam in base");
``T_world2cam`` = inv(T_cam2base) = what ``tf_link_to_cam`` /
``CameraInfo.extrinsic`` store.
"""

from __future__ import annotations

import os
import signal
import time
from queue import Empty, Queue
from typing import List, Optional, Tuple

import cv2
import lcm
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

from fr3.communication.lcm.subscriber.data_subscriber.CameraInfoSubscriber import (
    CameraInfoSubscriber,
)
from fr3.communication.lcm.subscriber.data_subscriber.JointMeasSubscriber import (
    JointMeasSubscriber,
)
from fr3.communication.lcm.subscriber.data_subscriber.RGBDSubscriber import RGBDSubscriber
from fr3.data_type.basic_types.CameraInfoData import CameraInfoData
from fr3.data_type.basic_types.JointMeasData import JointMeasData
from fr3.data_type.basic_types.RGBDData import RGBDData
from fr3.utils.calibration import apriltag_init as tag
from fr3.utils.calibration import silhouette_metrics as sm
from fr3.utils.calibration.diff_silhouette import (
    DiffSilhouetteRenderer,
    se3_from_wt,
    wt_from_se3,
)
from fr3.utils.calibration.robot_mesh_fk import RobotMeshFK
from fr3.utils.lie.se3 import invSE3


def _rgb_to_gray01(rgb: np.ndarray) -> np.ndarray:
    g = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    return np.clip(g / 255.0, 0.0, 1.0)


def _perturb_cam_in_base(
    T_cam2base: np.ndarray, rot_deg: float, trans_m: float, rng: np.random.Generator
):
    """Left-multiply a small SE(3) on cam-in-base (OpenCV eye / axes in world).

    Same convention as Manipulator-Software eyeball_camera_pose / EasyHeC:
    perturbing here moves/rotates the camera in the arm-base frame. Do **not**
    left-multiply noise onto T_world2cam — that warps the pose badly.
    """
    from fr3.utils.lie.so3 import exp_so3

    w = rng.normal(size=3)
    w = w / (np.linalg.norm(w) + 1e-9) * np.deg2rad(rot_deg)
    t = rng.normal(size=3)
    t = t / (np.linalg.norm(t) + 1e-9) * trans_m
    dT = np.eye(4, dtype=np.float64)
    dT[:3, :3] = exp_so3(w)
    dT[:3, 3] = t
    return (dT @ np.asarray(T_cam2base, dtype=np.float64)).astype(np.float32)


def _pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    """Position error (m: distance between the two origins) and rotation
    error (deg: angle of R_est R_gt^T) of two poses in the same frame."""
    t_err = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    dR = T_est[:3, :3] @ T_gt[:3, :3].T
    r_err = float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec())))
    return t_err, r_err


def _project(K: np.ndarray, T_world2cam: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Base-frame points (N, 3) -> pixels (N, 2)."""
    x = (T_world2cam[:3, :3] @ pts.T).T + T_world2cam[:3, 3]
    uv = (K @ x.T).T
    return uv[:, :2] / uv[:, 2:3]


def extrinsic_yaml_block(T_world2cam: np.ndarray) -> str:
    """``tf_link_to_cam`` block for configs/robot/zed.yaml."""
    q = Rotation.from_matrix(T_world2cam[:3, :3]).as_quat()  # x, y, z, w
    t = T_world2cam[:3, 3]
    return (
        "tf_link_to_cam:\n"
        f"  quaternion: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]   # x, y, z, w\n"
        f"  translation: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]"
    )


class CalibInterrupted(KeyboardInterrupt):
    """Raised when the user requests stop (SIGINT/SIGTERM or q in a window)."""


class DiffBaseToCamCalibrator:
    """LCM capture -> SAM2 masks -> initial guess -> Adam on T_cam2base -> evaluation."""

    def __init__(self, config: DictConfig, connect: bool = True):
        """``connect=False``: no LCM (offline use, e.g. scripts/calib_sim_offline.py,
        which sets the camera and the views itself)."""
        self.cfg = config
        self.num_joints = int(config.get("num_joints", 8))
        self.num_arm = int(config.get("num_arm_joints", 7))
        self.num_hand = int(config.get("num_hand_joints", 1))
        self._stop = False
        self._prev_sigint = None
        self._prev_sigterm = None

        # Ground truth = the extrinsic in the camera info is the TRUE pose
        # (MuJoCo). On a real camera it is only the configured value.
        self._has_gt = bool(config.get("ground_truth", False))
        # OpenCV windows during the refinement (false on a machine without a display).
        self._show = bool(config.get("show", True))
        self._init_mode = str(config.get("init", "apriltag"))
        if self._init_mode not in ("apriltag", "config", "gt_noise"):
            raise ValueError(f"init must be apriltag / config / gt_noise, got {self._init_mode!r}")
        if self._init_mode == "gt_noise" and not self._has_gt:
            raise ValueError("init: gt_noise needs ground_truth: true (simulation only)")
        self._loss_mode = str(config.get("loss", "boundary_then_area"))
        self._loss_stages = self._loss_mode.split("_then_")
        if not set(self._loss_stages) <= {"boundary", "area"}:
            raise ValueError(
                f"loss must be boundary / area / boundary_then_area, got {self._loss_mode!r}"
            )
        # Cap on the boundary loss's pixel distances (0 = none): limits how hard
        # a big mask mistake can pull, at the price of less pull from afar.
        self._boundary_clip_px = float(config.get("boundary_clip_px", 0.0))

        tcfg = config.get("apriltag", None)
        self._tag_cfg = tcfg
        if self._init_mode == "apriltag" and tcfg is None:
            raise ValueError("init: apriltag needs an 'apriltag:' section")
        if tcfg is not None:
            self._tag_detector = tag.make_detector(str(tcfg.get("family", "tag36h11")))
            self._tag_id = int(tcfg.get("id", 0))
            self._tag_size = float(tcfg["size"])
            self._T_base_tag = tag.tag_in_base(tcfg["tag_in_base"])

        self._rgb_q: Queue = Queue()
        self._info_q: Queue = Queue()
        self._arm_q: Queue = Queue()
        self._hand_q: Queue = Queue()
        if connect:
            self._lcm = lcm.LCM()
            cam_ch = config["rgbd_channel"]
            # Keep strong refs — LCM callbacks are bound methods; dropping the
            # subscriber objects can break delivery under GC pressure.
            self._subscribers = [
                RGBDSubscriber(self._lcm, self._rgb_q),
                CameraInfoSubscriber(self._lcm, self._info_q),
                JointMeasSubscriber(self._lcm, self._arm_q, self.num_arm),
                JointMeasSubscriber(self._lcm, self._hand_q, self.num_hand),
            ]
            self._subscribers[0].subscribe(cam_ch)
            self._subscribers[1].subscribe(cam_ch + "_info")
            self._subscribers[2].subscribe(config["arm_joint_meas_channel"])
            self._subscribers[3].subscribe(config["hand_joint_meas_channel"])
            print(
                f"[Calib] Subscribed: {cam_ch}, joints "
                f"{config['arm_joint_meas_channel']} ({self.num_arm}), "
                f"{config['hand_joint_meas_channel']} ({self.num_hand})"
            )
        # Joint commands come from GravCompController — this script does not publish ctrl.

        exclude_prefixes = config.get("exclude_link_prefixes", None)
        if exclude_prefixes is not None:
            exclude_prefixes = list(exclude_prefixes)
        self._fk = RobotMeshFK(
            urdf_path=config.get(
                "urdf_path", "assets/robot/urdf/fr3_robotiq.urdf"
            ),
            max_faces=int(config.get("max_faces", 0)),
            arm_only=bool(config.get("arm_only", False)),
            exclude_link_prefixes=exclude_prefixes,
        )
        self._sam2 = None
        self._renderer: Optional[DiffSilhouetteRenderer] = None
        self._K: Optional[np.ndarray] = None
        # Extrinsic from the camera info (OpenCV world2cam) and its inverse.
        self._T_info_w2c: Optional[np.ndarray] = None
        self._T_info_c2b: Optional[np.ndarray] = None
        self._H = int(config.get("height", 480))
        self._W = int(config.get("width", 640))
        self._last_rgb: Optional[np.ndarray] = None
        self._last_arm_q: Optional[np.ndarray] = None
        self._last_hand_q: Optional[np.ndarray] = None
        self._last_q: Optional[np.ndarray] = None
        self._joint_update_count = 0

    # ------------------------------------------------------------ plumbing
    def _on_signal(self, signum, frame) -> None:
        # Flag only — raising from here is unreliable inside OpenCV / CUDA C calls.
        self._stop = True
        print("\n[Calib] Interrupt received — exiting loops…", flush=True)

    def _install_signal_handlers(self) -> None:
        self._stop = False
        self._prev_sigint = signal.signal(signal.SIGINT, self._on_signal)
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._on_signal)

    def _restore_signal_handlers(self) -> None:
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)

    def _check_stop(self) -> None:
        if self._stop:
            raise CalibInterrupted("Calibration interrupted by user")

    def _wait_key(self, delay_ms: int = 15) -> int:
        """cv2.waitKey that also honors SIGINT/SIGTERM via self._stop."""
        self._check_stop()
        key = cv2.waitKey(max(1, int(delay_ms))) & 0xFF
        self._check_stop()
        return key

    def _wait_key_until(self, delay_ms: int = 0) -> int:
        """Like waitKey(0), but poll so Ctrl+C can break out of OpenCV."""
        if delay_ms > 0:
            return self._wait_key(delay_ms)
        while True:
            key = self._wait_key(50)
            if key != 255 and key != 0:
                return key

    def _spin_lcm(self, dt: float = 0.01) -> None:
        self._check_stop()
        self._lcm.handle_timeout(int(dt * 1000))

    def _drain(self, q: Queue):
        last = None
        while True:
            try:
                last = q.get_nowait()
            except Empty:
                break
        return last

    def _wait_camera_info(self, timeout: float = 30.0) -> CameraInfoData:
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._spin_lcm(0.05)
            info = self._drain(self._info_q)
            if isinstance(info, CameraInfoData):
                return info
        raise TimeoutError("Timed out waiting for CameraInfoData")

    def _poll_sensors(self) -> None:
        """Drain LCM and refresh cached RGB / q.

        Arm and hand are updated independently. Requiring both messages in the
        *same* drain (old behavior) dropped updates whenever one queue was empty
        and made `_last_q` look stuck at the startup pose.
        """
        self._spin_lcm(0.01)
        rgb_data = self._drain(self._rgb_q)
        if isinstance(rgb_data, RGBDData):
            self._last_rgb = rgb_data.get_rgb_image()

        arm = self._drain(self._arm_q)
        hand = self._drain(self._hand_q)
        updated = False
        if isinstance(arm, JointMeasData):
            self._last_arm_q = arm.get_q()
            updated = True
        if isinstance(hand, JointMeasData):
            self._last_hand_q = hand.get_q()
            updated = True
        if self._last_arm_q is not None and self._last_hand_q is not None:
            self._last_q = np.concatenate([self._last_arm_q, self._last_hand_q])
            if updated:
                self._joint_update_count += 1
        elif self._last_arm_q is not None and self.num_hand == 0:
            self._last_q = self._last_arm_q.copy()
            if updated:
                self._joint_update_count += 1

    def _init_sam2(self):
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2_camera_predictor

        repo = self.cfg.get("sam2_hf_repo", "facebook/sam2.1-hiera-tiny")
        fname = self.cfg.get("sam2_hf_filename", "sam2.1_hiera_tiny.pt")
        cfg_file = self.cfg.get("sam2_config_file", "configs/sam2.1/sam2.1_hiera_t.yaml")
        ckpt = hf_hub_download(repo_id=repo, filename=fname)
        self._sam2 = build_sam2_camera_predictor(
            cfg_file, ckpt, device="cuda", vos_optimized=True
        )
        print("[Calib] SAM2 ready.")

    def sam2_from_clicks(self, rgb: np.ndarray, points, labels) -> np.ndarray:
        """SAM2 mask (bool HxW) from clicks: points (N, 2) pixels (x, y),
        labels 1 = robot, 0 = not robot."""
        if self._sam2 is None:
            self._init_sam2()
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        lbl = np.asarray(labels, dtype=np.int32).reshape(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._sam2.load_first_frame(rgb)
            _, _, logits = self._sam2.add_new_prompt(frame_idx=0, obj_id=1, points=pts, labels=lbl)
        return (logits[0] > 0.0).squeeze(0).detach().cpu().numpy().astype(bool)

    def _sam2_mask(self, rgb: np.ndarray, prompt=None) -> np.ndarray:
        """Interactive segmentation: click, see the mask, fix it, accept.

        Left click = robot, right click = NOT robot (removes a part SAM2
        added wrongly, e.g. the table or a cable). The mask is recomputed
        after every click and shown in green. ENTER accepts, R starts over
        (without the automatic prompt). ``prompt``: initial ``(points,
        labels)`` (``_auto_prompt``)."""
        clicks: List[Tuple[int, int]] = [] if prompt is None else [tuple(map(int, p)) for p in prompt[0]]
        labels: List[int] = [] if prompt is None else [int(x) for x in prompt[1]]
        win = "SAM2: LMB robot, RMB not robot, ENTER accept, R reset"
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        state = {"mask": None, "dirty": bool(clicks)}

        def on_mouse(event, x, y, flags, param):
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                clicks.append((x, y))
                labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
                state["dirty"] = True

        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        print("[Calib] The green mask must cover the whole robot, gripper included (not the wrist "
              "camera, cables or table). Left-click what is missing, right-click what does not "
              "belong, then ENTER.")
        while True:
            if state["dirty"]:
                state["mask"] = self.sam2_from_clicks(rgb, clicks, labels)
                state["dirty"] = False
            vis = base.copy()
            if state["mask"] is not None:
                green = np.zeros_like(vis)
                green[state["mask"]] = (0, 255, 0)
                vis = cv2.addWeighted(vis, 1.0, green, 0.4, 0)
            for (x, y), l in zip(clicks, labels):
                cv2.circle(vis, (x, y), 5, (0, 255, 0) if l else (0, 0, 255), -1)
            cv2.imshow(win, vis)
            key = self._wait_key(20)
            if key in (13, 10) and state["mask"] is not None and not state["dirty"]:
                break
            if key in (ord("r"), ord("R")):
                clicks.clear()
                labels.clear()
                state["mask"] = None
            if key in (ord("q"), 27):
                cv2.destroyWindow(win)
                raise RuntimeError("SAM2 selection cancelled")
        cv2.destroyWindow(win)
        return state["mask"]

    def _auto_prompt(self, rgb: np.ndarray, q: np.ndarray, det, n_in: int = 12, n_out: int = 8):
        """SAM2 prompt from the tag's camera guess: ``n_in`` points spread over
        the robot's projected outline (eroded, so a few cm of guess error stay
        inside; farthest-point spread reaches the gripper), ``n_out`` in a ring
        around it. ``None`` without a tag detection."""
        if det is None or self._tag_cfg is None:
            return None
        if self._renderer is None:
            self._renderer = DiffSilhouetteRenderer(
                self._K, self._H, self._W, gray=float(self.cfg.get("render_gray", 0.7)), device="cuda"
            )
        mesh = self._fk.mesh_at_q(q)
        v = {"vertices": np.asarray(mesh.vertices, np.float32), "faces": np.asarray(mesh.faces, np.int32)}
        sil = self._render_mask(v, tag.cam_in_base_from_tag(self._T_base_tag, det.T_cam_tag)).astype(np.uint8)
        k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731
        inner = cv2.erode(sil, k(6)) > 0
        ring = (cv2.dilate(sil, k(30)) > 0) & ~(cv2.dilate(sil, k(14)) > 0)

        def spread(m, n):
            ys, xs = np.nonzero(m)
            if not len(xs):
                return np.zeros((0, 2), int)
            P = np.c_[xs, ys]
            idx = [int(np.argmin(P[:, 0] + P[:, 1]))]
            d = np.linalg.norm(P - P[idx[0]], axis=1)
            for _ in range(min(n, len(P)) - 1):
                idx.append(int(np.argmax(d)))
                d = np.minimum(d, np.linalg.norm(P - P[idx[-1]], axis=1))
            return P[idx]

        pin, pout = spread(inner, n_in), spread(ring, n_out)
        return np.r_[pin, pout], np.r_[np.ones(len(pin), int), np.zeros(len(pout), int)]

    def load_views(self, path: str) -> List[dict]:
        """Views from an earlier run's ``views.npz`` (images, joint angles),
        segmented again; the camera must not have moved since."""
        info = self._wait_camera_info()
        self._K = info.get_intrinsic().astype(np.float64)
        self._T_info_w2c = info.get_extrinsic().astype(np.float64)
        self._T_info_c2b = invSE3(self._T_info_w2c)
        self._H, self._W = info.get_img_dim()
        d = np.load(path)
        views = []
        for i, (rgb, q) in enumerate(zip(d["rgb"], d["q"])):
            det = self._detect_tag(rgb)
            print(f"[Calib] Saved view {i + 1}/{len(d['q'])}"
                  + (f", tag reproj {det.reproj_err_px:.2f} px" if det is not None else ", tag not seen"))
            mask = self._sam2_mask(rgb, self._auto_prompt(rgb, q, det))
            mesh = self._fk.mesh_at_q(q)
            views.append({"rgb": rgb, "q": q, "mask": mask, "tag": det,
                          "vertices": np.asarray(mesh.vertices, dtype=np.float32).copy(),
                          "faces": np.asarray(mesh.faces, dtype=np.int32).copy()})
        return views

    def _detect_tag(self, rgb: np.ndarray):
        if self._tag_cfg is None:
            return None
        return tag.detect_tag(rgb, self._K, self._tag_id, self._tag_size, self._tag_detector)

    # ------------------------------------------------------------ 1. capture
    def capture_views(self) -> List[dict]:
        """Live RGB preview; user poses under grav-comp and presses SPACE to capture."""
        info = self._wait_camera_info()
        self._K = info.get_intrinsic().astype(np.float64)
        self._T_info_w2c = info.get_extrinsic().astype(np.float64)
        self._T_info_c2b = invSE3(self._T_info_w2c)
        self._H, self._W = info.get_img_dim()
        print(f"[Calib] Camera {self._W}x{self._H}, K =\n{np.round(self._K, 2)}")
        if self._has_gt:
            eye = self._T_info_c2b[:3, 3]
            print(f"[Calib] Ground truth (sim) camera position in base: {np.round(eye, 3).tolist()} m")

        n_want = int(self.cfg.get("num_captures", 6))
        win = "Calib RGB — pose the arm; SPACE=capture, ESC=finish"
        cv2.namedWindow(win)
        print(
            f"[Calib] Pose the arm so the camera sees it, differently each time. "
            f"SPACE to capture ({n_want} wanted), ESC to finish early. "
            f"Ctrl+C in the terminal also exits."
        )

        views: List[dict] = []
        while len(views) < n_want:
            self._poll_sensors()
            rgb = self._last_rgb
            q = self._last_q
            if rgb is not None:
                vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
                det = self._detect_tag(rgb)
                if det is not None:
                    cv2.polylines(vis, [det.corners.astype(np.int32)], True, (255, 128, 0), 2)
                cv2.putText(
                    vis,
                    f"captured {len(views)}/{n_want}  |  SPACE=capture  ESC=done",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                if self._tag_cfg is not None:
                    cv2.putText(
                        vis,
                        "tag: seen" if det is not None else "tag: NOT seen",
                        (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 128, 0) if det is not None else (0, 0, 255),
                        1,
                    )
                if q is None:
                    cv2.putText(
                        vis,
                        "waiting for joint_meas...",
                        (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 128, 255),
                        2,
                    )
                else:
                    q_arm = np.rad2deg(q[: self.num_arm])
                    q_txt = "arm_deg=[" + ",".join(f"{v:.0f}" for v in q_arm) + "]"
                    cv2.putText(
                        vis,
                        f"{q_txt}  updates={self._joint_update_count}",
                        (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 128),
                        1,
                    )
                cv2.imshow(win, vis)
            key = self._wait_key(15)
            if key == 27:  # ESC
                break
            if key != ord(" "):
                continue
            # Re-poll right before freezing so q matches the displayed RGB.
            self._poll_sensors()
            rgb = self._last_rgb
            q = self._last_q
            if rgb is None or q is None:
                print("[Calib] No RGB/q yet — wait a moment and try SPACE again.")
                continue

            # Freeze this frame for SAM2 (keep polling so LCM does not stall).
            snap_rgb = rgb.copy()
            snap_q = q.copy()
            det = self._detect_tag(snap_rgb)
            print(
                f"[Calib] Capture {len(views) + 1}/{n_want}  "
                f"arm_q(rad)={np.array2string(snap_q[: self.num_arm], precision=3)}  "
                + (f"tag reproj {det.reproj_err_px:.2f} px" if det is not None else "tag not seen")
            )
            mask = self._sam2_mask(snap_rgb, self._auto_prompt(snap_rgb, snap_q, det))
            mesh = self._fk.mesh_at_q(snap_q)
            views.append(
                {
                    "rgb": snap_rgb,
                    "q": snap_q,
                    "mask": mask,
                    "tag": det,
                    "vertices": np.asarray(mesh.vertices, dtype=np.float32).copy(),
                    "faces": np.asarray(mesh.faces, dtype=np.int32).copy(),
                }
            )
            print(
                f"[Calib] Stored view {len(views)}/{n_want}. "
                "Pose again and SPACE for the next capture."
            )

        cv2.destroyWindow(win)
        n_hold = int(self.cfg.get("num_holdout", 2))
        if len(views) - n_hold < 2:
            raise RuntimeError(
                f"Need at least 2 fit views + {n_hold} held-out views, got {len(views)} captures"
            )
        print(f"[Calib] {len(views)} views: {len(views) - n_hold} to fit, {n_hold} held out.")
        return views

    # ------------------------------------------------------------ 2. initial guess
    def initial_guess(self, views: List[dict]) -> np.ndarray:
        """T_cam2base to start the refinement from (``init``)."""
        if self._init_mode == "gt_noise":
            rng = np.random.default_rng(int(self.cfg.get("seed", 0)))
            rot = float(self.cfg.get("init_rot_noise_deg", 5.0))
            trans = float(self.cfg.get("init_trans_noise_m", 0.03))
            print(f"[Calib] Init: ground truth + noise ({rot} deg, {trans} m)")
            return _perturb_cam_in_base(self._T_info_c2b, rot, trans, rng).astype(np.float64)
        if self._init_mode == "config":
            print("[Calib] Init: the extrinsic in the camera info (config)")
            return self._T_info_c2b.copy()
        dets = [v["tag"] for v in views if v["tag"] is not None]
        if not dets:
            raise RuntimeError(
                "init: apriltag, but the tag was not seen in any capture. "
                "Keep it in view (the arm must not cover it in every capture)."
            )
        # Camera and tag are both fixed: every detection measures the same
        # pose, so average them (it reduces the pixel noise of the corners).
        T_cam_tag = tag.average_poses([d.T_cam_tag for d in dets])
        T = tag.cam_in_base_from_tag(self._T_base_tag, T_cam_tag)
        print(
            f"[Calib] Init: AprilTag {self._tag_id} seen in {len(dets)}/{len(views)} captures "
            f"(corner re-projection {np.mean([d.reproj_err_px for d in dets]):.2f} px); "
            f"camera at {np.round(T[:3, 3], 3).tolist()} m in the base"
        )
        return T

    # ------------------------------------------------------------ 3. refinement
    def optimize(self, views: List[dict], T_init: np.ndarray) -> np.ndarray:
        """Refine T_cam2base on the fit views, one Adam run per loss stage
        (``loss: boundary_then_area`` = boundary, then area from its result)."""
        T = T_init
        for stage in self._loss_stages:
            T = self._refine(views, T, stage)
        return T

    def _refine(self, views: List[dict], T_init: np.ndarray, loss_name: str) -> np.ndarray:
        """Adam on (axis-angle, translation) of T_cam2base over the fit views."""
        fit = [v for v in views if not v.get("holdout")]
        omega, t = wt_from_se3(T_init)
        omega = omega.cuda().detach().requires_grad_(True)
        t = t.cuda().detach().requires_grad_(True)

        gray = float(self.cfg.get("render_gray", 0.7))
        lr = float(self.cfg.get("lr", 3e-3))
        lr_final = float(self.cfg.get("lr_final", lr))
        n_iters = int(self.cfg.get("n_iters", 2000))
        early_stop = int(self.cfg.get("early_stop", 200))
        viz_every = int(self.cfg.get("viz_every", 5))
        opt = torch.optim.Adam([omega, t], lr=lr)
        # Exponential decay lr -> lr_final over n_iters: big steps to get
        # there, small ones to settle (Adam's step is ~lr whatever the
        # gradient size, so a constant lr keeps jittering by ~lr at the end).
        gamma = (lr_final / lr) ** (1.0 / max(n_iters, 1))
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

        targets = []
        for v in fit:
            m = v["mask"].astype(np.float32)
            d_out, d_in = sm.distance_maps(v["mask"])
            if self._boundary_clip_px > 0:
                d_out = np.minimum(d_out, self._boundary_clip_px)
                d_in = np.minimum(d_in, self._boundary_clip_px)
            targets.append(
                {
                    "obs": torch.as_tensor(m, device="cuda"),
                    "d_out": torch.as_tensor(d_out, device="cuda"),
                    "d_in": torch.as_tensor(d_in, device="cuda"),
                }
            )

        win = "Calib: obs | rend | |diff|"
        print(
            f"[Calib] Refining T_cam2base on {len(fit)} views: loss={loss_name}, "
            f"up to {n_iters} iters, lr {lr} -> {lr_final}, early stop {early_stop}"
        )
        if n_iters <= 0:
            return T_init
        best_loss = float("inf")
        best_state = (omega.detach().clone(), t.detach().clone())
        last_improve = 0
        for it in range(n_iters):
            self._check_stop()
            opt.zero_grad()
            T_c2b = se3_from_wt(omega, t)
            loss = torch.zeros((), device="cuda")
            mosaics = []
            for v, tg in zip(fit, targets):
                self._renderer.set_mesh(v["vertices"], v["faces"])
                rend = (self._renderer.render(T_c2b) / max(gray, 1e-3)).clamp(0.0, 1.0)
                if loss_name == "boundary":
                    loss = loss + sm.boundary_loss(rend, tg["d_out"], tg["d_in"])
                else:
                    loss = loss + sm.area_loss(rend, tg["obs"])
                if self._show and it % viz_every == 0:
                    o = tg["obs"].cpu().numpy()
                    r = rend.detach().cpu().numpy()
                    mosaics.append(np.concatenate([o, r, np.abs(o - r)], axis=1))
            loss = loss / len(fit)
            # Keep the parameters this loss belongs to (before the step).
            loss_f = float(loss.detach())
            if loss_f < best_loss * (1 - 1e-6):
                best_loss = loss_f
                best_state = (omega.detach().clone(), t.detach().clone())
                last_improve = it
            loss.backward()
            opt.step()
            sched.step()

            if self._show and it % viz_every == 0:
                panel = np.concatenate(mosaics[: min(3, len(mosaics))], axis=0)
                panel_bgr = cv2.cvtColor((np.clip(panel, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    panel_bgr,
                    f"iter {it}  loss={loss_f:.1f}  best={best_loss:.1f}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(win, panel_bgr)
                key = self._wait_key(1)
                if key in (ord("q"), 27):
                    self._stop = True
                    self._check_stop()
            if it % 50 == 0:
                print(f"[Calib] iter {it:4d}  loss={loss_f:.1f}  best={best_loss:.1f}")

            if it - last_improve >= early_stop:
                print(f"[Calib] Early stop at iter {it} (no improvement for {early_stop} iters).")
                break

        T_cam2base = se3_from_wt(best_state[0], best_state[1]).detach().cpu().numpy().astype(np.float64)
        if self._show:
            cv2.destroyWindow(win)
        return T_cam2base

    # ------------------------------------------------------------ 4. evaluation
    def _render_mask(self, v: dict, T_cam2base: np.ndarray) -> np.ndarray:
        self._renderer.set_mesh(v["vertices"], v["faces"])
        with torch.no_grad():
            img = self._renderer.render(torch.as_tensor(T_cam2base, dtype=torch.float32, device="cuda"))
        return img.cpu().numpy() > 0.5 * self._renderer.gray

    def evaluate(self, views: List[dict], T_cam2base: np.ndarray) -> dict:
        """Silhouette agreement per view, the tag re-projection, and (sim) the
        error vs the true pose."""
        out = {"views": []}
        for i, v in enumerate(views):
            rend = self._render_mask(v, T_cam2base)
            b = sm.boundary_error(v["mask"], rend)
            out["views"].append(
                {"i": i, "holdout": bool(v.get("holdout")), "iou": sm.iou(v["mask"], rend),
                 "edge_mean_px": b["mean"], "edge_p95_px": b["p95"], "rend": rend}
            )
        for split in ("fit", "holdout"):
            rows = [r for r in out["views"] if r["holdout"] == (split == "holdout")]
            if rows:
                out[split] = {
                    "iou": float(np.mean([r["iou"] for r in rows])),
                    "edge_mean_px": float(np.mean([r["edge_mean_px"] for r in rows])),
                    "edge_p95_px": float(np.max([r["edge_p95_px"] for r in rows])),
                }
        if self._tag_cfg is not None:
            # Where the MEASURED tag pose lands in the image under this
            # camera pose vs where the tag was detected: a few px means the
            # tag measurement and the robot silhouettes agree.
            corners = (self._T_base_tag @ np.c_[tag.tag_corners_3d(self._tag_size), np.ones(4)].T).T[:, :3]
            errs = []
            for v in views:
                if v["tag"] is not None:
                    uv = _project(self._K, invSE3(T_cam2base), corners)
                    errs.append(np.sqrt(np.mean(np.sum((uv - v["tag"].corners) ** 2, axis=1))))
            if errs:
                out["tag_reproj_px"] = float(np.mean(errs))
                # The tag pose this camera pose implies vs the measured one:
                # how far your hand measurement of the tag was off (if the
                # camera pose is right).
                dets = [v["tag"] for v in views if v["tag"] is not None]
                T_base_tag_implied = T_cam2base @ tag.average_poses([d.T_cam_tag for d in dets])
                out["tag_vs_measured"] = _pose_error(T_base_tag_implied, self._T_base_tag)
        if self._has_gt:
            out["gt_error"] = _pose_error(T_cam2base, self._T_info_c2b)
        return out

    def report(self, results: dict) -> None:
        names = list(results)
        print("\n[Calib] ===================== evaluation =====================")
        print("  per view: IoU (1 = identical masks) | outline error mean / p95 (px)")
        n = len(results[names[0]]["views"])
        for i in range(n):
            hold = results[names[0]]["views"][i]["holdout"]
            cells = []
            for k in names:
                r = results[k]["views"][i]
                cells.append(f"{k}: {r['iou']:.3f} | {r['edge_mean_px']:5.1f} / {r['edge_p95_px']:5.1f}")
            print(f"  view {i} ({'HELD OUT' if hold else 'fit     '})  " + "   ".join(cells))
        for split in ("fit", "holdout"):
            if split in results[names[0]]:
                cells = [
                    f"{k}: IoU {results[k][split]['iou']:.3f}, outline {results[k][split]['edge_mean_px']:.1f} px"
                    for k in names
                ]
                print(f"  {split:8s} mean   " + "   ".join(cells))
        for k in names:
            r = results[k]
            line = f"  {k:7s}"
            if "tag_reproj_px" in r:
                dt, dr = r["tag_vs_measured"]
                line += (f" tag re-projection {r['tag_reproj_px']:.1f} px (implied tag pose vs "
                         f"measured: {dt * 1000:.1f} mm, {dr:.2f} deg)")
            if "gt_error" in r:
                dt, dr = r["gt_error"]
                line += f" | vs TRUE pose: {dt * 1000:.1f} mm, {dr:.2f} deg"
            print(line)
        print("[Calib] ========================================================")

    def save(self, views, T_init, T_final, results) -> str:
        out_dir = os.path.join(
            str(self.cfg.get("output_dir", "outputs/calibration")),
            f"{self.cfg['rgbd_channel']}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(out_dir, exist_ok=True)
        # Overlays: observed mask outline (red), rendered outline (green),
        # detected tag (orange), measured tag re-projected (blue).
        for name, T in (("init", T_init), ("final", T_final)):
            for v, r in zip(views, results[name]["views"]):
                img = cv2.cvtColor(v["rgb"], cv2.COLOR_RGB2BGR).copy()
                img[sm.outline(v["mask"], skip_border=False)] = (0, 0, 255)
                img[sm.outline(r["rend"], skip_border=False)] = (0, 255, 0)
                if v["tag"] is not None:
                    cv2.polylines(img, [v["tag"].corners.astype(np.int32)], True, (0, 128, 255), 1)
                    corners = (self._T_base_tag @ np.c_[tag.tag_corners_3d(self._tag_size), np.ones(4)].T).T[:, :3]
                    uv = _project(self._K, invSE3(T), corners)
                    cv2.polylines(img, [uv.astype(np.int32)], True, (255, 0, 0), 1)
                label = f"{name} view {r['i']}{' HELD OUT' if r['holdout'] else ''}  IoU {r['iou']:.3f}  outline {r['edge_mean_px']:.1f}px"
                cv2.putText(img, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imwrite(os.path.join(out_dir, f"view{r['i']}_{name}.png"), img)

        T_w2c = invSE3(T_final)
        summary = {
            "rgbd_channel": str(self.cfg["rgbd_channel"]),
            "init": self._init_mode,
            "loss": self._loss_mode,
            "T_cam2base": T_final.tolist(),
            "T_world2cam": T_w2c.tolist(),
            "T_cam2base_init": T_init.tolist(),
        }
        for k in ("init", "final"):
            summary[f"metrics_{k}"] = {
                kk: vv for kk, vv in results[k].items() if kk in ("fit", "holdout", "tag_reproj_px")
            }
            if "gt_error" in results[k]:
                summary[f"metrics_{k}"]["gt_error_mm_deg"] = [results[k]["gt_error"][0] * 1000, results[k]["gt_error"][1]]
        with open(os.path.join(out_dir, "result.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.create(summary)))
            f.write("\n# Paste into configs/robot/zed.yaml under the camera:\n")
            f.write("\n".join("# " + l for l in extrinsic_yaml_block(T_w2c).splitlines()) + "\n")
        np.savez(os.path.join(out_dir, "views.npz"),
                 q=np.stack([v["q"] for v in views]), masks=np.stack([v["mask"] for v in views]),
                 rgb=np.stack([v["rgb"] for v in views]),
                 holdout=np.array([bool(v.get("holdout")) for v in views]))
        return out_dir

    # ------------------------------------------------------------ main
    def calibrate(self, views: List[dict]):
        """Steps 2-4 on captured views: initial guess, refinement, evaluation,
        output. Returns (T_cam2base, output directory)."""
        n_hold = int(self.cfg.get("num_holdout", 2))
        for i, v in enumerate(views):
            v["holdout"] = i >= len(views) - n_hold

        T_init = self.initial_guess(views)
        self._renderer = DiffSilhouetteRenderer(
            self._K, self._H, self._W, gray=float(self.cfg.get("render_gray", 0.7)), device="cuda"
        )
        T_final = self.optimize(views, T_init)
        results = {"init": self.evaluate(views, T_init), "final": self.evaluate(views, T_final)}
        self.report(results)
        out_dir = self.save(views, T_init, T_final, results)

        T_w2c = invSE3(T_final)
        print(f"[Calib] T_cam2base (camera pose in the base):\n{np.round(T_final, 5)}")
        print(f"[Calib] T_world2cam (the extrinsic):\n{np.round(T_w2c, 5)}")
        print("[Calib] For configs/robot/zed.yaml:\n" + extrinsic_yaml_block(T_w2c))
        print(f"[Calib] Overlays and result.yaml in {out_dir}")
        return T_final, out_dir

    def run(self) -> np.ndarray:
        self._install_signal_handlers()
        try:
            src = self.cfg.get("capture_from", None)
            views = self.load_views(str(src)) if src else self.capture_views()
            T_final, out_dir = self.calibrate(views)
            print("[Calib] Press any key in an image window (or Ctrl+C) to close.")
            cv2.imshow("final overlay (view 0)", cv2.imread(os.path.join(out_dir, "view0_final.png")))
            self._wait_key_until(0)
            return T_final
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self._restore_signal_handlers()
