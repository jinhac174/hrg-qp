from fr3.sensor.BaseSensorManager import BaseSensorManager
from fr3.sensor.sensor.zed.ZED import ZED
from fr3.sensor.sensor_info.realsense.CameraInfo import CameraInfo


class ZEDManager(BaseSensorManager):
    """ZED cameras (the DROID setup: two ZED 2 + a ZED Mini on the wrist).

    Config (``configs/robot/zed.yaml``): one entry per camera under
    ``zed:``; its RGB-D goes to ``pub_manager.zed.<key>_channel`` and its
    camera info (intrinsics read from the camera, extrinsics from the config)
    to ``<channel>_info``.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._zed_configs = config["zed"]
        self._pub_manager_config = config["pub_manager"]

    def initialize_sensors_and_sensor_info(self):
        for key, cfg in self._zed_configs.items():
            name = self._pub_manager_config["zed"][f"{key}_channel"]
            if int(cfg["serial"]) == 0:  # not set in the config: camera not used
                print(f"[ZED {name}] serial 0 (not set): skipped")
                continue
            cam = ZED(
                name,
                cfg["serial"],
                resolution=cfg.get("resolution", "HD720"),
                width=cfg.get("width", 0),
                height=cfg.get("height", 0),
                frequency=cfg.get("frequency", 30),
                depth_mode=cfg.get("depth_mode", "NEURAL"),
                min_depth_mm=cfg.get("min_depth_mm", 150.0),
                max_depth_mm=cfg.get("max_depth_mm", 3000.0),
                image_flip=cfg.get("image_flip", False),
            )
            self._sensors[name] = cam
            self._sensor_info[name] = CameraInfo(
                name + "_info",
                cfg["serial"],
                cfg.get("fixed", True),
                cfg.get("attached_body", None),
                cam.height,
                cam.width,
                cfg.get("frequency", 30),
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
                1000,  # depth in mm
                tf_link_to_cam=cfg["tf_link_to_cam"],
                channel_name=name + "_info",
            )
