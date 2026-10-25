"""Stereolabs ZED camera (ZED 2 / ZED 2i / ZED Mini): the DROID cameras.

Publishes the same ``RGBDData`` as the RealSense driver, so everything
downstream (perception, calibration) is unchanged: RGB (uint8) from the left
lens, and the left-aligned depth in millimetres as uint16 (0 = no depth),
i.e. ``depth_factor: 1000``.

Needs the ZED SDK (CUDA) and its Python API ``pyzed`` on the computer the
cameras are plugged into (https://www.stereolabs.com/developers/release;
then ``python /usr/local/zed/get_python_api.py`` inside the conda env).
"""

from __future__ import annotations

import numpy as np

from fr3.data_type.basic_types.RGBDData import RGBDData
from fr3.sensor.BaseSensor import BaseSensor


class ZED(BaseSensor):
    def __init__(
        self,
        name: str,
        serial: int,
        *args,
        resolution: str = "HD720",
        width: int = 0,
        height: int = 0,
        frequency: int = 30,
        depth_mode: str = "NEURAL",
        min_depth_mm: float = 150.0,
        max_depth_mm: float = 3000.0,
        image_flip: bool = False,
        **kwargs,
    ):
        """``resolution``: the camera's capture mode (HD2K / HD1080 / HD720 /
        VGA). ``width`` / ``height``: published image size (0 = native); the
        intrinsics are scaled to it. ``image_flip``: rotate the image (and
        depth, intrinsics) by 180 deg in the SDK, for a camera mounted upside
        down. Extrinsics must be calibrated with the same setting."""
        super().__init__(name, *args, sensor_info_flag=True, **kwargs)
        import pyzed.sl as sl

        self._sl = sl
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init.camera_fps = int(frequency)
        init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        init.coordinate_units = sl.UNIT.MILLIMETER
        init.depth_minimum_distance = float(min_depth_mm)
        init.depth_maximum_distance = float(max_depth_mm)
        init.camera_image_flip = sl.FLIP_MODE.ON if image_flip else sl.FLIP_MODE.OFF
        init.set_from_serial_number(int(serial))
        self._cam = sl.Camera()
        err = self._cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED {serial} ({name}): {err}")

        info = self._cam.get_camera_information()
        conf = getattr(info, "camera_configuration", info)  # SDK 4 / SDK 3
        native = conf.resolution
        self.width = int(width or native.width)
        self.height = int(height or native.height)
        self._res = sl.Resolution(self.width, self.height)
        left = conf.calibration_parameters.left_cam
        sx, sy = self.width / native.width, self.height / native.height
        # Intrinsics of the published (left, rectified) image.
        self.fx, self.fy = left.fx * sx, left.fy * sy
        self.cx, self.cy = left.cx * sx, left.cy * sy
        print(f"[ZED {name}] serial {serial}, {resolution} -> {self.width}x{self.height}, "
              f"fx {self.fx:.1f} fy {self.fy:.1f} cx {self.cx:.1f} cy {self.cy:.1f}")

    def _run(self):
        sl = self._sl
        runtime = sl.RuntimeParameters()
        image, depth = sl.Mat(), sl.Mat()
        try:
            while not self._stop_event.is_set():
                if self._cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                # Capture time (host epoch clock), not the time after the depth
                # computation: a wrist camera's pose is looked up at this time.
                t_img = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() * 1e-9
                now = self.get_time()
                if not 0.0 <= now - t_img < 1.0:
                    t_img = now
                self._cam.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, self._res)
                self._cam.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU, self._res)
                rgb = np.ascontiguousarray(image.get_data()[:, :, 2::-1])  # BGRA -> RGB
                d = depth.get_data()
                d = np.where(np.isfinite(d), d, 0.0)
                d = np.clip(d, 0, 65535).astype(np.uint16)  # mm, 0 = unknown

                data = RGBDData(
                    height=self.height,
                    width=self.width,
                    rgb_channel_type=1,
                    depth_channel_type=3,
                    name=self.name,
                )
                data.set_data(
                    t=t_img,
                    rgb_channel_type=1,
                    rgb_image=rgb,
                    depth_channel_type=3,
                    depth_image=d,
                )
                self.pub_data_queue.put(data)
        finally:
            self._cam.close()
