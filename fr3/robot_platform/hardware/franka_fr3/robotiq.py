"""Robotiq 2F-85 driver: Modbus RTU over the gripper's USB / RS-485 adapter.

This is how the DROID setup connects the gripper (``/dev/ttyUSB0`` on the
computer that runs the robot; DROID uses polymetis' driver, this is a small
standalone one). Plug the gripper's USB adapter into the workstation that
runs ``run_hardware.py`` and give your user access to the port, e.g.
``sudo usermod -aG dialout $USER`` (log out and in) or
``sudo chmod a+rw /dev/ttyUSB0``.

:class:`RobotiqGripper` offers the same blocking API as the Franka Hand had in
libfranka -- ``move`` / ``grasp`` / ``stop`` / ``homing`` / ``state`` in metres
and newtons -- so ``gripper_cmd_t`` and every controller stay unchanged.

Protocol (Robotiq 2F-85 manual, "Control" / "Modbus RTU"):

* 115200 baud, 8N1, slave id 9, CRC-16/Modbus.
* Write 3 registers at 0x03E8 (function 0x10), 6 bytes:
  ``[ACTION, 0, 0, POSITION, SPEED, FORCE]``; ACTION bits: rACT (0) activate,
  rGTO (3) go to the requested position, rATR (4) automatic release.
* Read 3 input registers at 0x07D0 (function 0x04), 6 bytes:
  ``[STATUS, 0, FAULT, POSITION ECHO, POSITION, CURRENT]``; STATUS bits:
  gACT (0), gGTO (3), gSTA (4-5, 3 = activated), gOBJ (6-7: 0 moving,
  1 stopped by an object while opening, 2 ... while closing, 3 arrived, no
  object).
* POSITION 0 = open .. 255 = closed; SPEED / FORCE 0..255 span the datasheet
  ranges (20-150 mm/s, 20-235 N).
"""

from __future__ import annotations

import threading
import time

import numpy as np

from fr3.data_type.basic_types.GripperCmdData import GripperCmdData
from fr3.robot_platform.gripper import robotiq_2f85 as rq

SLAVE_ID = 9
REG_WRITE = 0x03E8
REG_READ = 0x07D0

OBJ_MOVING, OBJ_OPENING, OBJ_CLOSING, OBJ_NONE = 0, 1, 2, 3


def crc16(frame: bytes) -> bytes:
    """Modbus CRC-16 (poly 0xA001, init 0xFFFF), little-endian."""
    crc = 0xFFFF
    for b in frame:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes([crc & 0xFF, crc >> 8])


def write_frame(data: bytes) -> bytes:
    body = bytes([SLAVE_ID, 0x10, REG_WRITE >> 8, REG_WRITE & 0xFF, 0x00, 0x03, 0x06]) + data
    return body + crc16(body)


def read_frame() -> bytes:
    body = bytes([SLAVE_ID, 0x04, REG_READ >> 8, REG_READ & 0xFF, 0x00, 0x03])
    return body + crc16(body)


class SerialTransport:
    """The real port (pyserial, imported only here).

    ``port`` is a device (``/dev/ttyUSB0``) or a pyserial URL, e.g.
    ``socket://192.168.0.127:5020`` when the gripper is plugged into another
    computer (the DROID NUC) that bridges its port to TCP:
    ``python3 -m serial.tools.tcp_serial_redirect -P 5020 /dev/ttyUSB0 115200``.
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.2):
        import serial

        self._s = serial.serial_for_url(
            port, baudrate=baudrate, bytesize=8, parity="N", stopbits=1, timeout=timeout
        )

    def exchange(self, request: bytes, reply_len: int) -> bytes:
        self._s.reset_input_buffer()
        self._s.write(request)
        return self._s.read(reply_len)


class RobotiqGripper:
    """Blocking move / grasp / stop / homing / state on a 2F-85.

    Width <-> register position is linear between the positions the gripper
    reports fully open and fully closed; :meth:`homing` measures them (until
    then the typical values 3 / 230 are used).
    """

    MOVE_FORCE = 70.0  # N, force setting while moving (an obstacle stops it)
    POLL_DT = 0.01  # s between status reads while waiting
    TIMEOUT = 5.0  # s, longest a move / grasp may take

    def __init__(self, port: str = "/dev/ttyUSB0", transport=None):
        self._io = transport if transport is not None else SerialTransport(port)
        self._lock = threading.Lock()  # one transaction at a time (action + state threads)
        self.pos_open, self.pos_closed = 3, 230
        self.max_width = rq.MAX_WIDTH
        self._grasp_window = None  # (lo, hi) after a successful grasp
        self._stop = threading.Event()

    # ------------------------------------------------------------ registers
    def _write(self, act=1, gto=1, pos=0, speed=255, force=0):
        data = bytes([(act & 1) | (gto & 1) << 3, 0, 0, int(pos), int(speed), int(force)])
        with self._lock:
            reply = self._io.exchange(write_frame(data), 8)
        if len(reply) != 8 or reply[:2] != bytes([SLAVE_ID, 0x10]) or crc16(reply[:6]) != reply[6:]:
            raise IOError(f"Robotiq: bad write reply {reply.hex()}")

    def status(self) -> dict:
        with self._lock:
            reply = self._io.exchange(read_frame(), 11)
        if len(reply) != 11 or reply[:3] != bytes([SLAVE_ID, 0x04, 0x06]) or crc16(reply[:9]) != reply[9:]:
            raise IOError(f"Robotiq: bad read reply {reply.hex()}")
        st, flt, pr, po, cu = reply[3], reply[5], reply[6], reply[7], reply[8]
        return {
            "act": st & 1, "gto": (st >> 3) & 1, "sta": (st >> 4) & 3, "obj": (st >> 6) & 3,
            "fault": flt & 0x0F, "pos_request": pr, "pos": po, "current": cu,
        }

    # ------------------------------------------------------------ conversions
    def width_to_pos(self, width: float) -> int:
        f = 1.0 - float(np.clip(width, 0.0, self.max_width)) / self.max_width
        return int(round(self.pos_open + f * (self.pos_closed - self.pos_open)))

    def pos_to_width(self, pos: int) -> float:
        f = (pos - self.pos_open) / max(self.pos_closed - self.pos_open, 1)
        return float(np.clip((1.0 - f) * self.max_width, 0.0, self.max_width))

    @staticmethod
    def _byte(value, lo, hi) -> int:
        return int(round(255 * (float(np.clip(value, lo, hi)) - lo) / (hi - lo)))

    # ------------------------------------------------------------ actions
    def _go(self, pos, speed, force) -> dict:
        """Command a target and wait until the gripper stops (or stop())."""
        self._stop.clear()
        self._write(pos=pos, speed=self._byte(speed, *rq.SPEED_RANGE), force=self._byte(force, *rq.FORCE_RANGE))
        t0 = time.time()
        while True:
            time.sleep(self.POLL_DT)
            s = self.status()
            if s["fault"]:
                raise IOError(f"Robotiq fault 0x{s['fault']:X}")
            # gOBJ reads "moving" until the new request is taken up
            if s["pos_request"] == pos and s["obj"] != OBJ_MOVING:
                return s
            if self._stop.is_set():
                return s
            if time.time() - t0 > self.TIMEOUT:
                raise TimeoutError("Robotiq: action did not finish")

    def move(self, width: float, speed: float) -> bool:
        self._grasp_window = None
        target = self.width_to_pos(width)
        s = self._go(target, speed, self.MOVE_FORCE)
        return s["obj"] == OBJ_NONE or abs(self.pos_to_width(s["pos"]) - width) < 0.002

    def grasp(self, width, speed, force, eps_in, eps_out) -> bool:
        """Close (or open) until something stops the fingers; success if it
        stopped within [width - eps_in, width + eps_out]. The gripper keeps
        squeezing with ``force`` afterwards."""
        self._grasp_window = None
        cur = self.pos_to_width(self.status()["pos"])
        target = self.pos_closed if width <= cur else self.pos_open
        s = self._go(target, speed, force)
        w = self.pos_to_width(s["pos"])
        ok = s["obj"] in (OBJ_OPENING, OBJ_CLOSING) and width - eps_in <= w <= width + eps_out
        if ok:
            self._grasp_window = (width - eps_in, width + eps_out)
        return ok

    def preempt(self) -> None:
        """Stop *waiting* for the running move (it keeps going to its set-point
        until the next request replaces it): a stream of width set-points
        (reactive grasping) then follows the newest one."""
        self._stop.set()

    def stop(self) -> bool:
        """Stop where the fingers are (rGTO = 0)."""
        self._stop.set()
        s = self.status()
        self._write(gto=0, pos=s["pos"])
        return True

    def homing(self) -> bool:
        """Activate (reset + rACT) and measure the open / closed positions."""
        self._grasp_window = None
        self._write(act=0, gto=0)
        self._write(act=1, gto=0)
        t0 = time.time()
        while self.status()["sta"] != 3:
            if time.time() - t0 > self.TIMEOUT:
                raise TimeoutError("Robotiq: activation did not finish")
            time.sleep(0.05)
        for attr, target in (("pos_closed", 255), ("pos_open", 0)):
            s = self._go(target, rq.SPEED_RANGE[1], rq.FORCE_RANGE[0])
            if s["obj"] == OBJ_NONE:  # reached the end stop
                setattr(self, attr, s["pos"])
            else:  # something between the fingers: keep the previous value
                print(f"[Robotiq] homing: blocked at {s['pos']} (object in the gripper?); "
                      f"keeping {attr} = {getattr(self, attr)}")
        return True

    def action(self, mode, width, speed, force, eps_in, eps_out) -> bool:
        """``gripper_cmd_t`` dispatch (MOVE / GRASP / HOMING)."""
        if mode == GripperCmdData.MOVE:
            return self.move(width, speed)
        if mode == GripperCmdData.GRASP:
            return self.grasp(width, speed, force, eps_in, eps_out)
        if mode == GripperCmdData.HOMING:
            return self.homing()
        return True

    def state(self):
        """(width, max_width, is_grasped), like libfranka's GripperState."""
        s = self.status()
        w = self.pos_to_width(s["pos"])
        win = self._grasp_window
        held = win is not None and s["obj"] in (OBJ_OPENING, OBJ_CLOSING) and win[0] <= w <= win[1]
        return w, self.max_width, bool(held)


class FakeRobotiqTransport:
    """A simulated 2F-85 behind the Modbus protocol, for tests and the fake
    backend: fingers move at the commanded speed and stop at ``object_width``
    (if set) when closing."""

    def __init__(self, object_width: float | None = None):
        self.object_width = object_width
        self.act = self.gto = 0
        self.sta = 0
        self.req = self.speed = self.force = 0
        self.pos = 3.0
        self._t = time.time()
        self.writes = []

    def _pos_of_width(self, w):
        return 3 + (1 - w / rq.MAX_WIDTH) * 227

    def _update(self):
        t = time.time()
        dt, self._t = t - self._t, t
        if not (self.act and self.gto):
            return
        rate = 227 / rq.MAX_WIDTH * (0.02 + self.speed / 255 * 0.13)  # bits / s
        lim = 230 if self.req >= 230 else max(3, self.req)
        if self.object_width is not None and lim > self.pos:
            lim = min(lim, self._pos_of_width(self.object_width))
        step = rate * dt
        self.pos += float(np.clip(lim - self.pos, -step, step))

    def _obj(self):
        if not (self.act and self.gto):
            return OBJ_NONE
        target = min(max(self.req, 3), 230)
        if abs(self.pos - target) < 0.5:
            return OBJ_NONE
        if self.object_width is not None and self.req > self.pos and abs(self.pos - self._pos_of_width(self.object_width)) < 0.5:
            return OBJ_CLOSING
        return OBJ_MOVING

    def exchange(self, request: bytes, reply_len: int) -> bytes:
        assert crc16(request[:-2]) == request[-2:], "bad CRC from the driver"
        self._update()
        if request[1] == 0x10:
            d = request[7:13]
            self.writes.append(bytes(d))
            act, gto = d[0] & 1, (d[0] >> 3) & 1
            if act and not self.act:
                self.sta = 3  # activation completes instantly here
            if not act:
                self.sta = 0
            self.act, self.gto = act, gto
            self.req, self.speed, self.force = d[3], d[4], d[5]
            body = request[:6]
            return body + crc16(body)
        st = self.act | (self.gto << 3) | (self.sta << 4) | (self._obj() << 6)
        body = bytes([SLAVE_ID, 0x04, 0x06, st, 0, 0, self.req, int(round(self.pos)), self.force // 4])
        return body + crc16(body)
