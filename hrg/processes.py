"""Start this repo's processes and listen on LCM (evaluation, videos, tests).

    with Procs(out_dir) as procs, Listener({"franka_object_pose": se3_pose_t.decode}) as lst:
        env = procs.start("env", "run_env.py", "--config", "configs/sim/env.yaml", "--vis_mode", "vis_off")
        lst.wait_for(lambda: "franka_object_pose" in lst.last, 60)
        ctrl = procs.start("ctrl", "run_controller.py", "--config", "configs/controller.yaml")
        ctrl.key("v")

Processes run in a pseudo-terminal (they read keys from it) and log to
``<out_dir>/<name>.log``. Everything is released on exit: processes,
log files, and the LCM sockets (a leaked ``Listener`` keeps its sockets
subscribed to the bus, and the kernel keeps copying every message into them).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import lcm

from hrg.paths import REPO

PY = sys.executable


class Proc:
    def __init__(self, args, log_path: str):
        self.log_path = log_path
        self._log = open(log_path, "w")
        if sys.platform == "darwin":  # BSD script: flags, the typescript file, then the command
            argv = ["script", "-qeF", "/dev/null", PY, *args]
        else:  # util-linux script
            argv = ["script", "-qfec", " ".join([PY, *args]), "/dev/null"]
        self.p = subprocess.Popen(argv, cwd=REPO, stdout=self._log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.PIPE, preexec_fn=os.setsid)

    def key(self, keys: str) -> None:
        self.p.stdin.write(keys.encode())
        self.p.stdin.flush()

    def log(self) -> str:
        with open(self.log_path, errors="ignore") as f:
            return f.read()

    def stop(self) -> None:
        for sig, wait in ((signal.SIGINT, 2.0), (signal.SIGKILL, 0.5)):
            try:
                os.killpg(self.p.pid, sig)
            except (ProcessLookupError, PermissionError):  # already gone (macOS: EPERM)
                break
            t0 = time.time()
            while time.time() - t0 < wait and self.p.poll() is None:
                time.sleep(0.05)
        for f in (self.p.stdin, self._log):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass


class Procs:
    """Starts processes; stops every one of them on exit, even on failure."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.procs: list[Proc] = []

    def start(self, name: str, *args: str) -> Proc:
        p = Proc([str(a) for a in args], str(self.out_dir / f"{name}.log"))
        self.procs.append(p)
        return p

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for p in reversed(self.procs):
            p.stop()


class Listener:
    """Collects decoded LCM messages: ``last[channel]``, ``count[channel]``."""

    def __init__(self, channels: dict, url: str | None = None):
        self.lc = lcm.LCM(url) if url else lcm.LCM()
        self.last: dict = {}
        self.count = {c: 0 for c in channels}
        self._subs = [self.lc.subscribe(ch, self._cb(ch, decode)) for ch, decode in channels.items()]

    def _cb(self, ch, decode):
        def f(_c, data):
            self.count[ch] += 1
            self.last[ch] = decode(data)
        return f

    def spin(self, seconds: float) -> None:
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.lc.handle_timeout(10)

    def wait_for(self, pred, timeout: float, step: float = 0.2) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.spin(step)
            if pred():
                return True
        return False

    def reset_counts(self) -> None:
        for c in self.count:
            self.count[c] = 0

    def close(self) -> None:
        """Release the sockets: the subscriptions' callbacks refer back to
        this object, a cycle the garbage collector cannot break."""
        if self.lc is not None:
            for s in self._subs:
                self.lc.unsubscribe(s)
            self._subs, self.lc = [], None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def lcm_ok(timeout: float = 1.0) -> bool:
    """Loopback self-test of the LCM bus (the multicast route exists)."""
    try:
        with Listener({"hrg_selftest": bytes}) as lst:
            t0 = time.time()
            while time.time() - t0 < timeout and not lst.count["hrg_selftest"]:
                lst.lc.publish("hrg_selftest", b"x")
                lst.lc.handle_timeout(50)
            return lst.count["hrg_selftest"] > 0
    except Exception:  # noqa: BLE001
        return False
