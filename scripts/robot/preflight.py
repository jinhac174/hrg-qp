"""Robot-workstation preflight: is this machine ready to run the real FR3?

    python scripts/robot/preflight.py                  # checks, no robot motion
    python scripts/robot/preflight.py --hostname 172.16.0.2 --connect

Checks (FAIL = will not work, WARN = may work badly):
  * real-time kernel (PREEMPT_RT) and real-time priority limit
  * panda-py importable; libfranka version it was built against
  * robot reachable (ping); with --connect: panda-py connects and reads the
    state (no motion, no torque control)
  * LCM: multicast route and UDP receive buffer (camera images are large)
  * hardware config (hostname, safety limits within the FR3 limits)
  * gripper: the Robotiq 2F-85's USB port (with --connect: reads its status)
  * optional: ZED cameras, GPU for FoundationPose
"""

from __future__ import annotations

import argparse
import os
import platform
import resource
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

RESULTS = []


def report(level, name, msg=""):
    RESULTS.append(level)
    print(f"[{level:4}] {name}" + (f": {msg}" if msg else ""))


def check_kernel():
    v = platform.uname().version
    rt = os.path.exists("/sys/kernel/realtime") and open("/sys/kernel/realtime").read().strip() == "1"
    if rt or "PREEMPT_RT" in v or "PREEMPT RT" in v:
        report("OK", "real-time kernel", platform.release())
    else:
        report("FAIL", "real-time kernel",
               f"{platform.release()} is not PREEMPT_RT; libfranka needs it for the 1 kHz loop "
               "(Franka docs: 'Setting up the real-time kernel').")
    soft, hard = resource.getrlimit(resource.RLIMIT_RTPRIO)
    if hard >= 90 or hard == resource.RLIM_INFINITY:
        report("OK", "rtprio limit", str(hard))
    else:
        report("FAIL", "rtprio limit",
               f"{hard}; add your user to a 'realtime' group with rtprio 99 / memlock unlimited "
               "in /etc/security/limits.conf, then log in again.")
    gov = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if os.path.exists(gov):
        g = open(gov).read().strip()
        report("OK" if g == "performance" else "WARN", "CPU governor",
               g + ("" if g == "performance" else " (use 'performance' while controlling the robot)"))


def check_panda_py():
    try:
        import panda_py  # noqa: F401
        from panda_py import libfranka  # noqa: F401
    except Exception as e:  # noqa: BLE001
        report("FAIL", "panda-py", f"not importable ({e}). Install the FR3 build from the "
               "panda-py GitHub releases matching your libfranka (see requirements.txt).")
        return False
    ver = getattr(panda_py, "__version__", "?")
    lf = None
    try:
        out = subprocess.run(["pip", "show", "panda-python"], capture_output=True, text=True).stdout
        lf = next((line for line in out.splitlines() if line.startswith("Version")), None)
    except Exception:  # noqa: BLE001
        pass
    report("OK", "panda-py", f"{ver} ({lf or 'pip info n/a'})")
    if "0.9.2" in str(lf or ""):
        report("FAIL", "libfranka version", "PyPI wheel = libfranka 0.9.2: Panda only, not FR3.")
    else:
        report("INFO", "libfranka version",
               "make sure it matches your FR3 system version (Franka compatibility table)")
    return True


def check_robot(hostname, connect):
    if shutil.which("ping"):
        ok = subprocess.run(["ping", "-c", "2", "-W", "1", hostname], capture_output=True).returncode == 0
        report("OK" if ok else "FAIL", f"ping {hostname}",
               "" if ok else "no answer: cable / IP (static 172.16.0.x) / control box on?")
        if not ok:
            return
    if not connect:
        report("INFO", "robot connection", "skipped (use --connect)")
        return
    try:
        import panda_py

        panda = panda_py.Panda(hostname)
        st = panda.get_state()
        report("OK", "panda-py connect", f"q = {[round(x, 3) for x in st.q]}")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "panda-py connect",
               f"{e}  (Desk: joints unlocked? FCI activated? Single point of control?)")


def check_gripper(port, connect):
    """Robotiq 2F-85 on its USB / RS-485 adapter (Modbus RTU), local or
    bridged from another computer (``socket://host:port``)."""
    if "://" in port:
        report("INFO", "Robotiq port", f"{port} (network bridge: scripts/robotiq_tcp_bridge.py on that host)")
    elif not os.path.exists(port):
        report("FAIL", "Robotiq port", f"{port} missing: USB plugged in? ls /dev/ttyUSB*")
        return
    if "://" not in port and not os.access(port, os.R_OK | os.W_OK):
        report("FAIL", "Robotiq port", f"no access to {port}: sudo usermod -aG dialout $USER (log in again)")
        return
    if "://" not in port:
        report("OK", "Robotiq port", port)
    try:
        import serial  # noqa: F401
    except ImportError:
        report("FAIL", "pyserial", "pip install pyserial (the Robotiq driver needs it)")
        return
    if not connect:
        return
    try:
        from fr3.robot_platform.hardware.franka_fr3.robotiq import RobotiqGripper

        s = RobotiqGripper(port).status()
        report("OK" if not s["fault"] else "WARN", "Robotiq status",
               f"activated {s['sta'] == 3}, position {s['pos']}/255, fault 0x{s['fault']:X}")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "Robotiq status", f"{e}  (power on? red light = needs activation, fine)")


def check_lcm():
    url = os.environ.get("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=0 (default)")
    report("INFO", "LCM_DEFAULT_URL", url)
    out = subprocess.run(["ip", "route", "show"], capture_output=True, text=True).stdout
    if "224.0.0.0/4" in out or "default" in out:
        report("OK", "multicast route", "224.0.0.0/4" if "224.0.0.0/4" in out else "via default route")
    else:
        report("WARN", "multicast route",
               "none: sudo ip route add 224.0.0.0/4 dev lo  (single machine)")
    try:
        rmem = int(open("/proc/sys/net/core/rmem_max").read())
        ok = rmem >= 8 * 1024 * 1024
        report("OK" if ok else "WARN", "UDP receive buffer", f"rmem_max {rmem} B" + (
            "" if ok else " (camera frames are ~2 MB: sudo sysctl -w net.core.rmem_max=2147483647 "
            "net.core.rmem_default=2147483647; see docs/robot.md)"))
    except OSError:
        pass
    try:
        import lcm

        lc = lcm.LCM()
        got = []
        lc.subscribe("PREFLIGHT", lambda c, d: got.append(d))
        lc.publish("PREFLIGHT", b"x")
        lc.handle_timeout(500)
        report("OK" if got else "FAIL", "LCM loopback", "" if got else "published but not received")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "LCM", str(e))


def check_config(path, hostname_arg):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    host = hostname_arg or str(cfg.robot.hostname)
    report("INFO", "hardware config", f"{path}: hostname {cfg.robot.hostname}, backend {cfg.robot.backend}")
    if cfg.robot.backend != "pandapy":
        report("WARN", "backend", f"{cfg.robot.backend} (pandapy drives the robot)")
    import numpy as np

    from fr3.robot_platform.hardware.franka_fr3.HardwareFrankaFR3 import FR3_DQ_MAX, FR3_TAU_MAX

    s = cfg.safety
    ok = (np.all(np.asarray(s.tau_max) <= FR3_TAU_MAX) and np.all(np.asarray(s.max_joint_vel) <= FR3_DQ_MAX)
          and float(s.max_joint_error) <= 0.2 and float(s.ctrl_timeout) <= 0.2)
    report("OK" if ok else "WARN", "safety limits",
           f"max_joint_error {s.max_joint_error} rad, tau_rate_max {s.tau_rate_max} Nm/s, "
           f"ctrl_timeout {s.ctrl_timeout} s")
    return host


def cfg_gripper_port(path):
    from omegaconf import OmegaConf

    return OmegaConf.load(path).get("gripper", {}).get("port", "/dev/ttyUSB0")


def check_optional():
    try:
        import pyzed.sl as sl

        devs = [d.serial_number for d in sl.Camera.get_device_list()]
        report("OK" if devs else "WARN", "ZED cameras",
               f"serials {devs} (put them in configs/robot/zed.yaml)" if devs else "none connected")
    except Exception:  # noqa: BLE001
        report("INFO", "ZED cameras", "pyzed not installed (ZED SDK; only needed for perception)")
    try:
        import torch

        report("OK" if torch.cuda.is_available() else "WARN", "GPU (FoundationPose / SAM2)",
               torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available")
    except Exception:  # noqa: BLE001
        report("INFO", "GPU", "torch not installed (only needed for perception)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/robot/hardware.yaml")
    ap.add_argument("--hostname", default="")
    ap.add_argument("--connect", action="store_true", help="connect with panda-py (no motion)")
    args = ap.parse_args()

    print("== system"); check_kernel()
    print("== robot software"); has_pp = check_panda_py()
    print("== config"); host = check_config(args.config, args.hostname)
    print("== robot"); check_robot(host, args.connect and has_pp)
    print("== gripper"); check_gripper(str(cfg_gripper_port(args.config)), args.connect)
    print("== communication"); check_lcm()
    print("== optional"); check_optional()

    n_fail, n_warn = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print(f"\n{n_fail} FAIL, {n_warn} WARN")
    if n_fail == 0:
        print("Next: python run_hardware.py --config configs/robot/hardware.yaml --robot.backend fake "
              "(dry run), then docs/robot.md.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
