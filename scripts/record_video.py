"""Videos of the method at work (sim trials, or whatever is on the LCM bus).

Runs a trial like ``scripts/eval_sim.py`` (sim env + controller headless,
``v`` pressed after 1 s), samples the LCM state at 30 Hz while it runs and,
once it is over, renders the samples with ``hrg.render``: the robot, the
object, the obstacles, the planned fingertip paths (eq. 3), the fingertips
and their targets (eq. 12), a close-up on the object, and a caption (field,
time, controller state, fingertip error, perturbations). Rendering after the
trial keeps the CPU to the real-time simulation and the frame timing exact.

    # path field vs the eq. (2) baseline from the same start, side by side
    python scripts/record_video.py --object blue_mug --pose left_bottom --seed 1 --fields path linear
    # Fig. 7: pushes and a moved object
    python scripts/record_video.py --object blue_bowl --pose center_up --push_at 1.5 4.0 --jump_at 3.0
    # Fig. 8: moving obstacles
    python scripts/record_video.py --env configs/sim/env_obstacles.yaml --object wine_glass --pose center_up
    # the robot (or a sim you started yourself): record the bus for 20 s
    python scripts/record_video.py --live --object blue_bowl --duration 20

Output: ``outputs/videos/<name>.mp4`` (H.264, needs ``ffmpeg`` on the PATH).
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")  # offscreen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LCM_DEFAULT_URL", f"udpm://239.255.76.67:{random.randint(30000, 39999)}?ttl=0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from fr3.lcm_type.gripper.gripper_state_t import gripper_state_t  # noqa: E402
from fr3.lcm_type.joint.joint_ctrl_t import joint_ctrl_t  # noqa: E402
from fr3.lcm_type.joint.joint_meas_t import joint_meas_t  # noqa: E402
from fr3.lcm_type.pose.se3_pose_t import se3_pose_t  # noqa: E402
from fr3.lcm_type.vector.vec_list_t import vec_list_t  # noqa: E402
from hrg.paths import REPO, resolve  # noqa: E402
from hrg.processes import Listener, Procs  # noqa: E402
from hrg.render import SceneRenderer  # noqa: E402
from hrg.study import ORIENTATIONS, START_POSES  # noqa: E402

FPS = 30
CHANNELS = {
    "franka_arm_joint_meas": joint_meas_t.decode,
    "franka_arm_joint_ctrl": joint_ctrl_t.decode,
    "franka_gripper_state": gripper_state_t.decode,
    "franka_object_pose": se3_pose_t.decode,
    "franka_obstacles": vec_list_t.decode,
    "franka_hrg_fingertips": vec_list_t.decode,
    "franka_hrg_paths": vec_list_t.decode,
}
FIELD_LABEL = {"path": "Path field, eqs. (3)-(5)", "linear": "Linear field, eq. (2) (baseline)"}
WHITE, GREY, YELLOW = (255, 255, 255), (200, 200, 200), (255, 220, 80)


def _vec(q) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in q) + "]"


class FFmpegWriter:
    def __init__(self, path: str, w: int, h: int):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on the PATH")
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
            stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        self.p.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()


def _text(img, s, org, scale=0.7, color=WHITE, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _fingertip_overlay(last: dict):
    ft, P = last.get("franka_hrg_fingertips"), last.get("franka_hrg_paths")
    x = x_tgt = paths = None
    if ft is not None:
        p = {n: np.asarray(v) for n, v in zip(ft.name_list, ft.vec_list)}
        x = np.stack([p["thumb_cur"], p["index_cur"]])
        x_tgt = np.stack([p["thumb_tgt"], p["index_tgt"]])
    if P is not None:
        paths = [np.asarray([v for n, v in zip(P.name_list, P.vec_list) if n.startswith(f + "_")])
                 for f in ("thumb", "index")]
    return x, x_tgt, paths


class Recorder:
    """Renders the bus into frames: main view + close-up inset + caption."""

    def __init__(self, object_config: str, obstacle_halves, width: int, height: int, title: str, subtitle: str):
        self.r = SceneRenderer(object_config, obstacle_halves, width, height)
        self.w, self.h = width, height
        self.title, self.subtitle = title, subtitle
        self.reach_visible = False

    def frame(self, last: dict, t: float, state: str, events: list[str]) -> np.ndarray:
        """One frame from ``last`` (the latest message per channel at that instant)."""
        m, gs = last.get("franka_arm_joint_meas"), last.get("franka_gripper_state")
        pose, obs = last.get("franka_object_pose"), last.get("franka_obstacles")
        self.r.set_state(q_arm=None if m is None else m.q, width=None if gs is None else gs.width,
                         obj_pos=None if pose is None else pose.position,
                         obj_quat_wxyz=None if pose is None else pose.quat_wxyz,
                         obstacle_rows=None if obs is None or not obs.num_vecs else obs.vec_list)
        x, x_tgt, paths = _fingertip_overlay(last) if state == "reach" else (None, None, None)
        img = self.r.render(x, x_tgt, paths).copy()
        close = self.r.render(x, x_tgt, paths, closeup=True)
        iw, ih = int(self.w * 0.42), int(self.h * 0.42)
        inset = cv2.resize(close, (iw, ih), interpolation=cv2.INTER_AREA)
        y0, x0 = self.h - ih - 44, self.w - iw - 16
        img[y0:y0 + ih, x0:x0 + iw] = inset
        cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + iw + 1, y0 + ih + 1), GREY, 2)
        _text(img, "close-up", (x0 + 8, y0 + 24), 0.55, GREY, 1)

        s = self.h / 720.0
        _text(img, self.title, (20, int(40 * s)), 0.9 * s, WHITE, 2)
        _text(img, self.subtitle, (20, int(72 * s)), 0.6 * s, GREY, 1)
        line = f"t = {t:4.1f} s    state: {state}"
        if x is not None:
            line += f"    |x - x*| = {100 * np.linalg.norm(x - x_tgt, axis=1).max():4.1f} cm"
        _text(img, line, (20, int(104 * s)), 0.65 * s, WHITE, 2)
        for k, e in enumerate(events):
            _text(img, e, (int(self.w * 0.55), int((40 + 34 * k) * s)), 0.8 * s, YELLOW, 2)
        _text(img, "small spheres: fingertips (red = left pad, blue = right pad)   large: their targets (eq. 12)"
              "   tubes: planned fingertip paths (eq. 3)", (20, self.h - 16), 0.5 * s, GREY, 1)
        return img


def _state(log: str) -> str:
    m = re.findall(r"\[HRG\] \w+ -> (\w+)", log)
    return m[-1] if m else "idle"


class Sampler:
    """Snapshots of the bus taken at ``FPS`` while the trial runs (cheap:
    references to the latest decoded messages), rendered afterwards on an
    exact time grid. Rendering during the trial would compete with the
    real-time simulation for the CPU and, whenever a frame took longer than
    1 / FPS, leave the video uneven (frames written at a fixed rate but
    sampled at irregular wall-clock times)."""

    def __init__(self):
        self.samples: list[tuple[float, dict, float, str, list[str]]] = []  # (wall time, last, t, state, events)

    def take(self, t_wall: float, lst, t: float, state: str, events: list[str]) -> None:
        self.samples.append((t_wall, dict(lst.last), t, state, list(events)))

    def render(self, rec: Recorder, writer: FFmpegWriter) -> None:
        """Every frame of the uniform grid shows the latest sample at or
        before its time (the sim state is held, never interpolated)."""
        if not self.samples:
            return
        gaps = np.diff([s[0] for s in self.samples])
        if gaps.size and gaps.max() > 3.0 / FPS:
            print(f"  note: sampling fell behind by up to {gaps.max():.2f} s (CPU contention)")
        t_last, k = self.samples[-1][0], 0
        for n in range(int(t_last * FPS) + 1):
            tf = n / FPS
            while k + 1 < len(self.samples) and self.samples[k + 1][0] <= tf:
                k += 1
            _, last, t, state, events = self.samples[k]
            writer.write(rec.frame(last, t, state, events))


def record_trial(path: str, obj: str, pose: str, seed: int, field: str, orientation: str, env_cfgs,
                 push_at, jump_at, duration: float, width: int, height: int, title: str) -> dict:
    env_cfg = OmegaConf.merge(*[OmegaConf.load(resolve(f)) for f in env_cfgs])
    halves = [o["half"] for o in (env_cfg.get("obstacles") or [])]
    sub = (f"{obj}, gripper start {pose.replace('_', '-')}, {orientation}, placement {seed}"
           "   |   FR3 + Robotiq 2F-85, MuJoCo, real time")
    out_dir = tempfile.mkdtemp(prefix="hrg_video_")
    info = dict(lifted=False)
    sampler = Sampler()
    with Listener(CHANNELS) as lst, Procs(out_dir) as procs:
        env = procs.start("env", "run_env.py", "--config", *env_cfgs, "--vis_mode", "vis_off",
                          "--object_config", f"configs/objects/{obj}.yaml", "--obj_spawn_seed", str(seed),
                          "--platform.default_q_values", _vec(START_POSES[pose]),
                          "--obj_spawn_rpy", _vec(ORIENTATIONS[orientation]))
        if not lst.wait_for(lambda: "franka_object_pose" in lst.last and "franka_arm_joint_meas" in lst.last, 90):
            raise RuntimeError(f"env did not start:\n{env.log()[-2000:]}")
        z0 = float(lst.last["franka_object_pose"].position[2])
        ctrl = procs.start("ctrl", "run_controller.py", "--config", "configs/controller.yaml",
                           "--object_config", f"configs/objects/{obj}.yaml", "--hrg.field", field)
        if not lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, 90):
            raise RuntimeError(f"controller did not start:\n{ctrl.log()[-2000:]}")
        pending = sorted([(t, "k") for t in push_at] + [(t, "o") for t in jump_at])
        shown: list[tuple[float, str]] = []
        t_start, t_v, t_end = time.time(), None, None
        n, log = 0, ""
        while True:
            t_frame = t_start + n / FPS
            lst.spin(max(0.0, t_frame - time.time()))
            now = time.time()
            if t_v is None and now - t_start >= 1.0:
                ctrl.key("v")
                t_v = now
            t = 0.0 if t_v is None else now - t_v
            while t_v is not None and pending and t >= pending[0][0]:
                tk, key = pending.pop(0)
                env.key(key)
                shown.append((tk, "PUSH on the hand" if key == "k" else "OBJECT MOVED"))
            log = ctrl.log() if n % 3 == 0 else log
            state = _state(log)
            gs = lst.last.get("franka_gripper_state")
            lifted = (float(lst.last["franka_object_pose"].position[2]) - z0 > 0.05 and gs is not None
                      and gs.is_grasped)
            if lifted and not info["lifted"]:
                info["lifted"], info["t_lift"] = True, t
            if t_end is None and (state == "hold" or (t_v is not None and t >= duration)):
                t_end = now + 2.0
            events = [e for te, e in shown if 0.0 <= t - te < 1.2]
            sampler.take(now - t_start, lst, t, state, events)
            n += 1
            if t_end is not None and now >= t_end:
                break
        info["log"] = log
    shutil.rmtree(out_dir, ignore_errors=True)
    # the trial is over: render the samples with the machine to ourselves
    rec = Recorder(f"configs/objects/{obj}.yaml", halves, width, height, title, sub)
    writer = FFmpegWriter(path, width, height)
    sampler.render(rec, writer)
    writer.close()
    return info


def record_live(path: str, obj: str, duration: float, width: int, height: int, title: str) -> None:
    sampler = Sampler()
    with Listener(CHANNELS) as lst:
        t0 = time.time()
        n = 0
        while time.time() - t0 < duration:
            lst.spin(max(0.0, t0 + n / FPS - time.time()))
            t = time.time() - t0
            sampler.take(t, lst, t, "reach" if "franka_hrg_paths" in lst.last else "-", [])
            n += 1
    rec = Recorder(f"configs/objects/{obj}.yaml", [], width, height, title, "live from the LCM bus")
    writer = FFmpegWriter(path, width, height)
    sampler.render(rec, writer)
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="blue_bowl")
    ap.add_argument("--pose", default="center_up", choices=list(START_POSES))
    ap.add_argument("--orientation", default="upright", choices=list(ORIENTATIONS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fields", nargs="*", default=["path"], help="two fields: side by side")
    ap.add_argument("--env", nargs="*", default=[], help="extra env configs on top of configs/sim/env.yaml")
    ap.add_argument("--push_at", type=float, nargs="*", default=[])
    ap.add_argument("--jump_at", type=float, nargs="*", default=[])
    ap.add_argument("--duration", type=float, default=20.0, help="s after v at most (the paper's 20 s)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--live", action="store_true", help="record the bus (robot / your own sim) for --duration s")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.join(REPO, "outputs", "videos"), exist_ok=True)
    name = args.out or os.path.join(
        REPO, "outputs", "videos",
        f"{args.object}_{args.orientation}_{args.pose}_{args.seed}_{'_vs_'.join(args.fields)}.mp4")
    if args.live:
        record_live(name, args.object, args.duration, 1280, 720, args.title or "Hierarchical reactive grasping")
        print(name)
        return
    env_cfgs = ["configs/sim/env.yaml", *args.env]
    if len(args.fields) == 1:
        f = args.fields[0]
        info = record_trial(name, args.object, args.pose, args.seed, f, args.orientation, env_cfgs, args.push_at,
                            args.jump_at, args.duration, 1280, 720, args.title or FIELD_LABEL[f])
        print(f"{name}: {f} {'lifted at %.1f s' % info['t_lift'] if info['lifted'] else 'not lifted'}")
        return
    parts = []
    for f in args.fields:  # one after the other (never two real-time sims at once)
        part = name.replace(".mp4", f"_{f}.mp4")
        info = record_trial(part, args.object, args.pose, args.seed, f, args.orientation, env_cfgs, args.push_at,
                            args.jump_at, args.duration, 960, 540, FIELD_LABEL[f])
        print(f"{part}: {f} {'lifted at %.1f s' % info['t_lift'] if info['lifted'] else 'not lifted'}")
        parts.append(part)
    # side by side; the shorter clip holds its last frame
    inputs = sum([["-i", p] for p in parts], [])
    filt = "".join(f"[{i}:v]tpad=stop_mode=clone:stop_duration=60[v{i}];" for i in range(len(parts)))
    filt += "".join(f"[v{i}]" for i in range(len(parts))) + f"hstack=inputs={len(parts)}:shortest=0[v]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", filt, "-map", "[v]",
                    "-t", str(max(_duration(p) for p in parts)), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", name], check=True)
    for part in parts:
        os.remove(part)
    print(name)


def _duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout
    return float(out.strip() or 0.0)


if __name__ == "__main__":
    main()
