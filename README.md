# Hierarchical reactive grasping on the Franka FR3 + Robotiq 2F-85

An implementation of Lee, Lin, Alexiev, Kim, *Hierarchical Reactive Grasping
via Task-Space Velocity Fields and Joint-Space Quadratic Programming*
(arXiv 2509.01044, <https://reactivegrasp.github.io>; no public code) for a
Franka FR3 with a Robotiq 2F-85 parallel gripper, in MuJoCo and on the robot.

At 100 Hz, at the measured state: select fingertip targets on the object
(eq. 12), plan collision-free fingertip paths and take their initial
directions as velocity fields (eqs. 3-5), add the gripper-posture and
grasp-stability fields (eqs. 6-10), and track all of them with one joint-space
QP under the full robot's collision and limit constraints (eq. 11). At 1 kHz:
PD on the integrated QP velocity, with gravity compensation.

* **[docs/method.md](docs/method.md)**: the paper equation by equation, where each piece is in the code, and every FR3 / 2F-85 adaptation with its reason.
* **[docs/results.md](docs/results.md)**: the simulation studies (Table I, Figs. 6-8) and their limits.
* **[docs/robot.md](docs/robot.md)**: running it on the real robot.
* **`outputs/videos/`** (made by `scripts/record_video.py`): the method at work.

## Layout

```
hrg/                   the method and its runtime
  hand.py              2F-85 fingertips x(q, w), Jacobians, collision spheres / boxes
  sdf.py               d(x, O): point clouds (concave), boxes (convex, obstacles), the table
  targets.py           fingertip target pairs (generation) and their selection, eq. (12)
  path_opt.py          fingertip path optimisation, eq. (3) (via-point initial guess + SQP)
  fields.py            velocity fields, eqs. (2), (4)-(10)
  qp.py                the tracking QP, eq. (11)
  robot.py             FR3 kinematics / dynamics (Pinocchio), Gamma(q) (FCL)
  planner.py           one planning step, eqs. (2)-(12), a function of the state
  controller.py        the LCM controller: plan thread, grasp / lift, the 1 kHz control law
  sim_env.py           the MuJoCo study env (object, pushes, moved object, moving obstacles)
  objects.py, obstacles.py, study.py, viz.py, render.py, processes.py, cli.py, paths.py
fr3/                   the FR3 stack this runs on (from franka_visual_servoing, pruned):
                       LCM types / pub-sub, MuJoCo base envs, the FR3 sim platform and the
                       real hardware process (libfranka + Robotiq + safety layer), ZED and
                       FoundationPose wrappers
run_env.py             simulation              run_hardware.py    the real robot
run_controller.py      the controller          run_visualizer.py  browser 3-D view
run_sensors.py, run_perception.py, run_pose_bridge.py   cameras -> FoundationPose -> object pose
configs/               controller.yaml (the method's parameters), sim/, robot/, objects/, visualizer.yaml
assets/robot/          FR3 + 2F-85 MJCF / URDF / meshes;  assets/objects/<name>/: scans, meshes, targets
scripts/               prepare_objects.py, eval_sim.py, rollout.py, dissect.py, plot_trace.py, record_video.py, sim.sh
scripts/robot/         preflight.py, identify_payload.py, robotiq_tcp_bridge.py, publish_obstacles.py
tests/                 unit (method, models, safety layer) and integration (sim, robot pipeline)
```

## Setup

Python 3.12, `pip install -r requirements.txt` (the robot-only and GPU-only
packages are listed at its end; see docs/robot.md). `ffmpeg` on the PATH for
videos.

## Run the simulation

```bash
scripts/sim.sh blue_bowl          # tmux: env (MuJoCo viewer) | controller | browser view (http://localhost:8080)
```

or in three terminals:

```bash
python run_env.py --config configs/sim/env.yaml --object_config configs/objects/blue_bowl.yaml
python run_controller.py --config configs/controller.yaml --object_config configs/objects/blue_bowl.yaml
python run_visualizer.py --config configs/visualizer.yaml --object_mesh assets/objects/blue_bowl/visual.obj
```

Controller terminal: `v` reach and grasp, `x` open and stop, `h` home.
Env terminal or viewer: `k` push the hand, `o` move the object.
`--hrg.field linear` runs the eq. (2) baseline; `--config configs/sim/env.yaml
configs/sim/env_obstacles.yaml` adds moving obstacles. Objects:
`box`, `blue_bowl`, `blue_mug`, `wine_glass`, `white_plate`, `red_cup`, `tape`.

## Studies, videos, tests

```bash
python scripts/eval_sim.py --fields path linear --orientations all           # Table I (~3 h; run it alone)
python scripts/rollout.py --fields path linear --placements 20               # the planner alone, kinematic
python scripts/dissect.py outputs/eval/table1/<trial>/trace.npz --t_start 5  # replay a recorded trial: what binds?
python scripts/record_video.py --object blue_mug --pose left_bottom --fields path linear
python scripts/prepare_objects.py                                            # regenerate objects / targets
pytest -m "not integration"     # ~35 s
pytest -m integration           # ~1 min, starts the sim / controller / fake-hardware processes
```
