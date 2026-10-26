# On the real FR3 + Robotiq 2F-85 (the workstation)

On the robot exactly one process changes: `run_env.py` (MuJoCo) is replaced
by `run_hardware.py`, which talks to the FR3 (libfranka via panda-py) and the
2F-85 (Modbus over RS-485) and publishes **the same LCM channels**. The object
pose comes from FoundationPose instead of the simulator. The controller is the
same; `configs/robot/controller.yaml` on top of `configs/controller.yaml` sets
real-time mode, the paper's 1.5 cm trigger and the real cell's walls.

```
run_sensors.py (ZED)  --zed_ext1-->  run_perception.py (FoundationPose + SAM2, GPU)
     --fp_obj_pose-->  run_pose_bridge.py  --franka_object_pose-->  run_controller.py
run_hardware.py  <--joint_ctrl / gripper_cmd--  run_controller.py
run_hardware.py  --joint_meas / gripper_state / scene collision geometry-->  run_controller.py
(scripts/robot/publish_obstacles.py  --franka_obstacles-->  run_controller.py)
```

## 1. Once per workstation

1. **Real-time kernel** (`PREEMPT_RT`) and **libfranka** matching the FR3's
   system version; **panda-py built for that libfranka** (FR3 wheels on the
   panda-py GitHub releases; the PyPI wheel only talks to the older Panda).
2. Python env: `pip install -r requirements.txt` plus the robot block at its
   end (panda-py, pyserial).
3. **Desk**: unlock the joints, activate FCI, set the end-effector load to the
   2F-85: 0.925 kg, centre of mass 0.052 m along the flange z axis.
4. **Gripper**: its USB / RS-485 adapter on this machine (`gripper.port` in
   `configs/robot/hardware.yaml`, user in `dialout`), or, as in the DROID
   cell, on the NUC behind `scripts/robot/robotiq_tcp_bridge.py`
   (`port: socket://<nuc>:5020`).
5. **Payload** beyond the gripper (e.g. a wrist camera):
   `python scripts/robot/identify_payload.py`, then the `load:` block in
   `configs/robot/hardware.yaml`.
6. **LCM**: camera frames are ~2 MB, beyond the default UDP receive buffer:
   `sudo sysctl -w net.core.rmem_max=2147483647 net.core.rmem_default=2147483647`.
7. **Check**: `python scripts/robot/preflight.py` (no motion), then
   `--connect` (reads the robot and the gripper, no motion).

### FoundationPose + SAM2 (GPU)

`run_perception.py` imports `third_party.FoundationPose` and `sam2`. On the
lab workstation (conda env `franka`: Python 3.12, torch 2.11 cu128, CUDA 12.8,
RTX 5090) they were installed like this; nvdiffrast, open3d and sam2 were
already in the env (sam2: the real-time fork,
`git clone https://github.com/Gy920/segment-anything-2-real-time.git && pip install -e . --no-build-isolation`):

```bash
conda activate franka
export SETUPTOOLS_USE_DISTUTILS=local   # the shell sets "stdlib", which Python 3.12 no longer has: every source build fails
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
pip freeze | grep -iE "^(torch|torchvision|numpy|scipy|opencv-python)==" > /tmp/pins.txt   # keep the robot stack's versions
git clone --depth 1 https://github.com/NVlabs/FoundationPose.git third_party/FoundationPose
# Its requirements.txt minus pyrender (offscreen renderer only; it would downgrade PyOpenGL):
pip install -c /tmp/pins.txt scikit-learn h5py ruamel.yaml transformations "kornia>=0.7" warp-lang scikit-image \
    setuptools wheel ninja pybind11 gdown
# pytorch3d: FoundationPose needs only its Python side, a CPU build takes minutes, not hours
FORCE_CUDA=0 MAX_JOBS=16 pip install -c /tmp/pins.txt --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
# mycpp (pose clustering), with the system g++ so it matches the libstdc++ torch loads
cd third_party/FoundationPose/mycpp && mkdir -p build && cd build
CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") -DPython_EXECUTABLE=$(which python)
cmake --build . -j16 && cd ../..
# weights: weights/2023-10-28-18-33-37 (refiner) and weights/2024-01-11-20-02-45 (scorer), 247 MB
gdown --folder https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O weights
cd ../..
```

The SAM2 checkpoint (`facebook/sam2.1-hiera-tiny`) comes from HuggingFace on
first use.

### Camera calibration (where the ZED is in the robot base frame)

`configs/robot/zed.yaml`: the ZED serials and `ext1.tf_link_to_cam`
(`T_world2cam`: base point -> camera point). A 1 cm error there is a 1 cm
error in every grasp target. The values there are from 2026-10-19
(franka_visual_servoing's tutorial 14, which is where the method is explained).
Recalibrate whenever the camera may have moved.

The method: an AprilTag lying on the desk at a measured pose gives a first
guess (a few cm off), then the camera pose is refined until a render of the
robot (URDF meshes at the measured joints, nvdiffrast) matches the robot's
outline in the image (SAM2 masks you click) in 8 arm poses. The last two
poses are held out to check the result.

1. **Tag.** `tag36h11`, id 0, as large as practical (the 2026-10-19 one: a
   14.8 cm black square). Measure the black square's edge (`apriltag.size`)
   and where the tag lies (`apriltag.tag_in_base`: its centre in the base
   frame, x forward, y left, z up from the base's mounting surface, and its
   yaw) in `configs/robot/calib_zed_ext1.yaml`. A few mm / 1-2 deg off is
   fine: that is what the refinement fixes.
2. **Processes** (four terminals, `conda activate franka`):
   ```bash
   python run_hardware.py --config configs/robot/hardware.yaml
   python run_sensors.py --config configs/robot/zed.yaml
   python run_grav_comp.py --config configs/robot/grav_comp.yaml          # the arm floats: move it by hand
   python run_camera_calibration.py --config configs/robot/calib_zed_ext1.yaml
   ```
   (`--init config` instead starts from the current `zed.yaml` value: a
   re-check, or a small bump.) Quitting `run_grav_comp.py` (`q`) leaves the
   arm held where it is.
3. **Capture 8 poses**, spread over the image: arm left / right / reaching /
   folded, wrist turned. SPACE captures; then click the robot (several links
   and the gripper) until the green mask covers exactly the robot, right-click
   what does not belong (table, cables), ENTER. Keep your hands out of the
   robot's outline when you press SPACE.
4. **Judge it** (the printed table and `outputs/calibration/zed_ext1_<time>/`):
   held-out IoU close to the fit IoU, the green (rendered) outline on the red
   (mask) one in every `view*_final.png`, "implied tag pose vs measured"
   within ~1 cm.
5. **Install it**: paste the printed `tf_link_to_cam` block under `ext1` in
   `configs/robot/zed.yaml`, restart `run_sensors.py`.
6. **Physical check** before trusting it for grasping: with perception
   running (section 3), the object in the browser view must sit where it
   really is; or touch a tracked object's centre with the TCP and measure the
   miss.

## Objects: what the method needs per object

Yes, every object needs a **mesh of that exact object**. Three things come
from it, all in one object frame (upright, base on z = 0, centred in x / y):

| What | File | Used by |
|---|---|---|
| The object's pose | `visual_textured.obj` (or `visual.obj`) | FoundationPose tracks this mesh in the camera image: it is model-based, no mesh = no pose |
| Candidate fingertip targets | `fingertip_targets.npz` | the controller's eq. (12) target selection |
| The distance function d(x, O) | `surface_points.npy` | eq. (3) paths and the QP's hand-vs-object rows |

`configs/objects/<name>.yaml` points at them. Ready: `blue_bowl`,
`blue_mug`, `white_plate` (textured scans), `red_cup`, `tape`, `wine_glass`
(geometry only: those scans have no usable texture, so FoundationPose
matches shape and depth only). Use the **physical objects these were
scanned from**: a different bowl of a similar size gives a wrong pose and
wrong targets. The transparent wine glass is hard for the ZED's stereo depth
and FoundationPose alike; start with the bowl or the mug. `box` is synthetic
(no physical counterpart).

`gold_bowl` is a synthetic mesh from measurements (14 cm rim, 10 cm base,
5 cm tall, 2 mm walls, light gold with a white rim band): grasped and lifted
on the robot on 2026-10-26. It is rotationally symmetric, so run perception
with `--pose_filter.fix_x_rotation true` (its bounding box's x axis is the
bowl's axis) and `--auto_reset.enable false` (its tracking score sits above
the auto-reset threshold even when the fit is good):

```bash
python run_perception.py --config configs/robot/foundation_pose.yaml --mesh_file assets/objects/gold_bowl/visual.obj \
    --pose_filter.fix_x_rotation true --auto_reset.enable false
python run_pose_bridge.py --config configs/robot/pose_bridge.yaml --mesh_file assets/objects/gold_bowl/visual.obj
python run_controller.py --config configs/controller.yaml configs/robot/controller.yaml \
    --object_config configs/objects/gold_bowl.yaml
```

A plain cup or bowl (straight sloped walls, no handle) needs no scan: add a
`cup=dict(top=, bottom=, height=, wall=)` entry (outer diameters, m; optional
`band_z` + `rim_band` for a rim of another colour) to `OBJECTS` in
`scripts/prepare_objects.py`, as `gold_bowl` does, and run step 3 below.

A new scanned object:

1. Scan it (BundleSDF, or a phone photogrammetry app: a textured, metric
   (metres), closed-enough mesh) to `assets/objects/<name>/scan/textured_mesh.obj`
   (+ its `.mtl` / texture).
2. Add it to `OBJECTS` in `scripts/prepare_objects.py`: the rotation that
   stands it upright (`trimesh.poses.compute_stable_poses`), mass, colour.
3. `python scripts/prepare_objects.py --only <name>` (a few minutes: convex
   parts, surface points, fingertip targets, the textured copy, the config).
4. Try it in simulation first (`scripts/sim.sh <name>`).

## 2. Dry run without the robot

```bash
python run_hardware.py --config configs/robot/hardware.yaml --robot.backend fake
python run_controller.py --config configs/controller.yaml configs/robot/controller.yaml \
    --object_config configs/objects/blue_bowl.yaml
```

(`tests/integration/test_robot_pipeline.py` does this automatically.)

## 3. Perception check (arm not moving)

```bash
python run_sensors.py --config configs/robot/zed.yaml
python run_perception.py --config configs/robot/foundation_pose.yaml --mesh_file assets/objects/blue_bowl/visual_textured.obj
python run_pose_bridge.py --config configs/robot/pose_bridge.yaml --mesh_file assets/objects/blue_bowl/visual_textured.obj
python run_visualizer.py --config configs/visualizer.yaml --object_mesh assets/objects/blue_bowl/visual.obj
```

`run_perception.py` and `run_pose_bridge.py` must load the **same** mesh file
(the bridge undoes FoundationPose's bounding-box frame with it), in the object
frame: the object's `visual_textured.obj`, or `visual.obj` for the objects
without one. In the browser view the object must sit where it really is and
follow it when moved.

## 4. Grasping

```bash
python run_hardware.py --config configs/robot/hardware.yaml
python run_controller.py --config configs/controller.yaml configs/robot/controller.yaml \
    --object_config configs/objects/blue_bowl.yaml            # v = reach + grasp, x = open, h = home
python scripts/record_video.py --live --object blue_bowl --duration 30   # optional: a video of the bus
```

Hand on the user stop. `x` opens the gripper and stops; `h` goes home through
the same constrained QP.

Objects for the real experiments (paper Sec. III-B): `blue_bowl`, `red_cup`,
`wine_glass`, `tape` (`configs/objects/`). Obstacles: measure them into
`configs/robot/obstacles.yaml` and run `scripts/robot/publish_obstacles.py`.

## 5. To check on the robot

* **Gripper streaming.** While reaching, the width is streamed as `move`
  set-points at `width_stream_freq` (50 Hz); `run_hardware.py` lets a new
  `move` preempt the running one. Check how fast the 2F-85 follows over
  RS-485 (and the NUC bridge). If it lags, lower `width_stream_freq` or raise
  `width_max_lead`.
* **Trigger.** 1.5 cm (the paper's robot value) absorbs pose-estimation
  noise; watch the fingertip error the controller prints when it closes.
* **The cell.** The controller's walls (`configs/robot/controller.yaml`,
  `col_pairs`) are `assets/robot/scene_real_cell.xml`'s `floor`, `wall_front`,
  `wall_left`, published by `run_hardware.py`. They must match the real
  cell.
* **Table height.** `grasp.table_z` (0 = the robot base plane). If the table
  is not at the base height, set it (it is also the object's resting plane
  for the targets' validity).
