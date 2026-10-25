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

`run_perception.py` imports `third_party.FoundationPose` and `sam2`, set up as
in franka_visual_servoing's tutorial 13 (CUDA 12.8, torch cu128):

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/Gy920/segment-anything-2-real-time.git sam2 && (cd sam2 && pip install -e . --no-build-isolation)
git clone https://github.com/NVlabs/FoundationPose.git
cd ..
FORCE_CUDA=0 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -r third_party/FoundationPose/requirements.txt
# then FoundationPose's own build (mycpp, nvdiffrast) and its weights in third_party/FoundationPose/weights/
```

(or symlink `third_party` from franka_visual_servoing).

### Cameras

`configs/robot/zed.yaml`: the ZED serials and `ext1.tf_link_to_cam`, the
camera pose in the robot base frame. The values there were calibrated on
2026-10-19 with franka_visual_servoing's tutorial 14 (AprilTag + 8 views);
recalibrate there if the camera moved.

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
python run_perception.py --config configs/robot/foundation_pose.yaml --mesh_file assets/objects/blue_bowl/visual.obj
python run_pose_bridge.py --config configs/robot/pose_bridge.yaml --mesh_file assets/objects/blue_bowl/visual.obj
python run_visualizer.py --config configs/visualizer.yaml --object_mesh assets/objects/blue_bowl/visual.obj
```

The mesh must be the object's `visual.obj` in all three places: the
fingertip targets are defined in its frame. In the browser view the object
must sit where it really is and follow it when moved.

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
