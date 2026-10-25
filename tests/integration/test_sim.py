"""The method in the real-time simulation (sim env + controller over LCM).

The mug from beside it (spawn seed 1, gripper at left-bottom): the straight
line to the selected rim pinch crosses the mug. The path field (eqs. 3-5)
reaches, pinches the rim and lifts; the eq. (2) linear field stops ~4 cm
short with the hand on its 1 cm margin -- the local minimum of the paper's
Table I / Fig. 6. The same placement is stuck in the deterministic kinematic
rollout (``scripts/rollout.py``), so this is the field, not timing.
"""

from fr3.lcm_type.gripper.gripper_state_t import gripper_state_t
from fr3.lcm_type.joint.joint_ctrl_t import joint_ctrl_t
from fr3.lcm_type.pose.se3_pose_t import se3_pose_t
from hrg.processes import Listener
from hrg.study import START_POSES

OBJ = "configs/objects/blue_mug.yaml"
STARTUP = 90
CHANNELS = {"franka_object_pose": se3_pose_t.decode, "franka_gripper_state": gripper_state_t.decode,
            "franka_arm_joint_ctrl": joint_ctrl_t.decode}


def _run(procs, field: str, timeout: float) -> tuple[bool, str]:
    q0 = "[" + ",".join(str(v) for v in START_POSES["left_bottom"]) + "]"
    with Listener(CHANNELS) as lst:
        procs.start("env", "run_env.py", "--config", "configs/sim/env.yaml", "--vis_mode", "vis_off",
                    "--object_config", OBJ, "--obj_spawn_seed", "1", "--platform.default_q_values", q0)
        assert lst.wait_for(lambda: "franka_object_pose" in lst.last, STARTUP), "env did not publish"
        lst.spin(1.0)
        z0 = float(lst.last["franka_object_pose"].position[2])
        ctrl = procs.start("ctrl", "run_controller.py", "--config", "configs/controller.yaml",
                           "--object_config", OBJ, "--hrg.field", field)
        assert lst.wait_for(lambda: lst.count["franka_arm_joint_ctrl"] > 500, STARTUP), "no controller"
        ctrl.key("v")

        def lifted():
            gs = lst.last.get("franka_gripper_state")
            return float(lst.last["franka_object_pose"].position[2]) - z0 > 0.05 and gs is not None and gs.is_grasped

        ok = lst.wait_for(lifted, timeout)
        if ok:  # and it stays lifted and held
            lst.spin(2.0)
            ok = lifted()
        return ok, ctrl.log()


def test_path_field_grasps_the_mug_rim_and_holds_it(procs):
    ok, log = _run(procs, "path", 30)
    assert ok, log[-3000:]
    assert "Grasped after" in log and "Lifted" in log


def test_linear_field_is_stuck_beside_the_mug(procs):
    ok, log = _run(procs, "linear", 20)
    assert not ok, "the eq. (2) baseline was expected to stop short of the rim"
    assert "reach -> close" not in log
