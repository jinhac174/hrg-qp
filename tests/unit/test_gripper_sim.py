"""_GripperSim: the simulator's version of the gripper's move / grasp / stop /
homing API (driven by the sim with the measured width), for the Robotiq 2F-85
(0-85 mm, 20-235 N, 20-150 mm/s)."""

from fr3.data_type.basic_types.GripperCmdData import GripperCmdData as G
from fr3.robot_platform.gripper.robotiq_2f85 import MAX_WIDTH
from fr3.robot_platform.sim.franka_fr3.SimFrankaFR3 import _GripperSim

DT = 0.001


def cmd(seq, mode, width=0.0, speed=0.1, force=0.0, eps=0.005):
    c = G()
    c.set_data(0.0, seq, mode, width, speed, force, eps, eps)
    return c


def run(g, width, seconds, block_at=None):
    """Ideal fingers that follow the set-point, optionally stopped by an object
    of width ``block_at``. Returns the final width."""
    width_rate = 0.0
    for _ in range(int(seconds / DT)):
        ref, _ = g.step(width, width_rate, DT)
        new = max(ref, block_at) if block_at is not None else ref
        width_rate, width = (new - width) / DT, new
    return width


def test_move_completes():
    g = _GripperSim(0.08)
    g.command(cmd(1, G.MOVE, 0.04, speed=0.1), 0.08)
    assert g.busy
    w = run(g, 0.08, 1.0)
    assert abs(w - 0.04) < 1e-3
    assert not g.busy and g.last_seq == 1 and g.last_success


def test_grasp_object_succeeds_and_holds():
    g = _GripperSim(0.08)
    g.command(cmd(2, G.GRASP, 0.05, force=40.0, eps=0.01), 0.08)
    assert g.force == 40.0
    w = run(g, 0.08, 1.0, block_at=0.052)
    assert not g.busy and g.last_success
    assert g.is_grasped(w)


def test_grasp_nothing_fails():
    g = _GripperSim(0.08)
    g.command(cmd(3, G.GRASP, 0.05, force=40.0), 0.08)
    w = run(g, 0.08, 2.0)
    assert not g.busy and g.last_seq == 3 and not g.last_success
    assert not g.is_grasped(w)


def test_same_seq_is_ignored_and_stop_finishes():
    g = _GripperSim(0.08)
    g.command(cmd(4, G.MOVE, 0.0), 0.08)
    run(g, 0.08, 0.1)
    g.command(cmd(4, G.MOVE, 0.0), 0.07)  # resent (LCM) -> no restart
    assert g.busy
    g.command(cmd(5, G.STOP), 0.07)
    assert not g.busy and g.last_seq == 5 and g.last_success


def test_homing_opens_fully():
    g = _GripperSim(0.02)
    g.command(cmd(6, G.HOMING), 0.02)
    w = run(g, 0.02, 1.5)
    assert abs(w - MAX_WIDTH) < 1e-3 and g.last_success


def test_force_and_speed_clamped_to_2f85_range():
    """The 2F-85 squeezes with 20-235 N and moves at 20-150 mm/s."""
    g = _GripperSim(0.08)
    g.command(cmd(7, G.GRASP, 0.03, speed=1.0, force=500.0), 0.08)
    assert g.force == 235.0 and g.speed == 0.15
    g.command(cmd(8, G.GRASP, 0.03, speed=0.001, force=1.0), 0.08)
    assert g.force == 20.0 and g.speed == 0.02
