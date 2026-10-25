"""Robotiq 2F-85 driver (robot_platform/hardware/franka_fr3/robotiq.py): the
Modbus frames, and move / grasp / stop / homing against a simulated gripper
behind the same protocol (FakeRobotiqTransport)."""

import threading
import time

import pytest

from fr3.data_type.basic_types.GripperCmdData import GripperCmdData as G
from fr3.robot_platform.hardware.franka_fr3.robotiq import (
    FakeRobotiqTransport,
    RobotiqGripper,
    crc16,
    read_frame,
    write_frame,
)


def test_frames_match_the_manual():
    # Examples from the 2F-85 manual (Modbus RTU section).
    assert write_frame(bytes(6)).hex() == "091003e8000306000000000000" + "7330"
    assert crc16(bytes.fromhex("090307D00001")) == bytes.fromhex("85CF")
    assert crc16(bytes.fromhex("091003E8000306090000FFFFFF")) == bytes.fromhex("4229")
    assert read_frame()[:6] == bytes.fromhex("090407D00003")


@pytest.fixture
def gripper():
    g = RobotiqGripper(transport=FakeRobotiqTransport())
    assert g.homing()
    return g


def test_homing_activates_and_calibrates(gripper):
    assert gripper.status()["sta"] == 3
    assert (gripper.pos_open, gripper.pos_closed) == (3, 230)
    w, w_max, held = gripper.state()
    assert w == pytest.approx(w_max, abs=1e-3) and not held


def test_move_reaches_width(gripper):
    assert gripper.action(G.MOVE, 0.05, 0.15, 0, 0, 0)
    assert gripper.state()[0] == pytest.approx(0.05, abs=1e-3)


def test_homing_with_an_object_keeps_the_calibration():
    g = RobotiqGripper(transport=FakeRobotiqTransport(object_width=0.04))
    assert g.homing()
    assert (g.pos_open, g.pos_closed) == (3, 230)


def test_grasp_object_in_window_succeeds_and_holds():
    t = FakeRobotiqTransport()
    g = RobotiqGripper(transport=t)
    g.homing()
    t.object_width = 0.04
    assert g.action(G.GRASP, 0.04, 0.15, 100.0, 0.005, 0.005)
    w, _, held = g.state()
    assert w == pytest.approx(0.04, abs=1e-3) and held
    # force 100 N -> register (100 - 20) / 215 * 255
    assert t.writes[-1][5] == round(255 * 80 / 215)
    assert g.action(G.MOVE, 0.085, 0.15, 0, 0, 0)  # release
    assert not g.state()[2]


def test_grasp_wrong_width_or_nothing_fails():
    t = FakeRobotiqTransport()
    g = RobotiqGripper(transport=t)
    g.homing()
    t.object_width = 0.06
    assert not g.action(G.GRASP, 0.03, 0.15, 50.0, 0.005, 0.005)  # stopped at 6 cm
    assert not g.state()[2]
    g = RobotiqGripper(transport=FakeRobotiqTransport())
    g.homing()
    assert not g.action(G.GRASP, 0.03, 0.15, 50.0, 0.005, 0.005)  # closed on nothing


def test_stop_interrupts_a_move():
    g = RobotiqGripper(transport=FakeRobotiqTransport())
    g.homing()
    result = {}
    th = threading.Thread(target=lambda: result.setdefault("ok", g.move(0.0, 0.02)))
    th.start()
    time.sleep(0.3)
    assert g.stop()
    th.join(timeout=2.0)
    assert not th.is_alive()
    w = g.state()[0]
    assert 0.0 < w < 0.085  # stopped part-way
