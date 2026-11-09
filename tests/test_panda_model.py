"""Sanity checks of the Franka Panda model downloaded from MuJoCo Menagerie."""

import mujoco
import numpy as np
import pytest

from gesture_bc.sim.paths import require_panda_scene

# Gripper actuator command range (Menagerie remaps 0..0.04 m finger travel to 0..255).
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(require_panda_scene()))


def reset_to_home(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    home = model.key("home")
    mujoco.mj_resetDataKeyframe(model, data, home.id)
    data.ctrl[:] = home.ctrl
    return data


def simulate(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def test_model_structure(model):
    # 7 arm joints + 2 finger joints; 7 arm actuators + 1 actuator driving both fingers.
    assert model.nq == 9
    assert model.nu == 8
    assert model.key("home").ctrl[7] == GRIPPER_OPEN


def test_position_servos_hold_home_pose(model):
    data = reset_to_home(model)
    simulate(model, data, seconds=1.0)
    # PD servos fighting gravity leave a small steady-state error (~0.007 rad measured).
    home_arm = model.key("home").qpos[:7]
    assert np.max(np.abs(data.qpos[:7] - home_arm)) < 0.02


def test_gripper_closes_on_command(model):
    data = reset_to_home(model)
    data.ctrl[7] = GRIPPER_CLOSED
    simulate(model, data, seconds=1.0)
    finger_opening = data.qpos[7] + data.qpos[8]
    assert finger_opening < 0.002
