"""Hand features and the hand -> robot mapping, on synthetic landmarks (no camera needed)."""

import numpy as np
import pytest

from gesture_bc.control import CLOSE, OPEN
from gesture_bc.teleop.hand_features import (
    INDEX_TIP,
    PALM,
    THUMB_TIP,
    HandFeatures,
    extract_features,
)
from gesture_bc.teleop.hand_teleop import HandTeleop, HandTeleopConfig

WIDTH, HEIGHT = 640, 480
WORKSPACE_LOW, WORKSPACE_HIGH = (0.30, -0.30, 0.012), (0.70, 0.30, 0.35)
MAX_STEP_M = 0.015
START_TCP = np.array([0.45, 0.0, 0.30])


def open_hand(center=(0.5, 0.5), scale=1.0, pinched=False) -> np.ndarray:
    """Rough (21, 3) normalized landmarks of an open hand facing the camera.

    center = palm center (normalized image coordinates); scale > 1 = hand closer to the camera.
    """
    rng = np.random.default_rng(0)
    landmarks = np.zeros((21, 3))
    landmarks[:, 0] = rng.uniform(-0.1, 0.1, 21)  # fingers somewhere around the palm
    landmarks[:, 1] = rng.uniform(-0.2, 0.0, 21)
    landmarks[0] = [0.0, 0.1, 0.0]  # wrist
    landmarks[5] = [-0.05, -0.05, 0.0]  # index base
    landmarks[9] = [0.0, -0.06, 0.0]  # middle base
    landmarks[13] = [0.04, -0.05, 0.0]  # ring base
    landmarks[17] = [0.07, -0.03, 0.0]  # little-finger base
    landmarks[THUMB_TIP] = [-0.15, 0.0, 0.0]
    landmarks[INDEX_TIP] = [-0.14, 0.0, 0.0] if pinched else [-0.06, -0.25, 0.0]
    landmarks[:, :2] -= landmarks[PALM, :2].mean(axis=0)  # palm center at the origin
    landmarks[:, :2] *= scale
    landmarks[:, 0] += center[0]
    landmarks[:, 1] += center[1]
    return landmarks


def features(**kwargs) -> HandFeatures:
    return extract_features(open_hand(**kwargs), WIDTH, HEIGHT)


def make_teleop(**config) -> HandTeleop:
    return HandTeleop(HandTeleopConfig(**config), WORKSPACE_LOW, WORKSPACE_HIGH, MAX_STEP_M)


def obs_at(tcp) -> dict[str, np.ndarray]:
    return {"tcp_pos": np.array(tcp, dtype=float), "tcp_vel": np.zeros(3)}


# --- features ---------------------------------------------------------------------------------


def test_position_is_the_palm_center_in_image_widths():
    f = features(center=(0.5, 0.4))
    np.testing.assert_allclose(f.position, [0.5, 0.4 * HEIGHT / WIDTH])


def test_closer_hand_has_a_larger_palm_and_the_same_pinch():
    near, far = features(scale=1.5), features(scale=1.0)
    assert near.size == pytest.approx(1.5 * far.size)
    assert near.pinch == pytest.approx(far.pinch)  # relative to the palm size


def test_pinch_is_small_when_thumb_touches_index():
    assert features(pinched=True).pinch < 0.2 < 0.8 < features().pinch


def test_moving_the_fingers_does_not_move_the_palm_position():
    np.testing.assert_allclose(features(pinched=True).position, features().position)


# --- mapping ----------------------------------------------------------------------------------


def test_calibration_pose_maps_to_the_calibration_tcp():
    teleop = make_teleop()
    teleop.calibrate(features(), START_TCP)
    np.testing.assert_allclose(teleop.goal_for(features()), START_TCP)


@pytest.mark.parametrize(
    ("hand", "axis", "sign"),
    [
        ({"center": (0.6, 0.5)}, 1, +1),  # hand to the right -> TCP to the right (+y)
        ({"center": (0.5, 0.4)}, 2, +1),  # hand up (image y smaller) -> TCP up (+z)
        ({"scale": 1.2}, 0, +1),  # hand closer (larger) -> TCP towards the viewer (+x)
    ],
)
def test_hand_motion_moves_the_goal_in_the_matching_direction(hand, axis, sign):
    teleop = make_teleop()
    teleop.calibrate(features(), START_TCP)
    offset = teleop.goal_for(features(**hand)) - START_TCP
    assert sign * offset[axis] > 0.01
    assert np.count_nonzero(np.abs(offset) > 1e-9) == 1  # only that axis moves


def test_gains_scale_the_motion():
    teleop = make_teleop(lateral_gain=2.0)
    teleop.calibrate(features(), START_TCP)
    offset = teleop.goal_for(features(center=(0.55, 0.5))) - START_TCP
    assert offset[1] == pytest.approx(2.0 * 0.05)


def test_goal_stays_inside_the_workspace():
    teleop = make_teleop()
    teleop.calibrate(features(), START_TCP)
    goal = teleop.goal_for(features(center=(1.5, -1.0), scale=3.0))
    assert np.all(goal >= WORKSPACE_LOW) and np.all(goal <= WORKSPACE_HIGH)


def test_recalibration_keeps_the_robot_where_it_is():
    teleop = make_teleop()
    teleop.calibrate(features(), START_TCP)
    robot_now = np.array([0.55, 0.1, 0.10])
    teleop.calibrate(features(center=(0.3, 0.7)), robot_now)  # hand moved somewhere else
    np.testing.assert_allclose(teleop.goal_for(features(center=(0.3, 0.7))), robot_now)


# --- actions ----------------------------------------------------------------------------------


def test_robot_holds_still_before_calibration_and_when_the_hand_is_lost():
    teleop = make_teleop()
    action = teleop.act(obs_at(START_TCP), features(center=(0.8, 0.5)))  # not calibrated
    np.testing.assert_allclose(action, [0.0, 0.0, 0.0, OPEN])

    teleop.calibrate(features(), START_TCP)
    teleop.act(obs_at(START_TCP), features(center=(0.6, 0.5)))
    goal = teleop.goal.copy()
    teleop.act(obs_at(START_TCP), None)  # hand lost: keep going to the last goal
    np.testing.assert_allclose(teleop.goal, goal)


def test_action_moves_towards_the_hand_goal_within_the_action_range():
    teleop = make_teleop()
    teleop.calibrate(features(), START_TCP)
    action = teleop.act(obs_at(START_TCP), features(center=(0.7, 0.5)))  # far to the right
    assert action[1] == pytest.approx(1.0)  # saturated: full speed to the right
    assert np.all(np.abs(action[:3]) <= 1.0)


def test_gripper_uses_hysteresis():
    teleop = make_teleop(close_pinch=0.3, open_pinch=0.45)
    teleop.calibrate(features(), START_TCP)
    for pinch, expected in [(0.6, OPEN), (0.35, OPEN), (0.2, CLOSE), (0.4, CLOSE), (0.5, OPEN)]:
        f = HandFeatures(position=features().position, size=features().size, pinch=pinch)
        assert teleop.act(obs_at(START_TCP), f)[3] == expected, pinch


def test_reset_opens_the_gripper_but_keeps_the_calibration():
    teleop = make_teleop()
    teleop.calibrate(features(), START_TCP)
    teleop.act(obs_at(START_TCP), features(pinched=True))
    assert teleop.gripper_closed
    teleop.reset()
    assert not teleop.gripper_closed and teleop.goal is None and teleop.calibrated
