import itertools

import numpy as np
import pytest

from gesture_bc.sim.ik_controller import orientation_error
from gesture_bc.sim.pick_place_env import (
    OBS_KEYS,
    PickPlaceConfig,
    PickPlaceEnv,
    flatten_observation,
)

OPEN, CLOSE = -1.0, 1.0


@pytest.fixture(scope="module")
def env() -> PickPlaceEnv:
    return PickPlaceEnv()


def move_to(env, goal, grip, max_steps=150, tolerance=0.004):
    """Simple proportional controller in TCP space (used to test the environment)."""
    for _ in range(max_steps):
        direction = np.clip((goal - env._tcp_target) / env.config.max_step_m, -1.0, 1.0)
        obs, _, terminated, truncated, info = env.step(np.append(direction, grip))
        if np.linalg.norm(goal - obs["tcp_pos"]) < tolerance or terminated:
            break
    return obs, info


def hold(env, grip, steps):
    for _ in range(steps):
        obs, _, terminated, _, info = env.step([0.0, 0.0, 0.0, grip])
        if terminated:
            break
    return obs, info


# --- reset and randomization --------------------------------------------------------------------


def test_reset_is_deterministic_for_a_seed(env):
    first, _ = env.reset(seed=5)
    again, _ = env.reset(seed=5)
    other, _ = env.reset(seed=6)
    for key in OBS_KEYS:
        np.testing.assert_array_equal(first[key], again[key])
    assert not np.allclose(first["cube_pos"], other["cube_pos"])


def test_spawn_positions_follow_the_config(env):
    cfg = env.config
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        cube_xy, target_xy = obs["cube_pos"][:2], obs["target_pos"][:2]
        assert np.all(cube_xy >= cfg.cube_spawn_low) and np.all(cube_xy <= cfg.cube_spawn_high)
        assert np.all(target_xy >= cfg.target_spawn_low)
        assert np.all(target_xy <= cfg.target_spawn_high)
        assert np.linalg.norm(cube_xy - target_xy) >= cfg.min_cube_target_distance


def test_reset_options_fix_cube_and_target(env):
    obs, _ = env.reset(seed=0, options={"cube_xy": [0.5, 0.1], "target_xy": [0.6, -0.1]})
    np.testing.assert_allclose(obs["cube_pos"], [0.5, 0.1, env.cube_half_size], atol=1e-3)
    np.testing.assert_allclose(obs["target_pos"], [0.6, -0.1, env.cube_half_size], atol=1e-6)


def test_same_seed_and_actions_give_the_same_trajectory(env):
    actions = np.random.default_rng(0).uniform(-1, 1, size=(30, 4))

    def rollout():
        env.reset(seed=3)
        return np.array([env.step(a)[0]["tcp_pos"] for a in actions])

    np.testing.assert_array_equal(rollout(), rollout())


# --- observations -------------------------------------------------------------------------------


def test_observation_layout(env):
    obs, _ = env.reset(seed=0)
    assert list(obs) == OBS_KEYS
    for key, value in obs.items():
        assert value.dtype == np.float32
        assert value.shape == env.observation_space[key].shape
    np.testing.assert_allclose(obs["tcp_to_cube"], obs["cube_pos"] - obs["tcp_pos"], atol=1e-6)
    np.testing.assert_allclose(
        obs["cube_to_target"], obs["target_pos"] - obs["cube_pos"], atol=1e-6
    )
    assert flatten_observation(obs, OBS_KEYS).shape == (16,)


def test_observation_noise_only_when_enabled():
    noisy_env = PickPlaceEnv(PickPlaceConfig(obs_noise_std=0.01))
    noisy, _ = noisy_env.reset(seed=0)
    true_cube = noisy_env.data.xpos[noisy_env._cube_body_id]
    error = np.linalg.norm(noisy["cube_pos"] - true_cube)
    assert 0.0 < error < 0.06  # noisy, but within ~6 sigma


def test_velocity_observation_is_optional():
    env_with_velocity = PickPlaceEnv(PickPlaceConfig(include_velocity=True))
    env_with_velocity.reset(seed=0)
    obs, *_ = env_with_velocity.step([1.0, 0.0, 0.0, OPEN])
    assert obs["tcp_vel"].shape == (3,)
    assert obs["tcp_vel"][0] > 0.05  # moving forward [m/s]


# --- control ------------------------------------------------------------------------------------

# All 8 corners of the workspace box: every combination of (low or high) for x, y and z.
WORKSPACE_CORNERS = list(
    itertools.product(
        *zip(PickPlaceConfig.workspace_low, PickPlaceConfig.workspace_high, strict=True)
    )
)


@pytest.mark.parametrize("corner", WORKSPACE_CORNERS)
def test_every_workspace_corner_is_reachable_pointing_down(env, corner):
    target = np.array(corner)
    q = env.ik.solve(env._start_arm_qpos, target, env._tcp_quat, max_iterations=300)
    reached, rotation = env.ik.forward_kinematics(q)
    assert np.linalg.norm(reached - target) < 2e-3  # within 2 mm
    assert np.linalg.norm(orientation_error(env._tcp_quat, rotation.flatten())) < 0.01


def test_ik_does_not_change_the_simulation_state(env):
    env.reset(seed=0)
    qpos_before = env.data.qpos.copy()
    env.ik.solve(env._start_arm_qpos, np.array([0.6, 0.1, 0.1]), env._tcp_quat)
    np.testing.assert_array_equal(env.data.qpos, qpos_before)


def test_action_moves_tcp_in_the_commanded_direction(env):
    start, _ = env.reset(seed=0)
    for _ in range(10):
        obs, *_ = env.step([0.0, 1.0, 0.0, OPEN])  # +y, full speed: 10 * 1.5 cm = 15 cm
    obs, _ = hold(env, OPEN, 10)  # let the arm catch up
    moved = obs["tcp_pos"] - start["tcp_pos"]
    assert moved[1] == pytest.approx(0.15, abs=0.01)
    assert abs(moved[0]) < 0.005 and abs(moved[2]) < 0.005


def test_tcp_cannot_leave_the_workspace(env):
    env.reset(seed=0)
    for _ in range(60):  # push down much further than the table
        obs, _, _, _, info = env.step([0.0, 0.0, -1.0, OPEN])
    assert info["tcp_target"][2] == pytest.approx(env.config.workspace_low[2])
    assert obs["tcp_pos"][2] > env.config.workspace_low[2] - 0.003


def test_gripper_closes_and_opens(env):
    env.reset(seed=0)
    obs, _ = hold(env, CLOSE, 10)
    assert obs["gripper_opening"][0] < 0.005
    obs, _ = hold(env, OPEN, 10)
    assert obs["gripper_opening"][0] > 0.075


def test_episode_is_truncated_after_max_steps():
    short_env = PickPlaceEnv(PickPlaceConfig(max_episode_steps=5))
    short_env.reset(seed=0)
    for step in range(1, 6):
        _, _, terminated, truncated, _ = short_env.step(np.zeros(4))
        assert truncated == (step == 5) and not terminated


# --- task ---------------------------------------------------------------------------------------


def test_scripted_pick_and_place_succeeds(env):
    """End-to-end physics check: the grasp holds the cube while carrying it."""
    obs, _ = env.reset(seed=3)
    cube, target = obs["cube_pos"].copy(), obs["target_pos"].copy()

    move_to(env, cube + [0.0, 0.0, 0.10], OPEN)
    move_to(env, cube, OPEN)
    obs, info = hold(env, CLOSE, 10)
    assert info["is_grasping"]

    move_to(env, cube + [0.0, 0.0, 0.15], CLOSE)
    obs, info = hold(env, CLOSE, 20)  # 1 s in the air
    assert info["is_grasping"] and info["cube_was_lifted"]
    assert np.linalg.norm(obs["tcp_to_cube"]) < 0.015  # the cube moves with the gripper

    move_to(env, target + [0.0, 0.0, 0.10], CLOSE)
    move_to(env, target + [0.0, 0.0, 0.005], CLOSE)
    obs, info = hold(env, OPEN, 10)
    assert info["is_success"]


def test_success_requires_cube_in_zone_and_released(env):
    _, info = env.reset(seed=0, options={"cube_xy": [0.5, 0.0], "target_xy": [0.5, 0.0]})
    assert info["is_success"]  # cube on the target, gripper open

    _, info = env.reset(seed=0, options={"cube_xy": [0.5, 0.0], "target_xy": [0.5, 0.1]})
    assert not info["is_success"]  # 10 cm away from the target

    # Close the (empty) gripper above the placed cube. The first steps still count as success
    # (fingers not closed yet), so keep stepping instead of stopping at termination.
    env.reset(seed=0, options={"cube_xy": [0.5, 0.0], "target_xy": [0.5, 0.0]})
    for _ in range(10):
        _, _, _, _, info = env.step([0.0, 0.0, 0.0, CLOSE])
    assert not info["is_success"]  # gripper closed: the cube is not "released"


def test_cube_is_perturbed_once_before_the_grasp():
    config = PickPlaceConfig(perturb_cube=True, perturb_step_range=(1, 1))
    perturbed_env = PickPlaceEnv(config)
    start, _ = perturbed_env.reset(seed=0, options={"cube_xy": [0.52, 0.0]})
    obs, _, _, _, info = perturbed_env.step(np.zeros(4))
    shift = np.linalg.norm(obs["cube_pos"][:2] - start["cube_pos"][:2])
    assert info["perturbed"]
    assert (
        config.perturb_distance_range[0] - 1e-3 <= shift <= config.perturb_distance_range[1] + 1e-3
    )

    after_push = obs["cube_pos"].copy()
    obs, _ = hold(perturbed_env, OPEN, 5)
    np.testing.assert_allclose(obs["cube_pos"], after_push, atol=1e-3)  # only one push


@pytest.mark.render
def test_render_returns_an_rgb_image(env):
    env.reset(seed=0)
    for camera in ["front", "angled", "top"]:
        image = env.render_camera(camera)
        assert image.shape == (env.config.render_height, env.config.render_width, 3)
        assert image.dtype == np.uint8 and image.std() > 0  # not a blank image
