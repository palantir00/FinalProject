import numpy as np
import pytest

from gesture_bc.expert.scripted_expert import ExpertConfig, ScriptedExpert
from gesture_bc.rollout import NOISE_STREAM, replay_episode, run_episode
from gesture_bc.sim.pick_place_env import PickPlaceEnv


@pytest.fixture(scope="module")
def env() -> PickPlaceEnv:
    return PickPlaceEnv()


@pytest.fixture
def expert(env) -> ScriptedExpert:
    return ScriptedExpert(ExpertConfig(), env.config.max_step_m)


def test_same_seed_gives_an_identical_episode(env, expert):
    first = run_episode(env, expert, seed=4, noise_std=0.3)
    second = run_episode(env, expert, seed=4, noise_std=0.3)
    np.testing.assert_array_equal(first.executed_actions, second.executed_actions)
    np.testing.assert_array_equal(first.qpos, second.qpos)


def test_episode_arrays_are_aligned(env, expert):
    episode = run_episode(env, expert, seed=0)
    steps = episode.num_steps
    assert all(len(values) == steps for values in episode.obs.values())
    assert len(episode.executed_actions) == len(episode.tcp_targets) == len(episode.qpos) == steps
    assert episode.actions.dtype == np.float32


def test_noise_changes_the_executed_but_not_the_recorded_action(env, expert):
    clean = run_episode(env, expert, seed=3, noise_std=0.0)
    noisy = run_episode(env, expert, seed=3, noise_std=0.3)
    np.testing.assert_array_equal(clean.actions, clean.executed_actions)
    assert not np.allclose(noisy.actions[:, :3], noisy.executed_actions[:, :3])
    np.testing.assert_array_equal(noisy.actions[:, 3], noisy.executed_actions[:, 3])  # gripper
    # Same seed, same scene: noise must not change where the cube and target are.
    np.testing.assert_array_equal(clean.obs["cube_pos"][0], noisy.obs["cube_pos"][0])
    np.testing.assert_array_equal(clean.obs["target_pos"][0], noisy.obs["target_pos"][0])


def test_noise_stream_is_independent_of_the_environment_stream(env):
    """Why NOISE_STREAM exists: a generator seeded with the bare seed repeats the env's numbers."""
    obs, _ = env.reset(seed=5)
    low, high = env.config.cube_spawn_low, env.config.cube_spawn_high
    same_seed_draw = np.random.default_rng(5).uniform(low, high)
    np.testing.assert_allclose(same_seed_draw, obs["cube_pos"][:2], atol=1e-6)  # the trap
    noise_stream_draw = np.random.default_rng((5, NOISE_STREAM)).uniform(low, high)
    assert not np.allclose(noise_stream_draw, obs["cube_pos"][:2])


@pytest.mark.parametrize("noise_std", [0.0, 0.3])
def test_replay_reproduces_the_episode_exactly(env, expert, noise_std):
    episode = run_episode(env, expert, seed=6, noise_std=noise_std)
    assert replay_episode(env, episode) == 0.0
