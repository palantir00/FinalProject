import numpy as np
import pytest

from gesture_bc.expert.scripted_expert import ExpertConfig, Phase, ScriptedExpert
from gesture_bc.rollout import run_episode
from gesture_bc.sim.pick_place_env import PickPlaceConfig, PickPlaceEnv


@pytest.fixture(scope="module")
def env() -> PickPlaceEnv:
    return PickPlaceEnv()


def make_expert(env: PickPlaceEnv) -> ScriptedExpert:
    return ScriptedExpert(ExpertConfig(), env.config.max_step_m)


@pytest.mark.parametrize("seed", range(5))
def test_expert_solves_the_task(env, seed):
    episode = run_episode(env, make_expert(env), seed)
    assert episode.success
    assert episode.num_steps < 150  # about 70 steps (3.5 s) is typical


def test_actions_stay_in_the_action_range(env):
    episode = run_episode(env, make_expert(env), seed=0)
    assert np.all(np.abs(episode.actions) <= 1.0)
    assert set(np.unique(episode.actions[:, 3])) <= {-1.0, 1.0}  # gripper: open or close


def test_expert_goes_through_all_phases_in_order(env):
    expert = make_expert(env)
    phases = []
    run_episode(env, expert, seed=1, observer=lambda e, o, i: phases.append(expert.phase))
    visited = [phase for i, phase in enumerate(phases) if i == 0 or phase != phases[i - 1]]
    expected = [Phase.REACH, Phase.DESCEND, Phase.GRASP, Phase.LIFT, Phase.CARRY, Phase.LOWER]
    assert visited[: len(expected)] == expected
    assert Phase.RELEASE in visited


@pytest.mark.parametrize("seed", range(3))
def test_expert_follows_a_pushed_cube(seed):
    pushed_env = PickPlaceEnv(PickPlaceConfig(perturb_cube=True, perturb_step_range=(5, 15)))
    pushes = []
    episode = run_episode(
        pushed_env,
        make_expert(pushed_env),
        seed,
        observer=lambda e, o, info: pushes.append(info["perturbed"]),
    )
    assert pushes[-1], "the cube was not pushed in this episode"
    assert episode.success


def test_expert_starts_again_after_losing_the_cube(env):
    """Teleport the cube away while it is carried: the expert must notice and pick it up again."""
    expert = make_expert(env)
    teleported = []

    def snatch_cube(env, obs, info):
        if expert.phase == Phase.CARRY and not teleported:
            env._place_cube(np.array([0.45, 0.15]))
            env.data.ctrl[env._gripper_actuator_id] = 255.0
            teleported.append(True)

    episode = run_episode(env, expert, seed=2, observer=snatch_cube)
    assert teleported and episode.success


def test_expert_with_dart_noise_still_succeeds(env):
    successes = [
        run_episode(env, make_expert(env), seed, noise_std=0.3).success for seed in range(5)
    ]
    assert all(successes)
