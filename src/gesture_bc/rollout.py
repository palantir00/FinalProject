"""Run one episode of a policy in the environment and record it.

Used for collecting demonstrations (scripted expert now, teleoperation later) and for
evaluating learned policies, so all of them are recorded in exactly the same way.
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np

from gesture_bc.data.episode import Episode
from gesture_bc.sim.pick_place_env import PickPlaceEnv

# Second number of the noise generator's seed. Seeding the noise with the episode seed
# alone would give exactly the same random stream as the environment's generator
# (both are numpy PCG64 seeded with the same number), so noise and spawn positions would be
# correlated. A pair (seed, 1) gives an independent stream that is still reproducible.
NOISE_STREAM = 1


class Policy(Protocol):
    """Anything that can act in the environment: scripted expert, human, neural network."""

    def reset(self) -> None: ...

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...


# Called after reset and after every step, e.g. to render video frames.
Observer = Callable[[PickPlaceEnv, dict[str, np.ndarray], dict], None]


def run_episode(
    env: PickPlaceEnv,
    policy: Policy,
    seed: int,
    noise_std: float = 0.0,
    observer: Observer | None = None,
) -> Episode:
    """Reset the environment with ``seed`` and let ``policy`` act until the episode ends.

    noise_std > 0 adds Gaussian noise to the executed movement (not to the recorded action):
    the demonstrator then also shows how to recover from small deviations (DART, Laskey et
    al. 2017), which makes behavior cloning more robust.
    """
    obs, info = env.reset(seed=seed)
    policy.reset()
    noise_rng = np.random.default_rng((seed, NOISE_STREAM))
    if observer is not None:
        observer(env, obs, info)

    observations: dict[str, list[np.ndarray]] = {key: [] for key in obs}
    actions, executed_actions, tcp_targets, qpos, qvel = [], [], [], [], []
    while True:
        state_qpos, state_qvel = env.data.qpos.copy(), env.data.qvel.copy()
        # float32 = the precision stored in the file; the environment executes exactly the
        # stored numbers, so a replay from the file is bit-for-bit identical.
        action = np.asarray(policy.act(obs), dtype=np.float32)
        executed = action.copy()
        if noise_std > 0:
            noisy = executed[:3] + noise_rng.normal(0.0, noise_std, 3)
            executed[:3] = np.clip(noisy, -1.0, 1.0)
        next_obs, _, terminated, truncated, info = env.step(executed)

        for key, value in obs.items():
            observations[key].append(value)
        actions.append(action)
        executed_actions.append(executed)
        tcp_targets.append(info["tcp_target"])
        qpos.append(state_qpos)
        qvel.append(state_qvel)
        if observer is not None:
            observer(env, next_obs, info)

        obs = next_obs
        if terminated or truncated:
            break

    return Episode(
        obs={key: np.array(values, dtype=np.float32) for key, values in observations.items()},
        actions=np.array(actions, dtype=np.float32),
        executed_actions=np.array(executed_actions, dtype=np.float32),
        tcp_targets=np.array(tcp_targets, dtype=np.float32),
        qpos=np.array(qpos),
        qvel=np.array(qvel),
        seed=seed,
        success=bool(info["is_success"]),
    )


def replay_episode(env: PickPlaceEnv, episode: Episode) -> float:
    """Re-simulate an episode from its seed and executed actions.

    Returns the largest difference between the recorded and the re-simulated observations.
    The simulation is deterministic, so this should be 0: a check that the data file really
    describes what happened (and that nothing in the pipeline adds hidden randomness).
    """
    obs, _ = env.reset(seed=episode.seed)
    largest_difference = 0.0
    for t, action in enumerate(episode.executed_actions):
        for key, values in episode.obs.items():
            difference = np.max(np.abs(obs[key] - values[t]))
            largest_difference = max(largest_difference, float(difference))
        obs, *_ = env.step(action)
    return largest_difference
