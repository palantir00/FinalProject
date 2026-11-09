"""Turn recorded episodes into training arrays for behavior cloning."""

from dataclasses import dataclass

import numpy as np

from gesture_bc.data.episode import Episode


@dataclass
class BCArrays:
    """Samples for supervised learning: one row = one control step of one episode."""

    obs: np.ndarray  # (N, obs_dim) float32, entries concatenated in obs_keys order
    actions: np.ndarray  # (N, 4) float32, the demonstrator's actions
    episode_index: np.ndarray  # (N,) which episode a row came from

    def __len__(self) -> int:
        return len(self.actions)


def episodes_to_arrays(episodes: list[Episode], obs_keys: list[str]) -> BCArrays:
    obs = [np.concatenate([episode.obs[key] for key in obs_keys], axis=1) for episode in episodes]
    actions = [episode.actions for episode in episodes]
    index = [np.full(episode.num_steps, i) for i, episode in enumerate(episodes)]
    return BCArrays(
        obs=np.concatenate(obs).astype(np.float32),
        actions=np.concatenate(actions).astype(np.float32),
        episode_index=np.concatenate(index),
    )


def split_episodes(
    episodes: list[Episode], val_fraction: float, seed: int
) -> tuple[list[Episode], list[Episode]]:
    """Random split into training and validation episodes.

    The split is by EPISODE, not by step: consecutive steps of one episode are nearly
    identical, so splitting steps would put near-copies of training samples into the
    validation set and make the validation loss look better than it really is.
    """
    order = np.random.default_rng(seed).permutation(len(episodes))
    num_val = max(1, round(len(episodes) * val_fraction))
    val_ids = set(order[:num_val].tolist())
    train = [episode for i, episode in enumerate(episodes) if i not in val_ids]
    val = [episode for i, episode in enumerate(episodes) if i in val_ids]
    return train, val
