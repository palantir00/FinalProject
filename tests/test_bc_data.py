import numpy as np

from gesture_bc.data.dataset import episodes_to_arrays, split_episodes
from gesture_bc.data.episode import Episode
from gesture_bc.data.normalization import Normalizer


def make_episode(seed: int, steps: int) -> Episode:
    rng = np.random.default_rng(seed)
    return Episode(
        obs={
            "a": np.full((steps, 2), seed, np.float32),
            "b": np.full((steps, 1), -seed, np.float32),
        },
        actions=rng.uniform(-1, 1, (steps, 4)).astype(np.float32),
        executed_actions=np.zeros((steps, 4), np.float32),
        tcp_targets=np.zeros((steps, 3), np.float32),
        qpos=np.zeros((steps, 16)),
        qvel=np.zeros((steps, 15)),
        seed=seed,
        success=True,
    )


def test_arrays_concatenate_episodes_and_keys_in_order():
    episodes = [make_episode(1, 3), make_episode(2, 4)]
    arrays = episodes_to_arrays(episodes, ["a", "b"])
    assert arrays.obs.shape == (7, 3) and arrays.actions.shape == (7, 4)
    np.testing.assert_array_equal(arrays.obs[0], [1, 1, -1])  # key "a" first, then "b"
    np.testing.assert_array_equal(arrays.obs[-1], [2, 2, -2])
    np.testing.assert_array_equal(arrays.episode_index, [0, 0, 0, 1, 1, 1, 1])


def test_split_is_by_episode_and_reproducible():
    episodes = [make_episode(seed, 5) for seed in range(20)]
    train, val = split_episodes(episodes, val_fraction=0.1, seed=0)
    assert len(train) == 18 and len(val) == 2
    assert not {e.seed for e in train} & {e.seed for e in val}  # no episode in both
    train_again, val_again = split_episodes(episodes, val_fraction=0.1, seed=0)
    assert [e.seed for e in val] == [e.seed for e in val_again]
    _, val_other = split_episodes(episodes, val_fraction=0.1, seed=1)
    assert [e.seed for e in val] != [e.seed for e in val_other]


def test_normalizer_round_trip():
    data = np.random.default_rng(0).normal(5.0, 3.0, size=(1000, 4)).astype(np.float32)
    normalizer = Normalizer.fit(data)
    normalized = normalizer.normalize(data)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-3)
    np.testing.assert_allclose(normalizer.denormalize(normalized), data, atol=1e-4)


def test_constant_column_does_not_explode():
    data = np.column_stack([np.linspace(0, 1, 50), np.full(50, 0.025)]).astype(np.float32)
    normalized = Normalizer.fit(data).normalize(data)
    assert np.all(np.isfinite(normalized))
    # ~0 up to float32 rounding of the mean, divided by the 1e-3 floor of the std
    np.testing.assert_allclose(normalized[:, 1], 0.0, atol=1e-4)


def test_normalizer_survives_saving():
    normalizer = Normalizer.fit(np.random.default_rng(1).normal(size=(10, 3)).astype(np.float32))
    restored = Normalizer.from_dict(normalizer.to_dict())
    np.testing.assert_array_equal(restored.mean, normalizer.mean)
    np.testing.assert_array_equal(restored.std, normalizer.std)
