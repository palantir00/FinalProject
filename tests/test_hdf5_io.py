import h5py
import numpy as np
import pytest

from gesture_bc.data.episode import Episode
from gesture_bc.data.hdf5_io import DemoWriter, read_demos, read_metadata


def make_episode(seed: int, steps: int, success: bool) -> Episode:
    rng = np.random.default_rng(seed)
    return Episode(
        obs={
            "tcp_pos": rng.normal(size=(steps, 3)).astype(np.float32),
            "gripper_opening": rng.uniform(size=(steps, 1)).astype(np.float32),
        },
        actions=rng.uniform(-1, 1, size=(steps, 4)).astype(np.float32),
        executed_actions=rng.uniform(-1, 1, size=(steps, 4)).astype(np.float32),
        tcp_targets=rng.normal(size=(steps, 3)).astype(np.float32),
        qpos=rng.normal(size=(steps, 16)),
        qvel=rng.normal(size=(steps, 15)),
        seed=seed,
        success=success,
    )


def test_episodes_round_trip(tmp_path):
    path = tmp_path / "demos.hdf5"
    written = [make_episode(0, steps=7, success=True), make_episode(1, steps=4, success=False)]
    with DemoWriter(path, {"source": "test"}) as writer:
        for episode in written:
            writer.add(episode)

    loaded = read_demos(path)
    assert len(loaded) == 2
    for original, copy in zip(written, loaded, strict=True):
        assert (copy.seed, copy.success, copy.num_steps) == (
            original.seed,
            original.success,
            original.num_steps,
        )
        assert list(copy.obs) == list(original.obs)  # same keys, same order
        for key in original.obs:
            np.testing.assert_array_equal(copy.obs[key], original.obs[key])
        np.testing.assert_array_equal(copy.actions, original.actions)
        np.testing.assert_array_equal(copy.executed_actions, original.executed_actions)
        np.testing.assert_array_equal(copy.tcp_targets, original.tcp_targets)
        np.testing.assert_array_equal(copy.qpos, original.qpos)
        np.testing.assert_array_equal(copy.qvel, original.qvel)


def test_metadata_round_trip(tmp_path):
    path = tmp_path / "demos.hdf5"
    metadata = {
        "source": "scripted_expert",
        "noise_std": 0.25,
        "seeds": [0, 100],
        "env_config": {"control_hz": 20, "workspace_low": [0.3, -0.3, 0.012]},
    }
    with DemoWriter(path, metadata) as writer:
        writer.add(make_episode(0, steps=3, success=True))
        writer.add(make_episode(1, steps=3, success=False))

    loaded = read_metadata(path)
    for key, value in metadata.items():
        assert loaded[key] == value
    assert loaded["num_demos"] == 2 and loaded["num_success"] == 1
    assert loaded["obs_keys"] == ["tcp_pos", "gripper_opening"]
    assert loaded["format_version"] == 1


def test_reading_a_foreign_file_fails_clearly(tmp_path):
    path = tmp_path / "other.hdf5"
    with h5py.File(path, "w") as file:
        file.create_group("data")
    with pytest.raises(ValueError, match="not a"):
        read_demos(path)
