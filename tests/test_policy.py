import numpy as np
import pytest
import torch

from gesture_bc.data.normalization import Normalizer
from gesture_bc.models.mlp import MLP, count_parameters
from gesture_bc.models.policy import NeuralPolicy, load_policy, save_checkpoint
from gesture_bc.sim.pick_place_env import PickPlaceEnv


@pytest.fixture(scope="module")
def env() -> PickPlaceEnv:
    return PickPlaceEnv()


def identity_normalizer(size: int) -> Normalizer:
    return Normalizer(np.zeros(size, np.float32), np.ones(size, np.float32))


def test_mlp_shapes_and_size():
    model = MLP(obs_dim=19)
    assert model(torch.zeros(5, 19)).shape == (5, 4)  # 3 movement numbers + gripper logit
    # 19*256+256 + 2*(256*256+256) + 256*4+4: the size reported in the thesis
    assert count_parameters(model) == 137_732


def test_policy_returns_a_valid_action(env):
    torch.manual_seed(0)
    obs, _ = env.reset(seed=0)
    policy = NeuralPolicy(
        MLP(19), env.obs_keys, identity_normalizer(19), identity_normalizer(3), torch.device("cpu")
    )
    action = policy.act(obs)
    assert action.shape == (4,) and action.dtype == np.float32
    assert np.all(np.abs(action[:3]) <= 1.0)
    assert action[3] in (-1.0, 1.0)


def test_checkpoint_round_trip_gives_identical_actions(env, tmp_path):
    torch.manual_seed(1)
    model = MLP(19, hidden_sizes=(32, 32), dropout=0.0)
    obs_norm = Normalizer.fit(np.random.default_rng(0).normal(size=(50, 19)).astype(np.float32))
    move_norm = Normalizer.fit(np.random.default_rng(1).normal(size=(50, 3)).astype(np.float32))
    original = NeuralPolicy(model, env.obs_keys, obs_norm, move_norm, torch.device("cpu"))
    path = tmp_path / "policy.pt"
    config = {"obs_dim": 19, "hidden_sizes": [32, 32], "dropout": 0.0}
    save_checkpoint(path, model, config, env.obs_keys, obs_norm, move_norm, {"note": "test"})

    restored = load_policy(path, "cpu")
    obs, _ = env.reset(seed=3)
    np.testing.assert_array_equal(original.act(obs), restored.act(obs))
