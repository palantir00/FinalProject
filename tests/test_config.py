from pathlib import Path

import pytest

from gesture_bc.sim.pick_place_env import PickPlaceConfig
from gesture_bc.utils.config import load_config

ENV_CONFIG = Path(__file__).parents[1] / "configs" / "env" / "pick_place.yaml"


def test_env_yaml_matches_the_code_defaults():
    # The YAML documents the defaults; this test keeps the two from drifting apart.
    assert load_config(ENV_CONFIG, PickPlaceConfig) == PickPlaceConfig()


def test_missing_keys_keep_defaults(tmp_path):
    path = tmp_path / "env.yaml"
    path.write_text("max_episode_steps: 50\n", encoding="utf-8")
    config = load_config(path, PickPlaceConfig)
    assert config.max_episode_steps == 50
    assert config.control_hz == PickPlaceConfig().control_hz


def test_unknown_key_is_an_error(tmp_path):
    path = tmp_path / "env.yaml"
    path.write_text("max_episod_steps: 50\n", encoding="utf-8")  # typo
    with pytest.raises(TypeError):
        load_config(path, PickPlaceConfig)
