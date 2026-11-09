import csv

import pytest

from gesture_bc.data.hdf5_io import DemoWriter
from gesture_bc.expert.scripted_expert import ExpertConfig, ScriptedExpert
from gesture_bc.rollout import run_episode
from gesture_bc.sim.pick_place_env import PickPlaceEnv
from gesture_bc.train.trainer import TrainConfig, train


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    """Six expert demonstrations in a temporary HDF5 file."""
    path = tmp_path_factory.mktemp("data") / "demos.hdf5"
    env = PickPlaceEnv()
    expert = ScriptedExpert(ExpertConfig(), env.config.max_step_m)
    with DemoWriter(path, {"source": "test"}) as writer:
        for seed in range(6):
            writer.add(run_episode(env, expert, seed))
    return path


def test_training_learns_and_writes_its_files(small_dataset, tmp_path):
    config = TrainConfig(
        dataset=str(small_dataset),
        val_fraction=0.2,
        hidden_sizes=(64, 64),
        batch_size=32,  # ~10 updates per epoch on this tiny dataset
        max_epochs=40,
        patience=40,
        device="cpu",
    )
    result = train(config, tmp_path / "run")

    for name in ["best.pt", "config.yaml", "metrics.csv", "summary.json"]:
        assert (tmp_path / "run" / name).is_file()
    with open(tmp_path / "run" / "metrics.csv", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 40
    first_val = float(rows[0]["val_loss"])
    assert result.best_val_loss < 0.5 * first_val  # the network did learn something
    assert float(rows[result.best_epoch]["val_gripper_acc"]) > 0.9
