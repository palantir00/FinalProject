"""Behavior cloning: supervised training of a policy network on demonstrations.

Loss per sample:
    MSE(predicted movement, expert movement)                [3 numbers, normalized]
  + gripper_loss_weight * BCE(gripper logit, expert closed?) [binary decision]

The gripper is a yes/no decision, so it is learned as classification (binary cross-entropy)
instead of regressing the numbers -1 / +1.
"""

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from gesture_bc.data.dataset import episodes_to_arrays, split_episodes
from gesture_bc.data.hdf5_io import read_demos, read_metadata
from gesture_bc.data.normalization import Normalizer
from gesture_bc.models.mlp import MLP, MOVE_DIM, count_parameters
from gesture_bc.models.policy import save_checkpoint
from gesture_bc.utils.config import config_to_dict
from gesture_bc.utils.device import resolve_device
from gesture_bc.utils.provenance import git_commit
from gesture_bc.utils.seeding import seed_everything


@dataclass
class TrainConfig:
    """configs/train/mlp.yaml can override every value."""

    dataset: str = "data/expert_100_noise0.3.hdf5"
    max_demos: int = 0  # use only the first N successful demos (0 = all), for data-size studies
    val_fraction: float = 0.1  # share of EPISODES used for validation
    split_seed: int = 0  # same train/val split for every training seed
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # AdamW: pulls weights towards 0 (regularization)
    batch_size: int = 256
    max_epochs: int = 400
    patience: int = 40  # early stopping: stop after this many epochs without improvement
    gripper_loss_weight: float = 1.0
    device: str = "auto"
    seed: int = 0  # training seed: network initialization and batch order


@dataclass
class TrainResult:
    run_dir: Path
    best_epoch: int
    best_val_loss: float
    epochs_run: int
    num_parameters: int
    num_train_samples: int
    seconds: float


def _losses(model, obs, move, closed, gripper_weight):
    output = model(obs)
    move_loss = nn.functional.mse_loss(output[:, :MOVE_DIM], move)
    gripper_loss = nn.functional.binary_cross_entropy_with_logits(output[:, MOVE_DIM], closed)
    gripper_accuracy = ((output[:, MOVE_DIM] > 0).float() == closed).float().mean()
    return move_loss + gripper_weight * gripper_loss, move_loss, gripper_loss, gripper_accuracy


def train(config: TrainConfig, run_dir: str | Path) -> TrainResult:
    """Train one policy; writes config.yaml, metrics.csv, best.pt and summary.json to run_dir."""
    start = time.perf_counter()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = resolve_device(config.device)

    # 1) Data: successful demos only, split by episode, normalization from the train split.
    metadata = read_metadata(config.dataset)
    obs_keys = metadata["obs_keys"]
    episodes = [episode for episode in read_demos(config.dataset) if episode.success]
    if config.max_demos:
        episodes = episodes[: config.max_demos]
    train_episodes, val_episodes = split_episodes(episodes, config.val_fraction, config.split_seed)
    train_data = episodes_to_arrays(train_episodes, obs_keys)
    val_data = episodes_to_arrays(val_episodes, obs_keys)
    obs_normalizer = Normalizer.fit(train_data.obs)
    move_normalizer = Normalizer.fit(train_data.actions[:, :MOVE_DIM])

    def to_tensors(data):
        obs = torch.from_numpy(obs_normalizer.normalize(data.obs)).to(device)
        move = torch.from_numpy(move_normalizer.normalize(data.actions[:, :MOVE_DIM])).to(device)
        closed = torch.from_numpy((data.actions[:, MOVE_DIM] > 0).astype(np.float32)).to(device)
        return obs, move, closed

    # The whole dataset is a few MB: keep it on the GPU and draw batches there directly.
    train_obs, train_move, train_closed = to_tensors(train_data)
    val_obs, val_move, val_closed = to_tensors(val_data)

    # 2) Model and optimizer.
    obs_dim = train_obs.shape[1]
    model_config = {
        "type": "mlp",
        "obs_dim": obs_dim,
        "hidden_sizes": list(config.hidden_sizes),
        "dropout": config.dropout,
    }
    model = MLP(obs_dim, config.hidden_sizes, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoint_metadata = {
        "git_commit": git_commit(),
        "dataset": config.dataset,
        "dataset_git_commit": metadata.get("git_commit"),
        "train_config": config_to_dict(config),
        "num_train_episodes": len(train_episodes),
        "num_val_episodes": len(val_episodes),
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config_to_dict(config), file, sort_keys=False)

    # 3) Training loop with early stopping on the validation loss.
    best_val, best_epoch, epochs_without_improvement = float("inf"), -1, 0
    num_train = len(train_obs)
    columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_move_mse",
        "val_gripper_bce",
        "val_gripper_acc",
    ]
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=columns)
        writer.writeheader()
        for epoch in range(config.max_epochs):
            model.train()  # dropout on
            order = torch.randperm(num_train, device=device)
            epoch_loss = 0.0
            for first in range(0, num_train, config.batch_size):
                batch = order[first : first + config.batch_size]
                loss, *_ = _losses(
                    model,
                    train_obs[batch],
                    train_move[batch],
                    train_closed[batch],
                    config.gripper_loss_weight,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)

            model.eval()  # dropout off
            with torch.no_grad():
                val_loss, move_mse, gripper_bce, gripper_acc = _losses(
                    model, val_obs, val_move, val_closed, config.gripper_loss_weight
                )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": round(epoch_loss / num_train, 6),
                    "val_loss": round(val_loss.item(), 6),
                    "val_move_mse": round(move_mse.item(), 6),
                    "val_gripper_bce": round(gripper_bce.item(), 6),
                    "val_gripper_acc": round(gripper_acc.item(), 6),
                }
            )

            if val_loss.item() < best_val:
                best_val, best_epoch, epochs_without_improvement = val_loss.item(), epoch, 0
                save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    model_config,
                    obs_keys,
                    obs_normalizer,
                    move_normalizer,
                    {**checkpoint_metadata, "epoch": epoch, "val_loss": best_val},
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

    result = TrainResult(
        run_dir=run_dir,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        epochs_run=epoch + 1,
        num_parameters=count_parameters(model),
        num_train_samples=num_train,
        seconds=time.perf_counter() - start,
    )
    summary = {**result.__dict__, "run_dir": str(run_dir), "device": str(device)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result
