"""Train one behavior-cloning policy.

Usage (repository root, .venv activated):
    python scripts/train.py                                   # configs/train/mlp.yaml
    python scripts/train.py --seed 1 --dataset data/expert_100_noise0.0.hdf5
    python scripts/train.py --run-dir runs/my_experiment
"""

import argparse
import dataclasses
from pathlib import Path

from gesture_bc.train.trainer import TrainConfig, train
from gesture_bc.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a behavior-cloning policy.")
    parser.add_argument("--config", type=Path, default=Path("configs/train/mlp.yaml"))
    parser.add_argument("--dataset", help="override the dataset from the config")
    parser.add_argument("--seed", type=int, help="override the training seed")
    parser.add_argument("--run-dir", type=Path, help="default: runs/<dataset>_seed<seed>")
    args = parser.parse_args()

    config = load_config(args.config, TrainConfig)
    overrides = {"dataset": args.dataset, "seed": args.seed}
    config = dataclasses.replace(config, **{k: v for k, v in overrides.items() if v is not None})
    run_dir = args.run_dir or Path("runs") / f"mlp_{Path(config.dataset).stem}_seed{config.seed}"

    result = train(config, run_dir)
    print(f"Run directory: {result.run_dir}")
    print(
        f"  parameters: {result.num_parameters:,}, training samples: {result.num_train_samples:,}"
    )
    print(
        f"  best validation loss {result.best_val_loss:.4f} at epoch {result.best_epoch} "
        f"({result.epochs_run} epochs, {result.seconds:.1f} s)"
    )


if __name__ == "__main__":
    main()
