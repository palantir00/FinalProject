"""Phase 4 (feasibility test): train and evaluate every model with one command.

Experiments (every model is evaluated on the same 100 unseen scenes, seeds 100000-100099):
    expert     the scripted expert itself (reference)
    main       MLP trained on 100 demos, DART data vs clean data, 3 training seeds each
    data size  MLP trained on 10 / 25 / 50 demos of DART data, 3 training seeds each

Usage (repository root, .venv activated; needs data/expert_100_noise0.3.hdf5 and
data/expert_100_noise0.0.hdf5 from scripts/collect_expert.py):
    python scripts/run_feasibility.py            # everything (about 20-40 min)
    python scripts/run_feasibility.py --quick    # smoke test: 1 seed, 10 episodes

Outputs:
    runs/feasibility/<name>/            checkpoint, config, training log of every model
    results/feasibility/episodes.csv    one row per evaluation episode
    results/feasibility/models.csv      one row per model
    results/feasibility/table.md        success rate, mean +- std over training seeds
    results/feasibility/latency.csv     time of one policy step on CPU and GPU
"""

import argparse
import csv
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch

from gesture_bc.eval.evaluate import EVAL_SEED_START, evaluate_policy, measure_latency, summarize
from gesture_bc.expert.scripted_expert import ExpertConfig, ScriptedExpert
from gesture_bc.models.policy import load_policy
from gesture_bc.sim.pick_place_env import PickPlaceConfig, PickPlaceEnv
from gesture_bc.train.trainer import TrainConfig, train
from gesture_bc.utils.config import load_config

DATASETS = {
    "DART": "data/expert_100_noise0.3.hdf5",
    "clean": "data/expert_100_noise0.0.hdf5",
}
RUNS_DIR = Path("runs/feasibility")
RESULTS_DIR = Path("results/feasibility")


def plan(quick: bool) -> list[dict]:
    """The list of models to train: (data source, number of demos, training seed)."""
    seeds = [0] if quick else [0, 1, 2]
    models = []
    for data in ["DART", "clean"]:
        models += [{"data": data, "demos": 100, "seed": s} for s in seeds]
    for demos in [] if quick else [10, 25, 50]:
        models += [{"data": "DART", "demos": demos, "seed": s} for s in seeds]
    return models


def mean_std(values: list[float]) -> str:
    return f"{np.mean(values) * 100:.0f} ± {np.std(values) * 100:.0f}%"


def write_csv(path: Path, rows: list[dict]) -> None:
    """Rows may have different keys (the expert has no training columns): use all of them."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 feasibility experiment.")
    parser.add_argument("--quick", action="store_true", help="1 seed, 10 episodes")
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/mlp.yaml"))
    args = parser.parse_args()

    episodes = 10 if args.quick else 100
    eval_seeds = range(EVAL_SEED_START, EVAL_SEED_START + episodes)
    base_config = load_config(args.train_config, TrainConfig)
    env = PickPlaceEnv(load_config(Path("configs/env/pick_place.yaml"), PickPlaceConfig))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_episodes, model_rows = [], []
    start = time.perf_counter()

    # 1) The expert on the same scenes: the reference every model is compared to.
    expert = ScriptedExpert(ExpertConfig(), env.config.max_step_m)
    expert_results = evaluate_policy(env, expert, eval_seeds)
    summary = summarize(expert_results)
    model_rows.append({"name": "expert", "data": "-", "demos": "-", "seed": "-", **summary})
    all_episodes += [{"name": "expert", **dataclasses.asdict(r)} for r in expert_results]
    print(f"expert: success {summary['success_rate']:.0%}", flush=True)

    # 2) Train and evaluate every MLP.
    for model in plan(args.quick):
        name = f"mlp_{model['data']}_n{model['demos']}_seed{model['seed']}"
        config = dataclasses.replace(
            base_config,
            dataset=DATASETS[model["data"]],
            max_demos=model["demos"],
            seed=model["seed"],
        )
        result = train(config, RUNS_DIR / name)
        policy = load_policy(result.run_dir / "best.pt", device="cpu")
        results = evaluate_policy(env, policy, eval_seeds)
        summary = summarize(results)
        model_rows.append(
            {
                "name": name,
                **model,
                **summary,
                "best_val_loss": round(result.best_val_loss, 5),
                "epochs": result.epochs_run,
                "train_seconds": round(result.seconds, 1),
                "parameters": result.num_parameters,
            }
        )
        all_episodes += [{"name": name, **dataclasses.asdict(r)} for r in results]
        print(
            f"{name}: success {summary['success_rate']:.0%}  "
            f"(trained {result.seconds:.0f} s, {result.epochs_run} epochs)  "
            f"[{(time.perf_counter() - start) / 60:.1f} min total]",
            flush=True,
        )

    # 3) Latency of one policy step, CPU vs GPU (same network, same observation).
    checkpoint = RUNS_DIR / "mlp_DART_n100_seed0" / "best.pt"
    obs, _ = env.reset(seed=EVAL_SEED_START)
    latency_rows = []
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        timing = measure_latency(load_policy(checkpoint, device), obs)
        latency_rows.append({"device": device, **{k: round(v, 4) for k, v in timing.items()}})
        print(f"latency on {device}: {timing['median_ms']:.3f} ms (p95 {timing['p95_ms']:.3f})")

    # 4) Write results.
    write_csv(RESULTS_DIR / "episodes.csv", all_episodes)
    write_csv(RESULTS_DIR / "models.csv", model_rows)
    write_csv(RESULTS_DIR / "latency.csv", latency_rows)

    lines = [
        f"| Policy | Training data | Success rate ({episodes} unseen scenes, mean ± std) |",
        "|---|---|---|",
        f"| scripted expert | - | {model_rows[0]['success_rate'] * 100:.0f}% |",
    ]
    groups = {}
    for row in model_rows[1:]:
        groups.setdefault((row["data"], row["demos"]), []).append(row["success_rate"])
    for (data, demos), rates in groups.items():
        seeds = f"{len(rates)} seed" + ("s" if len(rates) > 1 else "")
        lines.append(f"| MLP | {demos} demos, {data} ({seeds}) | {mean_std(rates)} |")
    (RESULTS_DIR / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nDone in {(time.perf_counter() - start) / 60:.1f} min. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
