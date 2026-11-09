"""Evaluate a trained policy (or the scripted expert) on new, unseen scenes.

Usage (repository root, .venv activated):
    python scripts/evaluate.py --checkpoint runs/mlp_expert_100_noise0.3_seed0/best.pt
    python scripts/evaluate.py --expert --episodes 100
    python scripts/evaluate.py --checkpoint ... --episodes 20 --out results/eval.csv
"""

import argparse
import time
from pathlib import Path

from gesture_bc.eval.evaluate import (
    EVAL_SEED_START,
    evaluate_policy,
    summarize,
    write_results_csv,
)
from gesture_bc.expert.scripted_expert import ExpertConfig, ScriptedExpert
from gesture_bc.models.policy import load_policy
from gesture_bc.sim.pick_place_env import PickPlaceConfig, PickPlaceEnv
from gesture_bc.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy on unseen scenes.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="trained network (best.pt)")
    source.add_argument("--expert", action="store_true", help="evaluate the scripted expert")
    parser.add_argument("--env-config", type=Path, default=Path("configs/env/pick_place.yaml"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=EVAL_SEED_START)
    parser.add_argument("--device", default="cpu", help="cpu is faster for one sample at a time")
    parser.add_argument("--out", type=Path, help="CSV with one row per episode")
    args = parser.parse_args()

    env = PickPlaceEnv(load_config(args.env_config, PickPlaceConfig))
    if args.expert:
        policy, name = ScriptedExpert(ExpertConfig(), env.config.max_step_m), "scripted expert"
    else:
        policy, name = load_policy(args.checkpoint, args.device), str(args.checkpoint)

    seeds = range(args.seed_start, args.seed_start + args.episodes)
    start = time.perf_counter()
    results = evaluate_policy(env, policy, seeds)
    summary = summarize(results)
    print(
        f"{name}: success {summary['success_rate']:.0%} on {len(results)} new scenes "
        f"(seeds {seeds.start}-{seeds.stop - 1}), {time.perf_counter() - start:.0f} s"
    )
    print(f"  outcomes: { {k: v for k, v in summary.items() if k.startswith('n_')} }")
    print(
        f"  mean length of successful episodes: {summary['mean_steps_when_successful']:.1f} steps"
    )
    if args.out:
        write_results_csv(args.out, results, {"policy": name})
        print(f"  per-episode results: {args.out}")


if __name__ == "__main__":
    main()
