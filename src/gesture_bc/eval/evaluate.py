"""Run a policy on many new scenes and report how often (and how) it succeeds.

Evaluation seeds start at EVAL_SEED_START, far away from the training seeds (0, 1, 2, ...),
so a policy is never evaluated on a scene that was in its training data.
"""

import csv
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gesture_bc.rollout import Policy, run_episode
from gesture_bc.sim.pick_place_env import PickPlaceEnv

EVAL_SEED_START = 100_000

# Outcome of one episode (a coarse failure analysis; Phase 8 refines it).
SUCCESS = "success"
MISSED_GRASP = "missed_grasp"  # never held the cube with both fingers
STILL_HOLDING = "still_holding"  # time ran out while holding the cube
NOT_PLACED = "not_placed"  # grasped it at some point, but it ended outside the target zone
OUTCOMES = [SUCCESS, MISSED_GRASP, STILL_HOLDING, NOT_PLACED]


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    outcome: str
    steps: int
    cube_to_target_cm: float  # distance in the table plane at the end of the episode


def classify_outcome(success: bool, ever_grasped: bool, holding_at_end: bool) -> str:
    if success:
        return SUCCESS
    if not ever_grasped:
        return MISSED_GRASP
    if holding_at_end:
        return STILL_HOLDING
    return NOT_PLACED


def evaluate_policy(
    env: PickPlaceEnv,
    policy: Policy,
    seeds: range | list[int],
    make_observer=None,
) -> list[EpisodeResult]:
    """One episode per seed. make_observer(seed) may return an extra observer (e.g. video)."""
    results = []
    for seed in seeds:
        flags = {"ever_grasped": False, "holding": False, "cube_to_target": 0.0}
        extra = make_observer(seed) if make_observer else None

        def observer(env_, obs, info, flags=flags, extra=extra):
            flags["ever_grasped"] |= info["is_grasping"]
            flags["holding"] = info["is_grasping"]
            flags["cube_to_target"] = float(np.linalg.norm(obs["cube_to_target"][:2]))
            if extra is not None:
                extra(env_, obs, info)

        episode = run_episode(env, policy, seed, observer=observer)
        outcome = classify_outcome(episode.success, flags["ever_grasped"], flags["holding"])
        results.append(
            EpisodeResult(
                seed=seed,
                success=episode.success,
                outcome=outcome,
                steps=episode.num_steps,
                cube_to_target_cm=round(flags["cube_to_target"] * 100, 2),
            )
        )
    return results


def summarize(results: list[EpisodeResult]) -> dict:
    successes = [r for r in results if r.success]
    counts = Counter(r.outcome for r in results)
    return {
        "episodes": len(results),
        "success_rate": len(successes) / len(results),
        "mean_steps_when_successful": float(np.mean([r.steps for r in successes]))
        if successes
        else float("nan"),
        **{f"n_{outcome}": counts.get(outcome, 0) for outcome in OUTCOMES},
    }


def write_results_csv(path: str | Path, results: list[EpisodeResult], extra: dict | None = None):
    """One row per episode; ``extra`` columns (e.g. model name, seed) are added to every row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**(extra or {}), **asdict(result)} for result in results]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure_latency(policy: Policy, obs: dict[str, np.ndarray], repeats: int = 1000) -> dict:
    """Time of one policy.act() call in milliseconds: median and 95th percentile.

    The first calls are excluded (warm-up: memory allocation, GPU kernel loading).
    """
    for _ in range(50):
        policy.act(obs)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        policy.act(obs)  # includes the copy back to the CPU, so GPU work has finished
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
