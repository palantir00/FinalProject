import pytest

from gesture_bc.eval.evaluate import (
    EVAL_SEED_START,
    MISSED_GRASP,
    NOT_PLACED,
    STILL_HOLDING,
    SUCCESS,
    classify_outcome,
    evaluate_policy,
    measure_latency,
    summarize,
)
from gesture_bc.expert.scripted_expert import ExpertConfig, ScriptedExpert
from gesture_bc.sim.pick_place_env import PickPlaceEnv


@pytest.mark.parametrize(
    ("success", "ever_grasped", "holding_at_end", "expected"),
    [
        (True, True, False, SUCCESS),
        (False, False, False, MISSED_GRASP),
        (False, True, True, STILL_HOLDING),
        (False, True, False, NOT_PLACED),
    ],
)
def test_outcome_classification(success, ever_grasped, holding_at_end, expected):
    assert classify_outcome(success, ever_grasped, holding_at_end) == expected


def test_evaluation_seeds_are_far_from_training_seeds():
    assert EVAL_SEED_START >= 10_000  # datasets use seeds 0, 1, 2, ...


def test_expert_evaluation_and_summary():
    env = PickPlaceEnv()
    expert = ScriptedExpert(ExpertConfig(), env.config.max_step_m)
    results = evaluate_policy(env, expert, range(EVAL_SEED_START, EVAL_SEED_START + 3))
    assert all(r.success and r.outcome == SUCCESS for r in results)
    assert all(r.cube_to_target_cm < 3.0 for r in results)  # inside the 3 cm success radius
    summary = summarize(results)
    assert summary["success_rate"] == 1.0 and summary["n_success"] == 3


def test_latency_measurement():
    env = PickPlaceEnv()
    obs, _ = env.reset(seed=0)
    timing = measure_latency(ScriptedExpert(ExpertConfig(), env.config.max_step_m), obs, 20)
    assert 0.0 < timing["median_ms"] <= timing["p95_ms"]
