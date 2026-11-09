"""Live demo for presentations: the trained network solves new random scenes, one by one.

One command, no arguments needed (uses the network trained by run_feasibility.py):
    python scripts/demo.py
    python scripts/demo.py --checkpoint runs/.../best.pt --speed 0.5

Keys (click the window first):
    E        switch between the network and the scripted expert (same scene, restarts it)
    N        next scene now
    P        pause / resume
    ESC / Q  quit
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from gesture_bc.expert.scripted_expert import ExpertConfig, ScriptedExpert
from gesture_bc.models.policy import load_policy
from gesture_bc.sim.pick_place_env import PickPlaceConfig, PickPlaceEnv
from gesture_bc.sim.rendering import draw_text_lines
from gesture_bc.ui.keys import collect_keys, window_closed
from gesture_bc.utils.config import load_config

DEFAULT_CHECKPOINT = Path("runs/feasibility/mlp_DART_n100_seed0/best.pt")
DEMO_SEED_START = 200_000  # new scenes: neither in the training data nor in the evaluation
WINDOW = "gesture_bc: live demo"
PAUSE_AFTER_EPISODE_S = 1.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Live demo of the trained policy.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    parser.add_argument("--seed", type=int, default=DEMO_SEED_START)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(
            f"No trained network at {args.checkpoint}. Run: python scripts/run_feasibility.py"
        )

    env = PickPlaceEnv(load_config(Path("configs/env/pick_place.yaml"), PickPlaceConfig))
    policies = {
        "learned network (MLP)": load_policy(args.checkpoint, "cpu"),
        "scripted expert": ScriptedExpert(ExpertConfig(), env.config.max_step_m),
    }
    names = list(policies)
    current = 0
    seed = args.seed
    scenes, successes = 0, 0
    period_s = 1.0 / (env.config.control_hz * args.speed)

    def start_episode():
        policies[names[current]].reset()
        return env.reset(seed=seed)

    obs, info = start_episode()
    paused, keys, last_toggle = False, set(), 0.0
    print(__doc__)
    while True:
        tick = time.perf_counter()
        if "esc" in keys or "q" in keys:
            break
        if "p" in keys and tick - last_toggle > 0.5:
            paused, last_toggle = not paused, tick
        if "e" in keys and tick - last_toggle > 0.5:
            current, last_toggle = (current + 1) % len(names), tick
            obs, info = start_episode()
        if "n" in keys:
            seed += 1
            obs, info = start_episode()

        if not paused:
            obs, _, terminated, truncated, info = env.step(policies[names[current]].act(obs))
            if terminated or truncated:
                scenes += 1
                successes += int(info["is_success"])

        result = ""
        if info["is_success"]:
            result = "SUCCESS"
        elif info["step"] >= env.config.max_episode_steps:
            result = "time out"
        lines = [
            f"policy: {names[current]}   (E = switch)",
            f"scene {seed}   t {info['step'] / env.config.control_hz:4.1f} s   {result}",
            f"solved so far: {successes} / {scenes}" + ("   PAUSED" if paused else ""),
        ]
        color = (80, 255, 80) if info["is_success"] else (255, 255, 255)
        frame = draw_text_lines(env.render_camera("angled"), lines, color=color)
        cv2.imshow(WINDOW, cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))

        if result and not paused:  # show the result for a moment, then the next scene
            keys = collect_keys(PAUSE_AFTER_EPISODE_S)
            seed += 1
            obs, info = start_episode()
        else:
            keys = collect_keys(period_s - (time.perf_counter() - tick))
        if window_closed(WINDOW):
            break

    cv2.destroyAllWindows()
    env.close()
    print(f"Solved {successes} of {scenes} scenes.")


if __name__ == "__main__":
    main()
