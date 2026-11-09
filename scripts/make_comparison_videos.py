"""Side-by-side videos: scripted expert (left) vs trained network (right), same unseen scene.

Usage (repository root, .venv activated):
    python scripts/make_comparison_videos.py
    python scripts/make_comparison_videos.py --checkpoint runs/.../best.pt --seeds 100000 100005

Writes results/videos/compare_seed<seed>.mp4 and, for the first seed, a still image strip
results/feasibility/compare_seed<seed>.png (for the report).
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from gesture_bc.eval.evaluate import EVAL_SEED_START
from gesture_bc.expert.scripted_expert import ExpertConfig, ScriptedExpert
from gesture_bc.models.policy import load_policy
from gesture_bc.rollout import run_episode
from gesture_bc.sim.pick_place_env import PickPlaceConfig, PickPlaceEnv
from gesture_bc.sim.rendering import VideoWriter, draw_text_lines
from gesture_bc.utils.config import load_config

DEFAULT_CHECKPOINT = Path("runs/feasibility/mlp_DART_n100_seed0/best.pt")


def record(env: PickPlaceEnv, policy, seed: int, label: str, camera: str) -> list[np.ndarray]:
    frames = []

    def observer(env_, obs, info):
        status = "   SUCCESS" if info["is_success"] else ""
        lines = [label, f"t {info['step'] / env_.config.control_hz:4.1f} s{status}"]
        frames.append(draw_text_lines(env_.render_camera(camera), lines))

    run_episode(env, policy, seed, observer=observer)
    return frames


def side_by_side(left: list[np.ndarray], right: list[np.ndarray]) -> list[np.ndarray]:
    """Pair frames in time; the shorter episode holds its last frame until the other ends."""
    length = max(len(left), len(right))
    left = left + [left[-1]] * (length - len(left))
    right = right + [right[-1]] * (length - len(right))
    return [np.hstack([a, b]) for a, b in zip(left, right, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert vs network comparison videos.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[EVAL_SEED_START + i for i in range(3)]
    )
    parser.add_argument("--camera", default="angled")
    parser.add_argument("--label", default="learned MLP (100 demos)")
    args = parser.parse_args()

    env = PickPlaceEnv(load_config(Path("configs/env/pick_place.yaml"), PickPlaceConfig))
    expert = ScriptedExpert(ExpertConfig(), env.config.max_step_m)
    network = load_policy(args.checkpoint, "cpu")
    for i, seed in enumerate(args.seeds):
        left = record(env, expert, seed, "scripted expert", args.camera)
        right = record(env, network, seed, args.label, args.camera)
        frames = side_by_side(left, right)
        path = Path("results/videos") / f"compare_seed{seed}.mp4"
        with VideoWriter(path, fps=env.config.control_hz) as video:
            for frame in frames:
                video.add(frame)
        print(f"{path}: expert {len(left)} steps, network {len(right)} steps")
        if i == 0:
            picks = np.linspace(0, len(frames) - 1, 4).astype(int)
            strip = np.vstack([frames[k][::2, ::2] for k in picks])
            still = Path("results/feasibility") / f"compare_seed{seed}.png"
            still.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(still, strip)
            print(f"{still}: still frames for the report")
    env.close()


if __name__ == "__main__":
    main()
