"""Collect scripted-expert demonstrations into an HDF5 file.

Every episode uses its own seed (seed_start, seed_start + 1, ...), so the dataset can be
recreated exactly. All episodes are saved, each with a success flag.

By default the executed movement gets DART noise (std 0.3 in action units): the expert
still succeeds in 100% of episodes, and the data then also shows how to correct small
deviations. --noise-std 0 records clean demonstrations.

Usage (repository root, .venv activated):
    python scripts/collect_expert.py                  # 100 demos -> data/expert_100_noise0.3.hdf5
    python scripts/collect_expert.py --noise-std 0    # clean   -> data/expert_100_noise0.0.hdf5
    python scripts/collect_expert.py --videos 3       # also save 3 videos to results/videos/
"""

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gesture_bc.data.hdf5_io import DemoWriter
from gesture_bc.expert.scripted_expert import ExpertConfig, ScriptedExpert
from gesture_bc.rollout import run_episode
from gesture_bc.sim.pick_place_env import PickPlaceConfig, PickPlaceEnv
from gesture_bc.sim.rendering import VideoWriter, draw_text_lines
from gesture_bc.utils.config import config_to_dict, load_config
from gesture_bc.utils.provenance import git_commit

VIDEO_DIR = Path("results/videos")


def video_observer(video: VideoWriter, expert: ScriptedExpert, seed: int):
    """Render every step into the video, labeled with the expert's current phase."""

    def observer(env, obs, info):
        seconds = info["step"] / env.config.control_hz
        lines = [
            f"scripted expert   seed {seed}",
            f"t {seconds:4.1f} s   phase: {expert.phase.value}",
        ]
        if info["is_success"]:
            lines.append("SUCCESS")
        video.add(draw_text_lines(env.render(), lines))

    return observer


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect scripted-expert demonstrations.")
    parser.add_argument("--env-config", type=Path, default=Path("configs/env/pick_place.yaml"))
    parser.add_argument("--expert-config", type=Path, default=Path("configs/expert/scripted.yaml"))
    parser.add_argument("--num-demos", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0, help="seed of the first episode")
    parser.add_argument(
        "--noise-std", type=float, default=0.3, help="DART noise on executed movement (0 = off)"
    )
    parser.add_argument("--out", type=Path, help="default: data/expert_<N>_noise<std>.hdf5")
    parser.add_argument("--videos", type=int, default=0, help="save videos of the first N demos")
    args = parser.parse_args()
    out = args.out or Path(f"data/expert_{args.num_demos}_noise{args.noise_std:.1f}.hdf5")

    env_config = load_config(args.env_config, PickPlaceConfig)
    expert_config = load_config(args.expert_config, ExpertConfig)
    env = PickPlaceEnv(env_config)
    expert = ScriptedExpert(expert_config, env_config.max_step_m)
    seeds = range(args.seed_start, args.seed_start + args.num_demos)
    metadata = {
        "source": "scripted_expert",
        "git_commit": git_commit(),
        "env_config": config_to_dict(env_config),
        "expert_config": config_to_dict(expert_config),
        "noise_std": args.noise_std,
        "seeds": [seeds.start, seeds.stop],  # half-open range [start, stop)
    }

    lengths, successes = [], []
    start = time.perf_counter()
    with DemoWriter(out, metadata) as writer:
        for i, seed in enumerate(tqdm(seeds, desc="demos", unit="demo")):
            video = None
            observer = None
            if i < args.videos:
                video = VideoWriter(VIDEO_DIR / f"expert_seed{seed:03d}.mp4", env_config.control_hz)
                observer = video_observer(video, expert, seed)
            episode = run_episode(env, expert, seed, args.noise_std, observer)
            if video is not None:
                video.close()
            writer.add(episode)
            lengths.append(episode.num_steps)
            successes.append(episode.success)
    elapsed = time.perf_counter() - start

    lengths = np.array(lengths)
    failed = [seed for seed, ok in zip(seeds, successes, strict=True) if not ok]
    print(f"\nSaved {len(lengths)} demos to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(
        f"  success: {sum(successes)}/{len(successes)}"
        + (f"  failed seeds: {failed}" if failed else "")
    )
    mean_seconds = lengths.mean() / env_config.control_hz
    print(
        f"  length: {lengths.mean():.1f} +- {lengths.std():.1f} steps "
        f"(min {lengths.min()}, max {lengths.max()}), i.e. {mean_seconds:.1f} s"
    )
    print(f"  time: {elapsed:.1f} s")
    if args.videos:
        print(f"  videos: {VIDEO_DIR}/expert_seed*.mp4")
    env.close()


if __name__ == "__main__":
    main()
