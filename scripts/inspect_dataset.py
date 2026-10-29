"""Summarize a demonstration file and check that its episodes replay exactly.

Usage (repository root, .venv activated):
    python scripts/inspect_dataset.py data/expert_100.hdf5
    python scripts/inspect_dataset.py data/expert_100.hdf5 --replay 10 --video 2
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from gesture_bc.data.hdf5_io import read_demos, read_metadata
from gesture_bc.rollout import replay_episode
from gesture_bc.sim.pick_place_env import PickPlaceConfig, PickPlaceEnv
from gesture_bc.sim.rendering import VideoWriter, draw_text_lines
from gesture_bc.utils.config import config_from_dict
from gesture_bc.utils.provenance import git_commit

VIDEO_DIR = Path("results/videos")


def print_summary(path: Path, metadata: dict, episodes: list) -> None:
    lengths = np.array([episode.num_steps for episode in episodes])
    successes = sum(episode.success for episode in episodes)
    hz = metadata["env_config"]["control_hz"]
    print(f"File {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"  source {metadata['source']}, created {metadata['created']}")
    print(f"  recorded with code version {metadata['git_commit']}")
    if metadata["git_commit"] != git_commit():
        print(f"  note: the current code is {git_commit()}, not the code that recorded the data")
    print(f"  noise_std {metadata.get('noise_std', 0.0)}, seeds {metadata.get('seeds')}")
    print(f"  demos: {len(episodes)}, successful: {successes} ({successes / len(episodes):.0%})")
    print(
        f"  length: {lengths.mean():.1f} +- {lengths.std():.1f} steps "
        f"(min {lengths.min()}, max {lengths.max()}) = {lengths.mean() / hz:.1f} s at {hz} Hz"
    )
    print(f"  observation keys: {metadata['obs_keys']}")

    actions = np.concatenate([episode.actions for episode in episodes])
    print(f"  actions: {len(actions)} steps in total")
    for i, name in enumerate(["dx", "dy", "dz"]):
        column = actions[:, i]
        saturated = np.mean(np.abs(column) > 0.99)
        print(
            f"    {name}: mean {column.mean():+.3f}  std {column.std():.3f}  "
            f"at the limit (|a| = 1) {saturated:.0%} of steps"
        )
    print(f"    gripper closed in {np.mean(actions[:, 3] > 0):.0%} of steps")


def render_from_states(env: PickPlaceEnv, episode, path: Path, label: str) -> None:
    """Re-render an episode from its stored simulation states, without simulating it."""
    with VideoWriter(path, fps=env.config.control_hz) as video:
        for t in range(episode.num_steps):
            env.data.qpos[:] = episode.qpos[t]
            env.data.qvel[:] = episode.qvel[t]
            mujoco.mj_forward(env.model, env.data)
            video.add(draw_text_lines(env.render(), [f"{label}   step {t}"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and verify a demonstration file.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--replay", type=int, default=5, help="re-simulate the first N demos")
    parser.add_argument("--video", type=int, default=0, help="re-render the first N demos")
    args = parser.parse_args()

    metadata = read_metadata(args.path)
    episodes = read_demos(args.path)
    print_summary(args.path, metadata, episodes)

    env = PickPlaceEnv(config_from_dict(PickPlaceConfig, metadata["env_config"]))
    if args.replay:
        print(f"\nReplay check (re-simulate {args.replay} demos from seed + executed actions):")
        for episode in episodes[: args.replay]:
            difference = replay_episode(env, episode)
            verdict = "OK, identical" if difference == 0.0 else "DIFFERENT"
            print(
                f"  seed {episode.seed:4d}: max observation difference {difference:.1e}  {verdict}"
            )
    for episode in episodes[: args.video]:
        path = VIDEO_DIR / f"replay_{args.path.stem}_seed{episode.seed:03d}.mp4"
        render_from_states(env, episode, path, f"{args.path.name}  seed {episode.seed}")
        print(f"  video: {path}")
    env.close()


if __name__ == "__main__":
    main()
