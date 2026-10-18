"""Drive the pick-and-place environment with the keyboard (debugging, manual testing).

The window starts with the "front" camera: you look at the robot across the table, like in
a mirror. Keys (click the window first; hold a key to keep moving):
    W / S    move the gripper away from you / towards you
    A / D    move left / right (as seen on the screen)
    R / F    move up / down
    SPACE    close / open the gripper
    N        new episode (next seed)
    C        switch camera (front, angled, top)
    V        start / stop recording a video to results/videos/
    ESC / Q  quit

Usage (repository root, .venv activated):
    python scripts/view_env.py
    python scripts/view_env.py --seed 7 --config configs/env/pick_place.yaml
    python scripts/view_env.py --mujoco-viewer    # explore the scene in MuJoCo's own viewer
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from gesture_bc.sim.pick_place_env import PickPlaceConfig, PickPlaceEnv
from gesture_bc.sim.rendering import VideoWriter, draw_text_lines
from gesture_bc.utils.config import load_config

WINDOW = "gesture_bc: pick and place (keyboard)"
CAMERAS = ["front", "angled", "top"]
VIDEO_DIR = Path("results/videos")
ESC = 27

# Key -> direction of the TCP in world coordinates (x away from the robot base, y to the
# robot's left, z up). In the front view +y is screen right and -x is "into the screen".
MOVE_KEYS = {
    "w": np.array([-1.0, 0.0, 0.0]),
    "s": np.array([1.0, 0.0, 0.0]),
    "a": np.array([0.0, -1.0, 0.0]),
    "d": np.array([0.0, 1.0, 0.0]),
    "r": np.array([0.0, 0.0, 1.0]),
    "f": np.array([0.0, 0.0, -1.0]),
}
HELP_LINE = "W/S in/out  A/D left/right  R/F up/down  SPACE grip  N new  C camera  V rec  ESC quit"


def collect_keys(duration_s: float) -> set[str]:
    """Keys pressed during the next duration_s seconds (lowercase; ESC as "esc").

    Reading keys for the whole control period keeps the loop at a fixed rate and empties
    the queue of auto-repeat events, so the robot stops as soon as a key is released.
    """
    keys = set()
    end = time.perf_counter() + duration_s
    while (remaining_ms := int((end - time.perf_counter()) * 1000)) > 0:
        code = cv2.waitKey(remaining_ms)
        if code == -1:
            continue
        code &= 0xFF
        keys.add("esc" if code == ESC else chr(code).lower())
    return keys


def hud_lines(env, obs, info, seed, camera, gripper_closed, recording, episode_over) -> list[str]:
    tcp = obs["tcp_pos"]
    cube_to_target_cm = np.linalg.norm(obs["cube_to_target"][:2]) * 100
    lines = [
        f"seed {seed}   step {info['step']}/{env.config.max_episode_steps}   camera: {camera}"
        + ("   [REC]" if recording else ""),
        f"TCP  x {tcp[0]:.3f}  y {tcp[1]:+.3f}  z {tcp[2]:.3f} m",
        f"gripper: {'CLOSED' if gripper_closed else 'open'} "
        f"({obs['gripper_opening'][0] * 100:.1f} cm)   grasping: "
        f"{'yes' if info['is_grasping'] else 'no'}",
        f"cube -> target: {cube_to_target_cm:.1f} cm",
    ]
    if info["is_success"]:
        lines.append("SUCCESS!  press N for a new episode")
    elif episode_over:
        lines.append("TIME OUT  press N for a new episode")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard control of the pick-and-place env.")
    parser.add_argument("--config", type=Path, help="environment YAML (default: built-in values)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first episode")
    parser.add_argument("--mujoco-viewer", action="store_true", help="open MuJoCo's viewer")
    args = parser.parse_args()

    config = load_config(args.config, PickPlaceConfig) if args.config else PickPlaceConfig()
    env = PickPlaceEnv(config)
    seed = args.seed
    obs, info = env.reset(seed=seed)

    if args.mujoco_viewer:
        print("MuJoCo viewer: double-click a body, then Ctrl + right-drag to push it.")
        mujoco.viewer.launch(env.model, env.data)
        return

    print(__doc__)
    camera_index = 0
    gripper_closed = False
    episode_over = False
    video: VideoWriter | None = None
    keys: set[str] = set()
    period_s = 1.0 / config.control_hz

    while True:
        tick_start = time.perf_counter()
        # 1) React to the keys pressed during the previous control period.
        if "esc" in keys or "q" in keys:
            break
        if "n" in keys:
            seed += 1
            obs, info = env.reset(seed=seed)
            gripper_closed = False
            episode_over = False
        if "c" in keys:
            camera_index = (camera_index + 1) % len(CAMERAS)
        if "v" in keys:
            if video is None:
                path = VIDEO_DIR / f"keyboard_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                video = VideoWriter(path, fps=config.control_hz)
                print(f"Recording to {path}")
            else:
                video.close()
                print(f"Saved {video.path} ({video.n_frames} frames)")
                video = None
        if " " in keys:
            gripper_closed = not gripper_closed

        if not episode_over:
            move = sum((MOVE_KEYS[k] for k in keys if k in MOVE_KEYS), start=np.zeros(3))
            action = np.append(np.clip(move, -1.0, 1.0), 1.0 if gripper_closed else -1.0)
            obs, _, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

        # 2) Show the scene (and record it without the text overlay).
        camera = CAMERAS[camera_index]
        frame = env.render_camera(camera)
        if video is not None:
            video.add(frame)
        lines = hud_lines(
            env, obs, info, seed, camera, gripper_closed, video is not None, episode_over
        )
        color = (80, 255, 80) if info["is_success"] else (255, 255, 255)
        shown = draw_text_lines(frame, lines, color=color)
        shown = draw_text_lines(shown, [HELP_LINE], origin=(10, shown.shape[0] - 12))
        cv2.imshow(WINDOW, cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))

        # 3) Wait for the rest of the control period, collecting key presses.
        keys = collect_keys(period_s - (time.perf_counter() - tick_start))
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    if video is not None:
        video.close()
        print(f"Saved {video.path} ({video.n_frames} frames)")
    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
