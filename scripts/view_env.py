"""Drive the pick-and-place environment with the keyboard (debugging, manual testing).

Two views side by side:
    left   "front":  you face the robot across the table, like a mirror (control view)
    right  "angled": shows depth and the fingers opening/closing (C switches it to "top")

Keys (click the window first; hold a key to keep moving):
    LEFT / RIGHT  or  A / D    move left / right (as seen in the front view)
    UP / DOWN     or  R / F    move up / down
    W / S                      move away from you / towards you
    SPACE                      close / open the gripper (tap it)
    N                          new episode (next seed)
    C                          right view: angled / top
    V                          start / stop recording the angled view to results/videos/
    ESC / Q                    quit

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
CONTROL_CAMERA = "front"
SIDE_CAMERAS = ["angled", "top"]
VIDEO_CAMERA = "angled"
VIDEO_DIR = Path("results/videos")
ESC = 27
# Holding a key makes Windows repeat it; ignore SPACE for this long after a toggle,
# so one press toggles the gripper exactly once.
GRIPPER_TOGGLE_COOLDOWN_S = 0.5

# cv2.waitKeyEx returns the full key code; arrow-key codes differ between platforms.
ARROW_KEYS = {
    2424832: "left",  # Windows
    2490368: "up",
    2555904: "right",
    2621440: "down",
    65361: "left",  # Linux (GTK)
    65362: "up",
    65363: "right",
    65364: "down",
}

# Key -> direction of the TCP in world coordinates (x away from the robot base, y to the
# robot's left, z up). In the front view +y is screen right and -x is "into the screen".
MOVE_KEYS = {
    "left": np.array([0.0, -1.0, 0.0]),
    "a": np.array([0.0, -1.0, 0.0]),
    "right": np.array([0.0, 1.0, 0.0]),
    "d": np.array([0.0, 1.0, 0.0]),
    "up": np.array([0.0, 0.0, 1.0]),
    "r": np.array([0.0, 0.0, 1.0]),
    "down": np.array([0.0, 0.0, -1.0]),
    "f": np.array([0.0, 0.0, -1.0]),
    "w": np.array([-1.0, 0.0, 0.0]),
    "s": np.array([1.0, 0.0, 0.0]),
}
HELP_LINE = (
    "ARROWS or A/D R/F: left/right up/down   W/S: away/towards   SPACE: grip   "
    "N: new   C: view   V: rec   ESC: quit"
)


def collect_keys(duration_s: float) -> set[str]:
    """Names of the keys pressed during the next duration_s seconds.

    Reading keys for the whole control period keeps the loop at a fixed rate and empties
    the queue of auto-repeat events, so the robot stops as soon as a key is released.
    Letters are lowercase; special keys are "left", "right", "up", "down", "esc", " ".
    """
    keys = set()
    end = time.perf_counter() + duration_s
    while True:
        # Always wait at least 1 ms, even if rendering already used up the whole period:
        # waitKeyEx is also where the window processes its events. A window that never
        # gets here freezes and Windows reports it as "Not responding".
        remaining_ms = max(1, int((end - time.perf_counter()) * 1000))
        code = cv2.waitKeyEx(remaining_ms)
        if code in ARROW_KEYS:
            keys.add(ARROW_KEYS[code])
        elif code != -1:
            keys.add("esc" if code & 0xFF == ESC else chr(code & 0xFF).lower())
        if time.perf_counter() >= end:
            return keys


def describe_keys(keys: set[str]) -> str:
    if not keys:
        return "-   (nothing happens? click the window first)"
    return ", ".join(sorted("space" if key == " " else key for key in keys))


def hud_lines(
    env, obs, info, seed, keys, gripper_closed, recording, episode_over, loop_hz
) -> list[str]:
    tcp = obs["tcp_pos"]
    cube_to_target_cm = np.linalg.norm(obs["cube_to_target"][:2]) * 100
    too_slow = loop_hz < 0.75 * env.config.control_hz
    lines = [
        f"seed {seed}   step {info['step']}/{env.config.max_episode_steps}   "
        f"loop {loop_hz:.0f} Hz"
        + ("  SLOW (GPU?)" if too_slow else "")
        + ("   [REC]" if recording else ""),
        f"TCP  x {tcp[0]:.3f}  y {tcp[1]:+.3f}  z {tcp[2]:.3f} m",
        f"gripper: {'CLOSED' if gripper_closed else 'open'} "
        f"({obs['gripper_opening'][0] * 100:.1f} cm)   grasping: "
        f"{'yes' if info['is_grasping'] else 'no'}",
        f"cube -> target: {cube_to_target_cm:.1f} cm",
        f"keys: {describe_keys(keys)}",
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
    side_index = 0
    gripper_closed = False
    last_gripper_toggle = -np.inf
    episode_over = False
    video: VideoWriter | None = None
    keys: set[str] = set()
    period_s = 1.0 / config.control_hz
    loop_hz = float(config.control_hz)
    previous_tick = time.perf_counter()

    while True:
        tick_start = time.perf_counter()
        # Smoothed loop rate (shown in the HUD; below 20 Hz the simulation runs slow-motion).
        loop_hz = 0.9 * loop_hz + 0.1 / max(tick_start - previous_tick, 1e-6)
        previous_tick = tick_start
        # 1) React to the keys pressed during the previous control period.
        if "esc" in keys or "q" in keys:
            break
        if "n" in keys:
            seed += 1
            obs, info = env.reset(seed=seed)
            gripper_closed = False
            episode_over = False
        if "c" in keys:
            side_index = (side_index + 1) % len(SIDE_CAMERAS)
        if "v" in keys:
            if video is None:
                path = VIDEO_DIR / f"keyboard_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                video = VideoWriter(path, fps=config.control_hz)
                print(f"Recording to {path}")
            else:
                video.close()
                print(f"Saved {video.path} ({video.n_frames} frames)")
                video = None
        if " " in keys and tick_start - last_gripper_toggle > GRIPPER_TOGGLE_COOLDOWN_S:
            gripper_closed = not gripper_closed
            last_gripper_toggle = tick_start

        if not episode_over:
            move = sum((MOVE_KEYS[k] for k in keys if k in MOVE_KEYS), start=np.zeros(3))
            action = np.append(np.clip(move, -1.0, 1.0), 1.0 if gripper_closed else -1.0)
            obs, _, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

        # 2) Show both views; record the angled view without text overlays.
        side_camera = SIDE_CAMERAS[side_index]
        control_view = env.render_camera(CONTROL_CAMERA)
        side_view = env.render_camera(side_camera)
        if video is not None:
            video.add(side_view if side_camera == VIDEO_CAMERA else env.render_camera(VIDEO_CAMERA))

        lines = hud_lines(
            env, obs, info, seed, keys, gripper_closed, video is not None, episode_over, loop_hz
        )
        color = (80, 255, 80) if info["is_success"] else (255, 255, 255)
        control_view = draw_text_lines(control_view, lines, color=color)
        side_view = draw_text_lines(side_view, [f"view: {side_camera}  (C to switch)"])
        window_image = np.hstack([control_view, side_view])
        window_image = draw_text_lines(
            window_image, [HELP_LINE], origin=(10, window_image.shape[0] - 12)
        )
        cv2.imshow(WINDOW, cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))

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
