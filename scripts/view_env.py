"""Drive the pick-and-place environment with the keyboard (debugging, manual testing).

Two views side by side:
    left   "front":  you face the robot across the table, like a mirror (control view)
    right  "angled": shows depth and the fingers opening/closing (C switches it to "top")

Keys (click the window first; hold a key to keep moving):
    LEFT / RIGHT  or  A / D    move left / right (as seen in the front view)
    UP / DOWN     or  R / F    move up / down
    W / S                      move away from you / towards you
    SPACE                      close / open the gripper (tap it)
    P                          pause / resume (time stops while paused)
    N                          new episode (next seed)
    C                          right view: angled / top
    V                          start / stop recording the angled view to results/videos/
    ESC / Q                    quit

Usage (repository root, .venv activated):
    python scripts/view_env.py
    python scripts/view_env.py --max-steps 0      # practice without a time limit
    python scripts/view_env.py --seed 7 --config configs/env/pick_place.yaml
    python scripts/view_env.py --mujoco-viewer    # explore the scene in MuJoCo's own viewer
"""

import argparse
import dataclasses
import time
from dataclasses import dataclass, field
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
# Holding a key makes Windows repeat it; ignore a toggle key (SPACE, P) for this long after
# it fired, so one press toggles exactly once.
TOGGLE_COOLDOWN_S = 0.5
NO_LIMIT = 10**9  # "--max-steps 0": an episode limit that is never reached

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
    "P: pause   N: new   C: view   V: rec   ESC: quit"
)


@dataclass
class UiState:
    """Everything the window shows besides the simulation itself."""

    seed: int
    gripper_closed: bool = False
    paused: bool = False
    episode_over: bool = False
    recording: bool = False
    loop_hz: float = 20.0
    keys: set[str] = field(default_factory=set)
    last_toggle_time: dict[str, float] = field(default_factory=dict)

    def toggle_pressed(self, key: str, now: float) -> bool:
        """True once per press of a toggle key (auto-repeat of a held key is ignored)."""
        if (
            key not in self.keys
            or now - self.last_toggle_time.get(key, -np.inf) < TOGGLE_COOLDOWN_S
        ):
            return False
        self.last_toggle_time[key] = now
        return True


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


def hud_lines(env: PickPlaceEnv, obs: dict, info: dict, ui: UiState) -> list[str]:
    cfg = env.config
    tcp = obs["tcp_pos"]
    cube_to_target_cm = np.linalg.norm(obs["cube_to_target"][:2]) * 100
    elapsed_s = info["step"] / cfg.control_hz
    limit = (
        "no limit"
        if cfg.max_episode_steps >= NO_LIMIT
        else f"{cfg.max_episode_steps / cfg.control_hz:.0f} s"
    )
    too_slow = ui.loop_hz < 0.75 * cfg.control_hz
    lines = [
        f"seed {ui.seed}   time {elapsed_s:.1f} s / {limit}   loop {ui.loop_hz:.0f} Hz"
        + ("  SLOW (GPU?)" if too_slow else "")
        + ("   [REC]" if ui.recording else ""),
        f"TCP  x {tcp[0]:.3f}  y {tcp[1]:+.3f}  z {tcp[2]:.3f} m",
        f"gripper: {'CLOSED' if ui.gripper_closed else 'open'} "
        f"({obs['gripper_opening'][0] * 100:.1f} cm)   grasping: "
        f"{'yes' if info['is_grasping'] else 'no'}",
        f"cube -> target: {cube_to_target_cm:.1f} cm",
        f"keys: {describe_keys(ui.keys)}",
    ]
    if info["is_success"]:
        lines.append("SUCCESS!  press N for a new episode")
    elif ui.episode_over:
        lines.append("TIME OUT  press N for a new episode")
    elif ui.paused:
        lines.append("PAUSED  press P to resume")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard control of the pick-and-place env.")
    parser.add_argument("--config", type=Path, help="environment YAML (default: built-in values)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first episode")
    parser.add_argument(
        "--max-steps", type=int, help="episode limit in steps (0 = no limit; default: config)"
    )
    parser.add_argument("--mujoco-viewer", action="store_true", help="open MuJoCo's viewer")
    args = parser.parse_args()

    config = load_config(args.config, PickPlaceConfig) if args.config else PickPlaceConfig()
    if args.max_steps is not None:
        config = dataclasses.replace(config, max_episode_steps=args.max_steps or NO_LIMIT)
    env = PickPlaceEnv(config)
    ui = UiState(seed=args.seed, loop_hz=float(config.control_hz))
    obs, info = env.reset(seed=ui.seed)

    if args.mujoco_viewer:
        print("MuJoCo viewer: double-click a body, then Ctrl + right-drag to push it.")
        mujoco.viewer.launch(env.model, env.data)
        return

    print(__doc__)
    side_index = 0
    video: VideoWriter | None = None
    period_s = 1.0 / config.control_hz
    previous_tick = time.perf_counter()

    while True:
        tick_start = time.perf_counter()
        # Smoothed loop rate (shown in the HUD; below 20 Hz the simulation runs slow-motion).
        ui.loop_hz = 0.9 * ui.loop_hz + 0.1 / max(tick_start - previous_tick, 1e-6)
        previous_tick = tick_start
        # 1) React to the keys pressed during the previous control period.
        keys = ui.keys
        if "esc" in keys or "q" in keys:
            break
        if "n" in keys:
            ui.seed += 1
            obs, info = env.reset(seed=ui.seed)
            ui.gripper_closed = False
            ui.episode_over = False
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
            ui.recording = video is not None
        if ui.toggle_pressed(" ", tick_start):
            ui.gripper_closed = not ui.gripper_closed
        if ui.toggle_pressed("p", tick_start):
            ui.paused = not ui.paused

        if not ui.episode_over and not ui.paused:
            move = sum((MOVE_KEYS[k] for k in keys if k in MOVE_KEYS), start=np.zeros(3))
            action = np.append(np.clip(move, -1.0, 1.0), 1.0 if ui.gripper_closed else -1.0)
            obs, _, terminated, truncated, info = env.step(action)
            ui.episode_over = terminated or truncated

        # 2) Show both views; record the angled view without text overlays.
        side_camera = SIDE_CAMERAS[side_index]
        control_view = env.render_camera(CONTROL_CAMERA)
        side_view = env.render_camera(side_camera)
        if video is not None:
            video.add(side_view if side_camera == VIDEO_CAMERA else env.render_camera(VIDEO_CAMERA))

        lines = hud_lines(env, obs, info, ui)
        color = (80, 255, 80) if info["is_success"] else (255, 255, 255)
        control_view = draw_text_lines(control_view, lines, color=color)
        side_view = draw_text_lines(side_view, [f"view: {side_camera}  (C to switch)"])
        window_image = np.hstack([control_view, side_view])
        window_image = draw_text_lines(
            window_image, [HELP_LINE], origin=(10, window_image.shape[0] - 12)
        )
        cv2.imshow(WINDOW, cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))

        # 3) Wait for the rest of the control period, collecting key presses.
        ui.keys = collect_keys(period_s - (time.perf_counter() - tick_start))
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    if video is not None:
        video.close()
        print(f"Saved {video.path} ({video.n_frames} frames)")
    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
