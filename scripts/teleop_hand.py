"""Live hand teleoperation (preview; nothing is recorded): your hand drives the robot.

Three views: the webcam (left), the robot from the front (middle) and from an angle (right).
The webcam image is mirrored, like the front view: move the hand to the right and the
gripper moves to the right on screen.

How to use:
    1. Hold your open hand, palm towards the camera, about 50 cm away, in the middle of the
       image. Press C: this hand position now corresponds to the current gripper position.
    2. Move the hand: left/right and up/down move the gripper the same way; bringing the hand
       closer to the camera moves the gripper towards you (see the angled view).
    3. Pinch (thumb tip to index tip) to close the gripper; open the fingers to release.

Keys (click the window first):
    C        calibrate / re-center: the robot stays, the hand's current place becomes the center
    N        next scene
    P        pause / resume
    ESC / Q  quit

Usage (repository root, .venv activated):
    python scripts/teleop_hand.py
    python scripts/teleop_hand.py --max-steps 0     # practice without a time limit
    python scripts/teleop_hand.py --camera 1        # another camera
"""

import argparse
import dataclasses
import time
from pathlib import Path

import cv2
import numpy as np

from gesture_bc.sim.pick_place_env import PickPlaceConfig, PickPlaceEnv
from gesture_bc.sim.rendering import draw_text_lines
from gesture_bc.teleop.hand_features import HAND_CONNECTIONS
from gesture_bc.teleop.hand_teleop import HandTeleop, HandTeleopConfig
from gesture_bc.teleop.hand_tracker import HandTracker
from gesture_bc.teleop.live_tracking import HandSample, LiveHandTracking
from gesture_bc.teleop.one_euro import OneEuroFilter
from gesture_bc.ui.keys import collect_keys, window_closed
from gesture_bc.utils.config import load_config

WINDOW = "gesture_bc: hand teleoperation"
PANEL_SIZE = (426, 320)  # width, height of each of the three views (4:3, like the webcam)
PAUSE_AFTER_EPISODE_S = 1.5
NO_LIMIT = 10**9  # "--max-steps 0": an episode limit that is never reached
# Text colors (RGB)
WHITE, GREEN, YELLOW, ORANGE = (255, 255, 255), (80, 255, 80), (255, 230, 80), (255, 150, 50)


def draw_hand(image: np.ndarray, landmarks: np.ndarray, color: tuple[int, int, int]) -> None:
    """Draw the hand skeleton (landmarks normalized to 0..1) onto the image, in place."""
    height, width = image.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, points[a], points[b], color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)


def to_pixels(position: np.ndarray, image: np.ndarray) -> tuple[int, int]:
    """Palm position in image widths (x and y) -> pixel coordinates of the image."""
    width = image.shape[1]
    return int(position[0] * width), int(position[1] * width)


def camera_view(sample: HandSample | None, hand_visible: bool, teleop: HandTeleop) -> np.ndarray:
    if sample is None:
        blank = np.zeros((PANEL_SIZE[1], PANEL_SIZE[0], 3), np.uint8)
        return draw_text_lines(blank, ["starting the camera ..."])
    image = cv2.resize(sample.frame, PANEL_SIZE, interpolation=cv2.INTER_AREA)
    if sample.landmarks is not None:
        draw_hand(image, sample.landmarks, GREEN if hand_visible else ORANGE)
    if teleop.calibrated and sample.features is not None:
        # Calibration center (circle) and the palm now (dot): their offset moves the robot.
        center = to_pixels(teleop.reference[0].position, image)
        palm = to_pixels(sample.features.position, image)
        cv2.circle(image, center, 10, YELLOW, 2, cv2.LINE_AA)
        cv2.line(image, center, palm, YELLOW, 1, cv2.LINE_AA)
        cv2.circle(image, palm, 5, YELLOW, -1, cv2.LINE_AA)

    if not teleop.calibrated:
        status, color = "open hand, palm to camera, press C", YELLOW
    elif not hand_visible:
        status, color = "hand not visible: robot holds", ORANGE
    else:
        gripper = "CLOSED" if teleop.gripper_closed else "open"
        status, color = f"pinch {sample.features.pinch:.2f} -> gripper {gripper}", GREEN
    lines = [status, f"camera {sample.fps:.0f} FPS   hand found in {sample.detect_ms:.0f} ms"]
    return draw_text_lines(image, lines, color=color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live hand teleoperation (no recording).")
    parser.add_argument("--config", type=Path, default=Path("configs/teleop/hand.yaml"))
    parser.add_argument("--camera", type=int, help="camera index (default: from the config)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first scene")
    parser.add_argument(
        "--max-steps", type=int, help="episode limit in steps (0 = no limit; default: config)"
    )
    args = parser.parse_args()

    env_config = load_config(Path("configs/env/pick_place.yaml"), PickPlaceConfig)
    if args.max_steps is not None:
        env_config = dataclasses.replace(env_config, max_episode_steps=args.max_steps or NO_LIMIT)
    cfg = load_config(args.config, HandTeleopConfig)
    env = PickPlaceEnv(env_config)
    teleop = HandTeleop(
        cfg, env_config.workspace_low, env_config.workspace_high, env_config.max_step_m
    )
    tracking = LiveHandTracking(
        args.camera if args.camera is not None else cfg.camera_index,
        HandTracker(),
        OneEuroFilter(cfg.filter_min_cutoff_hz, cfg.filter_beta, cfg.filter_d_cutoff_hz),
        cfg.lost_timeout_s,
    )
    tracking.start()

    seed = args.seed
    obs, info = env.reset(seed=seed)
    scenes, successes = 0, 0
    paused, keys, last_toggle = False, set(), 0.0
    period_s = 1.0 / env_config.control_hz
    loop_hz, previous_tick = float(env_config.control_hz), time.perf_counter()
    limit = (
        "no limit"
        if env_config.max_episode_steps >= NO_LIMIT
        else f"{env_config.max_episode_steps * period_s:.0f} s"
    )
    print(__doc__)
    try:
        while True:
            tick = time.perf_counter()
            loop_hz = 0.9 * loop_hz + 0.1 / max(tick - previous_tick, 1e-6)
            previous_tick = tick
            if "esc" in keys or "q" in keys:
                break

            # 1) The newest hand; too old counts as "not visible".
            sample = tracking.latest()
            features = None
            if sample is not None and tick - sample.time_s < cfg.lost_timeout_s:
                features = sample.features
            hand_visible = features is not None

            # 2) Keys pressed during the previous control period.
            if "c" in keys and hand_visible:
                teleop.calibrate(features, obs["tcp_pos"])
            if "p" in keys and tick - last_toggle > 0.5:
                paused, last_toggle = not paused, tick
            if "n" in keys:
                seed += 1
                obs, info = env.reset(seed=seed)
                teleop.reset()

            # 3) One control step.
            if not paused:
                obs, _, terminated, truncated, info = env.step(teleop.act(obs, features))
                if terminated or truncated:
                    scenes += 1
                    successes += int(info["is_success"])

            # 4) Show the three views.
            result = ""
            if info["is_success"]:
                result = "SUCCESS"
            elif info["step"] >= env_config.max_episode_steps:
                result = "time out"
            front_lines = [
                f"scene {seed}   t {info['step'] * period_s:4.1f} s / {limit}   {result}",
                f"solved {successes} / {scenes}   loop {loop_hz:.0f} Hz"
                + ("   PAUSED" if paused else ""),
                "C calibrate  N next  P pause  ESC quit",
            ]
            tcp = obs["tcp_pos"]
            angled_lines = [
                f"TCP  x {tcp[0]:.2f}  y {tcp[1]:+.2f}  z {tcp[2]:.2f} m",
                f"grasping: {'yes' if info['is_grasping'] else 'no'}   cube -> target "
                f"{np.linalg.norm(obs['cube_to_target'][:2]) * 100:.1f} cm",
            ]
            views = [
                camera_view(sample, hand_visible, teleop),
                draw_text_lines(
                    cv2.resize(
                        env.render_camera("front"), PANEL_SIZE, interpolation=cv2.INTER_AREA
                    ),
                    front_lines,
                    color=GREEN if info["is_success"] else WHITE,
                ),
                draw_text_lines(
                    cv2.resize(
                        env.render_camera("angled"), PANEL_SIZE, interpolation=cv2.INTER_AREA
                    ),
                    angled_lines,
                ),
            ]
            cv2.imshow(WINDOW, cv2.cvtColor(np.hstack(views), cv2.COLOR_RGB2BGR))

            if result and not paused:  # show the result for a moment, then the next scene
                keys = collect_keys(PAUSE_AFTER_EPISODE_S)
                seed += 1
                obs, info = env.reset(seed=seed)
                teleop.reset()
            else:
                keys = collect_keys(period_s - (time.perf_counter() - tick))
            if window_closed(WINDOW):
                break
    finally:
        tracking.stop()
        cv2.destroyAllWindows()
        env.close()
    print(f"Solved {successes} of {scenes} scenes.")


if __name__ == "__main__":
    main()
