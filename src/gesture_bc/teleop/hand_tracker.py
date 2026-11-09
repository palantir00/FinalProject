"""MediaPipe Hand Landmarker: 21 landmarks of one hand in an RGB image."""

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from gesture_bc.teleop.paths import HAND_MODEL


class HandTracker:
    """Finds one hand per image.

    VIDEO mode: every call processes one frame and returns its result (no callbacks). Between
    frames MediaPipe tracks the hand instead of searching the whole image again, which is
    faster; that is why each frame needs a timestamp.
    """

    def __init__(self, model_path: Path = HAND_MODEL):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand model not found: {model_path}\n"
                "Download it with:  python scripts/fetch_assets.py"
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, rgb: np.ndarray, time_s: float) -> np.ndarray | None:
        """(21, 3) normalized landmarks of the hand in the RGB image, or None if there is none."""
        # MediaPipe requires strictly increasing timestamps in whole milliseconds.
        timestamp_ms = max(int(time_s * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        return np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])

    def close(self) -> None:
        self._landmarker.close()
