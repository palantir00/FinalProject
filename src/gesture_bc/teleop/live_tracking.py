"""Webcam + hand tracking in a background thread; the control loop reads the newest result.

Why a separate thread: the robot is controlled at a fixed 20 Hz (every 50 ms), while reading
a camera frame takes ~12 ms and finding the hand ~22 ms, sometimes 60 ms or more. In one
loop a slow frame would delay the robot. In a thread the camera runs at its own pace, and the
control loop simply takes the newest hand (it never waits for the camera).
OpenCV and MediaPipe release Python's GIL while they compute, so both really run in parallel.
"""

import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from gesture_bc.teleop.hand_features import HandFeatures, extract_features
from gesture_bc.teleop.hand_tracker import HandTracker
from gesture_bc.teleop.one_euro import OneEuroFilter


@dataclass(frozen=True)
class HandSample:
    """Result for one camera frame."""

    time_s: float  # when the frame was read (time.perf_counter)
    frame: np.ndarray  # mirrored RGB image, for display
    landmarks: np.ndarray | None  # (21, 3) normalized, None if no hand was found
    features: HandFeatures | None  # filtered position and size, raw pinch
    detect_ms: float  # time MediaPipe needed for this frame
    fps: float  # frames processed per second (smoothed)


class LiveHandTracking:
    def __init__(
        self,
        camera_index: int,
        tracker: HandTracker,
        position_filter: OneEuroFilter,
        lost_timeout_s: float,
    ):
        # On Windows the DirectShow backend opens most webcams much faster than the default.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(camera_index, backend)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera_index}. Windows: Settings > Privacy & security"
                " > Camera > allow desktop apps."
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._tracker = tracker
        self._filter = position_filter
        self._lost_timeout_s = lost_timeout_s
        self._latest: HandSample | None = None
        self._lock = threading.Lock()  # guards _latest: written by the thread, read by main
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._capture.release()
        self._tracker.close()

    def latest(self) -> HandSample | None:
        """Newest result (None before the first frame)."""
        with self._lock:
            return self._latest

    def _run(self) -> None:
        last_seen_s, fps, previous_s = -np.inf, 0.0, time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - previous_s, 1e-6)
            previous_s = now
            # Mirror the image: moving the hand to the right moves it to the right on screen.
            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]

            start = time.perf_counter()
            landmarks = self._tracker.detect(rgb, now)
            detect_ms = (time.perf_counter() - start) * 1000

            features = None
            if landmarks is not None:
                raw = extract_features(landmarks, width, height)
                if now - last_seen_s > self._lost_timeout_s:
                    self._filter.reset()  # the hand is back: do not smooth from an old position
                last_seen_s = now
                smooth = self._filter(np.append(raw.position, raw.size), now)
                features = HandFeatures(position=smooth[:2], size=float(smooth[2]), pinch=raw.pinch)

            sample = HandSample(now, rgb, landmarks, features, detect_ms, fps)
            with self._lock:
                self._latest = sample
