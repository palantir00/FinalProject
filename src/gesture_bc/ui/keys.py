"""Read keyboard input in an OpenCV window at a fixed loop rate."""

import time

import cv2

ESC = 27

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


def key_name(code: int) -> str:
    """Lowercase letter, " " for space, or "left"/"right"/"up"/"down"/"esc"."""
    if code in ARROW_KEYS:
        return ARROW_KEYS[code]
    return "esc" if code & 0xFF == ESC else chr(code & 0xFF).lower()


def collect_keys(duration_s: float) -> set[str]:
    """Names of the keys pressed during the next duration_s seconds.

    Reading keys for the whole control period keeps the loop at a fixed rate and empties
    the queue of auto-repeat events, so motion stops as soon as a key is released.
    """
    keys = set()
    end = time.perf_counter() + duration_s
    while True:
        # Always wait at least 1 ms, even if rendering already used up the whole period:
        # waitKeyEx is also where the window processes its events. A window that never
        # gets here freezes and Windows reports it as "Not responding".
        remaining_ms = max(1, int((end - time.perf_counter()) * 1000))
        code = cv2.waitKeyEx(remaining_ms)
        if code != -1:
            keys.add(key_name(code))
        if time.perf_counter() >= end:
            return keys


def window_closed(window: str) -> bool:
    return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
