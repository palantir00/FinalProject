"""Hand features for teleoperation, computed from MediaPipe's 21 hand landmarks.

Landmark numbering (MediaPipe Hands):
    0 wrist;  1-4 thumb (4 = tip);  5-8 index finger (5 = base joint, 8 = tip);
    9-12 middle finger;  13-16 ring finger;  17-20 little finger.
Each landmark is (x, y, z): x and y are normalized by the image width and height (0..1,
y pointing down); z is the depth relative to the wrist, on roughly the same scale as x
(MediaPipe documentation).

Three features drive the robot:
    position  palm center in the image          -> robot left/right and up/down
    size      apparent palm size                -> robot towards/away (closer hand = larger)
    pinch     thumb tip to index tip, relative  -> gripper
"""

from dataclasses import dataclass

import numpy as np

WRIST, THUMB_TIP, INDEX_BASE, INDEX_TIP = 0, 4, 5, 8
MIDDLE_BASE, RING_BASE, LITTLE_BASE = 9, 13, 17
# Landmarks of the palm: they do not move when the fingers move (e.g. when pinching).
PALM = [WRIST, INDEX_BASE, MIDDLE_BASE, RING_BASE, LITTLE_BASE]

# Bones of the hand, for drawing the skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # little finger and palm edge
]  # fmt: skip


@dataclass(frozen=True)
class HandFeatures:
    position: np.ndarray  # (2,) palm center: x to the right, y down [image widths]
    size: float  # palm size [image widths]; grows as the hand comes closer to the camera
    pinch: float  # thumb-index tip distance / palm size: about 0.1 pinched, about 1 open


def to_image_widths(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Express x, y and z in one unit (image widths), so that distances are not distorted.

    Normalized y is a fraction of the image HEIGHT; on a 640x480 image one unit of y is only
    0.75 of a unit of x.
    """
    return landmarks * np.array([1.0, height / width, 1.0])


def extract_features(landmarks: np.ndarray, width: int, height: int) -> HandFeatures:
    """Features of one hand from its (21, 3) normalized landmarks."""
    points = to_image_widths(landmarks, width, height)
    position = points[PALM, :2].mean(axis=0)
    # Palm size: mean side of the triangle wrist - index base - little-finger base. Measured
    # in 3D (with z), so tilting the hand changes it much less than a length in the image.
    size = np.mean(
        [
            np.linalg.norm(points[WRIST] - points[INDEX_BASE]),
            np.linalg.norm(points[WRIST] - points[LITTLE_BASE]),
            np.linalg.norm(points[INDEX_BASE] - points[LITTLE_BASE]),
        ]
    )
    # Relative to the palm size, so the same pinch gives the same value near and far.
    pinch = np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP]) / size
    return HandFeatures(position=position, size=float(size), pinch=float(pinch))
