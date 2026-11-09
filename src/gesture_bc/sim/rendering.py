"""Video files and text overlays for simulation frames."""

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


class VideoWriter:
    """Write RGB frames (H, W, 3, uint8) to an H.264 MP4 file that also plays in browsers.

    Usage:
        with VideoWriter("results/videos/episode.mp4", fps=20) as video:
            video.add(env.render())
    """

    def __init__(self, path: str | Path, fps: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # imageio uses the ffmpeg binary bundled with the imageio-ffmpeg package.
        self._writer = imageio.get_writer(self.path, fps=fps, codec="libx264", quality=8)
        self.n_frames = 0

    def add(self, frame: np.ndarray) -> None:
        self._writer.append_data(frame)
        self.n_frames += 1

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def draw_text_lines(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (10, 24),
    color: tuple[int, int, int] = (255, 255, 255),
    line_height: int = 22,
) -> np.ndarray:
    """Return a copy of the frame with text lines drawn on a dark band (readable on any scene)."""
    out = frame.copy()
    x, y = origin
    for i, line in enumerate(lines):
        baseline = y + i * line_height
        (width, height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(
            out, (x - 4, baseline - height - 5), (x + width + 4, baseline + 5), (0, 0, 0), -1
        )
        cv2.putText(out, line, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out
