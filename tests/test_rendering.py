import numpy as np

from gesture_bc.sim.rendering import VideoWriter, draw_text_lines


def test_video_writer_creates_an_mp4(tmp_path):
    path = tmp_path / "videos" / "test.mp4"  # parent folder does not exist yet
    with VideoWriter(path, fps=20) as video:
        for i in range(10):
            video.add(np.full((480, 640, 3), i * 20, dtype=np.uint8))
    assert video.n_frames == 10
    assert path.stat().st_size > 0


def test_text_overlay_does_not_modify_the_input():
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    out = draw_text_lines(frame, ["hello"])
    assert frame.max() == 0 and out.max() > 0
